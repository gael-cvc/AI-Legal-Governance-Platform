"""
=============================================================================
embedder.py — Encodage des chunks en vecteurs avec sentence-transformers
=============================================================================

RÔLE DE CE FICHIER :
Transformer chaque chunk de texte en un vecteur numérique (embedding).

DEFINITION EMBEDDING:
Un embedding = une liste de nombres (ex: [0.23, -0.81, 0.45, ...]) qui
représente le SENS du texte dans un espace mathématique à 384 dimensions.

Deux phrases qui ont le même sens auront des vecteurs proches.
Deux phrases qui parlent de sujets différents auront des vecteurs éloignés.

EXEMPLE :
"Le responsable de traitement doit informer les personnes"
→ [0.23, -0.81, 0.45, 0.12, ...]   (384 nombres)

"The data controller must notify individuals"
→ [0.24, -0.79, 0.46, 0.11, ...]   (très proche ! même sens, langue différente)

"La météo est agréable aujourd'hui"
→ [-0.55, 0.32, -0.12, 0.88, ...]  (très différent, sujet différent)

POURQUOI sentence-transformers ?
C'est la librairie open-source de référence pour les embeddings de texte.
On utilise le modèle "all-MiniLM-L6-v2" :
- Rapide : encode 2000 chunks en ~30 secondes sur CPU
- Léger : 80 Mo sur disque
- Performant : très bon rapport qualité/vitesse pour les textes juridiques courts
- Gratuit : pas d'API, tout tourne en local

FLUX DE DONNÉES :
    list[str] (textes)  →  [encode()]  →  np.ndarray (vecteurs 384D)

=============================================================================
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

# SentenceTransformer est la classe principale de la librairie sentence-transformers.
# Elle charge un modèle pré-entraîné et expose une méthode .encode() pour vectoriser.
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.embedder")

# ── Modèle choisi ──────────────────────────────────────────────────────────────
# "all-MiniLM-L6-v2" est un excellent modèle pour notre cas :
# - Dimension : 384 (compact, rapide à chercher dans FAISS)
# - Contexte max : 256 tokens (~1000 chars) — nos chunks sont calibrés pour ça
# - Score MTEB : 59.9 (très bon pour un modèle léger)
# - Taille : 80 Mo (se télécharge une seule fois, mis en cache automatiquement)
#
# Alternative plus puissante pour plus de précision (mais plus lent) :
# "BAAI/bge-large-en-v1.5" → dimension 1024, contexte 512 tokens

MODEL_NAME = "all-MiniLM-L6-v2"


class LegalEmbedder:
    """
    Encodeur de textes juridiques basé sur sentence-transformers.

    Encapsule le modèle d'embedding et expose des méthodes claires
    pour encoder des textes individuels ou des batches.

    PATTERN "SINGLETON LAZY" :
    Le modèle n'est chargé en mémoire qu'une seule fois, à la première utilisation.
    Charger un modèle prend 2-3 secondes — on ne veut pas le faire à chaque requête.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialise l'encodeur SANS charger le modèle tout de suite.
        Le modèle est chargé au premier appel à encode() ou à load().

        Paramètre :
            model_name : nom du modèle HuggingFace à utiliser
        """
        self.model_name = model_name
        # _model est None jusqu'au premier chargement (lazy loading)
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """
        Charge le modèle en mémoire.
        Télécharge automatiquement depuis HuggingFace si pas encore en cache.
        Le cache est dans ~/.cache/huggingface/ — le téléchargement ne se fait qu'une fois.
        """
        if self._model is not None:
            return  # Déjà chargé, rien à faire

        logger.info(f"Chargement du modèle d'embedding : {self.model_name}")
        t0 = time.perf_counter()

        # SentenceTransformer() télécharge et charge le modèle.
        # device="cpu" = on utilise le CPU (pas de GPU nécessaire pour ce projet).
        # Sur Mac M1/M2, on pourrait utiliser device="mps" pour accélérer,
        # mais "cpu" est universel et suffisant pour 2000 chunks.
        self._model = SentenceTransformer(self.model_name, device="cpu")

        elapsed = time.perf_counter() - t0
        logger.info(f"Modèle chargé en {elapsed:.2f}s")
        logger.info(f"Dimension des embeddings : {self._model.get_sentence_embedding_dimension()}")

    @property
    def dimension(self) -> int:
        """
        Retourne la dimension des vecteurs produits par ce modèle.
        Nécessaire pour initialiser l'index FAISS avec la bonne taille.
        Pour all-MiniLM-L6-v2 : 384
        """
        if self._model is None:
            self.load()
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs.

        PARAMÈTRES :
            texts        : liste de textes à encoder
            batch_size   : nombre de textes encodés en parallèle.
                           64 est optimal pour CPU — assez grand pour être efficace,
                           assez petit pour tenir en RAM.
            show_progress: affiche une barre de progression (utile pour 2000 chunks)

        RETOURNE :
            np.ndarray de forme (len(texts), 384)
            Chaque ligne = vecteur d'un texte
            Ex pour 2016 chunks : tableau de forme (2016, 384)

        NORMALISATION L2 :
            normalize_embeddings=True normalise chaque vecteur à longueur 1.
            Cela permet d'utiliser le produit scalaire (dot product) comme
            mesure de similarité, équivalent au cosinus mais plus rapide.
            FAISS IndexFlatIP (Inner Product) est optimisé pour ça.
        """
        if self._model is None:
            self.load()

        if not texts:
            # Cas edge : liste vide → retourne tableau vide de la bonne dimension
            return np.zeros((0, self.dimension), dtype=np.float32)

        logger.info(f"Encodage de {len(texts)} textes (batch_size={batch_size})...")
        t0 = time.perf_counter()

        # .encode() est la méthode principale de SentenceTransformer.
        # Elle gère automatiquement le découpage en batches.
        # convert_to_numpy=True → retourne np.ndarray (requis par FAISS)
        # normalize_embeddings=True → normalisation L2 pour cosine similarity
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # IMPORTANT pour FAISS IndexFlatIP
        )

        # On s'assure que le type est float32.
        # FAISS exige float32 — float64 causerait une erreur silencieuse.
        vectors = vectors.astype(np.float32)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Encodage terminé en {elapsed:.2f}s "
            f"— {len(texts)/elapsed:.0f} chunks/sec "
            f"— shape: {vectors.shape}"
        )

        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode une seule requête utilisateur en vecteur.
        Utilisé au moment de la recherche (pas de l'indexation).

        Retourne un vecteur de forme (1, 384) — compatible avec FAISS.search().
        """
        if self._model is None:
            self.load()

        # np.expand_dims ajoute une dimension : (384,) → (1, 384)
        # FAISS.search() attend un tableau 2D, pas un vecteur 1D.
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        return vector  # shape: (1, 384)