"""
embedder.py — Encodage des chunks en vecteurs avec sentence-transformers
=============================================================================

RÔLE DE CE FICHIER :
Transformer chaque chunk de texte juridique en un vecteur numérique
(embedding) qui représente son sens dans un espace mathématique à 384 dimensions.

QU'EST-CE QU'UN EMBEDDING ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Un embedding = une liste de 384 nombres flottants qui encode le SENS
d'un texte. Ce n'est pas une représentation des mots (comme TF-IDF),
c'est une représentation de la sémantique : deux textes qui veulent dire
la même chose produisent des vecteurs mathématiquement proches, même
s'ils utilisent des mots différents ou des langues différentes.

EXEMPLE CONCRET :
    "Le responsable de traitement doit informer les personnes"
    → vecteur A : [0.23, -0.81, 0.45, 0.12, ...]   (384 dimensions)

    "The data controller must notify individuals"
    → vecteur B : [0.24, -0.79, 0.46, 0.11, ...]

    cosine_similarity(A, B) ≈ 0.94  ← très proches (même sens, langues différentes)

    "La météo est agréable aujourd'hui"
    → vecteur C : [-0.55, 0.32, -0.12, 0.88, ...]

    cosine_similarity(A, C) ≈ 0.12  ← très éloignés (sujets différents)

POURQUOI C'EST PUISSANT POUR LA RECHERCHE JURIDIQUE :
Quand un utilisateur pose "What must a data controller do?", son vecteur
sera proche des chunks qui parlent des obligations du responsable de traitement
— même si ces chunks utilisent "controller shall", "processor must", etc.
C'est la recherche sémantique, bien supérieure à la recherche par mots-clés.

MODÈLE CHOISI : all-MiniLM-L6-v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DIMENSION   : 384 → compact, rapide à chercher dans FAISS
    CONTEXTE    : 256 tokens (~1000 chars) → nos chunks calibrés à 1200 chars
                  (la légère troncature ne perd pas les informations clés)
    PERFORMANCE : Score MTEB 59.9/100 → excellent rapport qualité/légèreté
    TAILLE      : 80 Mo → téléchargement unique, cache automatique HuggingFace
    VITESSE     : ~100 chunks/seconde sur CPU → 2016 chunks en ~20s

    Alternative plus précise (si on veut améliorer à l'avenir) :
    "BAAI/bge-large-en-v1.5" → dimension 1024, score MTEB 64.2, mais 1.3 Go

FLUX DE DONNÉES :
    list[str]  (textes des chunks)   →  encode()       →  np.ndarray (n, 384)
    str        (question utilisateur) →  encode_query() →  np.ndarray (1, 384)
=============================================================================
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

# SentenceTransformer est la classe principale de sentence-transformers.
# Elle télécharge et charge un modèle pré-entraîné depuis HuggingFace,
# puis expose une méthode .encode() simple pour vectoriser du texte.
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.embedder")

# Nom du modèle HuggingFace à utiliser.
# Constante en majuscules = convention Python pour les valeurs immuables.
# Changer cette ligne suffit pour changer de modèle dans tout le projet.
MODEL_NAME = "all-MiniLM-L6-v2"


class LegalEmbedder:
    """
    Encodeur de textes juridiques basé sur sentence-transformers.

    Cette classe encapsule le modèle d'embedding et expose deux méthodes :
    - encode()       : encoder un batch de chunks (phase d'indexation)
    - encode_query() : encoder une question utilisateur (phase de recherche)

    PATTERN LAZY LOADING :
    Le modèle (~80 Mo en RAM) n'est chargé qu'au premier appel à encode()
    ou à load(). On évite ainsi de charger un modèle lourd si la classe est
    instanciée mais jamais utilisée (ex: dans les tests unitaires).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialise l'encodeur SANS charger le modèle immédiatement.

        Paramètre :
            model_name : identifiant du modèle sur HuggingFace.
                         Défaut : "all-MiniLM-L6-v2"
                         Pour changer de modèle, modifier MODEL_NAME en haut du fichier.
        """
        self.model_name = model_name

        # _model vaut None jusqu'au premier appel à load() ou encode().
        # Le type hint "SentenceTransformer | None" documente les deux états possibles.
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """
        Charge le modèle en mémoire depuis le cache HuggingFace.

        PREMIER APPEL  : télécharge depuis huggingface.co (~80 Mo).
                         Durée : 30-60s selon la connexion internet.
        APPELS SUIVANTS : lit depuis ~/.cache/huggingface/ sur le disque local.
                          Durée : ~2-3 secondes.

        IDEMPOTENT : appeler load() plusieurs fois est sans danger —
        le if self._model is not None empêche tout rechargement inutile.
        """
        if self._model is not None:
            return  # Déjà en mémoire → rien à faire

        logger.info(f"Chargement du modèle : {self.model_name}")
        t0 = time.perf_counter()

        # SentenceTransformer() télécharge (si nécessaire) et charge le modèle.
        # device : détecté via la variable d'environnement DEVICE.
        #   DEVICE=mps  → Mac M1/M2/M3/M4 (Metal Performance Shaders, plus rapide)
        #   DEVICE=cpu  → Linux / Docker / tout autre environnement
        # Par défaut : "mps" (comportement local inchangé sur Mac).
        # Dans Docker : docker-compose.yml injecte DEVICE=cpu (MPS non disponible).
        device = os.getenv("DEVICE", "mps")
        logger.info(f"Device sélectionné : {device}")
        self._model = SentenceTransformer(self.model_name, device=device)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Modèle chargé en {elapsed:.2f}s | "
            f"dimension : {self._model.get_sentence_embedding_dimension()}"
        )

    @property
    def dimension(self) -> int:
        """
        Retourne la dimension des vecteurs produits par ce modèle.

        Une @property se lit comme un attribut (embedder.dimension)
        mais exécute du code. Ici : charge le modèle si pas encore fait,
        puis retourne la dimension via l'API sentence-transformers.

        Pourquoi cette méthode est nécessaire :
        VectorStore.build() doit créer faiss.IndexFlatIP(dim) avec la bonne
        dimension. Plutôt que de hardcoder 384, on interroge le modèle.
        Si on change de modèle (ex: bge-large → dim=1024), tout s'adapte.

        Pour all-MiniLM-L6-v2 : retourne 384.
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
        Encode un batch de textes en vecteurs. Utilisé pendant l'indexation.

        PARAMÈTRES :
            texts         : liste des textes à encoder (ex: les 2016 chunks)
            batch_size    : nombre de textes traités simultanément.
                            64 est optimal sur CPU :
                            → assez grand pour amortir les coûts de tokenisation batch
                            → assez petit pour tenir en RAM sans overflow
            show_progress : affiche une barre de progression tqdm
                            (utile pendant les 20s d'encodage de 2016 chunks)

        NORMALISATION L2 — pourquoi c'est critique :
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        normalize_embeddings=True normalise chaque vecteur à longueur 1 (norme = 1).
        Formule : v_normalisé = v / ||v||  (on divise chaque composante par la norme)

        Notre index FAISS utilise IndexFlatIP (Inner Product = produit scalaire).
        Avec des vecteurs de norme 1, le produit scalaire A·B est EXACTEMENT
        égal à la cosine similarity cos(θ) = A·B / (||A|| × ||B||).
        → On obtient une similarité entre 0 et 1 directement interprétable.
        Sans normalisation, IndexFlatIP calculerait autre chose que la cosine
        similarity, et les scores seraient non comparables entre eux.

        RETOURNE :
            np.ndarray de forme (len(texts), 384), dtype float32
            Chaque ligne i = vecteur 384D du texte texts[i]
            Ex pour 2016 chunks : tableau de 2016 × 384 = 776 448 floats
        """
        if self._model is None:
            self.load()

        if not texts:
            # Cas edge : liste vide → retourne tableau vide avec la bonne forme.
            # Évite une erreur FAISS si on tente d'indexer 0 vecteurs.
            return np.zeros((0, self.dimension), dtype=np.float32)

        logger.info(f"Encodage de {len(texts)} textes | batch_size={batch_size}...")
        t0 = time.perf_counter()

        vectors = self._model.encode(
            texts,
            batch_size           = batch_size,
            show_progress_bar    = show_progress,  # barre tqdm dans le terminal
            convert_to_numpy     = True,            # retourne np.ndarray (requis par FAISS)
            normalize_embeddings = True,            # normalisation L2 → IndexFlatIP = cosine
        )

        # On force float32 explicitement car FAISS l'exige.
        # sentence-transformers retourne généralement float32, mais ce cast
        # garantit la compatibilité et évite l'erreur cryptique FAISS
        # "vector type must be float32" si jamais le modèle retourne float64.
        vectors = vectors.astype(np.float32)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Encodage terminé en {elapsed:.2f}s "
            f"({len(texts) / elapsed:.0f} chunks/sec) "
            f"| shape: {vectors.shape}"
        )

        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode une seule question utilisateur. Utilisé pendant la recherche.

        DIFFÉRENCE AVEC encode() :
        encode()       → optimisé pour le débit (batch de 2016 chunks en 20s)
        encode_query() → optimisé pour la latence (1 question en ~10ms)

        COHÉRENCE DE NORMALISATION :
        Il est essentiel d'utiliser la MÊME normalisation ici que dans encode().
        Si les chunks sont normalisés mais pas la question (ou inversement),
        le produit scalaire ne serait plus égal à la cosine similarity
        et tous les scores de recherche seraient faux.

        RETOURNE :
            np.ndarray de forme (1, 384), dtype float32
            FAISS.index.search() exige un tableau 2D de shape (n_queries, dim).
            Pour une seule requête : shape (1, 384), pas (384,).
        """
        if self._model is None:
            self.load()

        # On encode une liste avec un seul élément [query].
        # SentenceTransformer retourne shape (1, 384) dans ce cas.
        vector = self._model.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,  # cohérence avec encode()
            show_progress_bar    = False, # désactivé : évite le crash sur Mac M1/M2/M4
                                          # la barre de progression tqdm interagit mal
                                          # avec le runtime Python sur Apple Silicon
                                          # lors d'encodages de séquences courtes
        ).astype(np.float32)

        return vector  # shape: (1, 384) — directement utilisable par FAISS
