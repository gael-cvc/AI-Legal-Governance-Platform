"""
reranker.py — Re-scoring des chunks avec un cross-encoder
=============================================================================

RÔLE DE CE FICHIER :
Après que FAISS a retourné ses k*2 meilleurs candidats par similarité
vectorielle, le cross-encoder lit chaque paire (question, chunk) ensemble
et produit un score de pertinence beaucoup plus précis.

POURQUOI UN DEUXIÈME MODÈLE ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le bi-encoder (all-MiniLM-L6-v2) encode la question et les chunks
SÉPARÉMENT, puis compare les vecteurs. C'est rapide mais il perd
les interactions fines entre les deux textes.

    Question : "What must a data controller do under GDPR?"
    Chunk A   : "Article 24 — The controller shall implement measures..."
    Chunk B   : "Recital 74 — The responsibility and liability of the controller..."

    Le bi-encoder donne à A et B des scores similaires car ils parlent
    tous deux du "controller". Il ne sait pas que A répond directement
    ("shall implement measures") et que B est un contexte secondaire.

Le cross-encoder résout ça :
    Input  : "[CLS] What must a data controller do? [SEP] Article 24 — The
              controller shall implement measures... [SEP]"
    Output : score unique de pertinence (ex: 8.2 pour A, 3.1 pour B)

    En lisant les deux textes côte à côte, le modèle voit les correspondances
    sémantiques directes : "must do" ↔ "shall implement", "controller" ↔ "controller".

ARCHITECTURE RETRIEVE & RERANK :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ┌──────────────────────────────────────────────────────────┐
    │  FAISS (bi-encoder)                                      │
    │  2016 chunks → top k*2 candidats en ~2ms               │
    │  (rapide mais approximatif)                              │
    └──────────────────────────────────────────────────────────┘
                           │
                           ▼ k*2 candidats (ex: 10)
    ┌──────────────────────────────────────────────────────────┐
    │  Cross-encoder                                           │
    │  10 paires (question, chunk) → 10 scores en ~200ms      │
    │  (lent mais très précis — ne peut pas lire 2016 chunks) │
    └──────────────────────────────────────────────────────────┘
                           │
                           ▼ top k résultats rerankés (ex: 5)
    LLM Claude

    Coût total : ~2ms (FAISS) + ~200ms (cross-encoder) = ~202ms
    vs cross-encoder seul sur 2016 chunks : ~40 secondes — inacceptable.

MODÈLE CHOISI : cross-encoder/ms-marco-MiniLM-L-6-v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DONNÉES D'ENTRAÎNEMENT : MS-MARCO — 340k paires (question, passage)
                             issues de Bing Search avec annotations humaines.
    ARCHITECTURE : MiniLM-L6 (6 couches Transformer) → rapide sur CPU/MPS
    SORTIE       : logit non normalisé (peut être négatif ou > 10)
                   Score élevé = très pertinent, score bas/négatif = peu pertinent
    TAILLE       : ~67 Mo — léger, chargement en ~1s

PATTERN SINGLETON (même que LegalEmbedder) :
Le modèle est chargé UNE SEULE FOIS au démarrage de l'API (dans main.py lifespan)
et réutilisé pour toutes les requêtes. Charger un modèle à chaque requête
coûterait ~1-2s et causerait des crashes sur Mac (allocation mémoire répétée).
=============================================================================
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("rag.reranker")

# Nom du modèle cross-encoder sur HuggingFace.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LegalReranker:
    """
    Re-classeur de chunks juridiques basé sur un cross-encoder.

    Cette classe encapsule le cross-encoder et expose une méthode rerank()
    qui prend une question et une liste de chunks, et retourne les chunks
    réordonnés par pertinence décroissante.

    PATTERN LAZY LOADING :
    Même stratégie que LegalEmbedder : le modèle (~67 Mo) n'est chargé
    qu'au premier appel à load() ou rerank(). En pratique, main.py
    appelle load() explicitement au démarrage.

    GESTION D'ERREUR GRACIEUSE :
    Si le cross-encoder échoue, rerank() retourne les chunks dans leur
    ordre FAISS original plutôt que de faire planter la requête.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """
        Charge le cross-encoder en mémoire depuis le cache HuggingFace.

        PREMIER APPEL  : télécharge depuis huggingface.co (~67 Mo).
                         Durée : 20-40s selon la connexion.
        APPELS SUIVANTS : lit depuis ~/.cache/huggingface/ sur le disque.
                          Durée : ~1-2 secondes.

        IDEMPOTENT : appeler load() plusieurs fois est sans danger.
        """
        if self._model is not None:
            return  # Déjà en mémoire → rien à faire

        logger.info(f"Chargement du cross-encoder : {self.model_name}")
        t0 = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder

            # device : détecté via la variable d'environnement DEVICE.
            #   DEVICE=mps  → Mac M1/M2/M3/M4 (Metal Performance Shaders, plus rapide)
            #   DEVICE=cpu  → Linux / Docker / tout autre environnement
            # Par défaut : "mps" (comportement local inchangé sur Mac).
            # Dans Docker : docker-compose.yml injecte DEVICE=cpu (MPS non disponible).
            device = os.getenv("DEVICE", "mps")
            logger.info(f"Device sélectionné : {device}")
            self._model = CrossEncoder(self.model_name, device=device)

            elapsed = time.perf_counter() - t0
            logger.info(f"Cross-encoder chargé en {elapsed:.2f}s")

        except Exception as e:
            logger.warning(
                f"Impossible de charger le cross-encoder : {e}\n"
                f"Le reranking sera désactivé — fallback ordre FAISS."
            )
            self._model = None

    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-score et réordonne les chunks par pertinence pour la question donnée.

        FONCTIONNEMENT :
        1. Construit les paires [(question, chunk_text_1), ...]
        2. Passe toutes les paires au cross-encoder en un seul appel batch
        3. Associe chaque score au chunk correspondant (rerank_score)
        4. Trie par score décroissant
        5. Retourne les top_k meilleurs

        FALLBACK GRACIEUX :
        Si le modèle n'est pas chargé ou si predict() lève une exception,
        on retourne les chunks[:top_k] dans l'ordre FAISS.

        Paramètres :
            question : question originale de l'utilisateur (non expansée)
            chunks   : liste de dicts avec au moins un champ "text"
            top_k    : nombre de chunks à retourner après reranking

        Retourne :
            list[dict] : chunks avec rerank_score ajouté, triés par score décroissant
                         ou chunks[:top_k] dans l'ordre FAISS si fallback
        """
        if len(chunks) <= 1:
            return chunks[:top_k]

        if self._model is None:
            logger.warning("Cross-encoder non disponible → fallback ordre FAISS")
            return chunks[:top_k]

        try:
            t0 = time.perf_counter()

            # Construction des paires (question, texte_du_chunk).
            # CrossEncoder.predict() attend une liste de tuples (str, str).
            pairs = [(question, chunk["text"]) for chunk in chunks]

            # predict() encode toutes les paires en batch.
            # show_progress_bar=False : évite le crash sur Mac Apple Silicon
            # en contexte async (même raison que encode_query()).
            scores = self._model.predict(pairs, show_progress_bar=False)

            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Reranking terminé en {elapsed:.0f}ms | "
                f"top score={reranked[0]['rerank_score']:.3f} | "
                f"bottom score={reranked[-1]['rerank_score']:.3f}"
            )

            return reranked[:top_k]

        except Exception as e:
            logger.warning(
                f"Reranking échoué : {e}\n"
                f"Fallback : ordre FAISS conservé."
            )
            return chunks[:top_k]

    @property
    def is_available(self) -> bool:
        """
        Indique si le cross-encoder est chargé et disponible.

        Utilisé par health.py pour indiquer l'état du reranker dans /health,
        et par search.py pour décider si le reranking est possible.
        """
        return self._model is not None
