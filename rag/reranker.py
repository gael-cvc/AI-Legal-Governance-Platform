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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DONNÉES D'ENTRAÎNEMENT : MS-MARCO — 340k paires (question, passage)
                             issues de Bing Search avec annotations humaines.
                             Très pertinent pour la recherche documentaire.
    ARCHITECTURE : MiniLM-L6 (6 couches Transformer) → rapide sur CPU/MPS
    SORTIE       : logit non normalisé (peut être négatif ou > 10)
                   Score élevé = très pertinent, score bas/négatif = peu pertinent
                   Attention : les scores NE SONT PAS des probabilités (pas de sigmoid)
    TAILLE       : ~67 Mo — léger, chargement en ~1s

PATTERN SINGLETON (même que LegalEmbedder) :
Le modèle est chargé UNE SEULE FOIS au démarrage de l'API (dans main.py lifespan)
et réutilisé pour toutes les requêtes. Charger un modèle à chaque requête
coûterait ~1-2s et causerait des crashes sur Mac (allocation mémoire répétée).
=============================================================================
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("rag.reranker")

# Nom du modèle cross-encoder sur HuggingFace.
# Alternatives possibles (plus précises mais plus lentes) :
# "cross-encoder/ms-marco-MiniLM-L-12-v2"  → 12 couches, ~10% plus précis, ~2x plus lent
# "cross-encoder/ms-marco-electra-base"     → ELECTRA, très précis, mais 440 Mo
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
    appelle load() explicitement au démarrage pour éviter toute latence
    sur la première requête utilisateur.

    GESTION D'ERREUR GRACIEUSE :
    Si le cross-encoder échoue (réseau coupé lors du téléchargement,
    mémoire insuffisante, bug PyTorch...), rerank() retourne les chunks
    dans leur ordre FAISS original plutôt que de faire planter la requête.
    L'API reste 100% fonctionnelle, juste sans l'amélioration du reranking.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        """
        Initialise le reranker SANS charger le modèle immédiatement.

        Paramètre :
            model_name : identifiant du modèle cross-encoder sur HuggingFace.
                         Défaut : "cross-encoder/ms-marco-MiniLM-L-6-v2"
        """
        self.model_name = model_name

        # _model vaut None jusqu'au premier appel à load() ou rerank().
        # On importe CrossEncoder ici pour éviter les imports circulaires
        # et permettre un chargement vraiment lazy.
        self._model = None

    def load(self) -> None:
        """
        Charge le cross-encoder en mémoire depuis le cache HuggingFace.

        PREMIER APPEL  : télécharge depuis huggingface.co (~67 Mo).
                         Durée : 20-40s selon la connexion.
        APPELS SUIVANTS : lit depuis ~/.cache/huggingface/ sur le disque.
                          Durée : ~1-2 secondes.

        DIFFÉRENCE AVEC LegalEmbedder.load() :
        CrossEncoder() de sentence-transformers prend un nom de modèle
        et un device. On utilise "mps" (Apple Silicon) pour la même raison
        que dans embedder.py : éviter les crashs PyTorch sur Mac M1/M2/M4
        en contexte async uvicorn.

        IDEMPOTENT : appeler load() plusieurs fois est sans danger.
        """
        if self._model is not None:
            return  # Déjà en mémoire → rien à faire

        logger.info(f"Chargement du cross-encoder : {self.model_name}")
        t0 = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder

            # device="mps" : même raison que embedder.py
            # évite les crashs PyTorch sur Mac Apple Silicon en contexte async
            self._model = CrossEncoder(self.model_name, device="mps")

            elapsed = time.perf_counter() - t0
            logger.info(f"Cross-encoder chargé en {elapsed:.2f}s")

        except Exception as e:
            # Si le chargement échoue, on log l'erreur mais on ne crash pas.
            # rerank() vérifiera si _model est None et retournera les chunks FAISS.
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
        1. Construit les paires [(question, chunk_text_1), (question, chunk_text_2), ...]
        2. Passe toutes les paires au cross-encoder en un seul appel batch
        3. Associe chaque score au chunk correspondant (rerank_score)
        4. Trie par score décroissant
        5. Retourne les top_k meilleurs

        POURQUOI LA QUESTION ORIGINALE (PAS L'EXPANSÉE) ?
        La query expansion ajoute des termes techniques pour guider FAISS.
        Ex: "data controller" → "data controller obligations responsibilities
            Article 24 GDPR implement appropriate technical organisational measures"
        Le cross-encoder lit les textes directement — lui passer la question
        concise et naturelle ("data controller") est plus efficace car il
        va chercher des correspondances directes avec le texte du chunk,
        sans être "noyé" par les termes d'expansion ajoutés.

        SCORES DU CROSS-ENCODER :
        Les scores sont des logits bruts (non normalisés), typiquement entre
        -5 et +10. Exemples observés sur notre corpus :
            > 8.0  = très pertinent (réponse directe à la question)
            3.0-8.0 = pertinent (information liée)
            < 0.0  = peu pertinent (sujet connexe mais indirect)
        Ces scores ne sont PAS comparables entre requêtes différentes —
        seul l'ordre relatif compte pour le tri.

        FALLBACK GRACIEUX :
        Si le modèle n'est pas chargé (échec lors de load()) ou si predict()
        lève une exception, on retourne les chunks[:top_k] dans l'ordre FAISS.
        Le champ rerank_score reste à None dans ce cas.

        Paramètres :
            question : question originale de l'utilisateur (non expansée)
            chunks   : liste de dicts avec au moins un champ "text"
            top_k    : nombre de chunks à retourner après reranking

        Retourne :
            list[dict] : chunks avec rerank_score ajouté, triés par score décroissant
                         ou chunks[:top_k] dans l'ordre FAISS si fallback
        """
        # Cas edge : 0 ou 1 chunk → pas besoin de reranker
        if len(chunks) <= 1:
            return chunks[:top_k]

        # Fallback si le modèle n'a pas pu se charger
        if self._model is None:
            logger.warning("Cross-encoder non disponible → fallback ordre FAISS")
            return chunks[:top_k]

        try:
            t0 = time.perf_counter()

            # Construction des paires (question, texte_du_chunk).
            # CrossEncoder.predict() attend une liste de tuples (str, str).
            # On utilise le texte brut du chunk — le modèle a été entraîné
            # sur des passages de longueur variable, jusqu'à ~512 tokens.
            pairs = [(question, chunk["text"]) for chunk in chunks]

            # predict() encode toutes les paires en batch et retourne un
            # numpy array de shape (len(chunks),) avec les scores float32.
            # show_progress_bar=False : évite le même crash que encode_query()
            # sur Mac Apple Silicon en contexte async.
            scores = self._model.predict(pairs, show_progress_bar=False)

            # Associe chaque score au chunk correspondant
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            # Tri décroissant par rerank_score → le plus pertinent en premier
            reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Reranking terminé en {elapsed:.0f}ms | "
                f"top score={reranked[0]['rerank_score']:.3f} | "
                f"bottom score={reranked[-1]['rerank_score']:.3f}"
            )

            return reranked[:top_k]

        except Exception as e:
            # Fallback gracieux : log l'erreur mais ne fait pas crasher la requête
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
