"""
search_test.py — Test du moteur de recherche vectoriel

Ce script pose 5 vraies questions juridiques à la plateforme
et affiche les chunks retrouvés avec leurs scores de similarité.

C'est la première fois qu'on voit le système "penser" :
  1. La question est transformée en vecteur (encode_query)
  2. FAISS cherche les vecteurs les plus proches (search)
  3. On affiche les chunks correspondants avec leurs métadonnées

LANCEMENT :
    python -m rag.search_test
"""

from __future__ import annotations

import logging
from pathlib import Path

from .embedder import LegalEmbedder
from .vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.search_test")

# ── Questions de test ──────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "question": "What are the obligations of a data controller under GDPR?",
        "filter":   "GDPR",
        "k":        5,
    },
    {
        "question": "High-risk AI systems requirements and conformity assessment",
        "filter":   "EU_AI_ACT",
        "k":        5,
    },
    {
        "question": "Automated decision making prohibition and exceptions Article 22",
        "filter":   None,
        "k":        5,
    },
    {
        "question": "Data minimisation and purpose limitation principles",
        "filter":   None,
        "k":        5,
    },
    {
        "question": "Recommandations CNIL sur les systèmes d'intelligence artificielle",
        "filter":   "CNIL",
        "k":        5,
    },
]


def run_search_test() -> None:
    """
    Lance les 5 requêtes de test et affiche les résultats dans le terminal.
    """

    # ── Chargement de l'index et du modèle ────────────────────────────────────
    logger.info("Chargement du vector store...")
    store = VectorStore()
    store.load()

    logger.info("Chargement du modèle d'embedding...")
    embedder = LegalEmbedder()
    embedder.load()

    sep = "=" * 70

    print(f"\n{sep}")
    print("  SEARCH TEST — AI Legal Governance Platform")
    print(f"  {store.n_vectors} vecteurs indexés · dimension 384 · IndexFlatIP")
    print(f"{sep}\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        question = query["question"]
        filtre   = query["filter"]
        k        = query["k"]

        print(f"{'─' * 70}")
        print(f"  REQUÊTE {i}/5")
        print(f"  ❓ {question}")
        if filtre:
            print(f"  🔍 Filtre : {filtre} uniquement")
        print(f"{'─' * 70}")

        # ── Encodage de la question ────────────────────────────────────────────
        # encode_query() transforme la question en vecteur (1, 384)
        # Le même modèle que pour les chunks → les espaces vectoriels sont alignés
        query_vector = embedder.encode_query(question)

        # ── Recherche dans FAISS ───────────────────────────────────────────────
        results = store.search(query_vector, k=k, regulation_filter=filtre)

        if not results:
            print("  ⚠️  Aucun résultat trouvé.\n")
            continue

        # ── Affichage des résultats ────────────────────────────────────────────
        for rank, chunk in enumerate(results, 1):
            score     = chunk.get("similarity_score", 0)
            reg       = chunk.get("regulation", "?")
            seg_id    = chunk.get("segment_id", "?")
            seg_type  = chunk.get("segment_type", "?")
            source    = chunk.get("source_file", "?")
            page      = chunk.get("page_start", "?")
            year      = chunk.get("year", "?")
            text      = chunk.get("text", "")

            # Barre visuelle du score (0.0 → 1.0)
            bar_len   = int(score * 30)
            score_bar = "█" * bar_len + "░" * (30 - bar_len)

            # Aperçu du texte (150 premiers caractères, sur une ligne)
            preview   = text[:180].replace("\n", " ").strip()

            print(f"\n  #{rank}  Score : {score:.4f}  [{score_bar}]")
            print(f"       {reg} ({year}) · {seg_id} [{seg_type}]")
            print(f"       Source : {source} · p.{page}")
            print(f"       ↳ {preview}...")

        print()

    print(f"{sep}")
    print("  Search test terminé.")
    print(f"{sep}\n")


if __name__ == "__main__":
    run_search_test()
