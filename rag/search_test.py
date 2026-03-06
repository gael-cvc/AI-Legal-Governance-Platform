"""
search_test.py — Validation du moteur de recherche vectoriel
=============================================================================

RÔLE DE CE FICHIER :
Script de test manuel pour vérifier que le pipeline de recherche fonctionne
correctement AVANT de connecter le LLM et de lancer l'API complète.

C'est la première fois dans le projet qu'on voit le système "penser" :
on pose de vraies questions juridiques et on observe quels textes il retrouve.

POURQUOI CE FICHIER EST IMPORTANT :
Avant de connecter Claude (LLM), on veut valider que la partie "recherche"
fonctionne bien. Un bon LLM avec une mauvaise recherche donnera de mauvaises
réponses — "garbage in, garbage out".

Ce script permet de diagnostiquer les problèmes de recherche :
- Scores trop bas → query expansion nécessaire
- Doublons → déduplication à corriger dans VectorStore
- Mauvais documents retournés → seuil min_score à ajuster
- Table des matières au lieu du contenu → chunking à retravailler

INTERPRÉTATION DES SCORES :
    > 0.70  = excellent  → le chunk répond directement à la question
    0.50-0.70 = bon      → le chunk est pertinent, contient l'information
    0.35-0.50 = correct  → le chunk est lié au sujet, utile comme contexte
    < 0.35  = faible     → filtré par MIN_SCORE, trop éloigné sémantiquement

LANCEMENT :
    python -m rag.search_test
=============================================================================
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
# Ces 5 questions couvrent les cas d'usage principaux de la plateforme :
# - Obligations (GDPR Article 24)
# - Conformité IA (EU AI Act)
# - Décision automatisée (GDPR Article 22)
# - Principe fondamental (GDPR Article 5)
# - Question en français (CNIL)
#
# Elles servent aussi à détecter les problèmes connus :
# - Query 1 : faible score attendu → "data controller obligations" ≠ "Article 24"
#   (problème de mapping sémantique, résolu par query expansion)
# - Query 3 : risque de retourner la table des matières
#   (problème de chunking, corrigé dans article_extractor.py v1.1)
TEST_QUERIES = [
    {
        "question": "What are the obligations of a data controller under GDPR?",
        "filter":   "GDPR",     # Cherche seulement dans les chunks GDPR
        "k":        5,          # Retourne les 5 meilleurs résultats
    },
    {
        "question": "High-risk AI systems requirements and conformity assessment",
        "filter":   "EU_AI_ACT",
        "k":        5,
    },
    {
        "question": "Automated decision making prohibition and exceptions Article 22",
        "filter":   None,       # Cherche dans tout le corpus (GDPR + EDPB)
        "k":        5,
    },
    {
        "question": "Data minimisation and purpose limitation principles",
        "filter":   None,       # Article 5 GDPR + CNIL → pas de filtre
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
    Lance les 5 requêtes de test et affiche les résultats formatés dans le terminal.

    FONCTIONNEMENT :
    Pour chaque question :
    1. encode_query() : transforme la question en vecteur (1, 384)
                        Utilise le même modèle que pour les chunks → espaces alignés
    2. store.search() : FAISS calcule les produits scalaires avec les 2016 vecteurs,
                        retourne les top-k après déduplication et filtrage
    3. Affichage      : on montre le score, la source, et un aperçu du texte
    """

    # ── Chargement des ressources ─────────────────────────────────────────────
    # On charge l'index et le modèle UNE SEULE FOIS avant la boucle de test.
    # Charger à chaque itération coûterait ~3s par question → inacceptable.
    logger.info("Chargement du vector store...")
    store = VectorStore()
    store.load()  # lit data/vector_store/index.faiss + metadata.json

    logger.info("Chargement du modèle d'embedding...")
    embedder = LegalEmbedder()
    embedder.load()  # charge all-MiniLM-L6-v2 depuis le cache HuggingFace

    # ── Affichage de l'en-tête ────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SEARCH TEST — AI Legal Governance Platform")
    print(f"  {store.n_vectors} vecteurs indexés · dimension 384 · IndexFlatIP")
    print(f"{sep}\n")

    # ── Boucle sur les 5 requêtes de test ─────────────────────────────────────
    for i, query in enumerate(TEST_QUERIES, 1):
        question = query["question"]
        filtre   = query["filter"]
        k        = query["k"]

        print(f"{'─' * 70}")
        print(f"  REQUÊTE {i}/{len(TEST_QUERIES)}")
        print(f"  ❓ {question}")
        if filtre:
            print(f"  🔍 Filtre réglementation : {filtre} uniquement")
        print(f"{'─' * 70}")

        # ── Encodage de la question ────────────────────────────────────────────
        # encode_query() transforme la question en vecteur (1, 384).
        # On utilise le MÊME modèle que pour les chunks (all-MiniLM-L6-v2) →
        # les espaces vectoriels sont alignés : questions et chunks sont
        # comparables mathématiquement.
        query_vector = embedder.encode_query(question)

        # ── Recherche FAISS ────────────────────────────────────────────────────
        # store.search() :
        # 1. FAISS calcule les produits scalaires (cosine similarity) entre
        #    query_vector et les 2016 vecteurs de l'index
        # 2. Retourne les k meilleurs après filtrage min_score + déduplication
        results = store.search(
            query_vector,
            k                 = k,
            regulation_filter = filtre,
        )

        if not results:
            print("  ⚠️  Aucun résultat au-dessus du seuil min_score=0.35.\n")
            print("  → Essayer de baisser min_score ou de reformuler la question.\n")
            continue

        # ── Affichage des résultats ────────────────────────────────────────────
        for rank, chunk in enumerate(results, 1):
            score    = chunk.get("similarity_score", 0.0)
            reg      = chunk.get("regulation",   "?")
            seg_id   = chunk.get("segment_id",   "?")
            seg_type = chunk.get("segment_type", "?")
            source   = chunk.get("source_file",  "?")
            page     = chunk.get("page_start",   "?")
            year     = chunk.get("year",         "?")
            text     = chunk.get("text",         "")

            # Barre visuelle du score : plus elle est longue, plus le score est élevé
            # int(score * 30) = 0 à 30 blocs selon le score (0.0 à 1.0)
            bar_len   = int(score * 30)
            score_bar = "█" * bar_len + "░" * (30 - bar_len)

            # Aperçu du texte : 180 premiers caractères, sur une seule ligne
            # .replace("\n", " ") : supprime les sauts de ligne pour l'affichage
            preview = text[:180].replace("\n", " ").strip()

            print(f"\n  #{rank}  Score : {score:.4f}  [{score_bar}]")
            print(f"       {reg} ({year}) · {seg_id} [{seg_type}]")
            print(f"       Source : {source} · page {page}")
            print(f"       ↳ {preview}...")

        print()  # Ligne vide entre les requêtes

    # ── Pied de page ──────────────────────────────────────────────────────────
    print(f"{sep}")
    print("  Search test terminé.")
    print("  Scores > 0.50 = bonne recherche, prêt pour la connexion LLM.")
    print("  Scores < 0.40 = activer query expansion dans api/search.py.")
    print(f"{sep}\n")


# Point d'entrée : python -m rag.search_test
if __name__ == "__main__":
    run_search_test()
