"""
=============================================================================
build_index.py — Orchestrateur : Silver → Embeddings → Index FAISS
=============================================================================

RÔLE DE CE FICHIER :
Script principal à lancer UNE SEULE FOIS pour construire l'index FAISS.
Après ça, l'index est sur disque et l'API peut le charger en < 1 seconde.

ÉTAPES :
1. Charge tous les chunks depuis data/silver/*.jsonl
2. Encode les textes en vecteurs avec sentence-transformers
3. Construit l'index FAISS
4. Sauvegarde index + métadonnées dans data/vector_store/

LANCEMENT :
    python -m rag.build_index

DURÉE ESTIMÉE :
    ~30-60 secondes sur Mac (CPU) pour 2016 chunks

=============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .embedder import LegalEmbedder
from .vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.build_index")

SILVER_DIR = Path("data/silver")


def load_chunks_from_silver() -> list[dict]:
    """
    Charge tous les chunks depuis les fichiers JSONL de la couche silver.
    Retourne une liste de dicts, un par chunk.
    """
    all_chunks = []
    files = sorted(SILVER_DIR.glob("*_chunks.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Aucun chunk trouvé dans {SILVER_DIR}. "
            f"Lance d'abord : python -m ingestion.pipeline"
        )

    for path in files:
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))
                    count += 1
        logger.info(f"  {path.name} : {count} chunks")

    logger.info(f"Total : {len(all_chunks)} chunks chargés")
    return all_chunks


def run_build_index() -> None:
    """
    Pipeline complet : charge les chunks, encode, indexe, sauvegarde.
    """
    logger.info("=" * 55)
    logger.info("BUILD INDEX — AI Legal Governance Platform")
    logger.info("=" * 55)
    t_total = time.perf_counter()

    # ── ÉTAPE 1 : Chargement des chunks ───────────────────────────────────────
    logger.info("Étape 1/3 : Chargement des chunks silver...")
    chunks = load_chunks_from_silver()

    # On extrait les textes (pour l'embedding) et les métadonnées (pour le stockage)
    # L'ordre doit être IDENTIQUE entre les deux listes.
    texts    = [chunk["text"] for chunk in chunks]
    # Pour les métadonnées, on garde tout SAUF le texte brut
    # (le texte est dans le vecteur, pas besoin de le stocker deux fois)
    # En fait on garde le texte aussi pour pouvoir le retourner dans les résultats
    metadata = chunks  # on garde tout, y compris le texte

    # ── ÉTAPE 2 : Embedding ───────────────────────────────────────────────────
    logger.info("Étape 2/3 : Encodage des textes en vecteurs...")
    embedder = LegalEmbedder()
    embedder.load()

    # encode() retourne un np.ndarray de forme (2016, 384)
    vectors = embedder.encode(texts, batch_size=64, show_progress=True)

    logger.info(f"Vecteurs produits : shape={vectors.shape}, dtype={vectors.dtype}")

    # ── ÉTAPE 3 : Construction et sauvegarde de l'index FAISS ─────────────────
    logger.info("Étape 3/3 : Construction de l'index FAISS...")
    store = VectorStore()
    store.build(vectors, metadata)
    store.save()

    # ── Résumé ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    logger.info("=" * 55)
    logger.info(f"Index construit avec succès en {elapsed:.1f}s")
    logger.info(f"Vecteurs indexés  : {store.n_vectors}")
    logger.info(f"Dimension         : {vectors.shape[1]}")
    logger.info(f"Sauvegardé dans   : data/vector_store/")
    logger.info("=" * 55)
    logger.info("Prochaine étape : python -m rag.search_test")


if __name__ == "__main__":
    run_build_index()