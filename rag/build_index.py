"""
build_index.py — Construction de l'index FAISS depuis la couche silver
=============================================================================

RÔLE DE CE FICHIER :
Script à lancer UNE SEULE FOIS (ou après chaque mise à jour du corpus)
pour transformer les chunks JSON de la couche silver en un index FAISS
interrogeable par l'API en temps réel.

POSITION DANS LE PIPELINE COMPLET :
    data/raw/       ← PDFs bruts
         │  python -m ingestion.pipeline
         ▼
    data/silver/    ← Chunks JSON (texte + métadonnées)
         │  python -m rag.build_index   ← CE FICHIER
         ▼
    data/vector_store/  ← Index FAISS + métadonnées (prêt pour l'API)
         │  uvicorn api.main:app
         ▼
    API FastAPI         ← Répond aux questions en ~500ms

POURQUOI SÉPARER "BUILD INDEX" DE L'API ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L'encodage de 2016 chunks prend ~20 secondes sur CPU.
On ne peut pas faire ça à chaque démarrage de l'API.
→ On fait le travail coûteux une fois, on sauvegarde le résultat sur disque.
→ L'API charge l'index en ~20ms depuis le disque à chaque démarrage.

C'est le même principe qu'une base de données : on insère les données une fois,
on les interroge des milliers de fois. Le coût d'insertion est amorti.

LANCEMENT :
    python -m rag.build_index

DURÉE ESTIMÉE :
    ~20-30 secondes sur Mac M1/M2 (CPU)
    (2016 chunks × ~10ms d'encodage par chunk en batch = ~20s)

RÉSULTAT :
    data/vector_store/index.faiss     ← ~3 Mo, contient les 2016 vecteurs 384D
    data/vector_store/metadata.json   ← ~2 Mo, métadonnées de chaque chunk
    data/vector_store/index_info.json ← quelques Ko, statistiques de l'index
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

    FORMAT JSONL (JSON Lines) :
    Chaque fichier *_chunks.jsonl contient une ligne JSON par chunk.
    On lit ligne par ligne pour rester efficace en mémoire, même si en pratique
    2016 chunks × ~1Ko chacun ≈ 2 Mo total — très raisonnable.

    ORDRE DE CHARGEMENT :
    sorted() garantit un ordre alphabétique déterministe des fichiers.
    Cela assure que l'index FAISS est construit dans le même ordre à chaque
    exécution, ce qui rend les identifiants de vecteurs reproductibles.
    (important pour le debug : le vecteur 42 sera toujours le même chunk)

    Retourne :
        list[dict] : tous les chunks de tous les fichiers, dans l'ordre

    Lève :
        FileNotFoundError si silver/ ne contient aucun fichier *_chunks.jsonl
        (= le pipeline d'ingestion n'a pas encore été lancé)
    """
    all_chunks: list[dict] = []

    # Cherche tous les fichiers *_chunks.jsonl dans silver/
    files = sorted(SILVER_DIR.glob("*_chunks.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier *_chunks.jsonl trouvé dans {SILVER_DIR}.\n"
            f"Lance d'abord le pipeline d'ingestion : python -m ingestion.pipeline"
        )

    # Lecture fichier par fichier
    for path in files:
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # ignore les lignes vides (fin de fichier)
                    # json.loads() désérialise une ligne JSON en dict Python
                    all_chunks.append(json.loads(line))
                    count += 1
        logger.info(f"  {path.name} : {count} chunks chargés")

    logger.info(f"Total : {len(all_chunks)} chunks chargés depuis {len(files)} fichiers")
    return all_chunks


def run_build_index() -> None:
    """
    Pipeline complet en 3 étapes : silver → embeddings → index FAISS → disque.

    ÉTAPE 1 : Chargement des chunks (data/silver/*.jsonl → list[dict])
    ÉTAPE 2 : Encodage des textes (list[str] → np.ndarray shape (n, 384))
    ÉTAPE 3 : Construction + sauvegarde (np.ndarray → index.faiss + metadata.json)
    """
    logger.info("=" * 55)
    logger.info("BUILD INDEX — AI Legal Governance Platform")
    logger.info("=" * 55)
    t_total = time.perf_counter()

    # ── ÉTAPE 1 : Chargement des chunks depuis silver/ ────────────────────────
    logger.info("Étape 1/3 : Chargement des chunks depuis data/silver/...")
    chunks = load_chunks_from_silver()

    # On extrait les textes dans une liste séparée pour l'embedding.
    # L'ordre DOIT être identique entre texts et chunks :
    # texts[i] est le texte de chunks[i], et vectors[i] sera le vecteur de texts[i].
    # → vectors[i] correspond à chunks[i] → FAISS index i retourne chunks[i]
    texts = [chunk["text"] for chunk in chunks]

    # On garde les chunks complets comme métadonnées (texte inclus).
    # Pourquoi inclure le texte dans les métadonnées ?
    # Pour pouvoir retourner le texte brut dans les réponses de l'API,
    # sans avoir à recharger le fichier silver à chaque requête.
    metadata = chunks

    # ── ÉTAPE 2 : Encodage en vecteurs ───────────────────────────────────────
    logger.info("Étape 2/3 : Encodage des textes en vecteurs 384D...")

    embedder = LegalEmbedder()
    embedder.load()  # charge all-MiniLM-L6-v2 depuis le cache HuggingFace

    # encode() retourne un np.ndarray de shape (n_chunks, 384), dtype float32
    # Les vecteurs sont normalisés L2 → prêts pour IndexFlatIP (cosine similarity)
    vectors = embedder.encode(texts, batch_size=64, show_progress=True)

    logger.info(f"Vecteurs produits : shape={vectors.shape}, dtype={vectors.dtype}")
    # Attendu : shape=(2016, 384), dtype=float32

    # ── ÉTAPE 3 : Construction + sauvegarde de l'index FAISS ─────────────────
    logger.info("Étape 3/3 : Construction de l'index FAISS et sauvegarde...")

    store = VectorStore()

    # build() :
    # 1. Crée faiss.IndexFlatIP(384)
    # 2. Appelle index.add(vectors) → copie les 2016 vecteurs dans FAISS
    store.build(vectors, metadata)

    # save() :
    # 1. Écrit data/vector_store/index.faiss (binaire FAISS)
    # 2. Écrit data/vector_store/metadata.json (métadonnées JSON)
    # 3. Écrit data/vector_store/index_info.json (stats)
    store.save()

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    logger.info("=" * 55)
    logger.info(f"Index construit avec succès en {elapsed:.1f}s")
    logger.info(f"Vecteurs indexés  : {store.n_vectors}")
    logger.info(f"Dimension         : {vectors.shape[1]}")
    logger.info(f"Fichiers créés    : data/vector_store/")
    logger.info("=" * 55)
    logger.info("Prochaine étape : python -m rag.search_test")
    logger.info("Ou lancer l'API : uvicorn api.main:app --reload --port 8000")


# Point d'entrée en ligne de commande.
# S'exécute uniquement si on lance : python -m rag.build_index
# Ne s'exécute PAS si ce module est importé par un autre fichier.
if __name__ == "__main__":
    run_build_index()
