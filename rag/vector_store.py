"""
vector_store.py — Construction et gestion de l'index FAISS

QU'EST-CE QUE FAISS ?
FAISS (Facebook AI Similarity Search) cherche très rapidement les vecteurs
les plus similaires dans une grande base de données.

ANALOGIE :
2016 chunks = 2016 points dans un espace à 384 dimensions.
Quand tu poses une question, on calcule son vecteur et on trouve
les points les plus proches = les chunks les plus pertinents.

INDEX CHOISI : IndexFlatIP
- "Flat" = exact (compare avec TOUS les vecteurs, pas d'approximation)
- "IP"   = Inner Product = produit scalaire
- Avec vecteurs normalisés L2 : IP = cosine similarity
- Parfait pour 2016 vecteurs (pour >100k on utiliserait IndexIVFFlat)

FICHIERS PRODUITS :
    data/vector_store/index.faiss    vecteurs (binaire FAISS)
    data/vector_store/metadata.json  métadonnées de chaque chunk
    data/vector_store/index_info.json infos de l'index

CORRECTIONS v1.1 :
- Déduplication par segment_id : un segment ne peut apparaître qu'une
  seule fois dans les résultats, même s'il a été découpé en plusieurs
  chunks très similaires (artefact de l'overlap).
- Score minimum : les résultats sous MIN_SCORE (0.35) sont filtrés car
  ils n'ont pas assez de pertinence sémantique pour être utiles.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("rag.vector_store")

VECTOR_STORE_DIR = Path("data/vector_store")

# ── Seuil de score minimum ─────────────────────────────────────────────────────
# En dessous de 0.35, le chunk n'est pas assez pertinent.
# Cosine similarity entre 0 et 1 (avec vecteurs normalisés L2) :
#   > 0.70 = très pertinent (sujet très proche)
#   0.50-0.70 = pertinent (sujet lié)
#   0.35-0.50 = acceptable (sujet connexe)
#   < 0.35 = trop faible (sujet différent)
MIN_SCORE = 0.35


class VectorStore:
    """
    Gère l'index FAISS et les métadonnées associées.

    Deux modes :
    1. Construction : build(vectors, metadata) → save()
    2. Utilisation  : load() → search(query_vector, k)
    """

    def __init__(self):
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []

    # ── CONSTRUCTION ──────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """
        Construit l'index FAISS depuis les vecteurs et métadonnées.

        IMPORTANT : vectors[i] doit correspondre à metadata[i].
        FAISS indexe les vecteurs par entier (0, 1, 2...) — on retrouve
        les métadonnées en utilisant cet entier comme clé de liste.
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Désalignement : {len(vectors)} vecteurs "
                f"mais {len(metadata)} métadonnées"
            )

        n, dim = vectors.shape
        logger.info(f"Construction index FAISS : {n} vecteurs, dim {dim}")
        t0 = time.perf_counter()

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._metadata = metadata

        elapsed = time.perf_counter() - t0
        logger.info(f"Index construit en {elapsed:.2f}s — {self._index.ntotal} vecteurs")

    def save(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Sauvegarde l'index FAISS et les métadonnées sur disque.
        """
        if self._index is None:
            raise RuntimeError("Index non construit. Appelle build() d'abord.")

        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        faiss.write_index(self._index, str(index_path))
        size_kb = index_path.stat().st_size / 1024
        logger.info(f"Index FAISS : {index_path} ({size_kb:.0f} KB)")

        metadata_path = directory / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Métadonnées : {metadata_path}")

        info = {
            "n_vectors":  self._index.ntotal,
            "dimension":  self._index.d,
            "index_type": "IndexFlatIP",
            "metric":     "cosine (normalized L2 + inner product)",
            "n_chunks":   len(self._metadata),
            "min_score":  MIN_SCORE,
        }
        with (directory / "index_info.json").open("w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Vector store sauvegardé dans {directory}/")

    # ── CHARGEMENT ────────────────────────────────────────────────────────────

    def load(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Charge l'index et les métadonnées depuis le disque.
        """
        index_path    = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index non trouvé : {index_path}\n"
                f"Lance d'abord : python -m rag.build_index"
            )

        logger.info(f"Chargement index FAISS depuis {directory}/")
        t0 = time.perf_counter()

        self._index = faiss.read_index(str(index_path))
        with metadata_path.open(encoding="utf-8") as f:
            self._metadata = json.load(f)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Index chargé en {elapsed:.2f}s "
            f"— {self._index.ntotal} vecteurs, dim {self._index.d}"
        )

    # ── RECHERCHE ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        regulation_filter: str | None = None,
        min_score: float = MIN_SCORE,
    ) -> list[dict]:
        """
        Recherche les k chunks les plus similaires à la requête.

        CORRECTIONS v1.1 :

        1. DÉDUPLICATION PAR segment_id :
           Le chunking avec overlap produit plusieurs chunks très similaires
           issus du même segment (ex: Article 5 chunk_0 et chunk_1 ont
           200 chars en commun → cosine similarity élevée).
           Sans déduplication, le même article apparaît 2-3 fois.
           Solution : on garde uniquement le chunk avec le meilleur score
           pour chaque segment_id unique (le premier = meilleur score car
           FAISS trie par score décroissant).

        2. SCORE MINIMUM (min_score=0.35) :
           En dessous de 0.35, le chunk n'est pas pertinent.
           On arrête dès qu'on passe sous le seuil — les résultats suivants
           seront encore moins bons (tri décroissant garanti par FAISS).

        PARAMÈTRES :
            query_vector      : vecteur de la question, shape (1, 384)
            k                 : nombre de résultats souhaités après filtrage
            regulation_filter : si "GDPR", ne retourne que les chunks GDPR
            min_score         : score minimum de similarité (défaut : 0.35)
        """
        if self._index is None:
            raise RuntimeError("Index non chargé. Appelle load() d'abord.")

        # On cherche k*5 candidats pour avoir assez après déduplication + filtrage
        search_k = k * 5

        scores, indices = self._index.search(query_vector, search_k)

        results: list[dict] = []

        # seen_segments : ensemble des segment_id déjà ajoutés.
        # Garantit qu'un même segment n'apparaît qu'une seule fois.
        seen_segments: set[str] = set()

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            # Score minimum — on arrête ici, les suivants sont encore moins bons
            if score < min_score:
                break

            chunk = dict(self._metadata[idx])
            chunk["similarity_score"] = float(score)

            # Filtre par réglementation
            if regulation_filter and chunk.get("regulation") != regulation_filter:
                continue

            # Déduplication par segment_id
            segment_id = chunk.get("segment_id", "")
            if segment_id in seen_segments:
                continue

            seen_segments.add(segment_id)
            results.append(chunk)

            if len(results) >= k:
                break

        return results

    @property
    def n_vectors(self) -> int:
        """Nombre de vecteurs dans l'index."""
        return self._index.ntotal if self._index else 0
