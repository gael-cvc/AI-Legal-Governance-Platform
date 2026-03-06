"""
vector_store.py — Index FAISS et recherche vectorielle
=============================================================================

RÔLE DE CE FICHIER :
Stocker les vecteurs des chunks juridiques et les retrouver rapidement
par similarité sémantique quand un utilisateur pose une question.

QU'EST-CE QUE FAISS ?
━━━━━━━━━━━━━━━━━━━━
FAISS (Facebook AI Similarity Search) est une librairie C++ avec bindings
Python, développée par Meta Research. Son rôle : trouver les N vecteurs
les plus proches d'un vecteur requête parmi des millions de candidats.

ANALOGIE GÉOMÉTRIQUE :
Imagine 2016 points dans un espace à 384 dimensions. Chaque point = un chunk.
Quand tu poses une question, son vecteur est un nouveau point dans cet espace.
FAISS trouve les points les plus proches de ce nouveau point.
Ces points proches = les chunks les plus pertinents pour ta question.

INDEX CHOISI : IndexFlatIP
━━━━━━━━━━━━━━━━━━━━━━━━━━
    "Flat" = recherche exacte (compare avec TOUS les vecteurs sans approximation)
    "IP"   = Inner Product = produit scalaire

    POURQUOI IndexFlatIP et pas IndexFlatL2 (distance euclidienne) ?
    Avec des vecteurs normalisés L2 (norme = 1), le produit scalaire A·B est
    exactement la cosine similarity. La cosine similarity mesure l'ANGLE entre
    les vecteurs, pas leur distance absolue — c'est ce qu'on veut pour la
    sémantique : deux textes "similaires" pointent dans la même direction
    quelle que soit leur longueur.

    POURQUOI PAS IndexIVFFlat (plus rapide mais approximatif) ?
    IndexIVFFlat divise l'espace en clusters (comme des zones géographiques)
    et ne cherche que dans les clusters proches. Pour 100k+ vecteurs, c'est
    indispensable (IndexFlatIP deviendrait trop lent).
    Pour nos 2016 vecteurs, IndexFlatIP est exact ET rapide (~1ms par requête).

CORRECTIONS v1.1 (deux bugs corrigés) :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    BUG 1 — Doublons dans les résultats :
    Le chunking avec overlap produit des chunks très similaires du même segment.
    Ex: Article 5 est découpé en chunk_0 et chunk_1 qui partagent 200 chars.
    Ces deux chunks ont une cosine similarity élevée avec la même question.
    Sans déduplication, Article 5 apparaissait 2 fois dans les résultats.
    CORRECTION : on filtre par segment_id unique, en gardant uniquement
    le chunk avec le meilleur score (= le premier, FAISS trie décroissant).

    BUG 2 — Résultats peu pertinents inclus :
    En dessous d'un certain score de cosine similarity, le chunk n'est
    pas sémantiquement lié à la question — c'est du bruit.
    CORRECTION : filtre MIN_SCORE = 0.35 (seuil calibré sur notre corpus).

FICHIERS PRODUITS PAR save() :
    data/vector_store/index.faiss     ← vecteurs FAISS (binaire)
    data/vector_store/metadata.json   ← métadonnées de chaque chunk (JSON)
    data/vector_store/index_info.json ← infos de l'index (stats)
=============================================================================
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

# ── Seuil de score minimum ────────────────────────────────────────────────────
# Cosine similarity entre 0 et 1 (avec vecteurs normalisés L2) :
#   > 0.70  = très pertinent : sujet identique ou très proche
#   0.50-0.70 = pertinent    : sujet directement lié
#   0.35-0.50 = acceptable   : sujet connexe, information utile
#   < 0.35  = trop faible    : sujet différent, résultat non pertinent (bruit)
#
# Calibré sur notre corpus : en dessous de 0.35, les résultats n'apportent
# pas d'information utile à la question juridique posée.
MIN_SCORE = 0.35


class VectorStore:
    """
    Gère l'index FAISS et les métadonnées associées.

    Cette classe a deux modes d'utilisation :

    MODE CONSTRUCTION (build_index.py) :
        store = VectorStore()
        store.build(vectors, metadata)  # charge les données en mémoire
        store.save()                    # écrit sur disque

    MODE RECHERCHE (API FastAPI) :
        store = VectorStore()
        store.load()                    # lit depuis disque
        results = store.search(query_vector, k=5)

    ALIGNEMENT VECTEURS ↔ MÉTADONNÉES :
    FAISS indexe les vecteurs par entier (0, 1, 2, ..., n-1).
    quand FAISS retourne l'index i, on retrouve les métadonnées via self._metadata[i].
    Il est donc IMPÉRATIF que vectors[i] corresponde à metadata[i].
    """

    def __init__(self):
        # L'index FAISS contient les vecteurs float32
        # None tant que build() ou load() n'a pas été appelé
        self._index: faiss.Index | None = None

        # Liste des métadonnées, alignée avec l'index FAISS :
        # self._metadata[i] = métadonnées du vecteur i dans l'index
        self._metadata: list[dict] = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONSTRUCTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """
        Construit l'index FAISS depuis les vecteurs encodés et leurs métadonnées.

        FONCTIONNEMENT INTERNE DE FAISS :
        faiss.IndexFlatIP(dim) crée un index vide qui attend des vecteurs de
        dimension `dim`. L'appel à .add(vectors) copie les vecteurs dans la
        mémoire interne de FAISS (structure de données optimisée pour la recherche).
        Après .add(), chaque vecteur a un identifiant entier (0, 1, 2...)
        qui correspond à son ordre d'insertion.

        VÉRIFICATION D'ALIGNEMENT :
        len(vectors) doit être égal à len(metadata). Si ce n'est pas le cas,
        les résultats de recherche retourneraient les mauvaises métadonnées
        (ex: rechercher le chunk 42 et retrouver les métadonnées du chunk 17).

        Paramètres :
            vectors  : np.ndarray de shape (n, 384), dtype float32
                       chaque ligne = vecteur d'un chunk, normalisés L2
            metadata : list[dict] de longueur n
                       metadata[i] = métadonnées du vecteur vectors[i]
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Désalignement critique : {len(vectors)} vecteurs "
                f"mais {len(metadata)} métadonnées. "
                f"Ils doivent avoir la même longueur."
            )

        n, dim = vectors.shape  # ex: n=2016, dim=384
        logger.info(f"Construction index FAISS : {n} vecteurs, dimension {dim}")
        t0 = time.perf_counter()

        # IndexFlatIP(dim) = index de recherche exacte par produit scalaire
        # dim = dimension des vecteurs (384 pour all-MiniLM-L6-v2)
        self._index = faiss.IndexFlatIP(dim)

        # .add() insère tous les vecteurs dans l'index d'un coup.
        # FAISS fait une copie interne : on peut modifier `vectors` après sans problème.
        self._index.add(vectors)

        # On garde les métadonnées en mémoire, alignées avec l'index
        self._metadata = metadata

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Index construit en {elapsed:.2f}s | "
            f"{self._index.ntotal} vecteurs indexés"
        )

    def save(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Sauvegarde l'index FAISS et les métadonnées sur le disque.

        TROIS FICHIERS ÉCRITS :
        1. index.faiss     : les vecteurs en format binaire FAISS
                             (non lisible par un humain, format propriétaire)
        2. metadata.json   : les métadonnées en JSON lisible
                             (un dict par chunk : regulation, article, texte, page...)
        3. index_info.json : statistiques de l'index (nombre de vecteurs, dimension...)
                             Utile pour vérifier l'index sans le charger complètement

        Paramètre :
            directory : répertoire de destination
                        Défaut : data/vector_store/
        """
        if self._index is None:
            raise RuntimeError(
                "Index non construit. Appelle build() avant save()."
            )

        directory.mkdir(parents=True, exist_ok=True)

        # ── Sauvegarde de l'index FAISS ───────────────────────────────────────
        # faiss.write_index() sérialise l'index en format binaire FAISS.
        # str(index_path) car FAISS attend une string, pas un Path object.
        index_path = directory / "index.faiss"
        faiss.write_index(self._index, str(index_path))
        size_kb = index_path.stat().st_size / 1024
        logger.info(f"Index FAISS : {index_path} ({size_kb:.0f} KB)")

        # ── Sauvegarde des métadonnées ────────────────────────────────────────
        # JSON lisible avec indent=2 pour faciliter le debug et l'audit.
        # ensure_ascii=False : conserve les caractères accentués (é, à, ü...)
        metadata_path = directory / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Métadonnées : {metadata_path} ({len(self._metadata)} chunks)")

        # ── Sauvegarde des infos de l'index ──────────────────────────────────
        info = {
            "n_vectors":  self._index.ntotal,     # nombre de vecteurs
            "dimension":  self._index.d,           # dimension des vecteurs (384)
            "index_type": "IndexFlatIP",           # type d'index FAISS
            "metric":     "cosine (L2-normalized inner product)",
            "n_chunks":   len(self._metadata),     # = n_vectors
            "min_score":  MIN_SCORE,               # seuil de filtrage
        }
        with (directory / "index_info.json").open("w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Vector store complet sauvegardé dans : {directory}/")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHARGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def load(self, directory: Path = VECTOR_STORE_DIR) -> None:
        """
        Charge l'index FAISS et les métadonnées depuis le disque en mémoire RAM.

        Appelé UNE SEULE FOIS au démarrage de l'API (dans main.py lifespan).
        Après le chargement, l'index reste en RAM et toutes les requêtes
        de recherche y accèdent directement — pas de lecture disque par requête.

        Durée : ~20ms (index.faiss = ~3 Mo, lecture séquentielle rapide)

        Lève FileNotFoundError si l'index n'existe pas encore
        (= build_index.py n'a pas encore été lancé).
        """
        index_path    = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS non trouvé : {index_path}\n"
                f"Crée d'abord l'index avec : python -m rag.build_index"
            )

        logger.info(f"Chargement index FAISS depuis {directory}/")
        t0 = time.perf_counter()

        # faiss.read_index() désérialise l'index binaire en mémoire.
        # L'index chargé est identique à celui créé par build() — même vecteurs,
        # même structure interne, même performance de recherche.
        self._index = faiss.read_index(str(index_path))

        # Chargement des métadonnées JSON
        with metadata_path.open(encoding="utf-8") as f:
            self._metadata = json.load(f)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Index chargé en {elapsed:.3f}s | "
            f"{self._index.ntotal} vecteurs | "
            f"dimension {self._index.d}"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RECHERCHE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        regulation_filter: str | None = None,
        min_score: float = MIN_SCORE,
    ) -> list[dict]:
        """
        Recherche les k chunks les plus similaires à la question posée.

        PIPELINE DE RECHERCHE INTERNE :
        ┌─────────────────────────────────────────────────────────────────┐
        │  1. self._index.search(query_vector, search_k)                  │
        │     FAISS calcule les produits scalaires entre query_vector      │
        │     et TOUS les vecteurs de l'index, retourne les k*5 meilleurs  │
        │     Durée : ~1ms pour 2016 vecteurs                             │
        │                                                                 │
        │  2. Filtrage score < min_score                                  │
        │     On ignore les chunks en dessous du seuil de pertinence      │
        │     (0.35 = seuil calibré sur notre corpus)                     │
        │                                                                 │
        │  3. Filtrage par réglementation (optionnel)                     │
        │     Si regulation_filter="GDPR", on ignore les chunks non-GDPR  │
        │                                                                 │
        │  4. Déduplication par segment_id                                │
        │     Un même article (ex: Article 5) peut avoir plusieurs chunks  │
        │     très similaires (chunk_0, chunk_1). On garde seulement      │
        │     le meilleur (le premier = plus grand score, FAISS tri desc.) │
        │                                                                 │
        │  5. On s'arrête dès qu'on a k résultats propres                 │
        └─────────────────────────────────────────────────────────────────┘

        POURQUOI search_k = k * 5 et pas juste k ?
        Si on cherche k=5 résultats mais que les 3 premiers sont dédupliqués
        (même segment_id) ou filtrés (mauvaise réglementation), on n'aurait
        que 2 résultats au final. En cherchant k*5 = 25 candidats initiaux,
        on s'assure d'avoir assez de matière pour obtenir k résultats propres.

        RETOUR DE FAISS :
        self._index.search() retourne deux tableaux numpy :
        - scores  : shape (1, search_k) — scores décroissants (le meilleur = premier)
        - indices : shape (1, search_k) — index entiers dans l'index FAISS
          (idx=42 signifie "le 42ème vecteur inséré" → self._metadata[42])
        - Si FAISS manque de candidats, les cases restantes valent -1 (indice invalide)

        Paramètres :
            query_vector      : vecteur de la question, shape (1, 384), normalisé L2
            k                 : nombre de résultats souhaités après tous les filtres
            regulation_filter : "GDPR" / "EU_AI_ACT" / "CNIL" / None (= pas de filtre)
            min_score         : score minimum de cosine similarity (défaut : 0.35)

        Retourne :
            list[dict] : chunks pertinents avec leurs métadonnées + similarity_score
                         Triés par score décroissant (le plus pertinent en premier)
        """
        if self._index is None:
            raise RuntimeError(
                "Index non chargé. Appelle load() avant search()."
            )

        # On cherche k*5 candidats bruts pour avoir assez de marge
        # après déduplication, filtrage réglementation et filtrage score
        search_k = k * 5

        # self._index.search() fait la recherche FAISS.
        # Retourne : (scores array (1, search_k), indices array (1, search_k))
        # [0] = première (et seule) requête dans le batch
        scores, indices = self._index.search(query_vector, search_k)

        results: list[dict] = []

        # seen_segments : ensemble des segment_id déjà ajoutés aux résultats.
        # Si un segment_id apparaît une 2ème fois, on l'ignore (déduplication).
        seen_segments: set[str] = set()

        for score, idx in zip(scores[0], indices[0]):

            # idx == -1 : FAISS n'a pas trouvé assez de vecteurs
            # (ne devrait pas arriver si search_k <= n_vectors)
            if idx == -1:
                continue

            # Filtrage par score minimum.
            # FAISS trie les résultats par score DÉCROISSANT → dès qu'on passe
            # sous le seuil, tous les suivants seront encore moins bons.
            # On peut donc faire un break immédiat (optimisation).
            if score < min_score:
                break

            # Copie défensive du dict de métadonnées pour ne pas modifier l'original
            chunk = dict(self._metadata[idx])

            # On ajoute le score de similarité pour que l'API puisse le retourner
            chunk["similarity_score"] = float(score)

            # ── Filtre par réglementation ─────────────────────────────────────
            # Si un filtre est spécifié, on ignore les chunks d'autres réglementations
            if regulation_filter and chunk.get("regulation") != regulation_filter:
                continue

            # ── Déduplication par segment_id ─────────────────────────────────
            # On ne retourne qu'un seul chunk par segment juridique.
            # Le premier rencontré = le meilleur score (tri FAISS décroissant).
            segment_id = chunk.get("segment_id", "")
            if segment_id in seen_segments:
                continue  # Ce segment est déjà dans les résultats → on saute

            seen_segments.add(segment_id)
            results.append(chunk)

            # On s'arrête dès qu'on a k résultats propres
            if len(results) >= k:
                break

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROPRIÉTÉS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def n_vectors(self) -> int:
        """
        Nombre de vecteurs dans l'index.

        Utilisé par :
        - /health pour retourner les stats de l'index
        - Les guards "if store.n_vectors == 0" dans l'API
        Retourne 0 si l'index n'est pas encore chargé (évite un AttributeError).
        """
        return self._index.ntotal if self._index else 0
