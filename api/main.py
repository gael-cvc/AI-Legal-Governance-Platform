"""
main.py — Point d'entrée de l'application FastAPI

DÉMARRAGE :
    uvicorn api.main:app --reload --port 8000

    --reload : redémarre automatiquement à chaque modification de fichier.
               NE PAS utiliser en production (instable, trop de ressources).
    --port   : port HTTP d'écoute (8000 par convention pour les APIs Python)

DOCUMENTATION AUTOMATIQUE GÉNÉRÉE PAR FASTAPI :
    http://localhost:8000/docs    → Swagger UI : interface interactive pour
                                    tester les endpoints directement depuis le navigateur
    http://localhost:8000/redoc  → ReDoc : documentation lisible, format alternatif

ARCHITECTURE DES SINGLETONS :
    L'index FAISS (2016 vecteurs × 384 dimensions), le modèle d'embedding
    (all-MiniLM-L6-v2, ~80MB) et le cross-encoder (ms-marco-MiniLM-L-6-v2,
    ~67MB) sont des ressources lourdes à charger.
    Les charger à chaque requête serait catastrophique (~3s/requête).

    Solution : on les charge UNE SEULE FOIS au démarrage dans des variables
    globales _vector_store, _embedder et _reranker. Toutes les requêtes
    partagent ces instances sans les recréer.

    Temps de démarrage : ~5-8 secondes (index + bi-encoder + cross-encoder)
    Temps par requête  : ~500-800ms (recherche + reranking + LLM)

POURQUOI LIFESPAN ET PAS @app.on_event ?
    FastAPI < 0.93 utilisait les décorateurs @app.on_event("startup") et
    @app.on_event("shutdown") pour gérer le cycle de vie.
    Ces décorateurs sont dépréciés depuis FastAPI 0.93.

    Le pattern lifespan (contextmanager async) est plus propre :
    - Le code AVANT yield = startup (chargement ressources)
    - Le yield = l'application est active et répond aux requêtes
    - Le code APRÈS yield = shutdown (libération ressources)

    Avantages du lifespan vs on_event :
    1. Tout le cycle de vie est dans une seule fonction → plus lisible
    2. Les variables du startup sont dans le scope du shutdown → pas de globals cachés
    3. La gestion d'erreur au démarrage est plus naturelle (try/except autour du yield)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Charge les variables d'environnement depuis le fichier .env à la racine.
# Le fichier .env contient typiquement :
#   ANTHROPIC_API_KEY=sk-ant-...
#   TRANSFORMERS_OFFLINE=1
# load_dotenv() doit être appelé le plus tôt possible, avant tout import
# qui pourrait lire os.getenv().
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api.main")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETONS GLOBAUX
# ══════════════════════════════════════════════════════════════════════════════

# Ces variables sont None avant le démarrage de l'application.
# Elles sont initialisées dans le lifespan() et accédées via les getters.
#
# POURQUOI DES GETTERS plutôt qu'accès direct ?
# Les autres modules (search.py, health.py) importent ces fonctions.
# Si on importait directement _vector_store, ils obtiendraient None (la valeur
# au moment de l'import, avant le startup). Avec les fonctions, ils obtiennent
# la valeur ACTUELLE au moment de l'appel (après le startup).

_vector_store = None   # instance VectorStore (index FAISS + métadonnées)
_embedder     = None   # instance LegalEmbedder (bi-encoder sentence-transformers)
_reranker     = None   # instance LegalReranker (cross-encoder ms-marco)


def get_vector_store():
    """Retourne l'instance VectorStore globale. None si pas encore chargée."""
    return _vector_store


def get_embedder():
    """Retourne l'instance LegalEmbedder globale. None si pas encore chargée."""
    return _embedder


def get_reranker():
    """
    Retourne l'instance LegalReranker (cross-encoder) globale.
    None si le lifespan n'a pas encore tourné ou si le chargement a échoué.
    search.py vérifie reranker.is_available avant d'appeler rerank().
    """
    return _reranker


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN : STARTUP ET SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie complet de l'application FastAPI.

    SÉQUENCE DE STARTUP (avant yield) :
    1. Chargement de l'index FAISS depuis vector_store/ (~20ms)
    2. Chargement du bi-encoder all-MiniLM-L6-v2 (~2s)
    3. Chargement du cross-encoder ms-marco-MiniLM-L-6-v2 (~1s)
    4. Vérification de la clé API Anthropic
    5. Log de confirmation + URLs

    SÉQUENCE DE SHUTDOWN (après yield) :
    Libération des références pour le garbage collector.

    GESTION D'ERREUR AU STARTUP :
    On ne raise pas d'exception si un composant ne se charge pas.
    L'API démarre quand même avec les composants disponibles.
    Le cross-encoder en particulier peut échouer sans bloquer le reste :
    search.py utilise l'ordre FAISS en fallback si reranker indisponible.
    """
    global _vector_store, _embedder, _reranker

    logger.info("=" * 60)
    logger.info("  Démarrage — AI Legal Governance Intelligence Platform  ")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # ── 1. Chargement index FAISS ─────────────────────────────────────────────
    try:
        from rag.vector_store import VectorStore
        _vector_store = VectorStore()
        _vector_store.load()
        logger.info(f"✓ Index FAISS : {_vector_store.n_vectors} vecteurs chargés")
    except FileNotFoundError:
        logger.error(
            "✗ Index FAISS non trouvé dans vector_store/. "
            "Lancez d'abord : python -m rag.build_index"
        )

    # ── 2. Chargement bi-encoder (embeddings) ─────────────────────────────────
    # device="mps" dans embedder.py pour éviter les crashs sur Mac Apple Silicon
    try:
        from rag.embedder import LegalEmbedder
        _embedder = LegalEmbedder()
        _embedder.load()
        logger.info(
            f"✓ Embedder : {_embedder.model_name} | "
            f"dimension={_embedder.dimension}"
        )
    except Exception as e:
        logger.error(f"✗ Erreur chargement embedder : {e}")

    # ── 3. Chargement cross-encoder (reranking) ───────────────────────────────
    # Ce modèle re-score les candidats FAISS en lisant (question + chunk) ensemble.
    # Plus précis que la similarité vectorielle seule.
    # En cas d'échec : reranking désactivé, order FAISS utilisé en fallback.
    try:
        from rag.reranker import LegalReranker
        _reranker = LegalReranker()
        _reranker.load()
        if _reranker.is_available:
            logger.info(f"✓ Reranker : {_reranker.model_name} chargé")
        else:
            logger.warning("⚠ Reranker : chargement échoué, fallback ordre FAISS")
    except Exception as e:
        logger.error(f"✗ Erreur chargement reranker : {e}")

    # ── 4. Vérification clé API Anthropic ────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"✓ Anthropic API : clé présente ({masked})")
    else:
        logger.warning(
            "⚠ ANTHROPIC_API_KEY absente — mode fallback sans LLM actif. "
            "Ajoutez ANTHROPIC_API_KEY=sk-ant-... dans .env"
        )

    # ── 5. Finalisation ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    from api.health import set_start_time
    set_start_time(time.time())

    logger.info(f"✓ API prête en {elapsed:.2f}s")
    logger.info("  Swagger UI   → http://localhost:8000/docs")
    logger.info("  Health check → http://localhost:8000/api/v1/health")
    logger.info("=" * 60)

    yield  # ← L'application est active ici : répond aux requêtes HTTP

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Arrêt de l'API — libération des ressources...")
    _vector_store = None
    _embedder     = None
    _reranker     = None
    logger.info("API arrêtée proprement.")


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "AI Legal Governance Intelligence Platform",
    description = (
        "**API de recherche juridique intelligente basée sur RAG.**\n\n"
        "Interrogez le corpus réglementaire européen en langage naturel.\n\n"
        "**Corpus indexé :** GDPR (2016) · EU AI Act (2024) · "
        "EDPB Guidelines · CNIL Recommendations · Data Governance Act (2022)\n\n"
        "**Pipeline :** Question → Query Expansion → FAISS (2016 vecteurs) → "
        "Reranking cross-encoder → Claude (synthèse + citations)\n\n"
        "**Stack :** FastAPI · sentence-transformers · FAISS · Anthropic Claude"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ══════════════════════════════════════════════════════════════════════════════
# CORS (Cross-Origin Resource Sharing)
# ══════════════════════════════════════════════════════════════════════════════

# EN DÉVELOPPEMENT : allow_origins=["*"] autorise tout le monde → pratique
# EN PRODUCTION : remplacer ["*"] par l'URL exacte du frontend déployé
#   ex: allow_origins=["https://ai-legal-platform.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTERS
# ══════════════════════════════════════════════════════════════════════════════

# Importés ICI (après les getters) pour éviter les imports circulaires.
# search.py et health.py importent get_vector_store/get_embedder/get_reranker.

from api.health import router as health_router
from api.search import router as search_router

app.include_router(health_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT RACINE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"], summary="Index de l'API")
async def root() -> dict:
    """Endpoint racine — répertoire des URLs disponibles."""
    return {
        "name":        "AI Legal Governance Intelligence Platform",
        "version":     "1.0.0",
        "description": "RAG-based legal research API for EU regulations",
        "endpoints": {
            "docs":        "GET  /docs                        → Swagger UI interactif",
            "redoc":       "GET  /redoc                       → Documentation ReDoc",
            "health":      "GET  /api/v1/health               → Statut + stats index",
            "search":      "POST /api/v1/search               → Recherche juridique RAG",
            "suggestions": "GET  /api/v1/search/suggestions   → Questions suggérées",
        },
    }
