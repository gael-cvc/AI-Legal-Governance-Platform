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
    L'index FAISS (2016 vecteurs × 384 dimensions) et le modèle d'embedding
    (all-MiniLM-L6-v2, ~90MB) sont des ressources lourdes à charger.
    Les charger à chaque requête serait catastrophique (~3s/requête).

    Solution : on les charge UNE SEULE FOIS au démarrage dans des variables
    globales _vector_store et _embedder. Toutes les requêtes partagent ces
    instances sans les recréer.

    Temps de démarrage : ~2-3 secondes (chargement index + modèle)
    Temps par requête  : ~200-500ms (recherche + reranking + LLM)

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
# load_dotenv() doit être appelé le plus tôt possible, avant tout import
# qui pourrait lire os.getenv().
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# Configuration du système de logging Python.
# Format : "2025-01-15 14:32:01,123 | INFO | api.main | Message"
# Chaque module crée son propre logger avec logging.getLogger(__name__).
# Le basicConfig ici définit le niveau et format par défaut pour tous.
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
# POURQUOI DES GETTERS (get_vector_store, get_embedder) PLUTÔT QU'ACCÈS DIRECT ?
# Les autres modules (search.py, health.py) importent ces fonctions.
# Si on importait directement _vector_store, ils obtiendraient None (la valeur
# au moment de l'import, avant le startup). Avec les fonctions, ils obtiennent
# la valeur ACTUELLE au moment de l'appel (après le startup).

_vector_store = None   # instance VectorStore (index FAISS + métadonnées)
_embedder     = None   # instance LegalEmbedder (modèle sentence-transformers)


def get_vector_store():
    """
    Retourne l'instance VectorStore globale.
    Retourne None si le lifespan n'a pas encore été exécuté.
    """
    return _vector_store


def get_embedder():
    """
    Retourne l'instance LegalEmbedder globale.
    Retourne None si le lifespan n'a pas encore été exécuté.
    """
    return _embedder


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN : STARTUP ET SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie complet de l'application FastAPI.

    SÉQUENCE DE STARTUP (avant yield) :
    1. Chargement de l'index FAISS depuis vector_store/ sur disque
    2. Chargement du modèle d'embedding (all-MiniLM-L6-v2)
    3. Vérification de la clé API Anthropic
    4. Log de confirmation + URL documentation

    SÉQUENCE DE SHUTDOWN (après yield) :
    1. Libération des références (permet au garbage collector de libérer la RAM)

    GESTION D'ERREUR AU STARTUP :
    On ne raise pas d'exception si l'index ou le modèle ne se charge pas.
    L'API démarre quand même — /health retournera "degraded" et /search
    retournera 503. C'est préférable à un crash total car :
    - Le monitoring peut détecter l'état "degraded" et alerter
    - Le développeur peut diagnostiquer via /health sans relancer uvicorn
    - Les endpoints qui ne dépendent pas de l'index (/docs, /) restent accessibles
    """
    global _vector_store, _embedder

    logger.info("=" * 60)
    logger.info("  Démarrage — AI Legal Governance Intelligence Platform  ")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # ── 1. Chargement index FAISS ─────────────────────────────────────────────
    # VectorStore.load() lit le fichier vector_store/index.faiss (vecteurs)
    # et vector_store/metadata.pkl (métadonnées des chunks).
    # Ces fichiers sont générés par : python -m rag.build_index
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
        # _vector_store reste None → /health retourne "degraded"

    # ── 2. Chargement modèle d'embedding ─────────────────────────────────────
    # LegalEmbedder.load() télécharge (premier lancement) ou charge depuis
    # le cache HuggingFace (~/.cache/huggingface/) le modèle all-MiniLM-L6-v2.
    # Taille : ~90MB. Dimension des vecteurs : 384.
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
        # _embedder reste None → /search retournera 503

    # ── 3. Vérification clé API Anthropic ────────────────────────────────────
    # On vérifie juste la présence de la clé — pas son validité.
    # Une clé invalide sera détectée au premier appel /search et activera
    # le fallback sans LLM (voir search.py generate_answer()).
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        # On masque la clé dans les logs (ne montrer que les 8 premiers caractères)
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"✓ Anthropic API : clé présente ({masked})")
    else:
        logger.warning(
            "⚠ ANTHROPIC_API_KEY absente — mode fallback sans LLM actif. "
            "Ajoutez ANTHROPIC_API_KEY=sk-ant-... dans .env"
        )

    # ── 4. Finalisation ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    # Enregistre le timestamp pour le calcul d'uptime dans /health
    from api.health import set_start_time
    set_start_time(time.time())

    logger.info(f"✓ API prête en {elapsed:.2f}s")
    logger.info("  Swagger UI   → http://localhost:8000/docs")
    logger.info("  Health check → http://localhost:8000/api/v1/health")
    logger.info("=" * 60)

    yield  # ← L'application est active ici : répond aux requêtes HTTP

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # Libère les références pour aider le garbage collector Python.
    # En pratique, le process OS est tué juste après — c'est surtout
    # utile dans les tests où on réinitialise l'app plusieurs fois.
    logger.info("Arrêt de l'API — libération des ressources...")
    _vector_store = None
    _embedder     = None
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

# POURQUOI CORS EST NÉCESSAIRE :
# Par défaut, les navigateurs bloquent les requêtes HTTP d'une origine
# (http://localhost:3000 = frontend React) vers une autre origine
# (http://localhost:8000 = notre API) → c'est la politique Same-Origin.
# Le middleware CORS ajoute les headers HTTP nécessaires pour autoriser
# ces requêtes cross-origin.
#
# EN DÉVELOPPEMENT : allow_origins=["*"] autorise tout le monde → pratique
# EN PRODUCTION : remplacer ["*"] par l'URL exacte du frontend déployé
#   ex: allow_origins=["https://ai-legal-platform.com"]
#   Ne jamais laisser ["*"] en production pour une API avec des données sensibles.

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # ← À restreindre en production
    allow_credentials = True,    # Autorise les cookies cross-origin (SSO, sessions)
    allow_methods     = ["*"],   # GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers     = ["*"],   # Authorization, Content-Type, etc.
)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTERS
# ══════════════════════════════════════════════════════════════════════════════

# On importe les routers ICI (après la définition de app et des fonctions
# get_vector_store/get_embedder) pour éviter les imports circulaires.
# search.py et health.py importent get_vector_store/get_embedder depuis main.py.
# Si on les importait en haut du fichier, main.py serait partiellement chargé
# quand ces modules essaieraient d'importer depuis lui.
#
# Le préfixe /api/v1 est une convention REST pour versionner l'API.
# Si on publie une v2 incompatible, on peut créer /api/v2 sans casser
# les clients qui utilisent encore /api/v1.

from api.health import router as health_router
from api.search import router as search_router

app.include_router(health_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")

# Résultat des préfixes :
#   GET  /api/v1/health               → health.py
#   POST /api/v1/search               → search.py
#   GET  /api/v1/search/suggestions   → search.py


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT RACINE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"], summary="Index de l'API")
async def root() -> dict:
    """
    Endpoint racine — retourne le répertoire des URLs disponibles.

    Utile pour les développeurs qui découvrent l'API : ils savent
    immédiatement où aller sans lire la documentation.
    C'est aussi une convention REST courante (HATEOAS simplifié).
    """
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
