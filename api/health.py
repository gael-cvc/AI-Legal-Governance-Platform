"""
health.py — Endpoint GET /api/v1/health

RÔLE DU HEALTH CHECK :
  Le health check est le premier endpoint qu'un système de monitoring interroge
  pour savoir si l'API est opérationnelle. Il a trois usages distincts :

  1. MONITORING INFRASTRUCTURE
     Les outils comme Prometheus, Datadog, ou les load balancers AWS/GCP
     envoient une requête GET /health toutes les 30 secondes. Si la réponse
     est 200 avec status="ok", le service est marqué comme "healthy".
     Si elle échoue ou retourne "degraded", une alerte est déclenchée.

  2. DIAGNOSTIC DÉVELOPPEUR
     Pendant le développement, le health check permet de vérifier rapidement
     que l'index FAISS est bien chargé, que le bon nombre de vecteurs est
     en mémoire, et que les réglementations attendues sont indexées.
     C'est le premier endroit à regarder si /search ne fonctionne pas.

  3. DOCUMENTATION VIVANTE
     La liste des réglementations retournée par /health informe le client
     de ce qui est disponible dans le corpus — sans avoir à lire le code.

SÉPARATION STATUS "degraded" / ERREUR 500 :
  On distingue deux cas d'anomalie :
  - status="degraded" → l'API répond mais l'index n'est pas chargé.
    Cause : build_index.py n'a pas été lancé. L'endpoint /search retournera
    503 mais le process FastAPI est vivant.
  - Erreur 500 → crash inattendu dans notre code. Signale un bug.

  Cette distinction aide les opérateurs à diagnostiquer rapidement :
  "degraded" = problème de données, pas de bug logiciel.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from .models import HealthResponse

router = APIRouter()

# Timestamp du démarrage de l'API.
# Initialisé à l'import, puis mis à jour par set_start_time() appelé dans
# le lifespan de main.py, juste après que les ressources sont chargées.
# Utilisé pour calculer l'uptime dans la réponse health.
_start_time: float = time.time()


def set_start_time(t: float) -> None:
    """
    Enregistre le timestamp de démarrage de l'API.

    Appelé depuis main.py dans le bloc lifespan après que l'index FAISS
    et le modèle d'embedding sont chargés.

    Pourquoi pas l'initialiser directement dans main.py ?
    Parce que _start_time vit dans health.py (là où il est utilisé).
    main.py n'a pas à connaître les détails d'implémentation du health check.
    Cette fonction est l'interface entre les deux modules.
    """
    global _start_time
    _start_time = t


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    description=(
        "Vérifie que l'API est opérationnelle et retourne les statistiques de l'index FAISS. "
        "Retourne status='ok' si l'index est chargé, 'degraded' sinon."
    ),
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Construit et retourne le rapport de santé de l'API.

    DÉROULEMENT :
    1. Récupère l'instance VectorStore globale via get_vector_store() (défini dans main.py)
    2. Si le store est chargé, extrait les stats (n_vectors, liste des réglementations)
    3. Calcule l'uptime depuis _start_time
    4. Construit et retourne le HealthResponse

    IMPORT CIRCULAIRE ÉVITÉ :
    On importe get_vector_store() à l'intérieur de la fonction (lazy import)
    plutôt qu'en haut du fichier. Pourquoi ?
    health.py est importé par main.py. Si health.py importait main.py en haut
    du fichier → import circulaire → ImportError au démarrage.
    L'import à l'intérieur de la fonction est exécuté à l'appel, pas à l'import,
    donc à ce moment main.py est déjà complètement chargé.
    """
    from .main import get_vector_store  # lazy import pour éviter la circularité

    store  = get_vector_store()
    uptime = time.time() - _start_time

    n_vectors:    int       = 0
    regulations:  list[str] = []

    if store is not None and store.n_vectors > 0:
        n_vectors = store.n_vectors

        # Extrait les réglementations uniques depuis les métadonnées FAISS.
        # store._metadata est une liste de dicts, un par chunk indexé.
        # Chaque dict a un champ "regulation" (ex: "GDPR", "EU_AI_ACT"...).
        # On utilise un set pour dédupliquer, puis on trie pour un affichage stable.
        regs: set[str] = set()
        for chunk_meta in store._metadata:
            reg = chunk_meta.get("regulation", "")
            if reg:
                regs.add(reg)
        regulations = sorted(regs)

    return HealthResponse(
        status         = "ok" if n_vectors > 0 else "degraded",
        version        = "1.0.0",
        index_loaded   = n_vectors > 0,
        n_vectors      = n_vectors,
        n_regulations  = len(regulations),
        regulations    = regulations,
        uptime_seconds = round(uptime, 1),
    )
