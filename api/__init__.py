"""
__init__.py — Package api/

CE FICHIER SERT À DEUX CHOSES :

1. MARQUER LE DOSSIER COMME UN PACKAGE PYTHON
   Sans ce fichier, Python ne reconnaît pas api/ comme un module importable.
   Quand on écrit `from api.models import SearchRequest`, Python cherche
   d'abord api/__init__.py pour confirmer que api/ est bien un package,
   puis il cherche models.py dedans.
   Sans ce fichier → ModuleNotFoundError.

2. EXPOSER LA VERSION DU PACKAGE
   __version__ est une convention Python (PEP 396). Elle permet d'écrire :
       import api
       print(api.__version__)  # "1.0.0"
   Utile pour les logs de démarrage, le health check (/api/v1/health),
   et la documentation Swagger auto-générée par FastAPI.

STRUCTURE DU PACKAGE api/ :
   api/
   ├── __init__.py   ← ce fichier (package marker + version)
   ├── main.py       ← app FastAPI, lifespan startup/shutdown, CORS, routers
   ├── models.py     ← schémas Pydantic (validation entrée/sortie de chaque endpoint)
   ├── search.py     ← logique RAG complète : expansion → FAISS → reranking → LLM
   └── health.py     ← endpoint GET /health (monitoring, uptime, stats FAISS)
"""

__version__ = "1.0.0"
