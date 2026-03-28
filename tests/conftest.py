"""
tests/conftest.py — Configuration globale de pytest

Ce fichier est chargé automatiquement par pytest avant les tests.
Il configure l'environnement de test pour éviter les effets de bord.
"""

import os
import sys
import tempfile

# ── Variables d'environnement de test ────────────────────────────────────────
# Configurées AVANT tout import des modules api.*
# Évite de charger les vrais modèles ML ou de contacter l'API Anthropic.

os.environ.setdefault("JWT_SECRET_KEY",    "test_secret_key_for_pytest_only")
os.environ.setdefault("DISABLE_DISCLAIMER","false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Dossier temporaire pour les logs d'audit — isolé par session de test
_TEMP_DIR = tempfile.mkdtemp(prefix="pytest_audit_")
os.environ.setdefault("AUDIT_LOG_DIR", _TEMP_DIR)

# Ajoute la racine du projet au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
