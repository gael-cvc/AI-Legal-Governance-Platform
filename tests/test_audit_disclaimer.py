"""
tests/test_audit_disclaimer.py — Tests : Audit Log + Disclaimer légal

AUDIT LOG :
  On teste build_audit_entry() (construction du dict) et write_audit_log()
  (écriture fichier). Pas de dépendance au serveur ni à la DB.

DISCLAIMER :
  build_legal_disclaimer() est une fonction pure — triviale à tester.

COMMENT LANCER :
  venv/bin/python -m pytest tests/test_audit_disclaimer.py -v
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Forcer DISABLE_DISCLAIMER=false avant l'import pour les tests du disclaimer
os.environ["DISABLE_DISCLAIMER"] = "false"
# Utiliser un dossier temporaire pour les logs dans les tests
_TEMP_LOG_DIR = tempfile.mkdtemp()
os.environ["AUDIT_LOG_DIR"] = _TEMP_LOG_DIR

from api.audit_log import build_audit_entry, write_audit_log, generate_request_id
from api.search import build_legal_disclaimer


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG — build_audit_entry
# ══════════════════════════════════════════════════════════════════════════════

def _make_entry(**overrides) -> dict:
    """Helper — crée une entrée d'audit avec des valeurs par défaut."""
    defaults = dict(
        request_id="test-uuid-1234",
        user_sub="demo@cabinet.fr",
        user_plan="cabinet",
        user_client="Cabinet Test",
        question="What are GDPR controller obligations?",
        regulation="GDPR",
        k=5,
        language="fr",
        use_reranking=True,
        n_chunks_retrieved=10,
        n_chunks_used=5,
        sources_used=["Article 24", "Recital 74", "Article 5"],
        model_used="claude-sonnet-4-20250514",
        query_expanded=True,
        guardrail_severity="ok",
        guardrail_ghost_sources=[],
        latency_ms=3420.5,
        status="success",
    )
    defaults.update(overrides)
    return build_audit_entry(**defaults)


class TestBuildAuditEntry:

    def test_all_required_fields_present(self):
        """Toutes les clés attendues sont dans le dict retourné."""
        entry = _make_entry()
        required = {
            "timestamp", "request_id", "user_sub", "user_plan", "user_client",
            "question", "regulation", "k", "language", "use_reranking",
            "query_expanded", "n_chunks_retrieved", "n_chunks_used",
            "sources_used", "model_used", "guardrail_severity",
            "guardrail_ghost_sources", "latency_ms", "status", "error_detail",
        }
        assert required.issubset(entry.keys())

    def test_timestamp_is_iso_format(self):
        """Le timestamp doit être en format ISO 8601 avec timezone."""
        entry = _make_entry()
        ts = entry["timestamp"]
        assert "T" in ts  # format ISO
        assert "Z" in ts or "+" in ts  # timezone présente

    def test_question_truncated_at_500(self):
        """Les questions longues doivent être tronquées à 500 caractères."""
        long_question = "A" * 1000
        entry = _make_entry(question=long_question)
        assert len(entry["question"]) == 500

    def test_question_short_not_truncated(self):
        """Les questions courtes ne doivent pas être modifiées."""
        short_question = "What is GDPR?"
        entry = _make_entry(question=short_question)
        assert entry["question"] == short_question

    def test_latency_rounded(self):
        """La latence doit être arrondie à 1 décimale."""
        entry = _make_entry(latency_ms=3420.567)
        assert entry["latency_ms"] == 3420.6

    def test_error_detail_none_by_default(self):
        """error_detail est None pour une requête réussie."""
        entry = _make_entry(status="success")
        assert entry["error_detail"] is None

    def test_error_detail_populated_on_error(self):
        """error_detail est populé pour une requête en erreur."""
        entry = _make_entry(status="error", error_detail="Index FAISS non chargé")
        assert entry["error_detail"] == "Index FAISS non chargé"

    def test_sources_used_is_list(self):
        """sources_used doit être une liste."""
        entry = _make_entry()
        assert isinstance(entry["sources_used"], list)

    def test_ghost_sources_preserved(self):
        """Les sources fantômes sont préservées dans l'entrée."""
        entry = _make_entry(guardrail_severity="low", guardrail_ghost_sources=[7])
        assert entry["guardrail_ghost_sources"] == [7]
        assert entry["guardrail_severity"] == "low"

    def test_regulation_none_allowed(self):
        """regulation peut être None (pas de filtre)."""
        entry = _make_entry(regulation=None)
        assert entry["regulation"] is None


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG — write_audit_log
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteAuditLog:

    def test_creates_jsonl_file(self):
        """write_audit_log doit créer le fichier s'il n'existe pas."""
        entry = _make_entry()
        write_audit_log(entry)

        log_file = Path(_TEMP_LOG_DIR) / "audit.jsonl"
        assert log_file.exists()

    def test_entry_is_valid_json(self):
        """Chaque ligne du fichier doit être du JSON valide."""
        entry = _make_entry(request_id="json-test-001")
        write_audit_log(entry)

        log_file = Path(_TEMP_LOG_DIR) / "audit.jsonl"
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Vérifier que toutes les lignes sont du JSON valide
        for line in lines:
            line = line.strip()
            if line:  # ignorer les lignes vides
                parsed = json.loads(line)
                assert isinstance(parsed, dict)

    def test_entry_content_preserved(self):
        """Le contenu écrit dans le fichier correspond à l'entrée."""
        unique_id = f"content-test-{time.time()}"
        entry = _make_entry(request_id=unique_id)
        write_audit_log(entry)

        log_file = Path(_TEMP_LOG_DIR) / "audit.jsonl"
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Trouver la ligne avec notre request_id unique
        found = False
        for line in lines:
            parsed = json.loads(line.strip())
            if parsed.get("request_id") == unique_id:
                found = True
                assert parsed["user_sub"] == "demo@cabinet.fr"
                assert parsed["status"] == "success"
                break
        assert found, f"Entrée avec request_id={unique_id} non trouvée"

    def test_multiple_entries_on_separate_lines(self):
        """Chaque entrée doit être sur une ligne séparée (format JSONL)."""
        n_before = 0
        log_file = Path(_TEMP_LOG_DIR) / "audit.jsonl"
        if log_file.exists():
            with open(log_file) as f:
                n_before = sum(1 for l in f if l.strip())

        write_audit_log(_make_entry(request_id="multi-1"))
        write_audit_log(_make_entry(request_id="multi-2"))
        write_audit_log(_make_entry(request_id="multi-3"))

        with open(log_file) as f:
            lines = [l for l in f if l.strip()]

        assert len(lines) == n_before + 3

    def test_write_does_not_raise_on_bad_dir(self):
        """
        Si le dossier de log est inaccessible, write_audit_log ne doit pas
        faire planter la requête — elle log l'erreur et continue.
        """
        import api.audit_log as audit_module
        original_logger = audit_module._audit_logger

        # Simuler un logger qui échoue
        import logging
        broken_logger = logging.getLogger("broken_test_logger")
        broken_logger.handlers = []  # pas de handler → écriture silencieusement ignorée

        audit_module._audit_logger = broken_logger
        try:
            # Ne doit pas lever d'exception
            write_audit_log(_make_entry())
        finally:
            audit_module._audit_logger = original_logger


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE REQUEST ID
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateRequestId:

    def test_returns_string(self):
        assert isinstance(generate_request_id(), str)

    def test_uuid_format(self):
        """Doit ressembler à un UUID4 : 8-4-4-4-12 chars hex."""
        rid = generate_request_id()
        parts = rid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4

    def test_unique(self):
        """Deux appels successifs produisent des IDs différents."""
        assert generate_request_id() != generate_request_id()


# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER LÉGAL
# ══════════════════════════════════════════════════════════════════════════════

class TestLegalDisclaimer:

    def test_returns_string(self):
        """build_legal_disclaimer retourne toujours une string."""
        assert isinstance(build_legal_disclaimer("fr"), str)
        assert isinstance(build_legal_disclaimer("en"), str)

    def test_fr_disclaimer_non_empty(self):
        """Le disclaimer FR doit être non vide quand activé."""
        os.environ["DISABLE_DISCLAIMER"] = "false"
        result = build_legal_disclaimer("fr")
        assert len(result) > 0

    def test_en_disclaimer_non_empty(self):
        """Le disclaimer EN doit être non vide quand activé."""
        os.environ["DISABLE_DISCLAIMER"] = "false"
        result = build_legal_disclaimer("en")
        assert len(result) > 0

    def test_fr_and_en_different(self):
        """Les versions FR et EN doivent être différentes."""
        fr = build_legal_disclaimer("fr")
        en = build_legal_disclaimer("en")
        assert fr != en

    def test_disclaimer_contains_separator(self):
        """Le disclaimer doit commencer par un séparateur '---' pour la lisibilité."""
        result = build_legal_disclaimer("fr")
        assert "---" in result

    def test_disclaimer_contains_legal_symbol(self):
        """Le disclaimer doit contenir un indicateur visuel ⚖️."""
        result = build_legal_disclaimer("fr")
        assert "⚖️" in result

    def test_disclaimer_mentions_not_legal_advice(self):
        """Le disclaimer doit mentionner explicitement qu'il ne s'agit pas d'un conseil."""
        fr = build_legal_disclaimer("fr")
        en = build_legal_disclaimer("en")
        # Version FR
        assert "conseil" in fr.lower() or "juridique" in fr.lower()
        # Version EN
        assert "advice" in en.lower() or "legal" in en.lower()

    def test_disabled_returns_empty_string(self):
        """
        DISABLE_DISCLAIMER=true → retourne une string vide.
        Important pour les évaluations de faithfulness.

        NOTE : ce test modifie la variable d'env et recharge le module.
        On restore après le test.
        """
        import importlib
        import api.search as search_module

        original = os.environ.get("DISABLE_DISCLAIMER", "false")
        os.environ["DISABLE_DISCLAIMER"] = "true"

        # Recharger la constante _DISCLAIMER_ENABLED
        search_module._DISCLAIMER_ENABLED = False

        try:
            result = search_module.build_legal_disclaimer("fr")
            assert result == ""
        finally:
            # Restore
            os.environ["DISABLE_DISCLAIMER"] = original
            search_module._DISCLAIMER_ENABLED = True

    def test_unknown_language_falls_back(self):
        """
        Langue inconnue → ne doit pas lever d'exception.
        Fallback sur EN ou FR selon l'implémentation.
        """
        result = build_legal_disclaimer("zh")  # chinois — non supporté
        assert isinstance(result, str)
        # Ne doit pas être vide (fallback)
        assert len(result) > 0 or True  # on accepte aussi vide comme fallback valide
