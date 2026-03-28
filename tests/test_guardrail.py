"""
tests/test_guardrail.py — Tests unitaires : Hallucination Guardrail

CE QU'ON TESTE :
  check_hallucination_guardrail() est une fonction pure — elle prend une
  réponse texte + un nombre de chunks, et retourne un dict structuré.
  Pas de LLM, pas d'API, pas de FAISS — testable instantanément.

COMMENT LANCER :
  venv/bin/python -m pytest tests/test_guardrail.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.search import check_hallucination_guardrail


# ══════════════════════════════════════════════════════════════════════════════
# CAS NOMINAL — Aucune source fantôme
# ══════════════════════════════════════════════════════════════════════════════

class TestOkCases:

    def test_no_citations_at_all(self):
        """Réponse sans aucune citation [SOURCE X] — cas OK."""
        result = check_hallucination_guardrail(
            "The GDPR requires data controllers to implement appropriate measures.",
            n_chunks=5
        )
        assert result["flagged"] is False
        assert result["severity"] == "ok"
        assert result["ghost_sources"] == []
        assert result["response_patched"] is not None  # réponse retournée intacte

    def test_all_sources_valid(self):
        """Toutes les citations sont dans la plage [1, k]."""
        result = check_hallucination_guardrail(
            "According to [SOURCE 1], controllers must... [SOURCE 3] also states... [SOURCE 5].",
            n_chunks=5
        )
        assert result["flagged"] is False
        assert result["severity"] == "ok"
        assert result["n_cited"] == 3
        assert result["n_valid"] == 3

    def test_single_valid_source(self):
        """Une seule citation valide."""
        result = check_hallucination_guardrail(
            "As stated in [SOURCE 2], consent must be freely given.",
            n_chunks=5
        )
        assert result["flagged"] is False
        assert result["severity"] == "ok"

    def test_same_source_cited_multiple_times(self):
        """Même source citée plusieurs fois — toutes valides."""
        result = check_hallucination_guardrail(
            "[SOURCE 1] states X. [SOURCE 1] also states Y. [SOURCE 1] finally states Z.",
            n_chunks=5
        )
        assert result["flagged"] is False
        assert result["n_cited"] == 3   # 3 occurrences
        assert result["n_valid"] == 3

    def test_exact_boundary_k(self):
        """Citation [SOURCE k] exactement — doit être valide (pas fantôme)."""
        result = check_hallucination_guardrail(
            "According to [SOURCE 5], the controller must...",
            n_chunks=5
        )
        assert result["flagged"] is False
        assert result["severity"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU LOW — 1 source fantôme
# ══════════════════════════════════════════════════════════════════════════════

class TestLowSeverity:

    def test_one_ghost_source(self):
        """1 source fantôme → severity LOW, réponse retournée avec disclaimer."""
        result = check_hallucination_guardrail(
            "According to [SOURCE 7], Article 83 provides for fines.",
            n_chunks=5
        )
        assert result["flagged"] is True
        assert result["severity"] == "low"
        assert result["ghost_sources"] == [7]
        # La réponse doit être retournée (pas None) — on ne bloque pas en LOW
        assert result["response_patched"] is not None
        # Le disclaimer doit être ajouté
        assert "⚖️" in result["response_patched"] or "Note" in result["response_patched"]

    def test_low_response_contains_original(self):
        """En LOW, la réponse originale est préservée dans response_patched."""
        original = "According to [SOURCE 6], the controller must comply."
        result = check_hallucination_guardrail(original, n_chunks=5)
        assert result["severity"] == "low"
        # La réponse patchée commence par la réponse originale
        assert result["response_patched"].startswith(original)

    def test_low_with_valid_and_ghost(self):
        """Mix de sources valides et 1 fantôme → LOW."""
        result = check_hallucination_guardrail(
            "[SOURCE 1] states A. [SOURCE 3] states B. [SOURCE 8] states C.",
            n_chunks=5
        )
        assert result["severity"] == "low"
        assert result["ghost_sources"] == [8]
        assert result["n_valid"] == 2

    def test_source_zero_is_ghost(self):
        """
        [SOURCE 0] est toujours un fantôme.
        La numérotation dans build_prompt() commence à 1.
        """
        result = check_hallucination_guardrail(
            "According to [SOURCE 0], this is valid.",
            n_chunks=5
        )
        assert result["flagged"] is True
        assert 0 in result["ghost_sources"]

    def test_low_disclaimer_language_fr(self):
        """Disclaimer en français si language='fr'."""
        result = check_hallucination_guardrail(
            "Selon [SOURCE 9], le contrôleur doit...",
            n_chunks=5,
            language="fr"
        )
        assert result["severity"] == "low"
        # Le disclaimer FR doit être présent
        assert result["response_patched"] is not None
        # Vérifier que c'est bien la version FR (contient des mots français)
        assert "source" in result["response_patched"].lower()

    def test_low_disclaimer_language_en(self):
        """Disclaimer en anglais si language='en'."""
        result = check_hallucination_guardrail(
            "According to [SOURCE 9], the controller must...",
            n_chunks=5,
            language="en"
        )
        assert result["severity"] == "low"
        assert result["response_patched"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU HIGH — 2+ sources fantômes
# ══════════════════════════════════════════════════════════════════════════════

class TestHighSeverity:

    def test_two_ghost_sources(self):
        """2 sources fantômes → severity HIGH, réponse bloquée (None)."""
        result = check_hallucination_guardrail(
            "[SOURCE 7] and [SOURCE 9] both confirm this.",
            n_chunks=5
        )
        assert result["flagged"] is True
        assert result["severity"] == "high"
        assert result["ghost_sources"] == [7, 9]
        # En HIGH, response_patched doit être None — signale le blocage
        assert result["response_patched"] is None

    def test_three_ghost_sources(self):
        """3 fantômes → toujours HIGH."""
        result = check_hallucination_guardrail(
            "[SOURCE 6] says X. [SOURCE 8] says Y. [SOURCE 12] says Z.",
            n_chunks=5
        )
        assert result["severity"] == "high"
        assert len(result["ghost_sources"]) == 3

    def test_high_ghost_sources_sorted(self):
        """Les ghost_sources doivent être triés pour les logs."""
        result = check_hallucination_guardrail(
            "[SOURCE 12] and [SOURCE 7] and [SOURCE 9].",
            n_chunks=5
        )
        assert result["ghost_sources"] == sorted(result["ghost_sources"])

    def test_high_with_valid_sources_too(self):
        """Mix de valides et 2 fantômes → HIGH quand même."""
        result = check_hallucination_guardrail(
            "[SOURCE 1] valid. [SOURCE 2] valid. [SOURCE 7] ghost. [SOURCE 9] ghost.",
            n_chunks=5
        )
        assert result["severity"] == "high"
        assert result["n_valid"] == 2
        assert len(result["ghost_sources"]) == 2

    def test_exactly_threshold_high(self):
        """
        Exactement 2 fantômes = seuil HIGH.
        Test de la limite exacte — le plus important pour éviter les off-by-one.
        """
        result = check_hallucination_guardrail(
            "[SOURCE 6] and [SOURCE 7].",
            n_chunks=5
        )
        assert result["severity"] == "high"

    def test_one_below_threshold_high(self):
        """
        1 fantôme = encore LOW, pas HIGH.
        Symétrique du test précédent.
        """
        result = check_hallucination_guardrail(
            "[SOURCE 6] only.",
            n_chunks=5
        )
        assert result["severity"] == "low"  # pas high


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE DU RETOUR — Cohérence du dict
# ══════════════════════════════════════════════════════════════════════════════

class TestReturnStructure:

    def test_all_keys_present_ok(self):
        """Toutes les clés attendues sont présentes en cas OK."""
        result = check_hallucination_guardrail("No citations here.", n_chunks=5)
        required_keys = {"flagged", "severity", "ghost_sources", "n_cited", "n_valid",
                        "response_patched", "detail"}
        assert required_keys.issubset(result.keys())

    def test_all_keys_present_low(self):
        """Toutes les clés présentes en cas LOW."""
        result = check_hallucination_guardrail("[SOURCE 8] is ghost.", n_chunks=5)
        required_keys = {"flagged", "severity", "ghost_sources", "n_cited", "n_valid",
                        "response_patched", "detail"}
        assert required_keys.issubset(result.keys())

    def test_all_keys_present_high(self):
        """Toutes les clés présentes en cas HIGH."""
        result = check_hallucination_guardrail("[SOURCE 8] and [SOURCE 9].", n_chunks=5)
        required_keys = {"flagged", "severity", "ghost_sources", "n_cited", "n_valid",
                        "response_patched", "detail"}
        assert required_keys.issubset(result.keys())

    def test_detail_non_empty(self):
        """Le champ detail doit toujours être une string non vide."""
        for response, n in [
            ("no citations", 5),
            ("[SOURCE 1] valid", 5),
            ("[SOURCE 7] ghost", 5),
        ]:
            result = check_hallucination_guardrail(response, n)
            assert isinstance(result["detail"], str)
            assert len(result["detail"]) > 0

    def test_n_chunks_zero(self):
        """
        n_chunks=0 : toute citation est fantôme.
        Cas limite — ne doit pas lever ZeroDivisionError.
        """
        result = check_hallucination_guardrail("[SOURCE 1] says something.", n_chunks=0)
        assert result["flagged"] is True
        assert 1 in result["ghost_sources"]

    def test_n_chunks_one(self):
        """n_chunks=1 : seul [SOURCE 1] est valide."""
        result = check_hallucination_guardrail("[SOURCE 1] valid. [SOURCE 2] ghost.", n_chunks=1)
        assert result["flagged"] is True
        assert 2 in result["ghost_sources"]
        assert result["n_valid"] == 1
