"""
eval_dataset.py — Dataset de référence pour l'évaluation du système RAG
=============================================================================

RÔLE DE CE FICHIER :
Définit les questions de référence avec les segments attendus (ground truth).
Chaque entrée = une question + la liste des segment_id qui DOIVENT apparaître
dans les résultats pour considérer la recherche comme réussie.

COMMENT ÇA MARCHE :
    recall@k = proportion de questions pour lesquelles au moins 1 segment
               attendu apparaît dans les k premiers résultats FAISS.

    faithfulness = proportion de claims dans la réponse Claude qui sont
                   directement supportés par les chunks sources retournés.
                   Mesuré via Claude lui-même (LLM-as-judge).

COMMENT ÉTENDRE CE DATASET :
    Ajouter des entrées dans EVAL_DATASET avec :
    - question      : la question en langage naturel
    - expected_ids  : liste des segment_id attendus (au moins 1 doit apparaître)
    - regulation    : filtre optionnel (None = tout le corpus)
    - segment_type  : filtre optionnel
    - notes         : description du cas de test
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalCase:
    """Un cas de test pour l'évaluation du système RAG."""
    question:     str
    expected_ids: list[str]          # segment_id attendus dans les résultats
    regulation:   Optional[str] = None
    segment_type: Optional[str] = None
    notes:        str           = ""


# ══════════════════════════════════════════════════════════════════════════════
# DATASET DE RÉFÉRENCE — 20 questions couvrant tout le corpus
# ══════════════════════════════════════════════════════════════════════════════

EVAL_DATASET: list[EvalCase] = [

    # ── GDPR : obligations controller ─────────────────────────────────────────
    EvalCase(
        question     = "What are the obligations of a data controller under GDPR?",
        expected_ids = ["Article 24", "Article 26", "Article 28"],
        regulation   = "GDPR",
        segment_type = "article",
        notes        = "Core controller obligations — Article 24 doit être #1"
    ),
    EvalCase(
        question     = "What is the principle of data minimisation?",
        expected_ids = ["Article 5"],
        regulation   = "GDPR",
        notes        = "Principe fondamental Art 5(1)(c)"
    ),
    EvalCase(
        question     = "What constitutes valid consent under GDPR?",
        expected_ids = ["Article 7", "Article 4"],
        regulation   = "GDPR",
        notes        = "Conditions du consentement valide"
    ),
    EvalCase(
        question     = "What are the lawful bases for processing personal data?",
        expected_ids = ["Article 6"],
        regulation   = "GDPR",
        notes        = "6 bases légales Article 6"
    ),
    EvalCase(
        question     = "What is the right to erasure under GDPR?",
        expected_ids = ["Article 17"],
        regulation   = "GDPR",
        notes        = "Droit à l'effacement / right to be forgotten"
    ),

    # ── GDPR : obligations organisationnelles ─────────────────────────────────
    EvalCase(
        question     = "When is a Data Protection Impact Assessment required?",
        expected_ids = ["Article 35"],
        regulation   = "GDPR",
        notes        = "DPIA — conditions de déclenchement obligatoire"
    ),
    EvalCase(
        question     = "What are the notification requirements for a personal data breach?",
        expected_ids = ["Article 33", "Article 34"],
        regulation   = "GDPR",
        notes        = "Notification violation 72h + communication personnes"
    ),
    EvalCase(
        question     = "What are the obligations for appointing a Data Protection Officer?",
        expected_ids = ["Article 37", "Article 38", "Article 39"],
        regulation   = "GDPR",
        notes        = "DPO — désignation, position, missions"
    ),
    EvalCase(
        question     = "What are the rules for international data transfers?",
        expected_ids = ["Article 44", "Article 45", "Article 46"],
        regulation   = "GDPR",
        notes        = "Transferts hors UE — décision adéquation + garanties"
    ),
    EvalCase(
        question     = "What are the special categories of personal data?",
        expected_ids = ["Article 9"],
        regulation   = "GDPR",
        notes        = "Données sensibles — santé, biométrie, religion..."
    ),

    # ── GDPR : droits des personnes ───────────────────────────────────────────
    EvalCase(
        question     = "What rights do data subjects have regarding automated decision making?",
        expected_ids = ["Article 22"],
        regulation   = "GDPR",
        notes        = "Décision automatisée + profilage Art 22"
    ),
    EvalCase(
        question     = "What is the right to data portability?",
        expected_ids = ["Article 20"],
        regulation   = "GDPR",
        notes        = "Portabilité des données Art 20"
    ),

    # ── EU AI Act ─────────────────────────────────────────────────────────────
    EvalCase(
        question     = "What AI systems are classified as high-risk under the EU AI Act?",
        expected_ids = ["Article 6", "Article 7"],
        regulation   = "EU_AI_ACT",
        notes        = "Classification systèmes IA à haut risque + Annex III"
    ),
    EvalCase(
        question     = "What are the prohibited AI practices under the AI Act?",
        expected_ids = ["Article 5"],
        regulation   = "EU_AI_ACT",
        notes        = "Pratiques IA interdites — risque inacceptable"
    ),
    EvalCase(
        question     = "What transparency obligations apply to AI systems interacting with humans?",
        expected_ids = ["Article 50"],
        regulation   = "EU_AI_ACT",
        notes        = "Obligations transparence Art 13 + Art 50 chatbots"
    ),
    EvalCase(
        question     = "What are the requirements for general purpose AI models?",
        expected_ids = ["Article 51", "Article 52", "Article 53"],
        regulation   = "EU_AI_ACT",
        notes        = "GPAI models — risque systémique + obligations"
    ),

    # ── EDPB Guidelines ───────────────────────────────────────────────────────
    EvalCase(
        question     = "What safeguards are required for solely automated decisions with legal effects?",
        expected_ids = ["Article 22"],
        regulation   = "GDPR",
        notes        = "EDPB guidelines sur Art 22 — droits + garanties humaines"
    ),

    # ── Data Governance Act ───────────────────────────────────────────────────
    EvalCase(
        question     = "What are the obligations of data intermediaries under the Data Governance Act?",
        expected_ids = ["Article 12", "Article 11"],
        regulation   = "DATA_GOVERNANCE",
        notes        = "Services d'intermédiation DGA"
    ),

    # ── CNIL ──────────────────────────────────────────────────────────────────
    EvalCase(
        question     = "What are the CNIL recommendations on AI training data?",
        expected_ids = [],   # freetext — pas de segment_id structuré
        regulation   = "CNIL",
        notes        = "Guidelines CNIL — données entraînement IA (freetext)"
    ),

    # ── Cross-corpus ──────────────────────────────────────────────────────────
    EvalCase(
        question     = "How does GDPR apply to AI systems that process personal data?",
        expected_ids = ["Article 5", "Article 6", "Article 22", "Article 35"],
        regulation   = None,  # tout le corpus
        notes        = "Question transversale GDPR + AI Act"
    ),
]
