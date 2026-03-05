"""
=============================================================================
article_extractor.py — Extraction des structures juridiques
=============================================================================

RÔLE DE CE FICHIER :
Ce fichier prend les pages brutes (RawPage) et les découpe en segments
juridiques structurés : Articles, Recitals, Annexes, Chapitres, Sections.

DOCUMENTS SUPPORTÉS ET LEURS STRUCTURES :
┌─────────────────────────┬──────────────────────────────────────────┐
│ Document                │ Structure détectée                       │
├─────────────────────────┼──────────────────────────────────────────┤
│ GDPR / RGPD             │ Article 5, Article 6... + Recitals (1)  │
│ EU AI Act               │ Article 5, Article 6... + Recitals (1)  │
│ Data Governance Act     │ Article 5, Article 6... + Recitals (1)  │
│ EDPB Guidelines         │ I. II. III.A. III.B.1. + ANNEX 1        │
│ CNIL Recommendations    │ I. II. III.A. III.B.1.                  │
└─────────────────────────┴──────────────────────────────────────────┘

PROBLÈME INITIAL AVEC EDPB/CNIL :
Les documents GDPR/AI Act utilisent "Article X" comme marqueur de section.
Les documents EDPB/CNIL utilisent une numérotation romaine/alphabétique :
    I. INTRODUCTION
    II. DEFINITIONS
    III. GENERAL PROVISIONS
    III.A. DATA PROTECTION PRINCIPLES
    III.B.1. Article 5(1)(a) - Lawful, fair and transparent

Sans pattern spécifique, ces documents tombaient en mode "freetext"
(une page = un segment), perdant toute structure logique.

SOLUTION AJOUTÉE :
Un pattern SECTION_OUTLINE_PATTERN détecte les structures I./II./III.A.
et crée des segments propres pour chaque section des guidelines.
La fonction _find_guideline_anchors() gère aussi la déduplication
(une même section apparaît dans la table des matières ET dans le corps).

FLUX DE DONNÉES :
    list[RawPage]  →  [extract_segments()]  →  list[LegalSegment]

=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_parser import RawPage


# =============================================================================
# PATTERNS REGEX — Détection des structures juridiques
# =============================================================================

# ── Pattern 1 : Articles législatifs (GDPR, AI Act, Data Governance Act) ──────
# Détecte : "Article 5", "Article 5a", "ARTICLE 12"
# Suivi d'un saut de ligne puis du titre de l'article.
# Exemple :
#   "Article 5
#    Principles relating to processing of personal data"
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(Article\s+(\d+\w?))\s*\n\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Pattern 2 : Recitals / Considérants ───────────────────────────────────────
# Détecte : "(1)", "(42)", "(173)" en début de paragraphe.
# Ces numéros entre parenthèses sont les "considérants" des règlements UE.
# {20,} = au moins 20 caractères après pour éviter les faux positifs courts.
RECITAL_PATTERN = re.compile(
    r"(?:^|\n)\s*\((\d{1,3})\)\s+([A-Z][^\n]{20,})",
    re.MULTILINE,
)

# ── Pattern 3 : Annexes ────────────────────────────────────────────────────────
# Détecte : "Annex I", "ANNEX II", "Annex 1 - Good Practice..."
# [IVXLCDM]+ = chiffres romains | \d+ = chiffres arabes
ANNEX_PATTERN = re.compile(
    r"(?:^|\n)\s*(Annex\s+([IVXLCDM]+|\d+))\b([^\n]*)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Pattern 4 : Chapitres (textes législatifs) ────────────────────────────────
# Détecte : "Chapter I", "CHAPTER II", "Chapter 3"
CHAPTER_PATTERN = re.compile(
    r"(?:^|\n)\s*(Chapter\s+([IVXLCDM]+|\d+))\s*\n\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Pattern 5 : Sections de guidelines (EDPB, CNIL) ──────────────────────────
# C'est le pattern AJOUTÉ pour corriger le problème EDPB/CNIL.
#
# STRUCTURE DES DOCUMENTS EDPB/CNIL :
#   I. INTRODUCTION
#   II. DEFINITIONS
#   III. GENERAL PROVISIONS ON PROFILING
#   III.A. DATA PROTECTION PRINCIPLES
#   III.B.1. Article 5(1)(a) - Lawful, fair and transparent
#   IV. SPECIFIC PROVISIONS
#
# EXPLICATION DU PATTERN :
#   (?:^|\n)         = début de ligne
#   \s*              = espaces optionnels
#   (                = début du groupe capturant l'identifiant complet
#     [IVXLCDM]{1,4} = 1 à 4 chiffres romains (I, II, III, IV, VI, VII...)
#     (?:            = groupe non-capturant pour la suite optionnelle
#       [.\-]        = point ou tiret séparateur
#       [A-Z0-9]     = lettre majuscule ou chiffre (ex: le "A" dans "III.A")
#     )*             = répété 0 ou plusieurs fois (couvre III, III.A, III.A.1)
#   )
#   \.               = point obligatoire après l'identifiant (ex: "III.")
#   \s+              = espace(s) obligatoire(s)
#   ([A-Z][^\n]{5,}) = titre en majuscule, au moins 5 caractères
#                      (évite de matcher des abréviations seules)
#
# EXEMPLES MATCHÉS :
#   "I. INTRODUCTION"            → id="I",     titre="INTRODUCTION"
#   "II. DEFINITIONS"            → id="II",    titre="DEFINITIONS"
#   "III.A. DATA PROTECTION"     → id="III.A", titre="DATA PROTECTION..."
#   "III.B.1. Article 5(1)(a)"   → id="III.B.1", titre="Article 5(1)(a)..."
#
# EXEMPLES NON MATCHÉS (faux positifs évités) :
#   "I think that..."  → pas de point immédiatement après "I"
#   "VI." seul         → titre trop court (< 5 chars)
SECTION_OUTLINE_PATTERN = re.compile(
    r"(?:^|\n)\s*([IVXLCDM]{1,4}(?:[.\-][A-Z0-9])*)\.\s*\n?\s*([A-Za-z][^\n]{5,})",
    re.MULTILINE,
)

# ── Ensemble des réglementations qui utilisent la structure "Article X" ────────
# Ces réglementations ont des textes législatifs avec articles numérotés.
# Les autres (EDPB, CNIL) utilisent des sections outline I./II./III.A.
LEGISLATIVE_REGULATIONS = {"GDPR", "EU_AI_ACT", "DATA_GOVERNANCE_ACT"}


# =============================================================================
# DATACLASS : LegalSegment
# =============================================================================
@dataclass
class LegalSegment:
    """
    Représente un segment structuré d'un texte juridique.

    Un segment = une unité logique du document :
    - Un article complet (ex: Article 5 du GDPR)
    - Un considérant/recital (ex: Recital 42 du GDPR)
    - Une annexe (ex: Annex I de l'EU AI Act)
    - Une section de guideline (ex: "Section III.A" de l'EDPB)  ← NOUVEAU
    - Un chapitre (ex: Chapter III du GDPR)
    """

    # Type de segment.
    # Valeurs possibles :
    #   "article"  = article législatif numéroté (GDPR, AI Act...)
    #   "recital"  = considérant (1), (42)...
    #   "annex"    = annexe (Annex I, Annex II...)
    #   "chapter"  = chapitre (Chapter I, Chapter II...)
    #   "section"  = section de guideline (I., II., III.A...)  ← NOUVEAU
    #   "freetext" = fallback : aucune structure détectée
    segment_type: str

    # Identifiant unique du segment.
    # Ex: "Article 5", "Recital 42", "Annex I", "Section III.A"
    segment_id: str

    # Titre du segment (la ligne qui suit immédiatement l'identifiant)
    title: str

    # Texte COMPLET du segment (de son en-tête jusqu'au prochain segment)
    text: str

    # Numéros de pages pour la traçabilité
    page_start: int
    page_end: int

    # Nom du fichier source ("edpb_automated_decision.pdf")
    source_file: str

    # Réglementation : "GDPR", "EU_AI_ACT", "EDPB", "CNIL", "DATA_GOVERNANCE_ACT"
    regulation: str

    # Calculé automatiquement
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def __repr__(self) -> str:
        return (
            f"LegalSegment({self.segment_type}, {self.segment_id!r}, "
            f"pages={self.page_start}-{self.page_end}, chars={self.char_count})"
        )


# =============================================================================
# FONCTION PRINCIPALE : extract_segments
# =============================================================================
def extract_segments(pages: list[RawPage], regulation: str) -> list[LegalSegment]:
    """
    Extrait les segments juridiques structurés depuis une liste de pages.

    STRATÉGIE ADAPTATIVE selon le type de document :

    Pour les textes législatifs (GDPR, AI Act, Data Governance Act) :
        → Cherche "Article X", Recitals "(1)", Annexes, Chapitres

    Pour les guidelines/recommandations (EDPB, CNIL) :
        → Cherche les sections "I.", "II.", "III.A.", "III.B.1." + Annexes

    Paramètres :
        pages      : liste des pages parsées (sorties de pdf_parser.parse_pdf)
        regulation : identifiant de la réglementation
                     Ex: "GDPR", "EU_AI_ACT", "EDPB", "CNIL"

    Retourne :
        list[LegalSegment] : segments trouvés, dans l'ordre d'apparition
    """

    # ── ÉTAPE 1 : Reconstituer le texte complet avec repères de position ───────
    page_boundaries: list[tuple[int, int]] = []
    full_text_parts: list[str] = []
    offset = 0

    for page in pages:
        page_boundaries.append((offset, page.page_number))
        full_text_parts.append(page.text)
        offset += len(page.text) + 1

    full_text = "\n".join(full_text_parts)

    # ── Fonction utilitaire : offset → numéro de page ─────────────────────────
    def _char_to_page(char_offset: int) -> int:
        """
        Retrouve le numéro de page pour une position donnée dans le texte complet.
        """
        page_num = pages[0].page_number
        for boundary_offset, pnum in page_boundaries:
            if char_offset >= boundary_offset:
                page_num = pnum
            else:
                break
        return page_num

    # ── ÉTAPE 2 : Choisir la stratégie selon le type de document ──────────────
    # Textes législatifs → Articles + Recitals + Chapitres
    # Guidelines/recommandations → Sections outline I./II./III.A. + Annexes
    if regulation in LEGISLATIVE_REGULATIONS:
        anchors = _find_legislative_anchors(full_text)
    else:
        anchors = _find_guideline_anchors(full_text)

    # ── CAS FALLBACK : aucune structure détectée ──────────────────────────────
    # Filet de sécurité absolu : une page = un segment freetext.
    if not anchors:
        return [
            LegalSegment(
                segment_type="freetext",
                segment_id=f"page_{p.page_number}",
                title="",
                text=p.text,
                page_start=p.page_number,
                page_end=p.page_number,
                source_file=p.source_file,
                regulation=regulation,
            )
            for p in pages
        ]

    # ── ÉTAPE 3 : Découper le texte entre les ancres ──────────────────────────
    segments: list[LegalSegment] = []

    for i, (start, seg_type, seg_id, title) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(full_text)
        segment_text = full_text[start:end].strip()

        # Ignore les segments trop courts (faux positifs du regex)
        if len(segment_text) < 30:
            continue

        segments.append(LegalSegment(
            segment_type=seg_type,
            segment_id=seg_id,
            title=title,
            text=segment_text,
            page_start=_char_to_page(start),
            page_end=_char_to_page(end - 1),
            source_file=pages[0].source_file if pages else "",
            regulation=regulation,
        ))

    return segments


# =============================================================================
# STRATÉGIE 1 : Documents législatifs (GDPR, AI Act, Data Governance Act)
# =============================================================================
def _find_legislative_anchors(full_text: str) -> list[tuple[int, str, str, str]]:
    """
    Détecte les ancres pour les textes législatifs.
    Cherche : Articles, Recitals, Annexes, Chapitres.

    Format retourné : (position, type_segment, identifiant, titre)
    """
    anchors: list[tuple[int, str, str, str]] = []

    for match in ARTICLE_PATTERN.finditer(full_text):
        anchors.append((match.start(), "article", match.group(1), match.group(3).strip()))

    for match in ANNEX_PATTERN.finditer(full_text):
        anchors.append((match.start(), "annex", match.group(1), match.group(3).strip()))

    for match in CHAPTER_PATTERN.finditer(full_text):
        anchors.append((match.start(), "chapter", match.group(1), match.group(3).strip()))

    for match in RECITAL_PATTERN.finditer(full_text):
        anchor_id = f"Recital {match.group(1)}"
        anchors.append((match.start(), "recital", anchor_id, match.group(2)[:80]))

    anchors.sort(key=lambda x: x[0])
    return anchors


# =============================================================================
# STRATÉGIE 2 : Guidelines et recommandations (EDPB, CNIL)
# =============================================================================
def _find_guideline_anchors(full_text: str) -> list[tuple[int, str, str, str]]:
    """
    Détecte les ancres pour les guidelines et recommandations EDPB/CNIL.

    Ces documents utilisent une structure outline numérotée :
        I. INTRODUCTION
        II. DEFINITIONS
        III.A. DATA PROTECTION PRINCIPLES
        III.B.1. Article 5(1)(a) - Lawful, fair and transparent
        ANNEX 1 - GOOD PRACTICE RECOMMENDATIONS

    PROBLÈME SPÉCIFIQUE : la table des matières (TOC).
    Le document EDPB commence par une table des matières qui liste
    toutes les sections. Ces mêmes titres réapparaissent dans le corps.
    Sans déduplication, chaque section serait détectée deux fois :
        1ère occurrence = ligne dans la TOC   → segment vide ou très court
        2ème occurrence = vraie section       → segment avec le vrai contenu

    SOLUTION : on garde la DERNIÈRE occurrence quand deux ancres avec le
    même identifiant sont détectées à moins de 3000 caractères d'écart.
    """
    anchors: list[tuple[int, str, str, str]] = []

    # Sections outline : "I. INTRODUCTION", "III.A. DATA PROTECTION PRINCIPLES"
    for match in SECTION_OUTLINE_PATTERN.finditer(full_text):
        section_id = match.group(1)          # "I", "II", "III.A", "III.B.1"
        title      = match.group(2).strip()  # "INTRODUCTION", "DEFINITIONS"...

        anchors.append((
            match.start(),
            "section",
            f"Section {section_id}",   # "Section I", "Section III.A"
            title
        ))

    # Annexes : présentes dans les guidelines aussi
    # Ex: "ANNEX 1 - GOOD PRACTICE RECOMMENDATIONS"
    for match in ANNEX_PATTERN.finditer(full_text):
        anchors.append((
            match.start(),
            "annex",
            match.group(1),         # "Annex 1"
            match.group(3).strip()  # "- GOOD PRACTICE RECOMMENDATIONS"
        ))

    # Tri par ordre d'apparition
    anchors.sort(key=lambda x: x[0])

    # ── Déduplication : TOC vs corps du document ──────────────────────────────
    # On parcourt les ancres et on remplace une occurrence par la suivante
    # si elles ont le même identifiant et sont proches (< 3000 chars).
    # Cela élimine les faux segments générés par la table des matières.
    deduplicated: list[tuple[int, str, str, str]] = []
    seen_ids: dict[str, int] = {}  # segment_id → index dans deduplicated

    for anchor in anchors:
        pos, seg_type, seg_id, title = anchor

        if seg_id in seen_ids:
            prev_idx = seen_ids[seg_id]
            prev_pos = deduplicated[prev_idx][0]

            if pos - prev_pos < 3000:
                # Proche → probablement TOC + corps.
                # On remplace l'ancienne entrée par la nouvelle (= corps réel).
                deduplicated[prev_idx] = anchor
            else:
                # Loin → deux vraies sections distinctes. On garde les deux.
                deduplicated.append(anchor)
                seen_ids[seg_id] = len(deduplicated) - 1
        else:
            deduplicated.append(anchor)
            seen_ids[seg_id] = len(deduplicated) - 1

    # Re-trier après déduplication (l'ordre peut avoir été perturbé)
    deduplicated.sort(key=lambda x: x[0])
    return deduplicated