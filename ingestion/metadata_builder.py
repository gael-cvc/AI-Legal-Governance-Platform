"""
=============================================================================
metadata_builder.py — Registre des sources et enrichissement des métadonnées
=============================================================================

RÔLE DE CE FICHIER :
Ce fichier fait deux choses simples mais essentielles :
1. Maintient un "annuaire" de tous nos documents sources (PDF)
2. Enrichit chaque segment avec les informations complètes sur sa source

POURQUOI LES MÉTADONNÉES SONT-ELLES IMPORTANTES ?
Sans métadonnées, un chunk de texte ressemble à :
    "Les données doivent être collectées pour des finalités déterminées..."

Avec métadonnées, le même chunk devient :
    Texte       : "Les données doivent être collectées pour des finalités déterminées..."
    Source      : GDPR
    Article     : Article 5
    Type        : regulation
    Année       : 2016
    Juridiction : EU
    Titre       : Regulation (EU) 2016/679 — General Data Protection Regulation

Quand l'IA répond à une question, elle peut citer :
    "Selon l'Article 5 du GDPR (2016)..."
au lieu de juste copier le texte sans référence.

COMMENT AJOUTER UN NOUVEAU DOCUMENT PDF :
Il suffit d'ajouter une nouvelle entrée dans SOURCE_REGISTRY ci-dessous.
Pas besoin de toucher au reste du code.

FLUX DE DONNÉES :
    LegalSegment  →  [enrich_segment_metadata()]  →  dict enrichi (couche bronze)

=============================================================================
"""

from __future__ import annotations

# dataclass avec frozen=True = un dataclass IMMUTABLE.
# "Immutable" = on ne peut PAS modifier les champs après création.
# C'est parfait pour les métadonnées : elles ne doivent jamais changer.
from dataclasses import dataclass

# Path pour manipuler les chemins de fichiers de façon propre et cross-platform.
from pathlib import Path


# =============================================================================
# DATACLASS : SourceMetadata
# =============================================================================
# frozen=True : cet objet est immuable après création.
# Si on essaie de faire metadata.year = 2025, Python lève une FrozenInstanceError.
# Pourquoi ? Pour éviter les bugs : les métadonnées d'une source ne changent jamais.
@dataclass(frozen=True)
class SourceMetadata:
    """
    Métadonnées complètes d'un document source.

    Ces informations sont FIXES pour chaque fichier PDF :
    elles décrivent d'où vient le document, quel règlement il représente, etc.
    """

    # Identifiant court de la réglementation.
    # Utilisé comme "tag" dans tous les segments et chunks.
    # Valeurs possibles : "GDPR", "EU_AI_ACT", "EDPB", "CNIL", "DATA_GOVERNANCE_ACT"
    regulation: str

    # Type de document juridique.
    # Exemples :
    #   "regulation"     = texte de loi contraignant (GDPR, EU AI Act)
    #   "guideline"      = ligne directrice interprétative (EDPB)
    #   "recommendation" = recommandation non contraignante (CNIL)
    #   "annex"          = annexes d'un texte de loi
    #   "recitals"       = considérants (partie "pourquoi" d'un règlement)
    document_type: str

    # Année de publication ou d'entrée en vigueur du document.
    year: int

    # Juridiction géographique :
    #   "EU" = Union Européenne (GDPR, EU AI Act, EDPB, Data Governance Act)
    #   "FR" = France (CNIL, versions françaises du RGPD)
    jurisdiction: str

    # Titre officiel complet du document.
    official_title: str

    # Langue du document.
    #   "EN" = anglais (versions officielles UE)
    #   "FR" = français (versions françaises)
    language: str


# =============================================================================
# SOURCE_REGISTRY : l'annuaire de tous nos PDFs
# =============================================================================
# C'est un dictionnaire Python :
#   clé    (str)            = nom exact du fichier PDF (ex: "gdpr_full.pdf")
#   valeur (SourceMetadata) = ses métadonnées complètes
#
# POURQUOI UN DICTIONNAIRE ?
# Accès en O(1) = instantané, quelle que soit la taille du registre.
# SOURCE_REGISTRY["gdpr_full.pdf"] retourne les métadonnées en une fraction de ms.
#
# COMMENT AJOUTER UN NOUVEAU DOCUMENT ?
# Ajouter une nouvelle entrée ici. C'est le SEUL endroit à modifier.
SOURCE_REGISTRY: dict[str, SourceMetadata] = {

    # ── GDPR — version anglaise ────────────────────────────────────────────────
    # Le texte de référence en anglais. Version officielle UE.
    "gdpr_full.pdf": SourceMetadata(
        regulation="GDPR",
        document_type="regulation",
        year=2016,          # Adopté en 2016, applicable depuis mai 2018
        jurisdiction="EU",
        official_title="Regulation (EU) 2016/679 — General Data Protection Regulation",
        language="EN",
    ),

    # ── RGPD — version française ───────────────────────────────────────────────
    # Même règlement que gdpr_full.pdf mais en français.
    # On garde regulation="GDPR" : c'est le même texte juridique, juste traduit.
    # La différence : language="FR" et official_title en français.
    "rgpd_full.pdf": SourceMetadata(
        regulation="GDPR",
        document_type="regulation",
        year=2016,
        jurisdiction="EU",
        official_title="Règlement (UE) 2016/679 — Règlement Général sur la Protection des Données",
        language="FR",
    ),

    # ── RGPD Recitals — considérants en français ──────────────────────────────
    # Les considérants sont la partie "pourquoi" du RGPD (173 recitals).
    # Ils expliquent l'intention du législateur derrière chaque article.
    # Exemple : le considérant 42 explique pourquoi le consentement doit être libre.
    # Juridiquement, ils ne sont pas contraignants mais servent à interpréter les articles.
    "rgpd_recitals.pdf": SourceMetadata(
        regulation="GDPR",
        document_type="recitals",
        year=2016,
        jurisdiction="EU",
        official_title="RGPD — Considérants 1 à 173",
        language="FR",
    ),

    # ── EU AI Act — texte principal ────────────────────────────────────────────
    # Le règlement européen sur l'IA, adopté en 2024.
    # Contient les 113 articles et les recitals.
    "ai_act_full.pdf": SourceMetadata(
        regulation="EU_AI_ACT",
        document_type="regulation",
        year=2024,
        jurisdiction="EU",
        official_title="Regulation (EU) 2024/1689 — Artificial Intelligence Act",
        language="EN",
    ),

    # ── EU AI Act — Annexes ────────────────────────────────────────────────────
    # Les annexes de l'EU AI Act dans un fichier séparé.
    # Contiennent les listes de systèmes IA interdits, à haut risque, etc.
    # (Annexes I à XIII)
    "ai_act_annexes.pdf": SourceMetadata(
        regulation="EU_AI_ACT",
        document_type="annex",
        year=2024,
        jurisdiction="EU",
        official_title="EU AI Act — Annexes I–XIII",
        language="EN",
    ),

    # ── EDPB — Guidelines sur la prise de décision automatisée ────────────────
    # L'EDPB (European Data Protection Board) publie des guidelines
    # qui interprètent le GDPR. Non contraignantes mais très importantes
    # car elles reflètent la position des autorités de contrôle européennes.
    "edpb_automated_decision.pdf": SourceMetadata(
        regulation="EDPB",
        document_type="guideline",
        year=2022,
        jurisdiction="EU",
        official_title="EDPB Guidelines on Automated Decision-Making",
        language="EN",
    ),

    # ── CNIL — Recommandations IA ──────────────────────────────────────────────
    # La CNIL est l'autorité française de protection des données.
    # Ses recommandations s'appliquent au contexte français.
    # Document volumineux (176 pages) sans structure "Article X" standard
    # → traité en mode freetext (une page = un segment).
    "cnil_ai_recommendations.pdf": SourceMetadata(
        regulation="CNIL",
        document_type="recommendation",
        year=2024,
        jurisdiction="FR",
        official_title="CNIL — AI Recommendations",
        language="EN",
    ),

    # ── CNIL — Guide de conformité ────────────────────────────────────────────
    # Guide pratique de la CNIL sur comment se conformer aux règles IA.
    "cnil_ai_how_to_comply.pdf": SourceMetadata(
        regulation="CNIL",
        document_type="recommendation",
        year=2024,
        jurisdiction="FR",
        official_title="CNIL — How to Comply with AI Rules",
        language="EN",
    ),

    # ── Data Governance Act ────────────────────────────────────────────────────
    # Règlement UE sur la gouvernance des données.
    # Complément important à l'AI Act : encadre le partage et la réutilisation
    # des données entre acteurs publics et privés en Europe.
    "data_governance_act.pdf": SourceMetadata(
        regulation="DATA_GOVERNANCE_ACT",
        document_type="regulation",
        year=2022,
        jurisdiction="EU",
        official_title="Regulation (EU) 2022/868 — Data Governance Act",
        language="EN",
    ),
}


# =============================================================================
# FONCTION : get_metadata
# =============================================================================
def get_metadata(filename: str) -> SourceMetadata:
    """
    Récupère les métadonnées d'un fichier PDF par son nom.

    Paramètre :
        filename (str) : nom ou chemin du fichier.
                         Ex: "gdpr_full.pdf"
                         Ex: "data/raw/gdpr_full.pdf" (chemin complet accepté aussi)

    Retourne :
        SourceMetadata : les métadonnées du fichier.

    Lève :
        ValueError : si le fichier n'est pas dans le registre.
                     Le message d'erreur indique exactement quoi faire.
    """
    # Path(filename).name extrait juste le nom de fichier depuis un chemin complet.
    # Ex: Path("data/raw/gdpr_full.pdf").name → "gdpr_full.pdf"
    # Cela rend la fonction robuste : on peut passer un chemin complet ou juste un nom.
    key = Path(filename).name

    if key not in SOURCE_REGISTRY:
        # Message d'erreur détaillé pour faciliter le débogage.
        # list(SOURCE_REGISTRY) retourne la liste de toutes les clés connues.
        raise ValueError(
            f"Source inconnue : {key!r}\n"
            f"Ajoutez-la dans SOURCE_REGISTRY dans metadata_builder.py\n"
            f"Sources connues : {list(SOURCE_REGISTRY)}"
        )
    return SOURCE_REGISTRY[key]


# =============================================================================
# FONCTION : enrich_segment_metadata
# =============================================================================
def enrich_segment_metadata(segment_dict: dict) -> dict:
    """
    Enrichit un dictionnaire de segment avec les métadonnées complètes de sa source.

    POURQUOI CETTE FONCTION PREND UN DICT ET PAS UN LegalSegment ?
    Dans pipeline.py, on sérialise les LegalSegment en dicts (via dataclasses.asdict())
    pour les écrire en JSON (format du Data Lake bronze).
    Cette fonction travaille directement sur ces dicts, évitant une re-conversion.

    COMMENT ÇA MARCHE :
    1. On récupère "source_file" depuis le dict (ex: "gdpr_full.pdf")
    2. On charge les métadonnées correspondantes depuis SOURCE_REGISTRY
    3. On fusionne le dict original avec les nouvelles métadonnées

    Paramètre :
        segment_dict (dict) : dictionnaire représentant un LegalSegment
                              (obtenu via dataclasses.asdict(segment))

    Retourne :
        dict : le même dictionnaire + les champs de métadonnées ajoutés
    """
    # On récupère les métadonnées pour ce fichier source.
    meta = get_metadata(segment_dict["source_file"])

    # On retourne un NOUVEAU dictionnaire qui combine :
    #   **segment_dict : tous les champs existants du segment (copiés)
    #   + les nouveaux champs de métadonnées
    #
    # Le "**" (double étoile) "décompresse" le dictionnaire :
    # c'est l'équivalent de copier toutes les paires clé-valeur.
    #
    # Si une clé existe des deux côtés (ex: "regulation"),
    # la valeur de DROITE écrase celle de gauche.
    # Ici, la valeur officielle du registre prend le dessus sur celle du segment.
    return {
        **segment_dict,                         # Tous les champs existants du segment
        "regulation":     meta.regulation,      # Écrase avec la valeur officielle du registre
        "document_type":  meta.document_type,   # Champ ajouté
        "year":           meta.year,            # Champ ajouté
        "jurisdiction":   meta.jurisdiction,    # Champ ajouté
        "official_title": meta.official_title,  # Champ ajouté
        "language":       meta.language,        # Champ ajouté
    }