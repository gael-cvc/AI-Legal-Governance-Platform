"""
=============================================================================
pdf_parser.py — Parseur de PDF avec PyMuPDF
=============================================================================

RÔLE DE CE FICHIER :
Ce fichier est la toute première étape du pipeline.
Il prend un fichier PDF brut et le transforme en texte lisible, page par page.

POURQUOI CE FICHIER EXISTE :
Les PDFs (GDPR, EU AI Act...) ne sont pas du texte simple.
Ce sont des fichiers binaires complexes avec mise en page, colonnes, en-têtes.
On ne peut pas juste faire open("fichier.pdf").read().
Il faut un outil spécialisé : PyMuPDF (aussi appelé "fitz" dans le code).

WHAT IS PyMuPDF / fitz ?
PyMuPDF est une librairie Python qui s'installe avec : pip install pymupdf
Quand tu fais "import fitz", c'est PyMuPDF que tu importes.
Le nom "fitz" vient du moteur C sous-jacent (MuPDF).
C'est la librairie la plus rapide et précise pour lire les PDFs en Python.

FLUX DE DONNÉES :
    fichier.pdf  →  [parse_pdf()]  →  liste de RawPage  →  article_extractor.py

=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────────

# "from __future__ import annotations" permet d'utiliser les annotations de type
# modernes (ex: list[str] au lieu de List[str]) même sur Python 3.9+.
# C'est une bonne pratique de l'écrire en haut de chaque fichier.
from __future__ import annotations

# "re" = Regular Expressions. C'est le module Python pour chercher des patterns
# dans du texte. Ex: trouver tous les "Article X" dans un texte.
# On l'utilise ici pour supprimer les numéros de page, en-têtes parasites, etc.
import re

# "dataclass" et "field" viennent du module "dataclasses" de Python.
# Un dataclass est une façon propre de créer des classes qui servent
# principalement à stocker des données.
# Au lieu d'écrire __init__(self, a, b, c), Python le génère automatiquement.
# "field(init=False)" = ce champ n'est PAS fourni par l'utilisateur,
#                       il est calculé automatiquement après __init__.
from dataclasses import dataclass, field

# "Path" vient de "pathlib" — c'est la façon moderne de gérer les chemins de fichiers.
# Au lieu de "data/raw/gdpr.pdf" (une string fragile), on utilise Path("data/raw/gdpr.pdf")
# qui est un objet intelligent qui fonctionne sur Windows, Mac, Linux.
from pathlib import Path

# "Iterator" est un type hint (annotation de type).
# Il sert uniquement à documenter : "cette fonction retourne un itérateur de RawPage".
# Ça n'affecte pas l'exécution, c'est pour la lisibilité et les outils d'analyse.
from typing import Iterator

# PyMuPDF s'importe sous le nom "fitz" (c'est son nom historique).
# C'est la librairie principale de ce fichier.
import fitz  # PyMuPDF


# =============================================================================
# DATACLASS : RawPage
# =============================================================================
# Un "dataclass" est une classe Python simplifiée pour stocker des données.
# @dataclass est un "décorateur" : il transforme automatiquement la classe
# en ajoutant __init__, __repr__, __eq__ sans que tu aies à les écrire.
#
# RawPage représente UNE page d'un PDF après extraction du texte.
# C'est notre unité de base à ce stade du pipeline.
@dataclass
class RawPage:
    # Numéro de la page dans le PDF (commence à 1, pas 0)
    page_number: int

    # Le texte extrait de cette page, après nettoyage
    text: str

    # Nom du fichier source (ex: "gdpr_full.pdf")
    # Utile pour la traçabilité : savoir d'où vient chaque morceau de texte
    source_file: str

    # char_count = nombre de caractères dans le texte
    # "field(init=False)" signifie : ce champ N'EST PAS un paramètre du constructeur.
    # Il est calculé automatiquement dans __post_init__ (juste après __init__).
    char_count: int = field(init=False)

    def __post_init__(self):
        """
        __post_init__ est appelé AUTOMATIQUEMENT par @dataclass juste après __init__.
        C'est ici qu'on calcule les champs qui dépendent d'autres champs.
        len(self.text) = nombre de caractères dans le texte.
        """
        self.char_count = len(self.text)


# =============================================================================
# PATTERNS DE NETTOYAGE
# =============================================================================
# Les PDFs de l'UE contiennent beaucoup de "bruit" à supprimer :
# - Numéros de page solitaires : "42"
# - En-têtes du Journal Officiel : "Official Journal of the European Union"
# - Références de pagination : "L 119/32  EN"
#
# re.compile() PRÉ-COMPILE le pattern regex pour être plus rapide.
# Si on compilait à chaque appel, ce serait lent sur 300+ pages.
#
# Explication des flags :
# - re.MULTILINE : ^ et $ matchent début/fin de CHAQUE LIGNE (pas juste du texte entier)
# - re.IGNORECASE : "article" matche aussi "Article", "ARTICLE"
_NOISE_PATTERNS: list[re.Pattern] = [

    # Pattern 1 : une ligne qui ne contient QUE des chiffres (numéro de page)
    # ^\s*  = début de ligne + espaces optionnels
    # \d{1,4} = 1 à 4 chiffres (couvre pages 1 à 9999)
    # \s*$  = espaces optionnels + fin de ligne
    re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE),

    # Pattern 2 : en-tête "Official Journal of the European Union"
    # .* = tout ce qui suit sur la même ligne
    re.compile(r"Official Journal of the European Union.*", re.IGNORECASE),

    # Pattern 3 : références de pagination du Journal Officiel
    # Exemple : "L 119/32  EN" ou "EN  L 119/1"
    re.compile(r"L \d+/\d+.*EN.*", re.IGNORECASE),

    # Pattern 4 : variante de l'en-tête OJ
    re.compile(r"EN\s+Official Journal.*", re.IGNORECASE),
]


def _clean_text(text: str) -> str:
    """
    Supprime le "bruit" d'une page PDF : numéros de page, en-têtes, etc.

    Le underscore devant "_clean_text" est une convention Python :
    cela signifie que cette fonction est PRIVÉE (interne à ce module).
    Elle ne sera pas importée si quelqu'un fait "from pdf_parser import *".

    Paramètre :
        text (str) : le texte brut d'une page, sorti directement de PyMuPDF

    Retourne :
        str : le texte nettoyé
    """
    # On applique chaque pattern de nettoyage, l'un après l'autre.
    # pattern.sub("", text) = "remplace toutes les occurrences du pattern par une chaîne vide"
    # Autrement dit : supprime les lignes parasites.
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # Normalise les sauts de ligne multiples.
    # \n{3,} = 3 sauts de ligne ou plus → remplacé par exactement 2 sauts de ligne.
    # Cela évite d'avoir de grands espaces vides entre les paragraphes.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # .strip() supprime les espaces/sauts de ligne au début et à la fin de la chaîne.
    return text.strip()


# =============================================================================
# FONCTION PRINCIPALE : parse_pdf
# =============================================================================
def parse_pdf(pdf_path: Path | str) -> list[RawPage]:
    """
    Lit un fichier PDF et retourne une liste de RawPage (une par page).

    C'est LA fonction principale de ce module.
    Elle est appelée par pipeline.py pour chaque PDF du répertoire data/raw/.

    Paramètres :
        pdf_path : chemin vers le fichier PDF.
                   Accepte soit un objet Path, soit une string simple.
                   Ex: parse_pdf("data/raw/gdpr_full.pdf")
                   Ex: parse_pdf(Path("data/raw/gdpr_full.pdf"))

    Retourne :
        list[RawPage] : liste des pages du PDF, dans l'ordre.
                        Seules les pages avec du texte sont incluses (pages vides ignorées).

    Lève :
        FileNotFoundError : si le fichier n'existe pas.
    """
    # On convertit en Path même si on reçoit une string.
    # Cela nous donne accès aux méthodes utiles comme .exists(), .name, etc.
    pdf_path = Path(pdf_path)

    # Vérification d'existence AVANT d'ouvrir.
    # Mieux vaut un message d'erreur clair qu'une erreur cryptique de fitz.
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    # Liste vide qui va accumuler les pages au fur et à mesure
    pages: list[RawPage] = []

    # fitz.open() ouvre le PDF.
    # Le bloc "with ... as doc:" garantit que le fichier est FERMÉ automatiquement
    # à la fin du bloc, même si une erreur survient.
    # C'est important pour ne pas laisser des fichiers ouverts en mémoire.
    with fitz.open(str(pdf_path)) as doc:

        # "enumerate(doc, start=1)" itère sur toutes les pages du PDF.
        # "start=1" : on commence à numéroter à 1 (pas 0), comme dans un vrai document.
        # page_num = numéro de la page (1, 2, 3, ...)
        # page = objet Page de PyMuPDF, contient toutes les données de la page
        for page_num, page in enumerate(doc, start=1):

            # page.get_text("text") extrait le texte de la page.
            # Le paramètre "text" signifie : texte simple, sans mise en forme.
            # Il existe d'autres modes : "html", "dict", "blocks"...
            # On choisit "text" car on veut le contenu brut pour notre NLP.
            raw_text = page.get_text("text")

            # On nettoie le texte (suppression des numéros de page, etc.)
            cleaned = _clean_text(raw_text)

            # On ignore les pages vides (après nettoyage, certaines pages
            # ne contiennent que des numéros de page qui ont été supprimés).
            # Un "if cleaned:" est vrai si la chaîne n'est pas vide.
            if cleaned:
                pages.append(RawPage(
                    page_number=page_num,
                    text=cleaned,
                    source_file=pdf_path.name,  # .name = juste "gdpr_full.pdf" sans le chemin
                ))

    return pages


# =============================================================================
# FONCTION ALTERNATIVE : iter_pages (version streaming)
# =============================================================================
def iter_pages(pdf_path: Path | str) -> Iterator[RawPage]:
    """
    Version "streaming" de parse_pdf, utilisant un générateur Python.

    DIFFÉRENCE AVEC parse_pdf :
    - parse_pdf() charge TOUTES les pages en mémoire d'un coup (list).
    - iter_pages() retourne les pages UNE PAR UNE avec "yield".

    POURQUOI C'EST UTILE :
    L'EU AI Act + annexes font 450+ pages.
    Si on charge tout en mémoire : ~50 Mo de texte.
    Avec un générateur, on traite une page à la fois → très peu de mémoire.

    COMMENT L'UTILISER :
        for page in iter_pages("data/raw/ai_act_full.pdf"):
            print(page.page_number, page.char_count)

    CONCEPT "yield" :
    Quand Python rencontre "yield", il "pause" la fonction et retourne la valeur.
    À la prochaine itération de la boucle, la fonction reprend là où elle s'est arrêtée.
    C'est ce qu'on appelle un "générateur".
    """
    pdf_path = Path(pdf_path)

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)

            if cleaned:
                # "yield" au lieu de "return" : on retourne UNE page à la fois
                # sans charger tout le document en mémoire.
                yield RawPage(
                    page_number=page_num,
                    text=cleaned,
                    source_file=pdf_path.name,
                )
