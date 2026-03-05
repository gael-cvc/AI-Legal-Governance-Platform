"""
=============================================================================
chunker.py — Découpage intelligent des segments en chunks
=============================================================================

RÔLE DE CE FICHIER :
Ce fichier prend les segments juridiques (ex: Article 5 du GDPR = 800 mots)
et les découpe en morceaux plus petits appelés "chunks".

POURQUOI DÉCOUPER EN CHUNKS ?
Les modèles d'embedding (sentence-transformers) ont une limite de tokens.
Un article long de 1500 mots ne peut pas être traité d'un coup.
De plus, des chunks trop longs donnent des embeddings "dilués" : le vecteur
essaie de représenter trop de concepts à la fois et devient peu précis.

Un bon chunk = 100 à 350 mots (environ 600-1400 caractères).

POURQUOI L'OVERLAP ?
Si on découpe "phrase 1 | phrase 2 | phrase 3 | phrase 4" en chunks de 2 phrases
SANS overlap :
    Chunk 1 : "phrase 1, phrase 2"
    Chunk 2 : "phrase 3, phrase 4"

Si une question porte sur le lien entre phrase 2 et phrase 3, aucun chunk
ne contient les deux → mauvaise réponse.

AVEC overlap (1 phrase de recouvrement) :
    Chunk 1 : "phrase 1, phrase 2"
    Chunk 2 : "phrase 2, phrase 3"   ← phrase 2 est répétée
    Chunk 3 : "phrase 3, phrase 4"

Maintenant le lien entre phrase 2 et 3 existe dans le chunk 2.
L'overlap garantit que le contexte n'est jamais perdu aux frontières.

FLUX DE DONNÉES :
    LegalSegment  →  [chunk_segment()]  →  list[Chunk]  →  couche silver

=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# On importe LegalSegment depuis notre propre module
from .article_extractor import LegalSegment


# =============================================================================
# DATACLASS : Chunk
# =============================================================================
# Un Chunk est l'unité finale qui sera :
# 1. Transformée en vecteur (embedding) par sentence-transformers
# 2. Stockée dans FAISS (notre base de vecteurs)
# 3. Retournée à l'IA comme contexte pour répondre aux questions
@dataclass
class Chunk:
    """
    Un morceau de texte prêt pour l'embedding et la recherche vectorielle.

    Chaque Chunk porte TOUTES les métadonnées nécessaires pour :
    - Savoir d'où il vient (regulation, article, page, fichier)
    - Construire une citation précise dans la réponse de l'IA
    - Filtrer les résultats (ex: "cherche seulement dans le GDPR")
    """

    # Identifiant unique de ce chunk dans toute la base de données
    # Format : "Article_5_chunk_0", "Article_5_chunk_1", "Annex_I_chunk_0"...
    # Permet de retrouver exactement ce chunk plus tard
    chunk_id: str

    # Le texte du chunk — c'est ce qui sera transformé en vecteur
    text: str

    # Identifiant du segment parent : "Article 5", "Annex I", "Recital 42"
    # Permet de savoir de quel article vient ce chunk
    segment_id: str

    # Type du segment parent : "article", "annex", "recital", "chapter", "freetext"
    segment_type: str

    # Métadonnées de la source (copiées depuis SourceMetadata via metadata_builder)
    regulation: str        # "GDPR", "EU_AI_ACT", "EDPB", "CNIL"
    document_type: str     # "regulation", "guideline", "recommendation", "annex"
    year: int              # 2016, 2024...
    jurisdiction: str      # "EU", "FR"
    official_title: str    # Titre complet du document
    language: str          # "EN"

    # Traçabilité jusqu'au PDF source
    source_file: str       # "gdpr_full.pdf"
    page_start: int        # Page où commence ce chunk
    page_end: int          # Page où se termine ce chunk

    # Position du chunk au sein de son segment parent
    # Utile pour reconstruire le contexte complet si nécessaire
    chunk_index: int       # 0, 1, 2... (position de ce chunk)
    total_chunks: int      # Nombre total de chunks dans ce segment

    # Calculé automatiquement
    char_count: int = field(init=False)

    def __post_init__(self):
        """Calcule la taille du chunk en caractères."""
        self.char_count = len(self.text)

    def to_metadata_dict(self) -> dict:
        """
        Retourne les métadonnées SANS le texte.

        POURQUOI ?
        FAISS stocke les vecteurs d'un côté et les métadonnées séparément.
        Le vecteur = représentation mathématique du texte.
        Les métadonnées = infos sur ce vecteur (source, article, page...).
        On sépare les deux pour l'architecture de stockage.
        """
        # On retourne tous les champs SAUF "text"
        # __dict__ = dictionnaire de tous les attributs de l'objet
        # {k: v for k, v in ...} = dict comprehension : filtre les clés
        return {k: v for k, v in self.__dict__.items() if k != "text"}


# =============================================================================
# DÉTECTION DES FRONTIÈRES DE PHRASES
# =============================================================================
# On veut découper le texte en phrases proprement, sans couper au milieu d'une phrase.
# Cette regex détecte les frontières de phrases :
#
# (?<=[.;])   = lookbehind : il y a un point ou point-virgule AVANT
#               Un "lookbehind" = condition sur ce qui précède, sans le consommer
# \s+         = un ou plusieurs espaces (l'espace entre deux phrases)
# (?=[A-Z(])  = lookahead : il y a une majuscule ou "(" APRÈS
#               Un "lookahead" = condition sur ce qui suit, sans le consommer
#
# EXEMPLES de découpes :
# "Les données doivent être licites. Elles doivent aussi être..." → découpe après le point
# "Le responsable doit: (a) informer; (b) sécuriser. Le sous-traitant..." → découpe après ";"
#
# NON-DÉCOUPES (évitées grâce au lookahead [A-Z(]) :
# "Art. 5 GDPR" → le point est suivi de "5" (chiffre), pas de majuscule → pas de découpe
# "Inc." → suivi d'espace + minuscule → pas de découpe
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    """
    Découpe un texte en phrases en utilisant le pattern de frontière.

    Paramètre :
        text (str) : texte à découper

    Retourne :
        list[str] : liste de phrases, chacune nettoyée (strip)
    """
    # re.split() découpe la chaîne aux endroits où le pattern matche.
    # Contrairement à str.split(), re.split() supporte les patterns complexes.
    parts = _SENTENCE_BOUNDARY.split(text)

    # On nettoie chaque partie et on ignore les parties vides
    # p.strip() = supprime espaces en début/fin
    # if p.strip() = ignore les chaînes vides ou avec que des espaces
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# FONCTION PRINCIPALE : chunk_segment
# =============================================================================
def chunk_segment(
    segment: LegalSegment,
    metadata: dict,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> list[Chunk]:
    """
    Découpe un LegalSegment en Chunks avec overlap.

    PARAMÈTRES ET LEURS JUSTIFICATIONS :

    max_chars = 1200 caractères ≈ 300 tokens ≈ 200 mots
        Pourquoi 1200 ? Le modèle d'embedding "all-MiniLM-L6-v2" a une limite
        de 256 tokens (≈ 1000 chars). "bge-large-en-v1.5" supporte 512 tokens.
        1200 chars est un bon compromis : suffisant pour le contexte, pas trop long.

    overlap_chars = 200 caractères ≈ 50 tokens ≈ 35 mots
        L'overlap représente ~17% de la taille du chunk. C'est la norme RAG.
        Trop petit : on perd le contexte aux frontières.
        Trop grand : les chunks se répètent trop, le corpus devient redondant.

    min_chars = 100 caractères ≈ 15 mots
        Un chunk plus court que ça est trop petit pour avoir du sens juridique.
        On le fusionne avec le chunk précédent.

    Paramètres :
        segment      : le LegalSegment à découper
        metadata     : dict de métadonnées enrichies (sortie de enrich_segment_metadata)
        max_chars    : taille cible maximale par chunk (en caractères)
        overlap_chars: chevauchement entre chunks consécutifs (en caractères)
        min_chars    : taille minimale d'un chunk (les plus petits sont fusionnés)

    Retourne :
        list[Chunk] : liste des chunks, dans l'ordre
    """

    # ── CAS SPÉCIAL : segment déjà assez court ────────────────────────────────
    # Si le segment entier tient dans max_chars, pas besoin de le découper.
    # On retourne un seul chunk qui contient tout le segment.
    # C'est le cas pour la plupart des articles courts du GDPR.
    if segment.char_count <= max_chars:
        return [
            Chunk(
                chunk_id=f"{segment.segment_id}_chunk_0",
                text=segment.text,
                segment_id=segment.segment_id,
                segment_type=segment.segment_type,
                chunk_index=0,
                total_chunks=1,
                **_extract_chunk_metadata(segment, metadata),
                # ** = décompresse le dict retourné par _extract_chunk_metadata
                # et injecte ses clé-valeurs comme arguments nommés
            )
        ]

    # ── DÉCOUPAGE EN PHRASES ──────────────────────────────────────────────────
    sentences = _split_sentences(segment.text)

    # Si pour une raison quelconque on n'a pas de phrases, on retourne vide
    if not sentences:
        return []

    # ── CONSTRUCTION DES CHUNKS AVEC OVERLAP ─────────────────────────────────
    raw_chunks: list[str] = []  # Liste des textes de chunks bruts

    # current = liste des phrases en cours d'accumulation pour le chunk actuel
    current: list[str] = []
    # current_len = taille totale en caractères du chunk en cours
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)

        # Si ajouter cette phrase dépasserait max_chars ET qu'on a déjà
        # du contenu dans current (sinon une seule phrase ultra-longue resterait bloquée)
        if current_len + s_len > max_chars and current:

            # ── Finaliser le chunk courant ──
            # " ".join(current) = coller les phrases avec un espace
            raw_chunks.append(" ".join(current))

            # ── Calculer l'overlap ──
            # On veut garder les dernières phrases pour former le début du prochain chunk.
            # On remonte dans current de la fin vers le début et on accumule
            # jusqu'à atteindre overlap_chars caractères.
            overlap_buffer: list[str] = []
            overlap_len = 0

            # reversed(current) = parcourir la liste à l'envers (de la fin)
            for sent in reversed(current):
                if overlap_len + len(sent) <= overlap_chars:
                    # insert(0, ...) = insérer en DÉBUT de liste (pour maintenir l'ordre)
                    overlap_buffer.insert(0, sent)
                    overlap_len += len(sent)
                else:
                    break  # On a assez de contexte, on s'arrête

            # Le nouveau "current" commence avec les phrases de l'overlap
            current = overlap_buffer
            current_len = overlap_len

        # Ajouter la phrase courante au chunk en cours
        current.append(sentence)
        current_len += s_len

    # ── Ne pas oublier le dernier chunk ──
    # Après la boucle, current contient les phrases non encore validées
    if current:
        raw_chunks.append(" ".join(current))

    # ── POST-TRAITEMENT : fusionner les mini-chunks ───────────────────────────
    # Certains chunks de fin peuvent être très courts (ex: une seule phrase de 80 chars).
    # On les fusionne avec le chunk précédent pour éviter des chunks "orphelins".
    final_chunks: list[str] = []

    for chunk_text in raw_chunks:
        if len(chunk_text) < min_chars and final_chunks:
            # Fusion : on ajoute ce mini-chunk à la fin du chunk précédent
            final_chunks[-1] += " " + chunk_text
        else:
            final_chunks.append(chunk_text)

    # ── CONSTRUIRE LES OBJETS Chunk ───────────────────────────────────────────
    total = len(final_chunks)  # Nombre total de chunks pour ce segment

    # List comprehension : crée la liste de Chunk directement
    # enumerate(final_chunks) = donne l'index i et le texte chunk_text en même temps
    return [
        Chunk(
            chunk_id=f"{segment.segment_id}_chunk_{i}",
            text=chunk_text,
            segment_id=segment.segment_id,
            segment_type=segment.segment_type,
            chunk_index=i,
            total_chunks=total,
            **_extract_chunk_metadata(segment, metadata),
        )
        for i, chunk_text in enumerate(final_chunks)
    ]


# =============================================================================
# FONCTION UTILITAIRE PRIVÉE : _extract_chunk_metadata
# =============================================================================
def _extract_chunk_metadata(segment: LegalSegment, metadata: dict) -> dict:
    """
    Extrait les métadonnées communes à tous les chunks d'un segment.

    POURQUOI CETTE FONCTION EXISTE ?
    Pour éviter de répéter le même code dans chunk_segment().
    Au lieu de réécrire tous les champs à chaque création de Chunk,
    on appelle cette fonction qui retourne le dict des métadonnées.
    On l'injecte ensuite avec ** dans le constructeur de Chunk.

    LOGIQUE DE PRIORITÉ DES MÉTADONNÉES :
    metadata.get("regulation", segment.regulation) signifie :
    - Utilise metadata["regulation"] si disponible (source officielle = métadonnées enrichies)
    - Sinon, utilise segment.regulation (valeur par défaut du segment)
    La métadonnée enrichie est toujours plus fiable que la valeur initiale du segment.

    Paramètres :
        segment  : le LegalSegment source
        metadata : dict de métadonnées enrichies (depuis enrich_segment_metadata)

    Retourne :
        dict : champs à injecter dans chaque Chunk créé depuis ce segment
    """
    return {
        # .get(key, default) = valeur du dict si la clé existe, sinon default
        "regulation":    metadata.get("regulation",    segment.regulation),
        "document_type": metadata.get("document_type", ""),
        "year":          metadata.get("year",          0),
        "jurisdiction":  metadata.get("jurisdiction",  ""),
        "official_title": metadata.get("official_title", ""),
        "language":      metadata.get("language",      "EN"),
        "source_file":   segment.source_file,
        "page_start":    segment.page_start,
        "page_end":      segment.page_end,
    }
