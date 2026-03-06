"""
pipeline.py — Orchestrateur du pipeline d'ingestion complet
=============================================================================

RÔLE DE CE FICHIER :
Ce fichier est le "chef d'orchestre" du pipeline de données.
Il coordonne tous les autres modules (pdf_parser, article_extractor,
metadata_builder, chunker) pour transformer les PDFs bruts en chunks
prêts à l'embedding.

ARCHITECTURE DU DATA LAKE :
Le pipeline produit deux couches de données ("bronze" et "silver"),
inspirées de l'architecture Medallion popularisée par Databricks :

    data/raw/          ← PDFs originaux (immuables, jamais modifiés)
         │
         ▼
    data/bronze/       ← Segments juridiques structurés (JSONL)
         │              Un fichier par PDF : gdpr_full_segments.jsonl
         │              Chaque ligne = un LegalSegment enrichi (Article, Recital...)
         ▼
    data/silver/       ← Chunks prêts pour l'embedding (JSONL)
                        Un fichier par PDF : gdpr_full_chunks.jsonl
                        Chaque ligne = un Chunk (≤1200 chars, avec métadonnées complètes)

POURQUOI DEUX COUCHES (BRONZE ET SILVER) ?
    Bronze = données "brutes mais structurées" :
    - On a extrait la structure (articles, recitals...) mais on n'a pas encore découpé
    - Utile pour débugger l'extraction : voir si les articles sont bien détectés
    - Réutilisable si on change la stratégie de chunking : on re-génère silver depuis bronze

    Silver = données "propres et prêtes à l'emploi" :
    - Découpées en chunks (≤1200 chars) avec overlap
    - Chaque chunk porte toutes ses métadonnées (regulation, article, page, fichier)
    - Directement consommables par rag/build_index.py pour créer les embeddings

FORMAT JSONL (JSON Lines) :
Chaque fichier .jsonl contient une ligne JSON par segment ou chunk.
Pourquoi JSONL plutôt qu'un JSON classique ?
- Streamable : on peut lire ligne par ligne sans charger tout le fichier en RAM
- Appendable : on peut ajouter des entrées sans réécrire tout le fichier
- Compatible avec les outils Big Data (Spark, DuckDB...)

UTILISATION :
    # Lancer le pipeline complet depuis la racine du projet :
    python -m ingestion.pipeline

    # Ou depuis le code :
    from ingestion.pipeline import run_full_ingestion
    summaries = run_full_ingestion()

FLUX COMPLET :
    PDF → parse_pdf() → pages → extract_segments() → segments
        → enrich_segment_metadata() → segments enrichis (bronze)
        → chunk_segment() → chunks (silver)
=============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

# Imports des modules du pipeline — chaque module fait une chose précise
from .article_extractor import extract_segments      # pages → segments structurés
from .chunker import chunk_segment                   # segments → chunks avec overlap
from .metadata_builder import enrich_segment_metadata, get_metadata  # enrichissement des métadonnées
from .pdf_parser import parse_pdf                    # PDF → pages

# ── Configuration du logger ───────────────────────────────────────────────────
# On utilise le module logging standard Python plutôt que print() car :
# - Les logs ont un niveau (INFO, WARNING, ERROR) qu'on peut filtrer
# - Le format inclut automatiquement timestamp, niveau et nom du module
# - En production, les logs peuvent être redirigés vers des fichiers ou des services
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingestion.pipeline")

# ── Chemins du Data Lake ──────────────────────────────────────────────────────
# On utilise Path() au lieu de strings pour la portabilité cross-platform
# (Windows utilise \ mais Mac/Linux utilisent /, Path() gère les deux)
RAW_DIR    = Path("data/raw")       # PDFs sources — jamais modifiés
BRONZE_DIR = Path("data/bronze")    # Segments structurés (LegalSegment)
SILVER_DIR = Path("data/silver")    # Chunks prêts pour l'embedding


# =============================================================================
# UTILITAIRE : Création des répertoires
# =============================================================================
def _ensure_dirs() -> None:
    """
    Crée les répertoires bronze/ et silver/ s'ils n'existent pas.

    parents=True : crée aussi les répertoires parents si nécessaire
                   Ex: si data/ n'existe pas non plus, il sera créé
    exist_ok=True : ne lève pas d'erreur si le répertoire existe déjà
                    (idempotent = peut être appelé plusieurs fois sans problème)
    """
    for d in [BRONZE_DIR, SILVER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAITEMENT D'UN SEUL DOCUMENT
# =============================================================================
def process_single_document(pdf_path: Path) -> dict:
    """
    Traite un PDF complet de bout en bout :
        raw (PDF) → bronze (segments) → silver (chunks)

    C'est la fonction centrale du pipeline. Elle est appelée une fois par PDF
    par run_full_ingestion(). Si elle échoue pour un PDF, l'erreur est catchée
    en dehors et le pipeline continue avec les autres PDFs.

    ÉTAPES INTERNES :
        1. parse_pdf()              : lit le PDF page par page (PyMuPDF)
        2. get_metadata()           : récupère les infos sur la source (SOURCE_REGISTRY)
        3. extract_segments()       : détecte les articles, recitals, sections...
        4. enrich_segment_metadata(): ajoute les métadonnées complètes à chaque segment
        5. Écriture bronze JSONL    : sauvegarde les segments enrichis
        6. chunk_segment()          : découpe chaque segment en chunks ≤1200 chars
        7. Écriture silver JSONL    : sauvegarde les chunks

    Paramètre :
        pdf_path (Path) : chemin vers le fichier PDF à traiter

    Retourne :
        dict : résumé du traitement (stats + chemins de sortie)
               Utilisé pour générer le rapport final ingestion_report.json
    """
    logger.info(f"Traitement : {pdf_path.name}")
    t0 = time.perf_counter()  # Mesure le temps de traitement de ce PDF

    # ── ÉTAPE 1 : Parsing PDF ─────────────────────────────────────────────────
    # parse_pdf() lit chaque page avec PyMuPDF et nettoie le bruit
    # (numéros de page solitaires, en-têtes "Official Journal"...)
    # Résultat : liste de RawPage, une par page non-vide
    pages = parse_pdf(pdf_path)

    # ── ÉTAPE 2 : Récupération des métadonnées source ─────────────────────────
    # get_metadata() cherche le nom de fichier dans SOURCE_REGISTRY
    # et retourne un SourceMetadata (regulation, year, jurisdiction, title...)
    # Cette info est essentielle pour extract_segments() qui a besoin de savoir
    # si c'est un texte législatif (→ cherche "Article X") ou une guideline
    # (→ cherche "I. II. III.A.")
    source_meta = get_metadata(pdf_path.name)

    # ── ÉTAPE 3 : Extraction des segments juridiques ──────────────────────────
    # extract_segments() analyse le texte complet pour détecter la structure :
    # - GDPR/AI Act → Article 5, Article 6, Recital (1), Annex I...
    # - EDPB/CNIL   → Section I., Section III.A., Annex 1...
    # Résultat : liste de LegalSegment (un par article/recital/section trouvé)
    segments = extract_segments(pages, regulation=source_meta.regulation)

    # Comptage par type pour les logs — utile pour vérifier que l'extraction marche bien
    n_articles = sum(1 for s in segments if s.segment_type == "article")
    n_annexes  = sum(1 for s in segments if s.segment_type == "annex")
    n_recitals = sum(1 for s in segments if s.segment_type == "recital")
    n_sections = sum(1 for s in segments if s.segment_type == "section")

    logger.info(
        f"  {len(pages)} pages | {len(segments)} segments "
        f"({n_articles} articles, {n_recitals} recitals, "
        f"{n_annexes} annexes, {n_sections} sections)"
    )

    # ── ÉTAPE 4 & 5 : Enrichissement + écriture Bronze ───────────────────────
    # Pour chaque segment :
    #   asdict(seg)               → convertit le dataclass LegalSegment en dict Python
    #                               (nécessaire pour la sérialisation JSON)
    #   enrich_segment_metadata() → ajoute regulation, year, jurisdiction, official_title, language
    #   json.dumps()              → sérialise en JSON
    # On écrit une ligne JSON par segment dans le fichier .jsonl
    #
    # JSONL = JSON Lines : chaque ligne est un JSON valide indépendant.
    # On peut lire ce fichier ligne par ligne sans charger tout en mémoire.
    bronze_path = BRONZE_DIR / f"{pdf_path.stem}_segments.jsonl"
    # pdf_path.stem = nom du fichier sans extension (ex: "gdpr_full" depuis "gdpr_full.pdf")

    segment_dicts: list[dict] = []  # On garde les dicts en mémoire pour l'étape suivante

    with bronze_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            # asdict() convertit récursivement le dataclass en dict Python pur
            # (tous les champs, y compris les dataclasses imbriquées)
            seg_dict = asdict(seg)

            # Enrichissement : ajoute les métadonnées de la source (year, jurisdiction...)
            enriched = enrich_segment_metadata(seg_dict)
            segment_dicts.append(enriched)

            # ensure_ascii=False : garde les caractères spéciaux (é, à, ü...) tels quels
            # plutôt que de les encoder en \u00e9 etc.
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    logger.info(f"  Bronze OK : {bronze_path.name} ({len(segment_dicts)} segments)")

    # ── ÉTAPE 6 & 7 : Chunking + écriture Silver ─────────────────────────────
    # Pour chaque segment enrichi, on récupère l'objet LegalSegment correspondant
    # (nécessaire car chunk_segment() attend un LegalSegment, pas un dict)
    # et on le découpe en chunks ≤1200 chars avec overlap de 200 chars.
    #
    # POURQUOI RETROUVER L'OBJET DEPUIS LE DICT ?
    # À l'étape précédente, on a converti les LegalSegment en dicts pour le JSON.
    # chunk_segment() a besoin de l'objet LegalSegment original (avec .char_count etc.).
    # On fait la correspondance via segment_id qui est unique dans ce document.
    all_chunks = []

    for seg_dict in segment_dicts:
        # next() avec une expression génératrice : trouve le premier LegalSegment
        # dont segment_id correspond. next() lève StopIteration si rien n'est trouvé
        # (ne devrait pas arriver car segment_dicts vient de segments).
        seg_obj = next(
            s for s in segments
            if s.segment_id == seg_dict["segment_id"]
        )

        # chunk_segment() retourne une liste de Chunk (peut être 1 si le segment est court)
        # max_chars=1200  : taille maximale d'un chunk en caractères
        # overlap_chars=200 : chevauchement entre chunks consécutifs
        chunks = chunk_segment(
            segment       = seg_obj,
            metadata      = seg_dict,
            max_chars     = 1200,
            overlap_chars = 200,
        )
        all_chunks.extend(chunks)  # .extend() ajoute tous les éléments de la liste

    logger.info(f"  {len(all_chunks)} chunks générés")

    # Écriture Silver : un chunk par ligne JSON
    silver_path = SILVER_DIR / f"{pdf_path.stem}_chunks.jsonl"

    with silver_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            # asdict() convertit le Chunk en dict pur (avec tous ses champs)
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    logger.info(f"  Silver OK : {silver_path.name} ({len(all_chunks)} chunks)")

    # ── Résumé du traitement de ce PDF ───────────────────────────────────────
    elapsed = time.perf_counter() - t0

    return {
        "source":      pdf_path.name,
        "regulation":  source_meta.regulation,
        "pages":       len(pages),
        "segments":    len(segments),
        "articles":    n_articles,
        "annexes":     n_annexes,
        "recitals":    n_recitals,
        "sections":    n_sections,
        "chunks":      len(all_chunks),
        "elapsed_s":   round(elapsed, 2),
        "bronze_path": str(bronze_path),
        "silver_path": str(silver_path),
    }


# =============================================================================
# PIPELINE COMPLET : tous les PDFs
# =============================================================================
def run_full_ingestion(raw_dir: Path = RAW_DIR) -> list[dict]:
    """
    Lance le pipeline d'ingestion sur tous les PDFs du répertoire raw/.

    STRATÉGIE D'ERREUR :
    Chaque PDF est traité dans un try/except indépendant.
    Si un PDF échoue (PDF corrompu, nom inconnu dans SOURCE_REGISTRY...),
    l'erreur est loggée et le pipeline CONTINUE avec les PDFs suivants.
    On ne bloque jamais l'ingestion complète pour un seul document problématique.

    À la fin, un fichier ingestion_report.json est écrit dans bronze/ avec
    les statistiques de chaque PDF (succès ou erreur).

    Paramètre :
        raw_dir (Path) : répertoire contenant les PDFs.
                         Valeur par défaut : data/raw/
                         Peut être changé pour les tests (ex: répertoire temporaire)

    Retourne :
        list[dict] : liste des résumés de traitement (un par PDF)
    """
    _ensure_dirs()

    # Cherche tous les fichiers .pdf dans le répertoire, triés alphabétiquement
    # sorted() assure un ordre de traitement déterministe (utile pour les logs et les tests)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"Aucun PDF trouvé dans {raw_dir}")
        return []

    logger.info(f"Pipeline lancé : {len(pdf_files)} documents à traiter")
    summaries: list[dict] = []

    for pdf_path in pdf_files:
        try:
            # Traitement d'un PDF complet : raw → bronze → silver
            summary = process_single_document(pdf_path)
            summaries.append(summary)
        except Exception as e:
            # exc_info=True : inclut la stacktrace complète dans les logs
            # (très utile pour débugger sans relancer le pipeline entier)
            logger.error(f"Échec pour {pdf_path.name}: {e}", exc_info=True)
            summaries.append({"source": pdf_path.name, "error": str(e)})

    # ── Rapport final ─────────────────────────────────────────────────────────
    # On sauvegarde un rapport JSON avec les stats de chaque PDF.
    # Utile pour auditer l'ingestion : combien d'articles extraits par PDF ?
    # Y a-t-il eu des erreurs ? Quel PDF est le plus lent à traiter ?
    report_path = BRONZE_DIR / "ingestion_report.json"
    with report_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    # Calcul des totaux pour le log final
    total_chunks   = sum(s.get("chunks",   0) for s in summaries)
    total_segments = sum(s.get("segments", 0) for s in summaries)
    successful     = sum(1 for s in summaries if "error" not in s)

    logger.info(
        f"\n{'='*50}\n"
        f"TERMINÉ : {successful}/{len(summaries)} documents OK\n"
        f"Total segments : {total_segments}\n"
        f"Total chunks   : {total_chunks}\n"
        f"Rapport écrit  : {report_path}\n"
        f"{'='*50}"
    )

    return summaries


# =============================================================================
# POINT D'ENTRÉE EN LIGNE DE COMMANDE
# =============================================================================
# Ce bloc s'exécute uniquement si on lance le fichier directement :
#     python -m ingestion.pipeline
# Il ne s'exécute PAS si le module est importé par un autre fichier.
# C'est la convention Python standard pour les scripts exécutables.
if __name__ == "__main__":
    run_full_ingestion()
