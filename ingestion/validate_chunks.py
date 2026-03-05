"""
=============================================================================
validate_chunks.py — Data Quality Report sur le corpus silver
=============================================================================

OBJECTIF :
Valider que le pipeline d'ingestion a produit des chunks de qualité
avant de les envoyer dans le vector store (FAISS).

Dans un projet IA en production, on ne passe JAMAIS à l'étape suivante
sans avoir validé la qualité des données de l'étape précédente.
Si les chunks sont mauvais, les embeddings sont mauvais,
et les réponses de l'IA sont mauvaises — on appelle ça "garbage in, garbage out".

Le but est de prouver que:
1. La taille des chunks est dans les bornes attendues (pas trop court, pas trop long)
2. Les métadonnées sont complètes et correctes pour chaque chunk
3. Les articles clés sont bien présents dans le corpus
4. Il n'y a pas de chunks vides ou corrompus

COMMANDE :
    python -m ingestion.validate_chunks

SORTIE :
    - Rapport affiché dans le terminal
    - data/bronze/quality_report.json  (rapport machine-readable)

=============================================================================
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
SILVER_DIR   = Path("data/silver")
BRONZE_DIR   = Path("data/bronze")

# ── Articles clés à vérifier absolument ───────────────────────────────────────
# Ce sont les articles les plus importants de chaque règlement.
# Si l'un d'eux est absent du corpus, c'est un problème sérieux.
CRITICAL_SEGMENTS = {
    "GDPR":               ["Article 5", "Article 6", "Article 9", "Article 17", "Article 22"],
    "EU_AI_ACT":          ["Article 5", "Article 6", "Article 9", "Article 10", "Article 13"],
    "DATA_GOVERNANCE_ACT":["Article 2", "Article 5", "Article 10"],
    "EDPB":               ["Section I", "Section II", "Section IV"],
    "CNIL":               [],  # Freetext — pas de segments nommés à vérifier
}

# ── Bornes de taille acceptables ──────────────────────────────────────────────
# En dessous de MIN_CHARS : chunk trop court, peu de contexte pour l'embedding
# Au dessus de MAX_CHARS : chunk trop long, risque de dépasser la fenêtre du modèle
MIN_CHARS = 100
MAX_CHARS = 2000


def load_all_chunks() -> list[dict]:
    """
    Charge tous les fichiers JSONL de la couche silver en une seule liste.
    Chaque ligne d'un fichier JSONL = un chunk = un dict Python.
    """
    all_chunks = []
    jsonl_files = sorted(SILVER_DIR.glob("*_chunks.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"Aucun fichier de chunks trouvé dans {SILVER_DIR}\n"
            f"Lance d'abord : python -m ingestion.pipeline"
        )

    for path in jsonl_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # ignore les lignes vides
                    all_chunks.append(json.loads(line))

    return all_chunks


def compute_size_stats(chunks: list[dict]) -> dict:
    """
    Calcule les statistiques de taille des chunks.

    Métriques calculées :
    - total     : nombre total de chunks
    - min/max   : taille minimale et maximale en caractères
    - moyenne   : taille moyenne
    - médiane   : taille médiane (plus robuste que la moyenne)
    - trop_courts : chunks sous MIN_CHARS (problématiques)
    - trop_longs  : chunks au-dessus de MAX_CHARS (problématiques)
    - distribution : répartition par tranches de taille
    """
    sizes = [len(c["text"]) for c in chunks]
    sizes_sorted = sorted(sizes)
    n = len(sizes)

    # Médiane : valeur du milieu quand on trie par ordre croissant
    median = sizes_sorted[n // 2] if n > 0 else 0

    # Distribution par tranches
    distribution = {
        "0-100 chars (trop court)":   sum(1 for s in sizes if s < 100),
        "100-500 chars":               sum(1 for s in sizes if 100 <= s < 500),
        "500-1000 chars":              sum(1 for s in sizes if 500 <= s < 1000),
        "1000-1500 chars (optimal)":   sum(1 for s in sizes if 1000 <= s < 1500),
        "1500-2000 chars":             sum(1 for s in sizes if 1500 <= s < 2000),
        "2000+ chars (trop long)":     sum(1 for s in sizes if s >= 2000),
    }

    return {
        "total_chunks": n,
        "min_chars":    min(sizes) if sizes else 0,
        "max_chars":    max(sizes) if sizes else 0,
        "avg_chars":    round(sum(sizes) / n) if n > 0 else 0,
        "median_chars": median,
        "trop_courts":  sum(1 for s in sizes if s < MIN_CHARS),
        "trop_longs":   sum(1 for s in sizes if s > MAX_CHARS),
        "distribution": distribution,
    }


def compute_coverage_stats(chunks: list[dict]) -> dict:
    """
    Calcule la répartition des chunks par réglementation et type de segment.

    Permet de vérifier que tous les documents sont bien représentés
    et qu'on n'a pas un déséquilibre trop fort (ex: 90% GDPR, 10% le reste).
    """
    # defaultdict(int) = dictionnaire où les valeurs manquantes valent 0 par défaut
    # Évite d'écrire "if key not in dict: dict[key] = 0" à chaque fois
    by_regulation   = defaultdict(int)
    by_segment_type = defaultdict(int)
    by_source       = defaultdict(int)

    for chunk in chunks:
        by_regulation[chunk.get("regulation", "UNKNOWN")]     += 1
        by_segment_type[chunk.get("segment_type", "UNKNOWN")] += 1
        by_source[chunk.get("source_file", "UNKNOWN")]        += 1

    return {
        "par_reglementation": dict(sorted(by_regulation.items())),
        "par_type_segment":   dict(sorted(by_segment_type.items())),
        "par_fichier_source": dict(sorted(by_source.items())),
    }


def check_critical_segments(chunks: list[dict]) -> dict:
    """
    Vérifie que les articles/sections critiques sont présents dans le corpus.

    Pour chaque règlement, on a défini une liste d'articles incontournables.
    Si l'un d'eux est absent, c'est un signal d'alarme : le parsing a raté
    quelque chose d'important.

    Retourne un dict avec pour chaque article : présent (True) ou absent (False).
    """
    # Construit un ensemble de tous les segment_id présents dans le corpus
    # Un ensemble (set) permet de faire la vérification en O(1)
    present_segment_ids = {
        chunk.get("segment_id", "") for chunk in chunks
    }

    results = {}

    for regulation, critical_list in CRITICAL_SEGMENTS.items():
        if not critical_list:
            results[regulation] = {"status": "N/A (freetext)", "details": {}}
            continue

        details = {}
        all_present = True

        for segment_id in critical_list:
            # On vérifie si l'article est présent dans n'importe quel chunk
            # (un article peut être découpé en plusieurs chunks)
            found = any(segment_id in sid for sid in present_segment_ids)
            details[segment_id] = "✅ présent" if found else "❌ ABSENT"
            if not found:
                all_present = False

        results[regulation] = {
            "status":  "✅ complet" if all_present else "⚠️  incomplet",
            "details": details,
        }

    return results


def check_metadata_completeness(chunks: list[dict]) -> dict:
    """
    Vérifie que tous les champs de métadonnées obligatoires sont remplis.

    Un chunk avec des métadonnées manquantes est inutilisable pour les citations :
    l'IA ne pourrait pas dire "selon l'Article X du Règlement Y (année Z)".
    """
    # Champs obligatoires pour qu'un chunk soit exploitable
    required_fields = [
        "chunk_id", "text", "segment_id", "segment_type",
        "regulation", "document_type", "year", "jurisdiction",
        "official_title", "source_file", "page_start",
    ]

    issues = defaultdict(list)  # champ → liste des chunk_ids problématiques

    for chunk in chunks:
        for field in required_fields:
            value = chunk.get(field)
            # On considère un champ manquant si : absent, None, chaîne vide, ou 0 pour year
            if value is None or value == "" or (field == "year" and value == 0):
                chunk_id = chunk.get("chunk_id", "UNKNOWN")
                issues[field].append(chunk_id)

    total = len(chunks)
    summary = {}

    for field in required_fields:
        n_issues = len(issues[field])
        if n_issues == 0:
            summary[field] = f"✅ 100% rempli ({total}/{total})"
        else:
            pct_ok = round(100 * (total - n_issues) / total, 1)
            summary[field] = f"⚠️  {pct_ok}% rempli ({total - n_issues}/{total}) — {n_issues} problèmes"

    return summary


def sample_random_chunks(chunks: list[dict], n: int = 5) -> list[dict]:
    """
    Tire N chunks aléatoires pour inspection manuelle.

    Dans le monde pro, l'inspection manuelle est irremplaçable.
    Les statistiques peuvent masquer des problèmes qualitatifs
    (ex: texte en mauvaise langue, caractères corrompus, mauvais découpage).
    On tire des chunks au hasard et on les lit pour vérifier visuellement.
    """
    sample_size = min(n, len(chunks))
    return random.sample(chunks, sample_size)


def print_report(stats: dict, coverage: dict, critical: dict,
                 metadata: dict, samples: list[dict]) -> None:
    """
    Affiche le rapport de qualité dans le terminal de façon lisible.
    """
    sep = "=" * 65

    print(f"\n{sep}")
    print("  DATA QUALITY REPORT — AI Legal Governance Platform")
    print(f"{sep}")

    # ── 1. Statistiques de taille ─────────────────────────────────────────────
    print("\n📊 1. STATISTIQUES DE TAILLE DES CHUNKS")
    print(f"   Total chunks      : {stats['total_chunks']}")
    print(f"   Taille min        : {stats['min_chars']} chars")
    print(f"   Taille max        : {stats['max_chars']} chars")
    print(f"   Taille moyenne    : {stats['avg_chars']} chars")
    print(f"   Taille médiane    : {stats['median_chars']} chars")
    print(f"   Chunks trop courts (<{MIN_CHARS})  : {stats['trop_courts']}")
    print(f"   Chunks trop longs  (>{MAX_CHARS}) : {stats['trop_longs']}")
    print("\n   Distribution :")
    for tranche, count in stats["distribution"].items():
        pct = round(100 * count / stats["total_chunks"], 1) if stats["total_chunks"] > 0 else 0
        bar = "█" * int(pct / 2)  # barre visuelle proportionnelle
        print(f"   {tranche:<30} {count:>5} ({pct:>5}%) {bar}")

    # ── 2. Couverture par réglementation ─────────────────────────────────────
    print("\n📚 2. COUVERTURE PAR RÉGLEMENTATION")
    total = stats["total_chunks"]
    for reg, count in coverage["par_reglementation"].items():
        pct = round(100 * count / total, 1) if total > 0 else 0
        print(f"   {reg:<25} {count:>5} chunks ({pct}%)")

    print("\n   Par type de segment :")
    for seg_type, count in coverage["par_type_segment"].items():
        pct = round(100 * count / total, 1) if total > 0 else 0
        print(f"   {seg_type:<15} {count:>5} chunks ({pct}%)")

    # ── 3. Articles critiques ─────────────────────────────────────────────────
    print("\n🎯 3. PRÉSENCE DES ARTICLES CRITIQUES")
    for regulation, result in critical.items():
        print(f"\n   {regulation} — {result['status']}")
        for article, status in result["details"].items():
            print(f"     {article:<20} {status}")

    # ── 4. Complétude des métadonnées ─────────────────────────────────────────
    print("\n🏷️  4. COMPLÉTUDE DES MÉTADONNÉES")
    for field, status in metadata.items():
        print(f"   {field:<20} {status}")

    # ── 5. Échantillons aléatoires ────────────────────────────────────────────
    print("\n🔍 5. ÉCHANTILLONS ALÉATOIRES (inspection manuelle)")
    for i, chunk in enumerate(samples, 1):
        print(f"\n   --- Chunk {i} ---")
        print(f"   ID         : {chunk.get('chunk_id')}")
        print(f"   Règlement  : {chunk.get('regulation')} ({chunk.get('year')})")
        print(f"   Segment    : {chunk.get('segment_id')} [{chunk.get('segment_type')}]")
        print(f"   Source     : {chunk.get('source_file')} p.{chunk.get('page_start')}")
        print(f"   Taille     : {len(chunk.get('text', ''))} chars")
        # On affiche les 200 premiers caractères du texte pour vérification visuelle
        text_preview = chunk.get("text", "")[:200].replace("\n", " ")
        print(f"   Aperçu     : {text_preview}...")

    print(f"\n{sep}")


def run_validation() -> dict:
    """
    Lance la validation complète et retourne le rapport.
    """
    print("Chargement des chunks depuis data/silver/...")
    chunks = load_all_chunks()
    print(f"{len(chunks)} chunks chargés.")

    # Calcul de toutes les métriques
    stats    = compute_size_stats(chunks)
    coverage = compute_coverage_stats(chunks)
    critical = check_critical_segments(chunks)
    metadata = check_metadata_completeness(chunks)
    samples  = sample_random_chunks(chunks, n=5)

    # Affichage dans le terminal
    print_report(stats, coverage, critical, metadata, samples)

    # Sauvegarde du rapport JSON (pour archivage ou CI/CD)
    report = {
        "size_stats":            stats,
        "coverage":              coverage,
        "critical_segments":     critical,
        "metadata_completeness": metadata,
    }

    report_path = BRONZE_DIR / "quality_report.json"
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nRapport JSON sauvegardé : {report_path}\n")
    return report


if __name__ == "__main__":
    run_validation()