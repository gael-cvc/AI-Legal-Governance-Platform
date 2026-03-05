"""pipeline.py"""
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .article_extractor import extract_segments
from .chunker import chunk_segment
from .metadata_builder import enrich_segment_metadata, get_metadata
from .pdf_parser import parse_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingestion.pipeline")

RAW_DIR    = Path("data/raw")
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")


def _ensure_dirs():
    for d in [BRONZE_DIR, SILVER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def process_single_document(pdf_path: Path) -> dict:
    logger.info(f"Traitement : {pdf_path.name}")
    t0 = time.perf_counter()

    pages = parse_pdf(pdf_path)
    source_meta = get_metadata(pdf_path.name)
    segments = extract_segments(pages, regulation=source_meta.regulation)

    n_articles = sum(1 for s in segments if s.segment_type == "article")
    n_annexes  = sum(1 for s in segments if s.segment_type == "annex")
    n_recitals = sum(1 for s in segments if s.segment_type == "recital")

    logger.info(f"  {len(pages)} pages | {len(segments)} segments ({n_articles} articles, {n_annexes} annexes, {n_recitals} recitals)")

    bronze_path = BRONZE_DIR / f"{pdf_path.stem}_segments.jsonl"
    segment_dicts = []
    with bronze_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            seg_dict = asdict(seg)
            enriched = enrich_segment_metadata(seg_dict)
            segment_dicts.append(enriched)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    logger.info(f"  Bronze OK : {bronze_path}")

    all_chunks = []
    for seg_dict in segment_dicts:
        seg_obj = next(s for s in segments if s.segment_id == seg_dict["segment_id"])
        chunks = chunk_segment(segment=seg_obj, metadata=seg_dict, max_chars=1200, overlap_chars=200)
        all_chunks.extend(chunks)

    logger.info(f"  {len(all_chunks)} chunks generes")

    silver_path = SILVER_DIR / f"{pdf_path.stem}_chunks.jsonl"
    with silver_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    logger.info(f"  Silver OK : {silver_path}")

    elapsed = time.perf_counter() - t0
    return {
        "source": pdf_path.name, "regulation": source_meta.regulation,
        "pages": len(pages), "segments": len(segments),
        "articles": n_articles, "annexes": n_annexes, "recitals": n_recitals,
        "chunks": len(all_chunks), "elapsed_s": round(elapsed, 2),
        "bronze_path": str(bronze_path), "silver_path": str(silver_path),
    }


def run_full_ingestion(raw_dir: Path = RAW_DIR) -> list[dict]:
    _ensure_dirs()
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"Aucun PDF trouve dans {raw_dir}")
        return []

    logger.info(f"Pipeline lance : {len(pdf_files)} documents")
    summaries = []

    for pdf_path in pdf_files:
        try:
            summaries.append(process_single_document(pdf_path))
        except Exception as e:
            logger.error(f"Echec pour {pdf_path.name}: {e}", exc_info=True)
            summaries.append({"source": pdf_path.name, "error": str(e)})

    report_path = BRONZE_DIR / "ingestion_report.json"
    with report_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    total_chunks   = sum(s.get("chunks",   0) for s in summaries)
    total_segments = sum(s.get("segments", 0) for s in summaries)
    successful     = sum(1 for s in summaries if "error" not in s)

    logger.info(
        f"\n{'='*50}\n"
        f"TERMINE : {successful}/{len(summaries)} documents OK\n"
        f"Segments : {total_segments} | Chunks : {total_chunks}\n"
        f"Rapport  : {report_path}\n"
        f"{'='*50}"
    )
    return summaries


if __name__ == "__main__":
    run_full_ingestion()
