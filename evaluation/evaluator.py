"""
evaluator.py — Évaluation automatique du système RAG
=============================================================================

RÔLE DE CE FICHIER :
Mesurer objectivement la qualité du système RAG sur deux dimensions :

  1. RECALL@K — Qualité de la RECHERCHE
     "Est-ce que le bon article juridique apparaît dans les k premiers résultats ?"
     → Mesure si FAISS + reranking retrouve les bonnes sources
     → Indépendant du LLM

  2. FAITHFULNESS — Fidélité de la GÉNÉRATION
     "Est-ce que la réponse Claude est bien supportée par les chunks fournis ?"
     → Mesure si Claude hallucine ou invente des informations
     → Utilise Claude lui-même comme juge (LLM-as-judge)

UTILISATION :
    # Depuis la racine du projet
    python -m eval.evaluator

    # Options
    python -m eval.evaluator --k 5          # recall@5 (défaut)
    python -m eval.evaluator --k 3          # recall@3
    python -m eval.evaluator --no-faithfulness   # skip faithfulness (plus rapide)
    python -m eval.evaluator --regulation GDPR   # évaluer uniquement GDPR

INTERPRÉTATION DES RÉSULTATS :
    recall@k ≥ 0.80 → bon système de recherche
    recall@k ≥ 0.60 → acceptable
    recall@k < 0.60 → revoir chunking, embeddings ou MIN_SCORE

    faithfulness ≥ 0.85 → Claude est fidèle aux sources
    faithfulness ≥ 0.70 → quelques hallucinations mineures
    faithfulness < 0.70 → revoir le prompt ou les chunks fournis
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Ajoute la racine du projet au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eval.evaluator")


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES COMPOSANTS
# ══════════════════════════════════════════════════════════════════════════════

def load_components():
    """
    Charge les 3 composants du pipeline RAG pour l'évaluation.
    Réutilise les mêmes classes que l'API — pas de duplication.

    Retourne : (vector_store, embedder, reranker)
    """
    logger.info("Chargement des composants RAG...")

    from rag.vector_store import VectorStore
    from rag.embedder import LegalEmbedder
    from rag.reranker import LegalReranker

    store = VectorStore()
    store.load()
    logger.info(f"✓ Index FAISS : {store.n_vectors} vecteurs")

    embedder = LegalEmbedder()
    embedder.load()
    logger.info(f"✓ Embedder : {embedder.model_name}")

    reranker = LegalReranker()
    reranker.load()
    logger.info(f"✓ Reranker : {'disponible' if reranker.is_available else 'indisponible'}")

    return store, embedder, reranker


# ══════════════════════════════════════════════════════════════════════════════
# RECALL@K
# ══════════════════════════════════════════════════════════════════════════════

def compute_recall_at_k(
    store:    object,
    embedder: object,
    reranker: object,
    dataset:  list,
    k:        int = 5,
    use_reranking: bool = True,
    regulation_filter: str | None = None,
) -> dict:
    """
    Calcule recall@k sur tout le dataset de référence.

    DÉFINITION RECALL@K :
    Pour chaque question, on récupère les k premiers résultats.
    La question est un succès si AU MOINS UN segment_id attendu
    apparaît dans ces k résultats.

    recall@k = nb_succès / nb_questions_avec_expected_ids

    POURQUOI "AU MOINS UN" et pas "TOUS" ?
    En RAG juridique, un article peut être découpé en plusieurs chunks.
    Ce qui compte c'est que le système trouve le bon texte — pas qu'il
    retrouve tous les chunks d'un article.

    Paramètres :
        store             : VectorStore chargé
        embedder          : LegalEmbedder chargé
        reranker          : LegalReranker chargé
        dataset           : liste d'EvalCase
        k                 : nombre de résultats à considérer
        use_reranking     : activer le reranking pour l'évaluation
        regulation_filter : filtrer le dataset par réglementation

    Retourne : dict avec métriques détaillées
    """
    from api.search import expand_query
    from rag.reranker import LegalReranker

    results_per_case = []
    hits             = 0
    total_evaluated  = 0

    # Filtre optionnel par réglementation
    cases = [
        c for c in dataset
        if regulation_filter is None or c.regulation == regulation_filter
    ]

    # On ignore les cas sans expected_ids (freetext CNIL par exemple)
    cases_with_ids = [c for c in cases if c.expected_ids]

    logger.info(
        f"Évaluation recall@{k} | "
        f"{len(cases_with_ids)} cas (sur {len(cases)} total) | "
        f"reranking={use_reranking}"
    )

    for i, case in enumerate(cases_with_ids, 1):
        t0 = time.perf_counter()

        # Query expansion (même logique que l'API)
        query, was_expanded = expand_query(case.question)

        # Encode la question
        query_vector = embedder.encode_query(query)

        # Recherche FAISS — k*2 candidats si reranking
        search_k = k * 2 if use_reranking else k
        raw_chunks = store.search(
            query_vector      = query_vector,
            k                 = search_k,
            regulation_filter = case.regulation,
            segment_type_filter = case.segment_type,
            min_score         = 0.30,  # seuil plus bas pour l'évaluation
        )

        # Reranking
        if use_reranking and reranker.is_available and len(raw_chunks) > 1:
            final_chunks = reranker.rerank(
                question = case.question,
                chunks   = raw_chunks,
                top_k    = k,
            )
        else:
            final_chunks = raw_chunks[:k]

        # Vérification du hit
        retrieved_ids = {c.get("segment_id", "") for c in final_chunks}
        hit = any(exp_id in retrieved_ids for exp_id in case.expected_ids)

        if hit:
            hits += 1

        total_evaluated += 1
        elapsed = (time.perf_counter() - t0) * 1000

        # Log détaillé par cas
        status = "✓ HIT" if hit else "✗ MISS"
        logger.info(
            f"  [{i:2d}/{len(cases_with_ids)}] {status} | "
            f"{case.regulation or 'ALL':15s} | "
            f"{case.question[:60]}..."
        )
        if not hit:
            logger.info(
                f"           Expected : {case.expected_ids}\n"
                f"           Got      : {sorted(retrieved_ids)}"
            )

        results_per_case.append({
            "question":     case.question,
            "regulation":   case.regulation,
            "expected_ids": case.expected_ids,
            "retrieved_ids": sorted(retrieved_ids),
            "hit":          hit,
            "expanded":     was_expanded,
            "latency_ms":   round(elapsed, 1),
        })

    recall = hits / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "metric":          f"recall@{k}",
        "score":           round(recall, 4),
        "hits":            hits,
        "total":           total_evaluated,
        "use_reranking":   use_reranking,
        "cases":           results_per_case,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FAITHFULNESS (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated legal answer is faithful to its source documents.

SOURCES PROVIDED TO THE AI:
{sources}

AI ANSWER TO EVALUATE:
{answer}

TASK:
Analyze each factual claim in the AI answer. Determine if it is:
- SUPPORTED: directly backed by the sources above
- UNSUPPORTED: not found in the sources (potential hallucination)
- NEUTRAL: general framing/transition without factual content

Return ONLY a JSON object with this exact structure:
{{
  "supported_claims": <integer>,
  "unsupported_claims": <integer>,
  "neutral_claims": <integer>,
  "faithfulness_score": <float between 0 and 1>,
  "unsupported_examples": [<list of up to 3 unsupported claim quotes>],
  "assessment": "<one sentence summary>"
}}

faithfulness_score = supported_claims / (supported_claims + unsupported_claims)
If there are no factual claims, return faithfulness_score: 1.0"""


async def evaluate_faithfulness_single(
    question: str,
    answer:   str,
    sources:  list[dict],
    client,
) -> dict:
    """
    Évalue la fidélité d'une réponse par rapport à ses sources via Claude.

    LLM-AS-JUDGE :
    On demande à Claude d'analyser sa propre réponse (ou une autre réponse)
    et de vérifier claim par claim si chaque affirmation est supportée par
    les sources fournies. C'est l'approche RAGAS simplifiée.

    Retourne un dict avec le score de faithfulness et les détails.
    """
    # Formate les sources pour le prompt du juge
    sources_text = "\n\n".join([
        f"[SOURCE {i+1}] {s.get('regulation')} — {s.get('segment_id')}\n{s.get('text', '')[:800]}"
        for i, s in enumerate(sources)
    ])

    prompt = FAITHFULNESS_PROMPT.format(
        sources = sources_text,
        answer  = answer,
    )

    try:
        message = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences si présentes
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        result["question"] = question[:80]
        return result

    except Exception as e:
        logger.warning(f"Faithfulness eval échouée : {e}")
        return {
            "question":            question[:80],
            "faithfulness_score":  None,
            "error":               str(e),
        }


async def compute_faithfulness(
    store:    object,
    embedder: object,
    reranker: object,
    dataset:  list,
    k:        int = 5,
    max_cases: int = 10,
    regulation_filter: str | None = None,
) -> dict:
    """
    Calcule la faithfulness moyenne sur un sous-ensemble du dataset.

    On limite à max_cases pour contrôler le coût API (chaque cas = 2 appels Claude).

    Retourne : dict avec score moyen et détails par cas.
    """
    import anthropic
    from api.search import expand_query, build_prompt

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY manquante — faithfulness ignorée"}

    client = anthropic.Anthropic(api_key=api_key)

    cases = [c for c in dataset if c.expected_ids]
    if regulation_filter:
        cases = [c for c in cases if c.regulation == regulation_filter]

    # Limite le nombre de cas pour maîtriser les coûts
    cases = cases[:max_cases]

    logger.info(
        f"Évaluation faithfulness | "
        f"{len(cases)} cas | k={k} | modèle=claude-sonnet-4-20250514"
    )

    scores  = []
    details = []

    for i, case in enumerate(cases, 1):
        logger.info(f"  [{i:2d}/{len(cases)}] {case.question[:60]}...")

        # 1. Récupère les chunks (même pipeline que l'API)
        query, _ = expand_query(case.question)
        query_vector = embedder.encode_query(query)

        raw_chunks = store.search(
            query_vector      = query_vector,
            k                 = k * 2,
            regulation_filter = case.regulation,
            min_score         = 0.30,
        )

        if reranker.is_available and len(raw_chunks) > 1:
            final_chunks = reranker.rerank(
                question = case.question,
                chunks   = raw_chunks,
                top_k    = k,
            )
        else:
            final_chunks = raw_chunks[:k]

        if not final_chunks:
            logger.warning(f"  Aucun chunk trouvé pour : {case.question[:60]}")
            continue

        # 2. Génère la réponse Claude
        try:
            prompt = build_prompt(case.question, final_chunks, language="en")
            message = client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 1000,
                messages   = [{"role": "user", "content": prompt}],
            )
            answer = message.content[0].text
        except Exception as e:
            logger.warning(f"  Génération réponse échouée : {e}")
            continue

        # 3. Évalue la faithfulness de cette réponse
        result = await evaluate_faithfulness_single(
            question = case.question,
            answer   = answer,
            sources  = final_chunks,
            client   = client,
        )

        if result.get("faithfulness_score") is not None:
            scores.append(result["faithfulness_score"])
            details.append(result)
            logger.info(
                f"     faithfulness={result['faithfulness_score']:.2f} | "
                f"supported={result.get('supported_claims', '?')} | "
                f"unsupported={result.get('unsupported_claims', '?')}"
            )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "metric":              "faithfulness",
        "score":               round(avg_score, 4),
        "n_cases_evaluated":   len(scores),
        "scores_distribution": {
            "min":    round(min(scores), 4) if scores else None,
            "max":    round(max(scores), 4) if scores else None,
            "mean":   round(avg_score, 4),
            "median": round(float(np.median(scores)), 4) if scores else None,
        },
        "cases": details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_report(recall_results: dict, faithfulness_results: dict | None) -> None:
    """Affiche un rapport lisible dans le terminal."""

    print("\n" + "=" * 60)
    print("  RAPPORT D'ÉVALUATION — AI Legal Governance Platform")
    print("=" * 60)

    # ── Recall@k ──────────────────────────────────────────────────
    r = recall_results
    score = r["score"]
    grade = "🟢 BON" if score >= 0.80 else ("🟡 ACCEPTABLE" if score >= 0.60 else "🔴 À AMÉLIORER")

    print(f"\n📊 RECALL@{r['metric'].split('@')[1]}")
    print(f"   Score  : {score:.1%}  {grade}")
    print(f"   Hits   : {r['hits']} / {r['total']}")
    print(f"   Reranking : {'activé' if r['use_reranking'] else 'désactivé'}")

    # Détail des MISS
    misses = [c for c in r["cases"] if not c["hit"]]
    if misses:
        print(f"\n   ✗ MISS ({len(misses)}) :")
        for m in misses:
            print(f"     • [{m['regulation']}] {m['question'][:55]}...")
            print(f"       Expected : {m['expected_ids']}")
            print(f"       Got      : {m['retrieved_ids'][:3]}")

    # ── Faithfulness ──────────────────────────────────────────────
    if faithfulness_results and "error" not in faithfulness_results:
        f = faithfulness_results
        fscore = f["score"]
        fgrade = "🟢 BON" if fscore >= 0.85 else ("🟡 ACCEPTABLE" if fscore >= 0.70 else "🔴 À AMÉLIORER")

        print(f"\n🎯 FAITHFULNESS (LLM-as-judge)")
        print(f"   Score  : {fscore:.1%}  {fgrade}")
        print(f"   Cas évalués : {f['n_cases_evaluated']}")
        dist = f.get("scores_distribution", {})
        if dist.get("min") is not None:
            print(f"   Min/Max/Médiane : {dist['min']:.1%} / {dist['max']:.1%} / {dist['median']:.1%}")

        # Exemples de claims non supportés
        for case in f.get("cases", []):
            examples = case.get("unsupported_examples", [])
            if examples:
                print(f"\n   ⚠ Claims non supportés dans : {case['question'][:50]}...")
                for ex in examples[:2]:
                    print(f"     → \"{ex[:80]}\"")

    elif faithfulness_results and "error" in faithfulness_results:
        print(f"\n🎯 FAITHFULNESS : {faithfulness_results['error']}")

    print("\n" + "=" * 60)


def save_results(
    recall_results:       dict,
    faithfulness_results: dict | None,
    output_path:          Path,
) -> None:
    """Sauvegarde les résultats en JSON pour archivage."""
    import datetime
    output = {
        "timestamp":     datetime.datetime.now().isoformat(),
        "recall":        recall_results,
        "faithfulness":  faithfulness_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Résultats sauvegardés : {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

async def main_async(args) -> None:
    """Pipeline d'évaluation complet."""
    from evaluation.eval_dataset import EVAL_DATASET

    store, embedder, reranker = load_components()

    # ── Recall@k ──────────────────────────────────────────────────────────────
    recall_results = compute_recall_at_k(
        store              = store,
        embedder           = embedder,
        reranker           = reranker,
        dataset            = EVAL_DATASET,
        k                  = args.k,
        use_reranking      = not args.no_reranking,
        regulation_filter  = args.regulation,
    )

    # ── Faithfulness ──────────────────────────────────────────────────────────
    faithfulness_results = None
    if not args.no_faithfulness:
        faithfulness_results = await compute_faithfulness(
            store              = store,
            embedder           = embedder,
            reranker           = reranker,
            dataset            = EVAL_DATASET,
            k                  = args.k,
            max_cases          = args.max_faithfulness_cases,
            regulation_filter  = args.regulation,
        )

    # ── Rapport ───────────────────────────────────────────────────────────────
    print_report(recall_results, faithfulness_results)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    if args.output:
        save_results(
            recall_results,
            faithfulness_results,
            Path(args.output),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation du système RAG — recall@k et faithfulness"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Nombre de résultats pour recall@k (défaut: 5)"
    )
    parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Ignorer l'évaluation faithfulness (plus rapide, économise les tokens)"
    )
    parser.add_argument(
        "--no-reranking", action="store_true",
        help="Désactiver le reranking pour mesurer l'impact du cross-encoder"
    )
    parser.add_argument(
        "--regulation", type=str, default=None,
        choices=["GDPR", "EU_AI_ACT", "CNIL", "EDPB", "DATA_GOVERNANCE"],
        help="Filtrer l'évaluation sur une réglementation"
    )
    parser.add_argument(
        "--max-faithfulness-cases", type=int, default=10,
        help="Nombre max de cas pour faithfulness (défaut: 10, pour maîtriser les coûts)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results/latest.json",
        help="Chemin du fichier JSON de résultats (défaut: eval/results/latest.json)"
    )

    args = parser.parse_args()

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
