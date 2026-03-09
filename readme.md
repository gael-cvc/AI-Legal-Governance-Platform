# AI Legal Governance Intelligence Platform

> RAG system for European regulatory law — GDPR · EU AI Act · DGA · EDPB Guidelines · CNIL

A production-grade Retrieval-Augmented Generation (RAG) API that answers legal questions in natural language with **full source traceability**. Every claim in every response is anchored to a specific article or recital from the official regulatory corpus.

Built for lawyers, DPOs, and compliance officers who need verifiable answers — not hallucinated summaries.

---

## Why RAG and not a chatbot?

A general-purpose LLM knows the GDPR — but you can't audit its answers. It completes from memory, mixes versions, and cites articles that may have been amended. RAG forces the model to answer **exclusively from the documents you provide**, with mandatory citations. Every response is traceable, auditable, and reproducible.

---

## Features

| Feature | Status |
|---|---|
| RAG pipeline (query expansion → FAISS → reranker → Claude) | ✅ |
| Two-stage retrieval (vector search + cross-encoder reranking) | ✅ |
| Metadata filtering (regulation, article, segment type, language) | ✅ |
| Prompt injection defense (18 regex + structural heuristics) | ✅ |
| Hallucination guardrail (ghost source detection, LOW/HIGH severity) | ✅ |
| Faithfulness evaluation — LLM-as-judge (88.5% @ k=5) | ✅ |
| Recall@5 evaluation pipeline (100% internal consistency) | ✅ |
| Docker containerization (MPS local / CPU cloud) | ✅ |
| Bilingual responses (FR / EN) | ✅ |
| Auth JWT + rate limiting | 🔄 In progress |
| Frontend (Next.js / Streamlit) | 📋 Roadmap |
| Audit log JSONL | 📋 Roadmap |

---

## Corpus

| Document | Regulation | Chunks |
|---|---|---|
| GDPR (Regulation 2016/679) | GDPR | ~600 |
| EU AI Act (Regulation 2024/1689) | EU_AI_ACT | ~500 |
| Data Governance Act | DATA_GOVERNANCE | ~200 |
| EDPB Guidelines (automated decisions, consent, transfers) | EDPB | ~400 |
| CNIL AI recommendations | CNIL | ~150 |
| + supplementary documents | — | ~166 |

**Total : 2,016 chunks · 8 PDFs · 384-dimensional vectors**

---

## Architecture

```
Question
    │
    ▼
[0] Prompt Injection Defense     — 18 regex + structural heuristics → HTTP 400
    │
    ▼
[1] Query Expansion              — 3 reformulations via LLM → reduces vocabulary mismatch
    │
    ▼
[2] FAISS Vector Search          — IndexFlatIP, cosine similarity, k*2 candidates
    │
    ▼
[3] Cross-Encoder Reranking      — ms-marco-MiniLM-L-6-v2, reads (question, chunk) together
    │
    ▼
[4] Claude Generation            — build_prompt() v1.1, FORBIDDEN memory completion rule
    │
    ▼
[5] Hallucination Guardrail      — ghost source detection, LOW disclaimer / HIGH HTTP 503
    │
    ▼
Structured JSON response (answer + citations + sources + metadata)
```

**Singleton pattern** — VectorStore, LegalEmbedder, and LegalReranker are loaded once at startup via FastAPI lifespan. Zero reloading overhead per request.

---

## Evaluation

### Recall@5 — 100% (internal consistency)

19/19 questions return at least one expected segment in the top 5 results with reranking enabled.

> ⚠️ **Methodological note**: the dataset was calibrated on the existing index — this measures internal consistency, not external validity. An independent benchmark (lawyer annotation, multi-value `expected_ids`) is planned for a future release.

### Faithfulness — 88.5% (LLM-as-judge, k=5, prompt v1.1)

Measures the share of claims in Claude's response that are directly traceable to the provided source chunks.

**3 runs documented:**

| Run | Config | Score | Delta |
|---|---|---|---|
| Run 1 | Prompt v1.0 · k=5 | 87.4% | baseline |
| Run 2 | Prompt v1.1 (FORBIDDEN rule) · k=5 | **88.5%** | +1.1% |
| Run 3 | Prompt v1.1 · k=7 | 87.4% | −1.1% |

**Structural ceiling ~88%** identified: the unsupported claims correspond to legally correct information that Claude has deeply internalized from training (GDPR is heavily represented). This is not hallucination — it's untraceable memory completion. The ceiling is a model-level constraint, not a retrieval problem. k=7 confirmed this: adding more chunks did not help.

### Hallucination Guardrail — Active

Detects **ghost sources**: `[SOURCE X]` references where X > number of chunks provided.

- **LOW** (1 ghost): response returned with disclaimer appended + warning log
- **HIGH** (2+ ghosts): HTTP 503, response blocked

This is a **Level 1 guardrail** (structural check, 0ms, 0 cost). Level 2 (semantic faithfulness per request via LLM-as-judge) is on the roadmap at ~$0.02/req.

---

## Stack

```
Python 3.12
FastAPI          — async REST API, Pydantic validation
FAISS            — IndexFlatIP, exact cosine similarity
sentence-transformers — all-MiniLM-L6-v2 (embedder) + ms-marco-MiniLM-L-6-v2 (reranker)
Anthropic API    — claude-sonnet-4-20250514
PyMuPDF          — PDF ingestion
Docker + Compose — containerization, DEVICE env var (mps/cpu)
```

---

## Quick Start

### Local (Mac M4 / MPS)

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform
cd AI-Legal-Governance-Platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

TRANSFORMERS_OFFLINE=1 uvicorn api.main:app --reload --port 8000
```

### Docker (CPU)

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

docker compose up --build
```

### Build the index (required on first run)

```bash
python -m rag.build_index
# Expected: 2016 vectors · 384D · IndexFlatIP
```

---

## API

### POST /search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the obligations of a data controller under GDPR?",
    "k": 5,
    "use_reranking": true,
    "language": "fr"
  }'
```

**Optional filters:**
```json
{
  "regulation": "GDPR",
  "segment_type": "article",
  "article_number": "Article 35",
  "language_filter": "en",
  "min_score": 0.35
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "vector_store": { "loaded": true, "n_vectors": 2016, "dimension": 384 },
  "embedder":     { "loaded": true, "model": "all-MiniLM-L6-v2", "device": "mps" },
  "reranker":     { "loaded": true, "model": "ms-marco-MiniLM-L-6-v2", "device": "mps" }
}
```

### GET /search/suggestions

Returns example questions by regulation for frontend onboarding.

---

## Evaluation CLI

```bash
# Recall@5 only (fast, free)
python -m evaluation.evaluator --no-faithfulness

# Recall@3 + faithfulness on 10 cases (~$0.20)
python -m evaluation.evaluator --k 3

# GDPR only
python -m evaluation.evaluator --regulation GDPR --no-faithfulness

# Measure reranking impact
python -m evaluation.evaluator --no-reranking --no-faithfulness

# Custom output path
python -m evaluation.evaluator --output evaluation/results/run_$(date +%Y-%m-%d).json
```

---

## Project Structure

```
.
├── api/
│   ├── main.py          # FastAPI app, lifespan, singletons
│   ├── search.py        # RAG pipeline, guardrails, prompt engineering
│   └── models.py        # Pydantic schemas
├── rag/
│   ├── build_index.py   # FAISS index construction
│   ├── vector_store.py  # FAISS search + metadata filters
│   └── embedder.py      # LegalEmbedder singleton
├── ingestion/           # raw → bronze → silver pipeline
├── evaluation/
│   ├── evaluator.py     # recall@k + faithfulness LLM-as-judge
│   └── eval_dataset.py  # 19 EvalCase definitions
├── data/
│   ├── raw/             # source PDFs (not versioned)
│   ├── bronze/          # parsed segments
│   ├── silver/          # final chunks
│   └── vector_store/    # FAISS index (not versioned)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required
DEVICE=mps                     # mps (Mac M4) or cpu (Docker/cloud)
TRANSFORMERS_OFFLINE=1         # Use cached models (recommended)
```

---

## Metrics Summary

| Metric | Value | Notes |
|---|---|---|
| Chunks | 2,016 | silver layer, ready to query |
| Vector dimensions | 384 | all-MiniLM-L6-v2 |
| FAISS index size | ~3MB | fits entirely in RAM |
| FAISS latency | < 5ms | exact search, 2016 vectors |
| Reranking latency | ~1-2s | cross-encoder, 6 pairs |
| Total /search latency | ~14s | query expansion + Claude |
| Recall@5 | 100% ⚠ | internal dataset |
| Faithfulness | 88.5% | k=5, prompt v1.1, LLM-as-judge |
| Injection patterns | 18 regex | + 3 structural heuristics |
| Guardrail | Active | LOW/HIGH severity |

---

## Roadmap

- [ ] Auth JWT + API key rotation + rate limiting (slowapi)
- [ ] Audit log JSONL (request_id, sources, latency, user_id)
- [ ] Legal disclaimer automatic injection
- [ ] Frontend (Next.js or Streamlit)
- [ ] Corpus update pipeline (incremental FAISS.add())
- [ ] Unit tests + CI/CD (GitHub Actions)
- [ ] Cloud deployment (Cloud Run)
- [ ] Independent benchmark (lawyer-annotated dataset)

---

## License

Private repository — all rights reserved.
