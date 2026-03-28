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
| **Auth JWT + API keys + rate limiting** | ✅ |
| Docker containerization (MPS local / CPU cloud) | ✅ |
| Bilingual responses (FR / EN) | ✅ |
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
[1] Query Expansion              — 3 reformulations → reduces vocabulary mismatch
    │
    ▼
[2] FAISS Vector Search          — IndexFlatIP, cosine similarity, k*2 candidates
    │
    ▼
[3] Cross-Encoder Reranking      — ms-marco-MiniLM, reads (question, chunk) together
    │
    ▼
[4] Claude Generation            — build_prompt() v1.1, FORBIDDEN memory completion rule
    │
    ▼
[5] Hallucination Guardrail      — ghost source detection, LOW disclaimer / HIGH HTTP 503
    │
    ▼
Structured JSON response
```

**Auth layer** sits before step [0] — unauthenticated requests never reach the RAG pipeline.

---

## Authentication

The API uses **JWT Bearer tokens** or **API keys**. Every request to `/search` must be authenticated.

### Get a token

```bash
# Option 1 — Demo token (no credentials, dev only)
curl -X POST http://localhost:8000/auth/token/demo

# Option 2 — Token with credentials
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@cabinet.fr", "password": "demo1234"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Use the token

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{"question": "What are GDPR controller obligations?", "k": 5}'
```

### Rate limits by plan

| Plan | Limit | Window | How to get |
|---|---|---|---|
| `demo` | 10 req | 60s | `/auth/token/demo` |
| `cabinet` | 30 req | 60s | `/auth/token` with credentials |
| `admin` | 200 req | 60s | `/auth/token` with admin account |

When the limit is exceeded, the API returns HTTP `429 Too Many Requests` with a `Retry-After` header.

### API Key (alternative to JWT)

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "k": 5}'
```

---

## Quick Start

### Local (Mac M4 / MPS)

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform
cd AI-Legal-Governance-Platform
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and generate JWT_SECRET_KEY:
# python3 -c "import secrets; print(secrets.token_hex(32))"

TRANSFORMERS_OFFLINE=1 venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

### Docker (CPU)

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and JWT_SECRET_KEY to .env

docker compose up --build
```

### Build the index (required on first run)

```bash
venv/bin/python -m rag.build_index
# Expected: 2016 vectors · 384D · IndexFlatIP
```

---

## API Reference

### POST /api/v1/search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer <token>" \
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

### GET /api/v1/health

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "vector_store": { "loaded": true, "n_vectors": 2016, "dimension": 384 },
  "embedder":     { "loaded": true, "model": "all-MiniLM-L6-v2", "device": "mps" },
  "reranker":     { "loaded": true, "model": "ms-marco-MiniLM-L-6-v2", "device": "mps" }
}
```

### GET /auth/me

```bash
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <token>"
```

```json
{
  "sub": "demo@cabinet.fr",
  "plan": "demo",
  "client": "Cabinet Demo",
  "is_admin": false,
  "rate_limit_max_requests": 10,
  "rate_limit_window_seconds": 60
}
```

---

## Evaluation

### Recall@5 — 100% (internal consistency)

19/19 questions return at least one expected segment in top 5 results with reranking.

> ⚠️ **Methodological note**: dataset calibrated on the existing index. Measures internal consistency, not external validity. Independent benchmark (lawyer annotation) planned.

### Faithfulness — 88.5% (LLM-as-judge, k=5, prompt v1.1)

Measures the share of claims directly traceable to provided source chunks.

| Run | Config | Score | Delta |
|---|---|---|---|
| Run 1 | Prompt v1.0 · k=5 | 87.4% | baseline |
| Run 2 | Prompt v1.1 (FORBIDDEN rule) · k=5 | **88.5%** | +1.1% |
| Run 3 | Prompt v1.1 · k=7 | 87.4% | −1.1% |

**Structural ceiling ~88%** — GDPR overrepresentation in Claude's training data. Not a retrieval problem, confirmed by k=7 test.

### Hallucination Guardrail

- **LOW** (1 ghost source): disclaimer appended + warning log
- **HIGH** (2+ ghost sources): HTTP 503, response blocked

### Evaluation CLI

```bash
# Recall@5 only (fast, free)
venv/bin/python -m evaluation.evaluator --no-faithfulness

# Faithfulness on 10 cases (~$0.20)
venv/bin/python -m evaluation.evaluator --k 5
```

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Auth — generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=your_generated_secret_here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Demo accounts (change in production)
DEMO_PASSWORD=demo1234
ADMIN_PASSWORD=your_secure_admin_password

# Disable demo token endpoint in production
DISABLE_DEMO_TOKEN=false

# Device (mps for Mac M4, cpu for Docker/cloud)
DEVICE=mps
TRANSFORMERS_OFFLINE=1
```

---

## Project Structure

```
.
├── api/
│   ├── main.py          # FastAPI app, lifespan, singletons
│   ├── auth.py          # JWT, API keys, rate limiting (NEW v1.5)
│   ├── auth_router.py   # /auth/token, /auth/me endpoints (NEW v1.5)
│   ├── search.py        # RAG pipeline, guardrails, prompt engineering
│   └── models.py        # Pydantic schemas
├── rag/
│   ├── build_index.py   # FAISS index construction
│   ├── vector_store.py  # FAISS search + metadata filters
│   └── embedder.py      # LegalEmbedder singleton
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

## Stack

```
Python 3.12
FastAPI          — async REST API, Pydantic validation
PyJWT            — JWT creation and verification (HS256)
FAISS            — IndexFlatIP, exact cosine similarity
sentence-transformers — all-MiniLM-L6-v2 (embedder) + ms-marco-MiniLM (reranker)
Anthropic API    — claude-sonnet-4-20250514
Docker + Compose — containerization, DEVICE env var (mps/cpu)
```

---

## Metrics Summary

| Metric | Value | Notes |
|---|---|---|
| Chunks | 2,016 | silver layer |
| Vector dimensions | 384 | all-MiniLM-L6-v2 |
| FAISS index size | ~3MB | fits in RAM |
| FAISS latency | < 5ms | exact search |
| Total /search latency | ~14s | query expansion + Claude |
| Recall@5 | 100% ⚠ | internal dataset |
| Faithfulness | 88.5% | k=5, prompt v1.1 |
| Injection patterns | 18 regex | + 3 structural heuristics |
| Guardrail | Active | LOW/HIGH severity |
| Auth | JWT + API keys | rate limiting per plan |

---

## Roadmap

- [x] Auth JWT + API keys + rate limiting
- [x] Hallucination guardrail (ghost source detection)
- [x] Faithfulness evaluation (LLM-as-judge, 88.5%)
- [ ] Audit log JSONL (request_id, sources, latency, user_id)
- [ ] Legal disclaimer automatic injection
- [ ] Frontend (Next.js or Streamlit)
- [ ] Corpus update pipeline (incremental FAISS.add())
- [ ] Unit tests + CI/CD (GitHub Actions)
- [ ] Cloud deployment (Cloud Run)
- [ ] Redis for distributed rate limiting
- [ ] PostgreSQL for user management (replace hardcoded accounts)
- [ ] Independent benchmark (lawyer-annotated dataset)

---

## License

Private repository — all rights reserved.
