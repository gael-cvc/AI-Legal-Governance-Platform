# AI Legal Governance Intelligence Platform

> RAG system for European regulatory law — GDPR · EU AI Act · DGA · EDPB Guidelines · CNIL

A production-grade Retrieval-Augmented Generation (RAG) API that answers legal questions in natural language with **full source traceability**. Every claim in every response is anchored to a specific article or recital from the official regulatory corpus.

Built for lawyers, DPOs, and compliance officers who need verifiable answers — not hallucinated summaries.

---

## Why RAG and not a chatbot?

A general-purpose LLM knows the GDPR — but you cannot audit its answers. It completes from memory, mixes versions, and cites articles that may have been amended. RAG forces the model to answer **exclusively from the documents you provide**, with mandatory citations. Every response is traceable, auditable, and reproducible.

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
| Auth JWT + API keys + rate limiting | ✅ |
| Audit log JSONL (request_id, user, sources, latency, guardrail) | ✅ |
| Legal disclaimer automatic injection (FR/EN) | ✅ |
| Unit tests — 100/100 PASSED | ✅ |
| CI/CD — GitHub Actions (Python 3.12, pytest) | ✅ |
| Docker containerization (MPS local / CPU cloud) | ✅ |
| Bilingual responses (FR / EN) | ✅ |
| Frontend (Next.js / Streamlit) | 📋 Roadmap |

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
[Auth]  JWT / API key verification + rate limiting → HTTP 401/429
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
[6] Audit Log                    — JSONL write: request_id, user, sources, latency, status
    │
    ▼
[7] Legal Disclaimer             — appended to response (FR/EN), outside faithfulness scope
    │
    ▼
Structured JSON response
```

---

## Authentication

Every request to `/search` must be authenticated. The API supports **JWT Bearer tokens** and **API keys**.

### Get a token

```bash
# Demo token (no credentials — dev only)
curl -X POST http://localhost:8000/auth/token/demo

# Token with credentials
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@cabinet.fr", "password": "demo1234"}'
```

### Use the token

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the obligations of a data controller under GDPR?", "k": 5}'
```

### Rate limits by plan

| Plan | Limit | Window | How to get |
|---|---|---|---|
| `demo` | 10 req | 60s | `/auth/token/demo` |
| `cabinet` | 30 req | 60s | `/auth/token` with credentials |
| `admin` | 200 req | 60s | `/auth/token` with admin account |

---

## Audit Log

Every successful `/search` request writes one JSON line to `logs/audit/audit.jsonl`:

```json
{
  "timestamp": "2026-03-28T14:32:01.123Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_sub": "demo@cabinet.fr",
  "user_plan": "cabinet",
  "question": "What are GDPR controller obligations?",
  "sources_used": ["Article 24", "Recital 74", "Article 5"],
  "guardrail_severity": "ok",
  "latency_ms": 3420.5,
  "status": "success"
}
```

- Rotating file handler: 10 MB x 5 files = 50 MB max
- Retention: 180 days (configurable via `AUDIT_RETENTION_DAYS`)
- `logs/` is in `.gitignore` — never versioned

---

## Legal Disclaimer

Automatically appended to every response (FR/EN):

```
---
⚖️ Ce système est un outil d'aide à la recherche documentaire juridique.
Les informations fournies ne constituent pas un conseil juridique.
Pour toute décision juridique, consultez un avocat qualifié.
```

Disable for evaluation runs: `DISABLE_DISCLAIMER=true` in `.env`.

---

## Tests

```bash
# Install
venv/bin/pip install pytest pytest-cov

# Run all tests
venv/bin/python -m pytest tests/ -v
```

```
collected 100 items

tests/test_audit_disclaimer.py  27 passed
tests/test_auth.py              22 passed
tests/test_guardrail.py         22 passed
tests/test_injection.py         24 passed

========================= 100 passed in 2.14s =========================
```

| File | Tests | Covers |
|---|---|---|
| `test_injection.py` | 24 | `detect_prompt_injection()` — legit questions + known attacks |
| `test_guardrail.py` | 22 | `check_hallucination_guardrail()` — OK / LOW / HIGH thresholds |
| `test_auth.py` | 22 | JWT create/decode, API key hash, rate limiting sliding window |
| `test_audit_disclaimer.py` | 27 | JSONL validity, UUID format, disclaimer FR/EN content |

CI runs automatically on every push to `main` via GitHub Actions.

---

## Quick Start

### Local (Mac M4 / MPS)

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform
cd AI-Legal-Governance-Platform
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env:
# ANTHROPIC_API_KEY=sk-ant-...
# JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

TRANSFORMERS_OFFLINE=1 venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

### Docker (CPU)

```bash
cp .env.example .env  # add ANTHROPIC_API_KEY and JWT_SECRET_KEY
docker compose up --build
```

### Build the index (required on first run)

```bash
venv/bin/python -m rag.build_index
# Expected: 2016 vectors · 384D · IndexFlatIP
```

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Auth
JWT_SECRET_KEY=your_generated_secret_here
ACCESS_TOKEN_EXPIRE_MINUTES=60
DEMO_PASSWORD=demo1234
ADMIN_PASSWORD=your_secure_admin_password
DISABLE_DEMO_TOKEN=false

# Disclaimer
DISABLE_DISCLAIMER=false

# Audit log
AUDIT_LOG_DIR=logs/audit
AUDIT_RETENTION_DAYS=180

# Device
DEVICE=mps
TRANSFORMERS_OFFLINE=1
```

---

## Project Structure

```
.
├── api/
│   ├── main.py           # FastAPI app, lifespan, singletons
│   ├── auth.py           # JWT, API keys, rate limiting
│   ├── auth_router.py    # /auth/token, /auth/me endpoints
│   ├── audit_log.py      # JSONL audit logging
│   ├── search.py         # RAG pipeline, guardrails, disclaimer
│   └── models.py         # Pydantic schemas
├── rag/
│   ├── build_index.py    # FAISS index construction
│   ├── vector_store.py   # FAISS search + metadata filters
│   └── embedder.py       # LegalEmbedder singleton
├── evaluation/
│   ├── evaluator.py      # recall@k + faithfulness LLM-as-judge
│   └── eval_dataset.py   # 19 EvalCase definitions
├── tests/
│   ├── conftest.py
│   ├── test_injection.py
│   ├── test_guardrail.py
│   ├── test_auth.py
│   └── test_audit_disclaimer.py
├── .github/
│   └── workflows/
│       └── tests.yml     # CI — Python 3.12, pytest, 100 tests
├── data/
│   ├── raw/              # source PDFs (not versioned)
│   ├── bronze/           # parsed segments
│   ├── silver/           # final chunks
│   └── vector_store/     # FAISS index (not versioned)
├── logs/                 # audit logs (not versioned)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Evaluation

### Recall@5 — 100% (internal consistency)

> Dataset calibrated on the existing index. Measures internal consistency. Independent benchmark (lawyer annotation) planned.

### Faithfulness — 88.5% (LLM-as-judge, k=5, prompt v1.1)

| Run | Config | Score | Delta |
|---|---|---|---|
| Run 1 | Prompt v1.0 · k=5 | 87.4% | baseline |
| Run 2 | Prompt v1.1 (FORBIDDEN rule) · k=5 | **88.5%** | +1.1% |
| Run 3 | Prompt v1.1 · k=7 | 87.4% | -1.1% |

Structural ceiling ~88% — GDPR overrepresentation in Claude training data. k=7 test confirmed.

---

## Metrics Summary

| Metric | Value | Notes |
|---|---|---|
| Chunks | 2,016 | silver layer |
| Vector dimensions | 384 | all-MiniLM-L6-v2 |
| FAISS latency | < 5ms | exact search |
| Total /search latency | ~14s | query expansion + Claude |
| Recall@5 | 100% | internal dataset |
| Faithfulness | 88.5% | k=5, prompt v1.1 |
| Injection patterns | 18 regex | + 3 structural heuristics |
| Guardrail | Active | LOW/HIGH severity |
| Auth | JWT + API keys | rate limiting per plan |
| Audit log | Active | JSONL rotating, 50MB max |
| Unit tests | 100/100 | 4 files, CI GitHub Actions |

---

## Roadmap

- [x] Auth JWT + API keys + rate limiting
- [x] Hallucination guardrail
- [x] Faithfulness evaluation (88.5%)
- [x] Audit log JSONL
- [x] Legal disclaimer (FR/EN)
- [x] Unit tests (100/100) + CI/CD GitHub Actions
- [ ] Frontend (Next.js or Streamlit)
- [ ] Corpus update pipeline (incremental FAISS.add())
- [ ] Redis for distributed rate limiting
- [ ] PostgreSQL for user management
- [ ] Cloud deployment (Cloud Run)
- [ ] Independent benchmark (lawyer-annotated dataset)

---

## License

Private repository — all rights reserved.
