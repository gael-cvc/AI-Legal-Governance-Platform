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
| **Frontend Streamlit — Lex AI** | ✅ |
| Bilingual interface (FR / EN) | ✅ |
| Corpus update pipeline (incremental FAISS.add()) | 📋 Roadmap |

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
Structured JSON response → Streamlit Frontend
```

---

## Frontend — Lex AI

```bash
# Install
venv/bin/pip install streamlit

# Run (requires API on port 8000)
venv/bin/streamlit run frontend/app.py
# → http://localhost:8501
```

### Features
- **Logo Lex AI / Themis** — sidebar, transparent background, base64 encoded (no blur)
- **Switch FR/EN** — full i18n: labels, placeholders, section titles
- **Corpus pills** with hover tooltips (GDPR, EU AI Act, DGA, EDPB, CNIL)
- **[SOURCE X] badges** in green with hover tooltip (title, regulation, year, page)
- **Clickable references** — scroll + teal highlight to corresponding source card
- **Source cards** with FAISS/Rerank scores, 200-char preview, full text expander
- **Legal disclaimer** — clean, no markdown artifacts
- **Back to top button** — fixed bottom-right, uses `st.components.v1.html()` for iframe access
- **Theme** — white professional, Playfair Display + Source Sans 3, teal `#2a9d8f` primary

---

## Authentication

Every request to `/search` must be authenticated.

```bash
# Demo token (no credentials — dev only)
curl -X POST http://localhost:8000/auth/token/demo

# Token with credentials
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@cabinet.fr", "password": "demo1234"}'

# Use the token
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the obligations of a data controller under GDPR?", "k": 5}'
```

### Rate limits

| Plan | Limit | Window |
|---|---|---|
| `demo` | 10 req | 60s |
| `cabinet` | 30 req | 60s |
| `admin` | 200 req | 60s |

---

## Tests

```bash
venv/bin/pip install pytest pytest-cov
venv/bin/python -m pytest tests/ -v
```

```
collected 100 items — 100 passed in 2.14s
```

| File | Tests | Covers |
|---|---|---|
| `test_injection.py` | 24 | `detect_prompt_injection()` |
| `test_guardrail.py` | 22 | `check_hallucination_guardrail()` |
| `test_auth.py` | 22 | JWT, API keys, rate limiting |
| `test_audit_disclaimer.py` | 27 | Audit log JSONL, disclaimer FR/EN |

CI runs on every push to `main` via GitHub Actions.

---

## Quick Start

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform
cd AI-Legal-Governance-Platform
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# ANTHROPIC_API_KEY=sk-ant-...
# JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Build index (first run)
venv/bin/python -m rag.build_index

# Terminal 1 — API
TRANSFORMERS_OFFLINE=1 venv/bin/python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
venv/bin/streamlit run frontend/app.py
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
JWT_SECRET_KEY=your_generated_secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
DEMO_PASSWORD=demo1234
ADMIN_PASSWORD=your_secure_password
DISABLE_DEMO_TOKEN=false
DISABLE_DISCLAIMER=false
AUDIT_LOG_DIR=logs/audit
AUDIT_RETENTION_DAYS=180
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
│   ├── auth_router.py    # /auth/token, /auth/me
│   ├── audit_log.py      # JSONL audit logging
│   ├── search.py         # RAG pipeline, guardrails, disclaimer
│   └── models.py         # Pydantic schemas
├── frontend/
│   ├── app.py            # Streamlit UI — Lex AI
│   ├── lexai_notext.png  # Logo sidebar
│   ├── lexai_logo.png    # Logo complet
│   └── .streamlit/
│       └── config.toml   # Theme teal
├── rag/
│   ├── build_index.py
│   ├── vector_store.py
│   └── embedder.py
├── evaluation/
│   ├── evaluator.py
│   └── eval_dataset.py
├── tests/
│   ├── conftest.py
│   ├── test_injection.py
│   ├── test_guardrail.py
│   ├── test_auth.py
│   └── test_audit_disclaimer.py
├── .github/workflows/tests.yml
├── logs/                 # audit logs (gitignored)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Metrics

| Metric | Value | Notes |
|---|---|---|
| Chunks | 2,016 | silver layer |
| FAISS latency | < 5ms | exact search |
| Total /search latency | ~14s | query expansion + Claude |
| Recall@5 | 100% ⚠ | internal dataset |
| Faithfulness | 88.5% | k=5, prompt v1.1 |
| Injection patterns | 18 regex | + 3 structural heuristics |
| Guardrail | Active | LOW/HIGH severity |
| Auth | JWT + API keys | rate limiting per plan |
| Audit log | Active | JSONL rotating, 50MB max |
| Unit tests | 100/100 | CI GitHub Actions |
| Frontend | Streamlit | Lex AI, FR/EN, tooltips |

---

## Roadmap

- [x] Auth JWT + API keys + rate limiting
- [x] Hallucination guardrail
- [x] Faithfulness evaluation (88.5%)
- [x] Audit log JSONL
- [x] Legal disclaimer (FR/EN)
- [x] Unit tests (100/100) + CI/CD GitHub Actions
- [x] Frontend Streamlit — Lex AI
- [ ] Corpus update pipeline (incremental FAISS.add())
- [ ] Redis for distributed rate limiting
- [ ] PostgreSQL for user management
- [ ] Cloud deployment (Cloud Run)
- [ ] Independent benchmark (lawyer-annotated dataset)

---

## License

Private repository — all rights reserved.
