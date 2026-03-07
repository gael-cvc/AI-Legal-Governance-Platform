# AI Legal Governance Intelligence Platform

> **[Français ci-dessous](#plateforme-dintelligence-juridique-ai)**

A production-ready RAG (Retrieval-Augmented Generation) system applied to European law. Answers precise legal questions by citing exact sources from a corpus of EU regulatory texts (GDPR, AI Act, DGA, CNIL, EDPB).

## Business Problem

Organizations deploying AI systems in Europe must comply with:
- GDPR requirements (Regulation EU 2016/679)
- EU AI Act high-risk obligations (Regulation EU 2024/1689)
- Data Governance Act (Regulation EU 2022/868)
- National authority guidelines (CNIL, EDPB)
- Automated decision-making restrictions

Legal documents are long (100–300+ pages), complex, frequently updated, and high-stakes in interpretation. Compliance teams need fast access to relevant articles, reliable citation, and auditability of AI responses.

This platform addresses that need by combining semantic search, cross-encoder reranking, and strict citation enforcement — making it suitable for high-risk regulatory environments.

## Stack

- **LLM** — Claude Sonnet 4 (Anthropic)
- **Embeddings** — all-MiniLM-L6-v2 (sentence-transformers)
- **Vector store** — FAISS IndexFlatIP
- **Reranker** — cross-encoder/ms-marco-MiniLM-L-6-v2
- **API** — FastAPI
- **Runtime** — Python 3.12, Apple Silicon M4 (MPS)

## System Architecture

**Data Layer**
Official regulatory PDFs → metadata extraction → chunking (raw → bronze → silver)

**Retrieval Layer**
Query expansion → sentence-transformer embeddings → FAISS vector search → cross-encoder reranking

**Generation Layer**
Strict prompt template — citation required, no speculation, refusal when context insufficient

## Installation

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform.git
cd AI-Legal-Governance-Platform

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file at the root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Build the pipeline

```bash
# 1. Ingestion — PDF → chunks (data/silver/)
python -m ingestion.pipeline

# 2. Indexing — chunks → FAISS (data/vector_store/)
python -m rag.build_index
```

## Run the API

```bash
TRANSFORMERS_OFFLINE=1 uvicorn api.main:app --reload --port 8000
```

Swagger UI available at `http://localhost:8000/docs`

## Usage

**Health check**
```bash
curl http://localhost:8000/health
```

**Search without reranking**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the obligations of a data controller under GDPR?", "k": 3, "use_reranking": false}'
```

**Search with reranking (recommended)**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the obligations of a data controller under GDPR?", "k": 3, "use_reranking": true}'
```

## Features

| Feature | Status |
|---------|--------|
| PDF ingestion pipeline (raw → bronze → silver) | ✅ Done |
| Sentence-transformer embeddings (MPS) | ✅ Done |
| FAISS vector search with deduplication | ✅ Done |
| Query expansion | ✅ Done |
| Cross-encoder reranking | ✅ Done |
| FastAPI with singleton pattern | ✅ Done |
| Strict citation enforcement | ✅ Done |
| Evaluation metrics (recall@k, faithfulness) | 🔄 In progress |
| Metadata filtering by regulation / article | 🔄 In progress |
| Prompt injection defense | 🔄 In progress |
| Docker + docker-compose | 🔄 In progress |

## Project structure

```
├── ingestion/        # Pipeline PDF → bronze → silver
│   ├── pdf_parser.py
│   ├── article_extractor.py
│   ├── chunker.py
│   └── pipeline.py
├── rag/              # Embedding, FAISS, reranker
│   ├── embedder.py
│   ├── vector_store.py
│   ├── reranker.py
│   └── build_index.py
├── api/              # FastAPI
│   ├── main.py
│   ├── search.py
│   ├── models.py
│   └── health.py
├── data/
│   ├── raw/          # Source PDFs (not versioned)
│   ├── bronze/       # Segments + metadata
│   ├── silver/       # Chunks JSONL
│   └── vector_store/ # FAISS index
└── docs/             # Full technical documentation
```

## Corpus

| Document | Type |
|----------|------|
| GDPR (EN + FR) | Regulation EU 2016/679 |
| RGPD Recitals | Recitals 1-173 |
| AI Act + Annexes | Regulation EU 2024/1689 |
| Data Governance Act | Regulation EU 2022/868 |
| EDPB Guidelines | Automated decision-making |
| CNIL Recommendations | AI compliance |

## Documentation

Full technical documentation available in [`docs/`](docs/project_doc.html) — architecture, technical decisions, results and logs.

---

# Plateforme d'Intelligence Juridique AI

Système RAG (Retrieval-Augmented Generation) appliqué au droit européen. Répond à des questions juridiques précises en citant les sources exactes depuis un corpus de textes réglementaires EU (RGPD, AI Act, DGA, CNIL, EDPB).

## Problème métier

Les organisations déployant des systèmes d'IA en Europe doivent se conformer à :
- Les exigences du RGPD (Règlement EU 2016/679)
- Les obligations de l'AI Act pour les systèmes à haut risque (Règlement EU 2024/1689)
- Le Data Governance Act (Règlement EU 2022/868)
- Les recommandations des autorités nationales (CNIL, EDPB)
- Les restrictions sur la prise de décision automatisée

Les textes réglementaires sont longs (100–300+ pages), complexes, fréquemment mis à jour, et à forts enjeux d'interprétation. Les équipes conformité ont besoin d'un accès rapide aux articles pertinents, d'une citation fiable, et d'une traçabilité des réponses IA.

Cette plateforme répond à ce besoin en combinant recherche sémantique, reranking cross-encoder, et enforcement strict des citations — adaptée aux environnements réglementaires à haut risque.

## Stack

- **LLM** — Claude Sonnet 4 (Anthropic)
- **Embeddings** — all-MiniLM-L6-v2 (sentence-transformers)
- **Vector store** — FAISS IndexFlatIP
- **Reranker** — cross-encoder/ms-marco-MiniLM-L-6-v2
- **API** — FastAPI
- **Runtime** — Python 3.12, Apple Silicon M4 (MPS)

## Architecture système

**Couche données**
PDFs réglementaires officiels → extraction metadata → chunking (raw → bronze → silver)

**Couche retrieval**
Query expansion → embeddings → recherche FAISS → reranking cross-encoder

**Couche génération**
Prompt strict — citation obligatoire, pas de spéculation, refus si contexte insuffisant

## Installation

```bash
git clone https://github.com/gael-cvc/AI-Legal-Governance-Platform.git
cd AI-Legal-Governance-Platform

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Créer un fichier `.env` à la racine :

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Construire le pipeline

```bash
# 1. Ingestion — PDF → chunks (data/silver/)
python -m ingestion.pipeline

# 2. Indexation — chunks → FAISS (data/vector_store/)
python -m rag.build_index
```

## Lancer l'API

```bash
TRANSFORMERS_OFFLINE=1 uvicorn api.main:app --reload --port 8000
```

Swagger UI disponible sur `http://localhost:8000/docs`

## Utilisation

**Health check**
```bash
curl http://localhost:8000/health
```

**Recherche sans reranking**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelles sont les obligations du responsable de traitement sous le RGPD ?", "k": 3, "use_reranking": false}'
```

**Recherche avec reranking (recommandé)**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelles sont les obligations du responsable de traitement sous le RGPD ?", "k": 3, "use_reranking": true}'
```

## Fonctionnalités

| Fonctionnalité | Statut |
|----------------|--------|
| Pipeline d'ingestion PDF (raw → bronze → silver) | ✅ Fait |
| Embeddings sentence-transformer (MPS) | ✅ Fait |
| Recherche FAISS avec déduplication | ✅ Fait |
| Query expansion | ✅ Fait |
| Reranking cross-encoder | ✅ Fait |
| FastAPI avec pattern singleton | ✅ Fait |
| Enforcement strict des citations | ✅ Fait |
| Métriques d'évaluation (recall@k, faithfulness) | 🔄 En cours |
| Filtrage par métadonnées (règlement, article) | 🔄 En cours |
| Défense contre les injections de prompt | 🔄 En cours |
| Docker + docker-compose | 🔄 En cours |

## Structure du projet

```
├── ingestion/        # Pipeline PDF → bronze → silver
│   ├── pdf_parser.py
│   ├── article_extractor.py
│   ├── chunker.py
│   └── pipeline.py
├── rag/              # Embedding, FAISS, reranker
│   ├── embedder.py
│   ├── vector_store.py
│   ├── reranker.py
│   └── build_index.py
├── api/              # FastAPI
│   ├── main.py
│   ├── search.py
│   ├── models.py
│   └── health.py
├── data/
│   ├── raw/          # PDFs sources (non versionnés)
│   ├── bronze/       # Segments + metadata
│   ├── silver/       # Chunks JSONL
│   └── vector_store/ # Index FAISS
└── docs/             # Documentation technique complète
```

## Corpus

| Document | Type |
|----------|------|
| GDPR (EN + FR) | Règlement EU 2016/679 |
| RGPD Recitals | Considérants 1-173 |
| AI Act + Annexes | Règlement EU 2024/1689 |
| Data Governance Act | Règlement EU 2022/868 |
| EDPB Guidelines | Décision automatisée |
| CNIL Recommendations | Conformité IA |

## Documentation

Documentation technique complète disponible dans [`docs/`](docs/project_doc.html) — architecture, décisions techniques, résultats et logs.
