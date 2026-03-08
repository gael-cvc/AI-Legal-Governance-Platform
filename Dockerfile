# ── Image de base ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Métadonnées ────────────────────────────────────────────────────────────────
LABEL maintainer="gael-cvc"
LABEL description="AI Legal Governance Intelligence Platform — FastAPI RAG API"

# ── Variables d'environnement runtime ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV DEVICE=cpu

# ── Répertoire de travail ──────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances système ────────────────────────────────────────────────────────
# build-essential nécessaire pour compiler certaines libs C (faiss, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ─────────────────────────────────────────────────────────
# IMPORTANT : copier requirements.txt EN PREMIER pour exploiter le cache Docker.
# Si seul le code change, cette couche reste en cache → rebuild en ~10s au lieu de 5min.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Code source ────────────────────────────────────────────────────────────────
COPY api/        api/
COPY rag/        rag/
COPY evaluation/ evaluation/

# ── Port exposé ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck interne ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Commande de démarrage ──────────────────────────────────────────────────────
# --host 0.0.0.0 : écoute sur toutes les interfaces (obligatoire dans Docker)
# pas de --reload : inutile et dangereux en prod
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
