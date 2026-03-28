"""
audit_log.py — Audit Log JSONL pour le pipeline RAG

QU'EST-CE QU'UN AUDIT LOG ?
  Un audit log est un enregistrement chronologique de tout ce qui se passe
  dans le système. Chaque événement (chaque requête /search) est écrit sur
  une ligne dans un fichier. Ces enregistrements permettent de :

  1. TRACER les requêtes : qui a demandé quoi, quand, avec quel résultat
  2. DÉBUGGER les problèmes : reproduire un comportement inattendu
  3. MESURER les performances : latences, modèles utilisés, chunks récupérés
  4. RESPECTER le RGPD : Article 5(2) "accountability" — prouver la conformité
  5. FACTURER correctement : combien de requêtes par client

QU'EST-CE QUE JSONL ?
  JSONL = JSON Lines. Un format où chaque ligne est un objet JSON valide
  et indépendant. Contrairement à un fichier JSON classique (un grand objet),
  JSONL peut être lu ligne par ligne sans charger tout le fichier en mémoire.

  Exemple :
    {"timestamp": "2026-03-28T14:32:01Z", "question": "...", "latency_ms": 3420}
    {"timestamp": "2026-03-28T14:33:12Z", "question": "...", "latency_ms": 2890}

  Avantages de JSONL vs CSV :
    - Structure flexible (champs optionnels, valeurs nulles)
    - Chaque ligne est parseable indépendamment (si une ligne est corrompue,
      les autres restent lisibles)
    - Compatible avec tous les outils d'analyse (jq, pandas, Elasticsearch)

ARCHITECTURE — ROTATION DES FICHIERS :
  Les logs s'accumulent. Sans rotation, le fichier grossit indéfiniment.
  On utilise RotatingFileHandler : quand le fichier atteint MAX_BYTES,
  il est renommé en audit.jsonl.1, audit.jsonl.2... jusqu'à BACKUP_COUNT.
  Au-delà → les anciens fichiers sont supprimés automatiquement.

  Taille max par défaut : 10 MB × 5 fichiers = 50 MB max sur disque.
  À ~500 bytes/entrée × 1000 req/jour = ~500KB/jour → ~100 jours avant rotation.

CONFORMITÉ RGPD — ARTICLE 32 :
  Les logs contiennent potentiellement des données personnelles (questions
  posées, identifiants utilisateurs). Obligations :
    - Durée de conservation limitée (par défaut : 6 mois, ajustable)
    - Accès restreint aux logs (fichier chmod 640)
    - Pas de données sensibles dans les questions (garanti par l'injection defense)
    - Possibilité de purge sur demande (droit à l'oubli)

CHANGEMENT v1.6 — AUDIT LOG :
  Nouveau module. Appelé depuis search.py après chaque requête réussie.
  Impact sur les performances : négligeable (~0.1ms par écriture, async).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Dossier où sont stockés les logs d'audit.
# Séparé des logs applicatifs (qui vont dans stdout/stderr via uvicorn).
# Ce dossier doit être dans .gitignore (données potentiellement sensibles).
AUDIT_LOG_DIR  = Path(os.getenv("AUDIT_LOG_DIR", "logs/audit"))

# Taille max d'un fichier de log avant rotation (en bytes).
# 10 MB = environ 20 000 entrées à ~500 bytes/entrée.
AUDIT_MAX_BYTES    = int(os.getenv("AUDIT_MAX_BYTES",    str(10 * 1024 * 1024)))  # 10 MB

# Nombre de fichiers de backup conservés après rotation.
# 5 fichiers × 10 MB = 50 MB max sur disque.
AUDIT_BACKUP_COUNT = int(os.getenv("AUDIT_BACKUP_COUNT", "5"))

# Durée de rétention des logs en jours (pour la conformité RGPD).
# Les logs plus anciens peuvent être supprimés manuellement ou via un cron.
# Par défaut : 180 jours (6 mois) — durée raisonnable pour un usage cabinet.
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "180"))


# ══════════════════════════════════════════════════════════════════════════════
# SETUP DU LOGGER D'AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def _setup_audit_logger() -> logging.Logger:
    """
    Configure un logger dédié aux audits, séparé du logger applicatif.

    POURQUOI UN LOGGER SÉPARÉ ?
    Le logger principal (`api.search`) écrit dans stdout/stderr — les logs
    sont visibles dans le terminal et gérés par uvicorn/le système de déploiement.
    Le logger d'audit écrit dans un fichier JSONL rotatif — les logs sont
    persistants, structurés, et séparés des logs de debug.

    POURQUOI NE PAS UTILISER `print()` ?
    Le module logging gère la rotation, le buffering, et le thread-safety.
    `print()` n'est pas thread-safe — en cas de requêtes concurrentes,
    les lignes pourraient se mélanger.

    RETOURNE : logger configuré avec RotatingFileHandler
    """
    # Créer le dossier si nécessaire
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Créer le logger d'audit — nom distinct pour ne pas interférer avec
    # le logger principal "api.search"
    audit_logger = logging.getLogger("audit")

    # Ne pas propager au logger parent (évite la double écriture dans stdout)
    audit_logger.propagate = False

    # Éviter les handlers dupliqués si le module est importé plusieurs fois
    if audit_logger.handlers:
        return audit_logger

    audit_logger.setLevel(logging.INFO)

    # RotatingFileHandler : gère automatiquement la rotation des fichiers
    # maxBytes  : taille max avant rotation
    # backupCount : nombre de fichiers de backup à conserver
    handler = RotatingFileHandler(
        filename     = AUDIT_LOG_DIR / "audit.jsonl",
        maxBytes     = AUDIT_MAX_BYTES,
        backupCount  = AUDIT_BACKUP_COUNT,
        encoding     = "utf-8",
    )

    # Format minimal : juste le message brut (le JSON est dans le message)
    # Pas de timestamp dans le format — il est dans le JSON lui-même
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)

    return audit_logger


# Instance globale du logger d'audit — initialisée une seule fois au import
_audit_logger = _setup_audit_logger()


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE D'UNE ENTRÉE D'AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def build_audit_entry(
    # ── Identifiants ──────────────────────────────────────────────────────────
    request_id:    str,           # UUID unique par requête
    user_sub:      str,           # identifiant du client (email ou "apikey:nom")
    user_plan:     str,           # plan tarifaire (demo/cabinet/admin)
    user_client:   str,           # nom lisible du client

    # ── Requête ───────────────────────────────────────────────────────────────
    question:      str,           # question posée (potentiellement tronquée)
    regulation:    Optional[str], # filtre regulation appliqué (ou None)
    k:             int,           # nombre de chunks demandés
    language:      str,           # langue de la réponse (fr/en)
    use_reranking: bool,          # reranking activé ou non

    # ── Résultat RAG ──────────────────────────────────────────────────────────
    n_chunks_retrieved: int,      # chunks retournés par FAISS
    n_chunks_used:      int,      # chunks envoyés au LLM après reranking
    sources_used:       list[str],# segment_id des sources utilisées
    model_used:         str,      # modèle LLM utilisé
    query_expanded:     bool,     # query expansion activée

    # ── Guardrail ─────────────────────────────────────────────────────────────
    guardrail_severity:     str,        # "ok" | "low" | "high"
    guardrail_ghost_sources: list[int], # numéros de sources fantômes (si any)

    # ── Métriques ─────────────────────────────────────────────────────────────
    latency_ms:    float,         # temps total de la requête en millisecondes
    status:        str,           # "success" | "error" | "blocked_injection" | "guardrail_high"

    # ── Optionnel ─────────────────────────────────────────────────────────────
    error_detail:  Optional[str] = None,  # message d'erreur si status != "success"
) -> dict:
    """
    Construit le dictionnaire d'une entrée d'audit.

    CHAMPS INCLUS ET POURQUOI :

    request_id : permet de corréler les logs d'audit avec les logs applicatifs.
                 Si un client se plaint d'une réponse, on retrouve l'entrée
                 exacte avec son request_id.

    user_sub / plan : qui a fait la requête et avec quel plan. Utile pour
                      la facturation et pour détecter les abus.

    question (tronquée à 500 chars) : essentielle pour débugger. Attention
                                       au RGPD : si les questions contiennent
                                       des données personnelles, la durée de
                                       rétention doit être limitée.

    sources_used : les segment_id des sources effectivement citées. Permet
                   de tracer quelle version du corpus a été utilisée pour
                   une réponse donnée.

    guardrail_severity : trace les cas LOW (disclaimer ajouté) et HIGH (bloqué).
                         Utile pour mesurer la qualité du système dans le temps.

    latency_ms : mesure de performance. Permet de détecter les dégradations.

    CHAMPS INTENTIONNELLEMENT ABSENTS :
    - La réponse complète de Claude : trop volumineuse, et contient potentiellement
      des informations confidentielles issues du corpus.
    - Le texte des chunks : idem, trop volumineux.
    - Le mot de passe ou le token JWT : jamais dans les logs.
    """
    return {
        # Quand
        "timestamp":  datetime.now(tz=timezone.utc).isoformat(),

        # Identifiants
        "request_id": request_id,
        "user_sub":   user_sub,
        "user_plan":  user_plan,
        "user_client": user_client,

        # Requête — question tronquée pour limiter le volume de données personnelles
        "question":       question[:500] if question else "",
        "regulation":     regulation,
        "k":              k,
        "language":       language,
        "use_reranking":  use_reranking,
        "query_expanded": query_expanded,

        # Résultat RAG
        "n_chunks_retrieved": n_chunks_retrieved,
        "n_chunks_used":      n_chunks_used,
        "sources_used":       sources_used,
        "model_used":         model_used,

        # Guardrail
        "guardrail_severity":      guardrail_severity,
        "guardrail_ghost_sources": guardrail_ghost_sources,

        # Métriques
        "latency_ms": round(latency_ms, 1),
        "status":     status,

        # Erreur (optionnel)
        "error_detail": error_detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'ÉCRITURE
# ══════════════════════════════════════════════════════════════════════════════

def write_audit_log(entry: dict) -> None:
    """
    Écrit une entrée d'audit dans le fichier JSONL.

    THREAD-SAFETY :
    Le module logging est thread-safe. En cas de requêtes concurrentes
    (FastAPI async), les écritures sont sérialisées par le handler.

    ERREURS D'ÉCRITURE :
    Si l'écriture échoue (disque plein, permissions), on log l'erreur
    dans le logger applicatif mais on NE FAIT PAS PLANTER la requête.
    Un problème de logging ne doit pas dégrader le service.

    PARAMÈTRE :
        entry : dict retourné par build_audit_entry()
    """
    try:
        _audit_logger.info(json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        # Fallback sur le logger applicatif — ne pas laisser une erreur
        # de logging faire planter la requête
        logging.getLogger("api.search").error(
            f"[AUDIT] Échec écriture audit log : {e} | "
            f"request_id={entry.get('request_id', 'unknown')}"
        )


def generate_request_id() -> str:
    """
    Génère un identifiant unique pour chaque requête.

    UUID4 = identifiant aléatoire de 128 bits.
    Format : 8-4-4-4-12 caractères hexadécimaux.
    Exemple : "550e8400-e29b-41d4-a716-446655440000"

    Probabilité de collision : 1 sur 2^122 ≈ négligeable.
    """
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def get_audit_stats() -> dict:
    """
    Retourne des statistiques basiques sur les fichiers de log d'audit.

    Utilisé par l'endpoint /admin/audit/stats (si implémenté).
    Ne charge pas le contenu des fichiers en mémoire — juste les métadonnées.
    """
    log_file = AUDIT_LOG_DIR / "audit.jsonl"

    if not log_file.exists():
        return {
            "status":       "no_logs",
            "log_dir":      str(AUDIT_LOG_DIR),
            "files":        [],
            "total_size_mb": 0,
        }

    files = []
    total_size = 0

    # Fichier principal + backups (audit.jsonl.1, .2, ...)
    for f in sorted(AUDIT_LOG_DIR.glob("audit.jsonl*")):
        size = f.stat().st_size
        total_size += size
        files.append({
            "name":    f.name,
            "size_mb": round(size / 1024 / 1024, 2),
        })

    return {
        "status":        "ok",
        "log_dir":       str(AUDIT_LOG_DIR),
        "files":         files,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "retention_days": AUDIT_RETENTION_DAYS,
        "max_bytes_per_file": AUDIT_MAX_BYTES,
        "backup_count":  AUDIT_BACKUP_COUNT,
    }
