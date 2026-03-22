"""
auth.py — Authentification JWT + API Keys + Rate Limiting

CONTEXTE — POURQUOI CE MODULE MAINTENANT :
  Jusqu'ici, l'API est entièrement ouverte. N'importe qui qui connaît
  l'URL peut appeler /search. Chaque appel coûte ~$0.015 d'API Anthropic.
  Un bot qui spam peut générer une facture significative en quelques minutes.

  Ce module ajoute deux couches de protection avant que la requête
  n'atteigne le pipeline RAG :

  1. AUTHENTIFICATION — "Qui appelle l'API ?"
     Deux modes supportés :
       - JWT Bearer token  : standard industrie, stateless, signé cryptographiquement
       - API Key statique  : plus simple, adapté aux intégrations machine-à-machine

  2. RATE LIMITING — "Combien de fois par minute ?"
     Limite le nombre de requêtes par token/IP dans une fenêtre glissante.
     Protège contre les abus même d'un utilisateur authentifié.

ARCHITECTURE — CHOIX TECHNIQUES :

  JWT vs Sessions :
    Les sessions stockent l'état côté serveur (base de données ou Redis).
    Le JWT est stateless : toutes les informations sont dans le token lui-même,
    signé avec une clé secrète. Le serveur n'a rien à stocker — il vérifie
    juste la signature. C'est l'architecture idéale pour une API REST.

  HS256 vs RS256 :
    HS256 (HMAC-SHA256) utilise une clé secrète symétrique.
    RS256 utilise une paire clé privée/publique asymétrique.
    On choisit HS256 ici : plus simple, suffisant pour un déploiement
    single-service. RS256 serait nécessaire si plusieurs services
    indépendants devaient valider les tokens (microservices).

  Rate limiting en mémoire vs Redis :
    En mémoire (dict Python) : simple, zéro dépendance, exact.
    Mais les compteurs sont perdus au redémarrage, et ne sont
    pas partagés entre plusieurs instances du serveur.
    Redis : partagé entre instances, persistant.
    → On implémente d'abord en mémoire. La migration vers Redis
    consiste à remplacer _rate_limit_store par un client Redis
    avec les mêmes méthodes get/set/expire.

  Fenêtre glissante vs Fenêtre fixe :
    Fenêtre fixe : 30 req de 00:00 à 01:00, compteur remis à zéro à 01:00.
    Un attaquant peut envoyer 30 req à 00:59 et 30 req à 01:01 → 60 req
    en 2 secondes. Problème dit "burst at boundary".
    Fenêtre glissante : les 30 req sont comptées sur les 60 dernières
    secondes, peu importe quand. Plus robuste, légèrement plus coûteux
    en mémoire. On utilise une fenêtre glissante ici.

CHANGEMENT v1.5 — AUTH + RATE LIMITING :
  Nouveau fichier. Intégré dans main.py via FastAPI dependency injection.
  Impact sur les endpoints existants : ajout d'un paramètre
  `current_user: TokenData = Depends(get_current_user)` ou
  `_ = Depends(require_auth)` selon le niveau de protection souhaité.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

import logging

logger = logging.getLogger("api.auth")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Clé secrète pour signer les JWT.
# CRITIQUE : doit être longue (>= 32 chars), aléatoire, et SECRÈTE.
# En production : stocker dans les secrets du cloud (GCP Secret Manager,
# AWS Secrets Manager), jamais dans le code ou dans un fichier versionné.
# Générer : python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_use_secrets_token_hex_32")
JWT_ALGORITHM  = "HS256"

# Durée de vie des tokens.
# ACCESS_TOKEN : durée courte par design — si intercepté, il expire vite.
# REFRESH_TOKEN : durée longue, permet de renouveler l'access token sans
#                 re-authentification (non implémenté ici, prévu en v2).
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",  "60"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",    "30"))

# API Keys statiques — alternatives au JWT pour les intégrations M2M.
# Format : dict[api_key_hash → metadata]
# IMPORTANT : on stocke le HASH de la clé, jamais la clé en clair.
# En production : stocker dans une base de données avec hashlib.sha256.
# Ici : chargé depuis une variable d'environnement pour la simplicité.
# Format env : "hash1:client_name1,hash2:client_name2"
_API_KEYS_RAW = os.getenv("API_KEYS", "")

def _load_api_keys() -> dict[str, str]:
    """
    Charge les API keys depuis l'environnement.
    Retourne un dict {sha256_hash: client_name}.

    En production, remplacer par une requête base de données.
    """
    if not _API_KEYS_RAW:
        return {}
    result = {}
    for entry in _API_KEYS_RAW.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result

_API_KEYS: dict[str, str] = _load_api_keys()


# Rate limits par type de client.
# Clé : nom du plan. Valeur : (max_requests, window_seconds).
# Ces valeurs sont conservatrices — ajuster selon les besoins réels.
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "demo":    (10,  60),   # 10 req / minute  — démo publique
    "cabinet": (30,  60),   # 30 req / minute  — client cabinet
    "admin":   (200, 60),   # 200 req / minute — admin / tests
    "default": (20,  60),   # 20 req / minute  — fallback
}


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenData:
    """
    Payload extrait d'un JWT ou d'une API key validée.

    Propagé dans toute la requête via FastAPI dependency injection.
    Accessible dans les endpoints via `current_user: TokenData = Depends(...)`.
    """
    sub:      str              # "subject" — identifiant unique du client
    plan:     str = "default"  # plan tarifaire → détermine le rate limit
    client:   str = ""         # nom lisible (pour les logs)
    exp:      Optional[int] = None  # timestamp expiration (Unix)
    is_admin: bool = False     # accès aux endpoints admin

    @property
    def rate_limit(self) -> tuple[int, int]:
        """Retourne (max_requests, window_seconds) selon le plan."""
        return RATE_LIMITS.get(self.plan, RATE_LIMITS["default"])


@dataclass
class RateLimitState:
    """
    État du rate limiting pour un identifiant donné.
    Utilise une deque pour la fenêtre glissante.
    """
    # timestamps des requêtes dans la fenêtre courante
    timestamps: deque = field(default_factory=deque)


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING — FENÊTRE GLISSANTE
# ══════════════════════════════════════════════════════════════════════════════

# Store en mémoire — clé : identifiant unique (sub + endpoint ou IP)
# En production : remplacer par Redis pour le partage multi-instances.
_rate_limit_store: dict[str, RateLimitState] = defaultdict(RateLimitState)


def check_rate_limit(identifier: str, max_requests: int, window_seconds: int) -> None:
    """
    Vérifie si l'identifiant a dépassé sa limite de requêtes.

    ALGORITHME — FENÊTRE GLISSANTE :
    1. Récupère la liste des timestamps des requêtes passées
    2. Supprime les timestamps plus vieux que `window_seconds`
    3. Si le nombre de timestamps restants >= max_requests → HTTP 429
    4. Sinon → ajoute le timestamp courant et laisse passer

    COMPLEXITÉ : O(n) où n = nombre de requêtes dans la fenêtre.
    En pratique : max_requests est petit (10-200), donc O(1) amorti.

    PARAMÈTRES :
        identifier    : clé unique par client (sub + endpoint)
        max_requests  : nombre maximum de requêtes autorisées
        window_seconds: taille de la fenêtre en secondes

    LÈVE :
        HTTPException 429 si la limite est dépassée,
        avec un header Retry-After indiquant quand réessayer.
    """
    now   = time.time()
    state = _rate_limit_store[identifier]

    # Expire les timestamps hors de la fenêtre glissante
    cutoff = now - window_seconds
    while state.timestamps and state.timestamps[0] < cutoff:
        state.timestamps.popleft()

    if len(state.timestamps) >= max_requests:
        # Calcule le temps à attendre avant que la plus ancienne requête expire
        oldest     = state.timestamps[0]
        retry_after = int(oldest + window_seconds - now) + 1

        logger.warning(
            f"[RATE LIMIT] identifier={identifier} | "
            f"{len(state.timestamps)}/{max_requests} req dans {window_seconds}s | "
            f"retry_after={retry_after}s"
        )
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail      = (
                f"Trop de requêtes. Limite : {max_requests} req/{window_seconds}s. "
                f"Réessayez dans {retry_after} secondes."
            ),
            headers     = {"Retry-After": str(retry_after)},
        )

    # Requête autorisée — enregistrer le timestamp
    state.timestamps.append(now)


# ══════════════════════════════════════════════════════════════════════════════
# JWT — CRÉATION ET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def create_access_token(
    sub:      str,
    plan:     str  = "default",
    client:   str  = "",
    is_admin: bool = False,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Crée un JWT signé avec HS256.

    STRUCTURE D'UN JWT :
        header.payload.signature

    PAYLOAD (claims standards + custom) :
        sub  : identifiant du sujet (obligatoire, RFC 7519)
        exp  : expiration Unix timestamp (obligatoire par bonne pratique)
        iat  : issued at — quand le token a été émis
        plan : claim custom — plan tarifaire du client
        client : claim custom — nom lisible du client
        is_admin : claim custom — flag admin

    SIGNATURE :
        HMAC-SHA256(base64(header) + "." + base64(payload), JWT_SECRET_KEY)
        Toute modification du payload invalide la signature.

    PARAMÈTRES :
        sub        : identifiant unique du client (email, uuid, slug)
        plan       : "demo" | "cabinet" | "admin" | "default"
        client     : nom lisible pour les logs
        is_admin   : True pour accès endpoints admin
        expires_delta : surcharge la durée par défaut

    RETOURNE : token JWT signé en string
    """
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub":      sub,
        "exp":      expire,
        "iat":      datetime.now(tz=timezone.utc),
        "plan":     plan,
        "client":   client,
        "is_admin": is_admin,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """
    Décode et valide un JWT.

    VALIDATIONS EFFECTUÉES PAR python-jwt :
        - Signature : vérifie que le token n'a pas été altéré
        - Expiration : vérifie que exp > maintenant
        - Algorithme : vérifie que l'algo est bien HS256 (pas "none" attack)

    ATTAQUE "algorithm confusion" :
        Un attaquant peut envoyer un token avec "alg": "none" pour contourner
        la vérification de signature. En spécifiant algorithms=["HS256"],
        on rejette tout token avec un algorithme différent.

    LÈVE :
        HTTPException 401 si le token est invalide, expiré, ou malformé.
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms = [JWT_ALGORITHM],
            options    = {"verify_exp": True},
        )
        return TokenData(
            sub      = payload["sub"],
            plan     = payload.get("plan", "default"),
            client   = payload.get("client", ""),
            exp      = payload.get("exp"),
            is_admin = payload.get("is_admin", False),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Token expiré. Générez un nouveau token via /auth/token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"[AUTH] Token invalide : {e}")
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Token invalide.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )


# ══════════════════════════════════════════════════════════════════════════════
# API KEYS — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _hash_api_key(key: str) -> str:
    """
    Hash une API key avec SHA-256 pour comparaison sécurisée.

    On ne stocke jamais les clés en clair — uniquement leurs hashes.
    Si la base de données est compromise, les clés réelles sont protégées.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def validate_api_key(api_key: str) -> TokenData:
    """
    Valide une API key statique.

    COMPARAISON EN TEMPS CONSTANT :
    hmac.compare_digest() évite les attaques par timing.
    Une comparaison naïve (==) s'arrête au premier caractère différent,
    révélant des informations sur le hash via le timing de la réponse.
    hmac.compare_digest() prend toujours le même temps.

    LÈVE :
        HTTPException 401 si la clé est inconnue.
    """
    key_hash = _hash_api_key(api_key)

    for stored_hash, client_name in _API_KEYS.items():
        if hmac.compare_digest(key_hash, stored_hash):
            logger.info(f"[AUTH] API key valide — client={client_name}")
            return TokenData(
                sub    = f"apikey:{client_name}",
                plan   = "cabinet",   # les API keys ont le plan cabinet par défaut
                client = client_name,
            )

    logger.warning(f"[AUTH] API key invalide — hash={key_hash[:12]}...")
    raise HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "API key invalide.",
        headers     = {"WWW-Authenticate": "ApiKey"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI SECURITY SCHEMES
# ══════════════════════════════════════════════════════════════════════════════

# HTTPBearer : extrait le token depuis le header "Authorization: Bearer <token>"
_bearer_scheme  = HTTPBearer(auto_error=False)

# APIKeyHeader : extrait la clé depuis le header "X-API-Key: <key>"
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    bearer:  Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key: Optional[str]                          = Security(_api_key_scheme),
) -> TokenData:
    """
    FastAPI dependency — Authentifie la requête.

    Essaie d'abord le JWT Bearer, puis l'API key.
    Si aucun des deux n'est fourni → HTTP 401.

    USAGE DANS UN ENDPOINT :
        @router.post("/search")
        async def search(
            request: SearchRequest,
            current_user: TokenData = Depends(get_current_user),
        ):
            # current_user.sub, current_user.plan, etc. sont disponibles

    POURQUOI Depends() et pas un middleware global ?
    FastAPI Depends() permet de choisir endpoint par endpoint quels
    routes sont protégées. /health peut rester public pendant que
    /search et /search/suggestions exigent une auth.
    Un middleware global s'appliquerait à tous les endpoints sans exception.
    """
    # Tentative JWT
    if bearer and bearer.credentials:
        token_data = decode_access_token(bearer.credentials)
        logger.debug(f"[AUTH] JWT OK — sub={token_data.sub} plan={token_data.plan}")
        return token_data

    # Tentative API key
    if api_key:
        token_data = validate_api_key(api_key)
        return token_data

    # Aucun credential fourni
    raise HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = (
            "Authentification requise. "
            "Fournissez un JWT via 'Authorization: Bearer <token>' "
            "ou une API key via 'X-API-Key: <key>'."
        ),
        headers     = {"WWW-Authenticate": "Bearer"},
    )


async def require_admin(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """
    FastAPI dependency — Exige les droits admin.

    Chaîne Depends(get_current_user) → vérifie l'auth, puis le flag admin.

    USAGE :
        @router.get("/admin/stats")
        async def admin_stats(admin: TokenData = Depends(require_admin)):
            ...
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Accès refusé. Droits administrateur requis.",
        )
    return current_user


async def authenticated_and_rate_limited(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """
    FastAPI dependency — Authentifie ET applique le rate limiting.

    C'est la dependency principale à utiliser sur /search.

    PIPELINE :
        1. get_current_user() → authentifie (JWT ou API key)
        2. Détermine max_requests et window_seconds depuis current_user.plan
        3. check_rate_limit() → vérifie la limite
        4. Si OK → retourne current_user pour usage dans l'endpoint

    L'identifiant du rate limit combine le sub et le nom de l'endpoint
    pour permettre des limites différentes par endpoint (ex: /search
    plus restrictif que /search/suggestions).

    NOTE : l'endpoint est fixé à "search" ici pour simplifier.
    Pour des limites par endpoint, passer l'endpoint en paramètre
    ou utiliser Request.url.path.
    """
    max_req, window = current_user.rate_limit
    identifier      = f"{current_user.sub}:search"

    check_rate_limit(
        identifier    = identifier,
        max_requests  = max_req,
        window_seconds= window,
    )

    return current_user


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT /auth/token — GÉNÉRATION DE TOKEN
# ══════════════════════════════════════════════════════════════════════════════

# IMPORTANT : en production, cet endpoint doit être protégé par :
# - Un rate limit strict sur /auth/token (5 req/minute par IP)
#   pour prévenir le brute-force
# - Une vraie base de données d'utilisateurs avec mots de passe hashés
#   (bcrypt, argon2)
# - HTTPS obligatoire (sinon les credentials transitent en clair)
#
# ICI : on simule des utilisateurs hardcodés pour le développement.
# Remplacer _DEMO_USERS par une vraie auth (PostgreSQL + bcrypt).

_DEMO_USERS: dict[str, dict] = {
    # Format : email → {password_hash (bcrypt en prod), plan, client}
    # AVERTISSEMENT : mots de passe en clair uniquement pour le dev local
    # En production : utiliser bcrypt.checkpw(password, stored_hash)
    "demo@cabinet.fr": {
        "password": os.getenv("DEMO_PASSWORD", "demo1234"),
        "plan":     "demo",
        "client":   "Cabinet Demo",
        "is_admin": False,
    },
    "admin@cabinet.fr": {
        "password": os.getenv("ADMIN_PASSWORD", "admin_change_me"),
        "plan":     "admin",
        "client":   "Admin",
        "is_admin": True,
    },
}

# Rate limit strict sur /auth/token — séparé du rate limit sur /search
_auth_rate_limit_store: dict[str, RateLimitState] = defaultdict(RateLimitState)


def get_token_for_credentials(email: str, password: str) -> str:
    """
    Valide les credentials et retourne un JWT.

    EN PRODUCTION, remplacer par :
        1. Requête PostgreSQL pour récupérer le hash du mot de passe
        2. bcrypt.checkpw(password.encode(), stored_hash)
        3. Si OK → create_access_token(...)

    LÈVE :
        HTTPException 401 si les credentials sont invalides.
        HTTPException 429 si trop de tentatives (brute-force protection).
    """
    # Rate limit strict sur les tentatives d'auth : 5 req/minute par email
    # Protège contre le brute-force des mots de passe.
    check_rate_limit(
        identifier     = f"auth:{email}",
        max_requests   = 5,
        window_seconds = 60,
    )

    user = _DEMO_USERS.get(email)

    # TIMING ATTACK PROTECTION :
    # On utilise hmac.compare_digest même pour les utilisateurs inexistants.
    # Une comparaison courte-circuit révélerait si l'email existe via le timing.
    stored_password = user["password"] if user else "dummy_password_to_prevent_timing"
    if not user or not hmac.compare_digest(password, stored_password):
        logger.warning(f"[AUTH] Échec authentification — email={email}")
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Email ou mot de passe incorrect.",
        )

    token = create_access_token(
        sub      = email,
        plan     = user["plan"],
        client   = user["client"],
        is_admin = user["is_admin"],
    )

    logger.info(f"[AUTH] Token émis — client={user['client']} plan={user['plan']}")
    return token
