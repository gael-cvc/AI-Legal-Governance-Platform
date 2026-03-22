"""
auth_router.py — Endpoints d'authentification

ENDPOINTS :
    POST /auth/token       — Échange email/password contre un JWT
    GET  /auth/me          — Infos sur le token courant (debug)
    POST /auth/token/test  — Génère un token de démo sans credentials (dev only)

CES ENDPOINTS SONT VOLONTAIREMENT SIMPLES.
En production, ajouter :
    - POST /auth/refresh   — Renouvelle un access token via refresh token
    - POST /auth/logout    — Révoque un token (nécessite une blacklist Redis)
    - POST /auth/register  — Inscription (si SaaS multi-tenant)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from .auth import (
    TokenData,
    authenticated_and_rate_limited,
    create_access_token,
    get_current_user,
    get_token_for_credentials,
)

import os
import logging

logger = logging.getLogger("api.auth_router")
router = APIRouter(prefix="/auth", tags=["Auth"])


# ── MODÈLES PYDANTIC ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """
    Corps de la requête POST /auth/token.

    SÉCURITÉ : doit impérativement transiter sur HTTPS.
    En HTTP, le password est visible dans les logs réseau.
    """
    email:    str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email":    "demo@cabinet.fr",
                "password": "demo1234",
            }
        }


class TokenResponse(BaseModel):
    """
    Réponse de POST /auth/token.

    access_token : JWT signé, à inclure dans les requêtes suivantes
                   via le header "Authorization: Bearer <access_token>"
    token_type   : toujours "bearer" (RFC 6750)
    expires_in   : durée de vie en secondes (pour que le client sache
                   quand renouveler sans parser le JWT)
    """
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int  # secondes


class MeResponse(BaseModel):
    """Infos sur le token courant."""
    sub:      str
    plan:     str
    client:   str
    is_admin: bool
    rate_limit_max_requests: int
    rate_limit_window_seconds: int


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@router.post(
    "/token",
    response_model = TokenResponse,
    summary        = "Obtenir un JWT",
    description    = (
        "Échange des credentials (email + password) contre un JWT. "
        "Le token doit ensuite être envoyé dans le header "
        "`Authorization: Bearer <token>` sur tous les endpoints protégés. "
        "\n\nComptes de démo :\n"
        "- `demo@cabinet.fr` / `demo1234` → plan demo (10 req/min)\n"
        "- `admin@cabinet.fr` / voir .env → plan admin (200 req/min)"
    ),
)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Génère un JWT après validation des credentials.

    Rate limit strict : 5 tentatives/minute par email.
    Protège contre le brute-force.
    """
    from .auth import ACCESS_TOKEN_EXPIRE_MINUTES

    token = get_token_for_credentials(request.email, request.password)

    return TokenResponse(
        access_token = token,
        token_type   = "bearer",
        expires_in   = ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get(
    "/me",
    response_model = MeResponse,
    summary        = "Infos sur le token courant",
    description    = "Retourne les informations décodées du JWT ou de l'API key courante.",
)
async def get_me(
    current_user: TokenData = Depends(get_current_user),
) -> MeResponse:
    """
    Endpoint de debug — vérifie que le token est valide et affiche son contenu.

    Utile pour :
    - Vérifier qu'un token fonctionne correctement
    - Connaître son plan et ses limites
    - Débugger les problèmes d'auth côté client
    """
    max_req, window = current_user.rate_limit
    return MeResponse(
        sub                       = current_user.sub,
        plan                      = current_user.plan,
        client                    = current_user.client,
        is_admin                  = current_user.is_admin,
        rate_limit_max_requests   = max_req,
        rate_limit_window_seconds = window,
    )


@router.post(
    "/token/demo",
    response_model = TokenResponse,
    summary        = "Token de démo (dev only)",
    description    = (
        "Génère un token de démo sans credentials. "
        "**Désactiver en production** via la variable d'environnement "
        "`DISABLE_DEMO_TOKEN=true`."
    ),
)
async def demo_token() -> TokenResponse:
    """
    Génère un token demo sans authentification.

    UNIQUEMENT POUR LE DÉVELOPPEMENT ET LES DÉMONSTRATIONS.
    Désactiver en production en settant DISABLE_DEMO_TOKEN=true.

    Utilisation typique : tester l'API rapidement sans configurer
    des credentials dans le client.
    """
    if os.getenv("DISABLE_DEMO_TOKEN", "false").lower() == "true":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "L'endpoint de démo est désactivé en production.",
        )

    from .auth import ACCESS_TOKEN_EXPIRE_MINUTES

    token = create_access_token(
        sub    = "demo:anonymous",
        plan   = "demo",
        client = "Anonymous Demo",
    )

    logger.info("[AUTH] Token demo généré (sans credentials)")

    return TokenResponse(
        access_token = token,
        token_type   = "bearer",
        expires_in   = ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
