"""
tests/test_auth.py — Tests unitaires : JWT, API Keys, Rate Limiting

PARTICULARITÉS :
  - JWT : on crée et valide de vrais tokens — pas de mock nécessaire.
  - Rate limiting : on teste avec une fenêtre très courte (1s) pour
    ne pas ralentir la suite de tests.
  - Les tests de rate limiting sont sensibles au temps CPU — on ajoute
    des marges généreuses.

COMMENT LANCER :
  venv/bin/python -m pytest tests/test_auth.py -v
"""

import sys
import os
import time
import hashlib
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch la variable d'env AVANT l'import pour avoir une clé connue
os.environ["JWT_SECRET_KEY"] = "test_secret_key_for_unit_tests_only_not_production"

from api.auth import (
    TokenData,
    check_rate_limit,
    create_access_token,
    decode_access_token,
    validate_api_key,
    _hash_api_key,
    _rate_limit_store,
    RATE_LIMITS,
)
from fastapi import HTTPException


# ══════════════════════════════════════════════════════════════════════════════
# JWT — CRÉATION ET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestJWT:

    def test_create_and_decode_basic(self):
        """Créer un token et le décoder — round-trip basique."""
        token = create_access_token(sub="test@cabinet.fr", plan="cabinet", client="Test")
        data  = decode_access_token(token)

        assert data.sub    == "test@cabinet.fr"
        assert data.plan   == "cabinet"
        assert data.client == "Test"
        assert data.is_admin is False

    def test_admin_flag(self):
        """Le flag is_admin doit être préservé dans le token."""
        token = create_access_token(sub="admin@test.fr", is_admin=True)
        data  = decode_access_token(token)
        assert data.is_admin is True

    def test_default_plan(self):
        """Plan par défaut si non spécifié."""
        token = create_access_token(sub="user@test.fr")
        data  = decode_access_token(token)
        assert data.plan == "default"

    def test_expired_token_raises_401(self):
        """Token expiré → HTTPException 401."""
        token = create_access_token(
            sub="user@test.fr",
            expires_delta=timedelta(seconds=-1)  # déjà expiré
        )
        try:
            decode_access_token(token)
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
            assert "expiré" in e.detail.lower() or "expired" in e.detail.lower()

    def test_tampered_token_raises_401(self):
        """Token modifié → signature invalide → HTTPException 401."""
        token = create_access_token(sub="user@test.fr")
        # Modifier un caractère dans le payload (partie du milieu)
        parts = token.split(".")
        # Ajouter un caractère à la fin du payload
        parts[1] = parts[1] + "X"
        tampered = ".".join(parts)

        try:
            decode_access_token(tampered)
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401

    def test_invalid_token_raises_401(self):
        """Chaîne aléatoire → HTTPException 401."""
        try:
            decode_access_token("not_a_jwt_at_all")
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401

    def test_empty_token_raises_401(self):
        """Token vide → HTTPException 401."""
        try:
            decode_access_token("")
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401

    def test_rate_limit_from_plan(self):
        """TokenData.rate_limit retourne le bon tuple selon le plan."""
        token_demo    = create_access_token(sub="u", plan="demo")
        token_cabinet = create_access_token(sub="u", plan="cabinet")
        token_admin   = create_access_token(sub="u", plan="admin")

        data_demo    = decode_access_token(token_demo)
        data_cabinet = decode_access_token(token_cabinet)
        data_admin   = decode_access_token(token_admin)

        assert data_demo.rate_limit    == RATE_LIMITS["demo"]
        assert data_cabinet.rate_limit == RATE_LIMITS["cabinet"]
        assert data_admin.rate_limit   == RATE_LIMITS["admin"]

    def test_unknown_plan_falls_back_to_default(self):
        """Plan inconnu → fallback sur le plan 'default'."""
        token = create_access_token(sub="u", plan="unknown_plan")
        data  = decode_access_token(token)
        assert data.rate_limit == RATE_LIMITS["default"]


# ══════════════════════════════════════════════════════════════════════════════
# API KEYS — HACHAGE ET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIKeys:

    def test_hash_is_sha256(self):
        """Le hash d'une clé doit être un SHA-256 hexadécimal (64 chars)."""
        h = _hash_api_key("my_secret_key")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self):
        """Même clé → même hash (déterministe)."""
        assert _hash_api_key("key123") == _hash_api_key("key123")

    def test_hash_different_keys(self):
        """Deux clés différentes → hashes différents."""
        assert _hash_api_key("key_a") != _hash_api_key("key_b")

    def test_hash_case_sensitive(self):
        """Le hachage est sensible à la casse."""
        assert _hash_api_key("MyKey") != _hash_api_key("mykey")

    def test_validate_unknown_key_raises_401(self):
        """Clé inconnue → HTTPException 401."""
        try:
            validate_api_key("totally_unknown_key_xyz_123")
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401

    def test_validate_empty_key_raises_401(self):
        """Clé vide → HTTPException 401."""
        try:
            validate_api_key("")
            assert False, "Doit lever une HTTPException"
        except HTTPException as e:
            assert e.status_code == 401


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING — FENÊTRE GLISSANTE
# ══════════════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    """
    Tests du rate limiting avec des fenêtres courtes (1 seconde)
    pour ne pas ralentir la CI.

    ISOLATION : chaque test utilise un identifiant unique pour ne pas
    interférer avec les autres tests ou avec une instance du serveur.
    """

    def _unique_id(self, suffix: str) -> str:
        """Génère un identifiant unique pour isoler les tests."""
        return f"test_rate_limit_{suffix}_{time.time()}"

    def test_under_limit_passes(self):
        """5 requêtes avec limite 10 → toutes passent sans exception."""
        identifier = self._unique_id("under")
        for _ in range(5):
            check_rate_limit(identifier, max_requests=10, window_seconds=60)
        # Si on arrive ici, aucune exception n'a été levée

    def test_exactly_at_limit_passes(self):
        """Exactement max_requests → la dernière doit encore passer."""
        identifier = self._unique_id("exact")
        for _ in range(10):
            check_rate_limit(identifier, max_requests=10, window_seconds=60)
        # La 10ème requête doit passer

    def test_over_limit_raises_429(self):
        """max_requests + 1 → HTTPException 429."""
        identifier = self._unique_id("over")
        # Remplir jusqu'à la limite
        for _ in range(5):
            check_rate_limit(identifier, max_requests=5, window_seconds=60)

        # La 6ème doit être bloquée
        try:
            check_rate_limit(identifier, max_requests=5, window_seconds=60)
            assert False, "Doit lever une HTTPException 429"
        except HTTPException as e:
            assert e.status_code == 429

    def test_429_has_retry_after_header(self):
        """Le 429 doit inclure un header Retry-After."""
        identifier = self._unique_id("retry")
        for _ in range(3):
            check_rate_limit(identifier, max_requests=3, window_seconds=60)

        try:
            check_rate_limit(identifier, max_requests=3, window_seconds=60)
            assert False
        except HTTPException as e:
            assert e.status_code == 429
            assert "Retry-After" in e.headers
            retry_after = int(e.headers["Retry-After"])
            assert retry_after > 0
            assert retry_after <= 60  # ne peut pas dépasser la fenêtre

    def test_window_expiry_resets_counter(self):
        """
        Après expiration de la fenêtre, les requêtes sont à nouveau acceptées.
        On utilise une fenêtre de 1 seconde pour que le test reste rapide.
        """
        identifier = self._unique_id("expiry")

        # Remplir la limite
        for _ in range(3):
            check_rate_limit(identifier, max_requests=3, window_seconds=1)

        # Vérifier que c'est bien bloqué
        try:
            check_rate_limit(identifier, max_requests=3, window_seconds=1)
            assert False, "Doit être bloqué"
        except HTTPException as e:
            assert e.status_code == 429

        # Attendre que la fenêtre expire
        time.sleep(1.1)

        # Maintenant ça doit de nouveau passer
        check_rate_limit(identifier, max_requests=3, window_seconds=1)
        # Si on arrive ici, le test passe

    def test_different_identifiers_independent(self):
        """
        Deux identifiants différents ont des compteurs indépendants.
        Le rate limit de l'un ne doit pas affecter l'autre.
        """
        id_a = self._unique_id("ind_a")
        id_b = self._unique_id("ind_b")

        # Remplir id_a jusqu'à la limite
        for _ in range(3):
            check_rate_limit(id_a, max_requests=3, window_seconds=60)

        # id_b doit encore passer sans problème
        check_rate_limit(id_b, max_requests=3, window_seconds=60)
        # Si on arrive ici, les compteurs sont bien indépendants

    def test_limit_one(self):
        """
        Cas limite : max_requests=1.
        La 1ère passe, la 2ème est bloquée.
        """
        identifier = self._unique_id("limit_one")
        check_rate_limit(identifier, max_requests=1, window_seconds=60)

        try:
            check_rate_limit(identifier, max_requests=1, window_seconds=60)
            assert False
        except HTTPException as e:
            assert e.status_code == 429


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN DATA — MODÈLE DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenData:

    def test_default_values(self):
        """TokenData avec valeurs par défaut."""
        td = TokenData(sub="user@test.fr")
        assert td.plan     == "default"
        assert td.client   == ""
        assert td.is_admin is False
        assert td.exp      is None

    def test_rate_limit_property(self):
        """La property rate_limit retourne le bon tuple."""
        td = TokenData(sub="u", plan="demo")
        max_req, window = td.rate_limit
        assert isinstance(max_req, int)
        assert isinstance(window, int)
        assert max_req > 0
        assert window > 0
