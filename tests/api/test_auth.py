
"""
tests/unit/test_auth.py

Lexora AI — Authentication & Authorization Unit Tests

Expected production module:
    backend.app.services.auth_service

Expected class:
    AuthService

Expected constructor:
    AuthService(
        user_repository=...,
        token_service=...,
        password_hasher=...,
        audit_service=...,
        max_login_attempts=5,
    )

Expected async methods:
    register_user(
        email,
        password,
        tenant_id,
        role="viewer",
    ) -> dict

    login(
        email,
        password,
        tenant_id,
    ) -> dict

    authenticate_token(
        access_token,
    ) -> dict

    refresh_access_token(
        refresh_token,
    ) -> dict

    logout(
        refresh_token,
    ) -> bool

    change_password(
        user_id,
        tenant_id,
        old_password,
        new_password,
    ) -> bool

Expected login response:
    {
        "access_token": str,
        "refresh_token": str,
        "token_type": "bearer",
        "user": {
            "id": str,
            "email": str,
            "tenant_id": str,
            "role": str,
        },
    }

Security requirements:
    - Never store plaintext passwords.
    - Verify passwords using a secure password hasher.
    - Reject invalid credentials without user enumeration.
    - Reject expired, revoked, or malformed tokens.
    - Validate JWT signature, issuer, audience and token type.
    - Enforce tenant membership and active account status.
    - Prevent privilege escalation during registration.
    - Revoke refresh tokens on logout.
    - Enforce password complexity and account lockout.
    - Avoid logging passwords or bearer tokens.

This suite defines a reference service contract.
It is not verified against the actual application.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import inspect
import logging
import re
import secrets
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

USER_A = "user_alpha"
USER_B = "user_beta"
ADMIN_ID = "admin_alpha"

EMAIL_A = "user.alpha@example.com"
EMAIL_B = "user.beta@example.com"
ADMIN_EMAIL = "admin@example.com"

PASSWORD = "StrongPassword@123!"
NEW_PASSWORD = "NewStrongPassword@456!"

INVALID_PASSWORD = "IncorrectPassword@999!"

MAX_LOGIN_ATTEMPTS = 5

VALID_ROLES = {
    "viewer",
    "editor",
    "reviewer",
    "admin",
}

NOW = datetime(
    2026, 10, 8, 12, 0, 0,
    tzinfo=timezone.utc,
)


# ============================================================
# IMPORT
# ============================================================

def load_auth_service_class():
    candidates = (
        "backend.app.services.auth_service",
        "backend.app.services.auth",
        "backend.app.security.auth_service",
        "app.services.auth_service",
        "app.services.auth",
        "app.security.auth_service",
    )

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(str(exc))
            continue

        auth_class = getattr(
            module,
            "AuthService",
            None,
        )

        if auth_class is not None:
            return auth_class

    pytest.fail(
        "AuthService could not be imported. "
        "Update the import path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# PASSWORD HASHER
# ============================================================

class FakePasswordHasher:
    """
    Test-only password hasher.

    Uses PBKDF2-HMAC-SHA256 with a random salt.
    Production should use an appropriately configured
    Argon2id or equivalent password hashing scheme.

    This fake is intended for isolated service tests,
    not production password storage.
    """

    def hash(self, password):
        if not isinstance(password, str):
            raise TypeError("Password must be a string")

        salt = secrets.token_bytes(16)

        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100_000,
        )

        return (
            "pbkdf2_sha256$"
            + salt.hex()
            + "$"
            + digest.hex()
        )

    def verify(self, password, encoded):
        try:
            algorithm, salt_hex, digest_hex = (
                encoded.split("$")
            )

            if algorithm != "pbkdf2_sha256":
                return False

            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)

            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                100_000,
            )

            return hmac.compare_digest(
                actual,
                expected,
            )

        except (ValueError, TypeError, AttributeError):
            return False


# ============================================================
# FAKE TOKEN SERVICE
# ============================================================

class FakeTokenService:
    """
    Deterministic-behavior token test double.

    Tokens are opaque random strings held in memory.
    This does NOT test actual JWT signing or decoding.

    JWT cryptographic validation must be covered
    by separate token-service integration tests.
    """

    def __init__(self):
        self.tokens = {}
        self.revoked = set()

    def _create(self, payload, token_type):
        token = secrets.token_urlsafe(32)

        self.tokens[token] = {
            **deepcopy(payload),
            "token_type": token_type,
            "exp": NOW + timedelta(
                minutes=15
                if token_type == "access"
                else 60 * 24 * 7
            ),
        }

        return token

    async def create_access_token(self, payload):
        return self._create(
            payload,
            "access",
        )

    async def create_refresh_token(self, payload):
        return self._create(
            payload,
            "refresh",
        )

    async def verify_token(self, token, token_type=None):
        if token in self.revoked:
            raise PermissionError("Token revoked")

        payload = self.tokens.get(token)

        if payload is None:
            raise PermissionError("Invalid token")

        if payload["exp"] <= NOW:
            raise PermissionError("Expired token")

        if (
            token_type is not None
            and payload["token_type"] != token_type
        ):
            raise PermissionError("Wrong token type")

        return deepcopy(payload)

    async def revoke_token(self, token):
        self.revoked.add(token)
        return True


# ============================================================
# USER FACTORY
# ============================================================

def make_user(
    user_id,
    email,
    password_hash,
    tenant_id=TENANT_A,
    role="viewer",
    is_active=True,
    failed_login_attempts=0,
    is_locked=False,
):
    return {
        "id": user_id,
        "email": email,
        "password_hash": password_hash,
        "tenant_id": tenant_id,
        "role": role,
        "is_active": is_active,
        "failed_login_attempts": failed_login_attempts,
        "is_locked": is_locked,
    }


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def password_hasher():
    return FakePasswordHasher()


@pytest.fixture
def token_service():
    return FakeTokenService()


@pytest.fixture
def user_repository(password_hasher):
    repository = MagicMock()

    users = [
        make_user(
            USER_A,
            EMAIL_A,
            password_hasher.hash(PASSWORD),
            tenant_id=TENANT_A,
        ),
        make_user(
            USER_B,
            EMAIL_B,
            password_hasher.hash(PASSWORD),
            tenant_id=TENANT_B,
        ),
        make_user(
            ADMIN_ID,
            ADMIN_EMAIL,
            password_hasher.hash(PASSWORD),
            tenant_id=TENANT_A,
            role="admin",
        ),
        make_user(
            "inactive_user",
            "inactive@example.com",
            password_hasher.hash(PASSWORD),
            tenant_id=TENANT_A,
            is_active=False,
        ),
        make_user(
            "locked_user",
            "locked@example.com",
            password_hasher.hash(PASSWORD),
            tenant_id=TENANT_A,
            is_locked=True,
        ),
    ]

    repository.users = {
        user["id"]: user
        for user in users
    }

    async def get_by_email(email, tenant_id):
        for user in repository.users.values():
            if (
                user["email"].lower() == email.lower()
                and user["tenant_id"] == tenant_id
            ):
                return deepcopy(user)

        return None

    async def get_by_id(user_id, tenant_id):
        user = repository.users.get(user_id)

        if user and user["tenant_id"] == tenant_id:
            return deepcopy(user)

        return None

    async def create(user_data):
        user_id = user_data.get(
            "id",
            f"user_{secrets.token_hex(8)}",
        )

        record = {
            **deepcopy(user_data),
            "id": user_id,
        }

        repository.users[user_id] = record

        return deepcopy(record)

    async def update(user_id, changes):
        user = repository.users.get(user_id)

        if user is None:
            raise LookupError("User not found")

        user.update(deepcopy(changes))

        return deepcopy(user)

    repository.get_by_email = AsyncMock(
        side_effect=get_by_email
    )
    repository.get_by_id = AsyncMock(
        side_effect=get_by_id
    )
    repository.create = AsyncMock(
        side_effect=create
    )
    repository.update = AsyncMock(
        side_effect=update
    )

    return repository


@pytest.fixture
def audit_service():
    service = MagicMock()

    service.record_event = AsyncMock(
        return_value=None
    )

    return service


@pytest.fixture
def auth_service(
    user_repository,
    token_service,
    password_hasher,
    audit_service,
):
    service_class = load_auth_service_class()

    return service_class(
        user_repository=user_repository,
        token_service=token_service,
        password_hasher=password_hasher,
        audit_service=audit_service,
        max_login_attempts=MAX_LOGIN_ATTEMPTS,
    )


# ============================================================
# HELPERS
# ============================================================

async def invoke(method, **kwargs):
    result = method(**kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


async def login(
    service,
    email=EMAIL_A,
    password=PASSWORD,
    tenant_id=TENANT_A,
):
    return await invoke(
        service.login,
        email=email,
        password=password,
        tenant_id=tenant_id,
    )


async def register(
    service,
    email,
    password=PASSWORD,
    tenant_id=TENANT_A,
    role="viewer",
):
    return await invoke(
        service.register_user,
        email=email,
        password=password,
        tenant_id=tenant_id,
        role=role,
    )


async def authenticate(service, access_token):
    return await invoke(
        service.authenticate_token,
        access_token=access_token,
    )


async def refresh(service, refresh_token):
    return await invoke(
        service.refresh_access_token,
        refresh_token=refresh_token,
    )


async def logout(service, refresh_token):
    return await invoke(
        service.logout,
        refresh_token=refresh_token,
    )


async def change_password(
    service,
    user_id=USER_A,
    tenant_id=TENANT_A,
    old_password=PASSWORD,
    new_password=NEW_PASSWORD,
):
    return await invoke(
        service.change_password,
        user_id=user_id,
        tenant_id=tenant_id,
        old_password=old_password,
        new_password=new_password,
    )


def assert_valid_login_response(result):
    assert isinstance(result, dict)

    required = {
        "access_token",
        "refresh_token",
        "token_type",
        "user",
    }

    assert required.issubset(result.keys())

    assert isinstance(result["access_token"], str)
    assert isinstance(result["refresh_token"], str)

    assert result["access_token"]
    assert result["refresh_token"]

    assert result["token_type"].lower() == "bearer"

    user = result["user"]

    assert isinstance(user, dict)

    assert {
        "id",
        "email",
        "tenant_id",
        "role",
    }.issubset(user.keys())

    assert "password" not in user
    assert "password_hash" not in user


# ============================================================
# INITIALIZATION
# ============================================================

async def test_auth_service_initialization(auth_service):
    assert auth_service is not None
    assert callable(auth_service.login)
    assert callable(auth_service.register_user)
    assert callable(auth_service.authenticate_token)
    assert callable(auth_service.refresh_access_token)
    assert callable(auth_service.logout)
    assert callable(auth_service.change_password)


# ============================================================
# PASSWORD HASHING
# ============================================================

async def test_password_hash_not_plaintext(
    password_hasher,
):
    hashed = password_hasher.hash(PASSWORD)

    assert hashed != PASSWORD
    assert PASSWORD not in hashed


async def test_password_hash_verification(
    password_hasher,
):
    hashed = password_hasher.hash(PASSWORD)

    assert password_hasher.verify(
        PASSWORD,
        hashed,
    )


async def test_wrong_password_hash_verification(
    password_hasher,
):
    hashed = password_hasher.hash(PASSWORD)

    assert not password_hasher.verify(
        INVALID_PASSWORD,
        hashed,
    )


async def test_password_hash_uses_unique_salt(
    password_hasher,
):
    first = password_hasher.hash(PASSWORD)
    second = password_hasher.hash(PASSWORD)

    assert first != second


async def test_password_hash_unicode(
    password_hasher,
):
    password = "Strongപാസ്‌വേഡ്@123!"

    hashed = password_hasher.hash(password)

    assert password_hasher.verify(
        password,
        hashed,
    )


# ============================================================
# LOGIN
# ============================================================

async def test_successful_login(auth_service):
    result = await login(auth_service)

    assert_valid_login_response(result)


async def test_login_returns_correct_user(
    auth_service,
):
    result = await login(auth_service)

    assert result["user"]["id"] == USER_A
    assert result["user"]["email"] == EMAIL_A
    assert result["user"]["tenant_id"] == TENANT_A


async def test_admin_login(auth_service):
    result = await login(
        auth_service,
        email=ADMIN_EMAIL,
    )

    assert result["user"]["role"] == "admin"


async def test_wrong_password_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            password=INVALID_PASSWORD,
        )


async def test_unknown_user_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            email="unknown@example.com",
        )


async def test_cross_tenant_login_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            email=EMAIL_A,
            tenant_id=TENANT_B,
        )


async def test_inactive_user_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            email="inactive@example.com",
        )


async def test_locked_user_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            email="locked@example.com",
        )


async def test_login_does_not_return_password(
    auth_service,
):
    result = await login(auth_service)

    serialized = repr(result)

    assert PASSWORD not in serialized
    assert "password_hash" not in serialized


# ============================================================
# LOGIN INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_email",
    [
        "",
        "   ",
        "invalid-email",
        "@example.com",
        "user@",
        None,
    ],
)
async def test_invalid_login_email(
    auth_service,
    invalid_email,
):
    with pytest.raises(
        (ValueError, TypeError, PermissionError)
    ):
        await login(
            auth_service,
            email=invalid_email,
        )


@pytest.mark.parametrize(
    "invalid_password",
    ["", "   ", None],
)
async def test_invalid_login_password(
    auth_service,
    invalid_password,
):
    with pytest.raises(
        (ValueError, TypeError, PermissionError)
    ):
        await login(
            auth_service,
            password=invalid_password,
        )


@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_login_tenant(
    auth_service,
    invalid_tenant,
):
    with pytest.raises(
        (ValueError, TypeError, PermissionError)
    ):
        await login(
            auth_service,
            tenant_id=invalid_tenant,
        )


# ============================================================
# REGISTRATION
# ============================================================

async def test_register_new_user(auth_service):
    result = await register(
        auth_service,
        "new.user@example.com",
    )

    assert isinstance(result, dict)
    assert result["email"] == "new.user@example.com"
    assert result["tenant_id"] == TENANT_A


async def test_registered_password_hashed(
    auth_service,
    user_repository,
):
    email = "hash.test@example.com"

    await register(
        auth_service,
        email,
    )

    users = [
        user
        for user in user_repository.users.values()
        if user["email"] == email
    ]

    assert len(users) == 1

    assert users[0]["password_hash"] != PASSWORD


async def test_default_registration_role(
    auth_service,
):
    result = await register(
        auth_service,
        "viewer.test@example.com",
    )

    assert result["role"] == "viewer"


async def test_duplicate_registration_rejected(
    auth_service,
):
    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await register(
            auth_service,
            EMAIL_A,
        )


@pytest.mark.parametrize(
    "weak_password",
    [
        "123456",
        "password",
        "abcdefgh",
        "Password123",
        "short!",
        "",
    ],
)
async def test_weak_password_rejected(
    auth_service,
    weak_password,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await register(
            auth_service,
            "weak.password@example.com",
            password=weak_password,
        )


async def test_self_registration_cannot_assign_admin(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await register(
            auth_service,
            "attacker@example.com",
            role="admin",
        )


async def test_registration_does_not_return_hash(
    auth_service,
):
    result = await register(
        auth_service,
        "safe.registration@example.com",
    )

    assert "password_hash" not in result
    assert "password" not in result


# ============================================================
# ACCESS TOKENS
# ============================================================

async def test_access_token_authentication(
    auth_service,
):
    session = await login(auth_service)

    user = await authenticate(
        auth_service,
        session["access_token"],
    )

    assert user["id"] == USER_A
    assert user["tenant_id"] == TENANT_A


async def test_invalid_access_token_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await authenticate(
            auth_service,
            "invalid.token.value",
        )


async def test_empty_access_token_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError, TypeError)
    ):
        await authenticate(
            auth_service,
            "",
        )


async def test_refresh_token_cannot_be_used_as_access(
    auth_service,
):
    session = await login(auth_service)

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await authenticate(
            auth_service,
            session["refresh_token"],
        )


async def test_expired_access_token_rejected(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    token = session["access_token"]

    token_service.tokens[token]["exp"] = (
        NOW - timedelta(seconds=1)
    )

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await authenticate(
            auth_service,
            token,
        )


async def test_revoked_access_token_rejected(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    token = session["access_token"]

    await token_service.revoke_token(token)

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await authenticate(
            auth_service,
            token,
        )


# ============================================================
# TOKEN CLAIMS
# ============================================================

async def test_access_token_contains_identity(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    claims = await token_service.verify_token(
        session["access_token"],
        token_type="access",
    )

    assert (
        claims.get("sub", claims.get("user_id"))
        == USER_A
    )


async def test_access_token_contains_tenant(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    claims = await token_service.verify_token(
        session["access_token"],
        token_type="access",
    )

    assert claims["tenant_id"] == TENANT_A


async def test_access_token_contains_role(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    claims = await token_service.verify_token(
        session["access_token"],
        token_type="access",
    )

    assert claims["role"] == "viewer"


# ============================================================
# REFRESH TOKENS
# ============================================================

async def test_refresh_access_token(
    auth_service,
):
    session = await login(auth_service)

    result = await refresh(
        auth_service,
        session["refresh_token"],
    )

    assert isinstance(result, dict)
    assert result["access_token"]


async def test_refresh_returns_new_access_token(
    auth_service,
):
    session = await login(auth_service)

    result = await refresh(
        auth_service,
        session["refresh_token"],
    )

    assert (
        result["access_token"]
        != session["access_token"]
    )


async def test_invalid_refresh_token_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await refresh(
            auth_service,
            "invalid-refresh-token",
        )


async def test_access_token_cannot_refresh(
    auth_service,
):
    session = await login(auth_service)

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await refresh(
            auth_service,
            session["access_token"],
        )


async def test_expired_refresh_token_rejected(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    token = session["refresh_token"]

    token_service.tokens[token]["exp"] = (
        NOW - timedelta(seconds=1)
    )

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await refresh(
            auth_service,
            token,
        )


# ============================================================
# LOGOUT
# ============================================================

async def test_logout_success(auth_service):
    session = await login(auth_service)

    result = await logout(
        auth_service,
        session["refresh_token"],
    )

    assert result is True


async def test_logout_revokes_refresh_token(
    auth_service,
):
    session = await login(auth_service)

    token = session["refresh_token"]

    await logout(
        auth_service,
        token,
    )

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await refresh(
            auth_service,
            token,
        )


async def test_repeated_logout_is_safe(
    auth_service,
):
    session = await login(auth_service)

    token = session["refresh_token"]

    await logout(auth_service, token)

    try:
        result = await logout(
            auth_service,
            token,
        )
    except (PermissionError, ValueError):
        return

    assert isinstance(result, bool)


# ============================================================
# PASSWORD CHANGES
# ============================================================

async def test_change_password(
    auth_service,
):
    result = await change_password(
        auth_service
    )

    assert result is True


async def test_old_password_rejected_after_change(
    auth_service,
):
    await change_password(
        auth_service
    )

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            password=PASSWORD,
        )


async def test_new_password_accepted(
    auth_service,
):
    await change_password(
        auth_service
    )

    result = await login(
        auth_service,
        password=NEW_PASSWORD,
    )

    assert_valid_login_response(result)


async def test_wrong_old_password_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await change_password(
            auth_service,
            old_password=INVALID_PASSWORD,
        )


async def test_cross_tenant_password_change_rejected(
    auth_service,
):
    with pytest.raises(
        (PermissionError, ValueError, LookupError)
    ):
        await change_password(
            auth_service,
            user_id=USER_A,
            tenant_id=TENANT_B,
        )


# ============================================================
# ACCOUNT LOCKOUT
# ============================================================

async def test_failed_login_increments_attempts(
    auth_service,
    user_repository,
):
    before = user_repository.users[
        USER_A
    ]["failed_login_attempts"]

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            password=INVALID_PASSWORD,
        )

    after = user_repository.users[
        USER_A
    ]["failed_login_attempts"]

    assert after > before


async def test_account_locked_after_max_attempts(
    auth_service,
    user_repository,
):
    for _ in range(MAX_LOGIN_ATTEMPTS):
        with pytest.raises(
            (PermissionError, ValueError)
        ):
            await login(
                auth_service,
                password=INVALID_PASSWORD,
            )

    assert user_repository.users[
        USER_A
    ]["is_locked"] is True


async def test_locked_account_rejects_correct_password(
    auth_service,
):
    for _ in range(MAX_LOGIN_ATTEMPTS):
        with pytest.raises(
            (PermissionError, ValueError)
        ):
            await login(
                auth_service,
                password=INVALID_PASSWORD,
            )

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            password=PASSWORD,
        )


async def test_successful_login_resets_failed_attempts(
    auth_service,
    user_repository,
):
    user_repository.users[
        USER_A
    ]["failed_login_attempts"] = 2

    await login(auth_service)

    assert user_repository.users[
        USER_A
    ]["failed_login_attempts"] == 0


# ============================================================
# MULTI-TENANT SECURITY
# ============================================================

async def test_tenant_a_token_identity(
    auth_service,
):
    session = await login(
        auth_service,
        email=EMAIL_A,
        tenant_id=TENANT_A,
    )

    user = await authenticate(
        auth_service,
        session["access_token"],
    )

    assert user["tenant_id"] == TENANT_A


async def test_tenant_b_token_identity(
    auth_service,
):
    session = await login(
        auth_service,
        email=EMAIL_B,
        tenant_id=TENANT_B,
    )

    user = await authenticate(
        auth_service,
        session["access_token"],
    )

    assert user["tenant_id"] == TENANT_B


async def test_cross_tenant_token_claim_not_trusted(
    auth_service,
    token_service,
):
    session = await login(auth_service)

    token = session["access_token"]

    # Simulates a validly decoded token containing
    # an unauthorized tenant claim.
    # This is NOT a JWT signature-forgery test.
    token_service.tokens[token]["tenant_id"] = TENANT_B

    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await authenticate(
            auth_service,
            token,
        )


# ============================================================
# ROLE SECURITY
# ============================================================

async def test_user_role_preserved(
    auth_service,
):
    session = await login(auth_service)

    assert session["user"]["role"] == "viewer"


async def test_admin_role_preserved(
    auth_service,
):
    session = await login(
        auth_service,
        email=ADMIN_EMAIL,
    )

    assert session["user"]["role"] == "admin"


async def test_registration_rejects_unknown_role(
    auth_service,
):
    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await register(
            auth_service,
            "unknown.role@example.com",
            role="super_root",
        )


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_successful_login_audited(
    auth_service,
    audit_service,
):
    await login(auth_service)

    audit_service.record_event.assert_awaited()


async def test_failed_login_audited(
    auth_service,
    audit_service,
):
    with pytest.raises(
        (PermissionError, ValueError)
    ):
        await login(
            auth_service,
            password=INVALID_PASSWORD,
        )

    audit_service.record_event.assert_awaited()


async def test_logout_audited(
    auth_service,
    audit_service,
):
    session = await login(auth_service)

    audit_service.record_event.reset_mock()

    await logout(
        auth_service,
        session["refresh_token"],
    )

    audit_service.record_event.assert_awaited()


# ============================================================
# SENSITIVE DATA PROTECTION
# ============================================================

async def test_password_not_logged(
    auth_service,
    caplog,
):
    with caplog.at_level(logging.DEBUG):
        await login(auth_service)

    assert PASSWORD not in caplog.text


async def test_invalid_password_not_logged(
    auth_service,
    caplog,
):
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(
            (PermissionError, ValueError)
        ):
            await login(
                auth_service,
                password=INVALID_PASSWORD,
            )

    assert INVALID_PASSWORD not in caplog.text


async def test_tokens_not_logged(
    auth_service,
    caplog,
):
    with caplog.at_level(logging.DEBUG):
        session = await login(auth_service)

    assert (
        session["access_token"]
        not in caplog.text
    )

    assert (
        session["refresh_token"]
        not in caplog.text
    )


async def test_password_hash_not_exposed(
    auth_service,
):
    session = await login(auth_service)

    assert "password_hash" not in repr(session)


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_logins(
    auth_service,
):
    results = await asyncio.gather(
        *[
            login(auth_service)
            for _ in range(5)
        ]
    )

    assert len(results) == 5

    tokens = [
        result["access_token"]
        for result in results
    ]

    assert len(tokens) == len(set(tokens))


async def test_concurrent_tenant_logins(
    auth_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        login(
            auth_service,
            email=EMAIL_A,
            tenant_id=TENANT_A,
        ),
        login(
            auth_service,
            email=EMAIL_B,
            tenant_id=TENANT_B,
        ),
    )

    assert tenant_a["user"]["tenant_id"] == TENANT_A
    assert tenant_b["user"]["tenant_id"] == TENANT_B


async def test_concurrent_token_authentication(
    auth_service,
):
    session = await login(auth_service)

    users = await asyncio.gather(
        *[
            authenticate(
                auth_service,
                session["access_token"],
            )
            for _ in range(10)
        ]
    )

    assert len(users) == 10

    assert all(
        user["id"] == USER_A
        for user in users
    )


# ============================================================
# END
# ============================================================
