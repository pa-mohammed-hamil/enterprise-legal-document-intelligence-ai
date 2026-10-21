"""
backend/app/middleware/auth.py

Authentication middleware for Enterprise Legal Document Intelligence AI.

Responsibilities
----------------
- Extract Bearer tokens
- Verify JWT access tokens
- Reject malformed / expired tokens
- Populate request.state with authenticated identity
- Preserve tenant / organization context
- Support configurable public routes
- Generate consistent 401 responses
- Avoid mixing authentication with RBAC authorization

Security:
- Does not trust client-supplied user/tenant headers.
- Uses identity claims only from verified tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class AuthenticationError(Exception):
    """Base authentication exception."""


class MissingTokenError(AuthenticationError):
    """Authentication token is missing."""


class InvalidTokenError(AuthenticationError):
    """Authentication token is invalid."""


class ExpiredTokenError(AuthenticationError):
    """Authentication token has expired."""


# ============================================================================
# Token verifier protocol
# ============================================================================


class TokenVerifier(Protocol):
    """
    Implemented by the application's JWT/auth service.

    Expected return value:

        {
            "sub": "user-id",
            "email": "user@example.com",
            "tenant_id": "tenant-id",
            "organization_id": "org-id",
            "roles": ["legal_reviewer"],
            "permissions": ["document:read"],
            "token_type": "access"
        }
    """

    def verify_access_token(
        self,
        token: str,
    ) -> Mapping[str, Any]:
        ...


# ============================================================================
# Authentication identity
# ============================================================================


@dataclass(frozen=True, slots=True)
class AuthenticatedIdentity:
    user_id: str

    email: str | None = None
    tenant_id: str | None = None
    organization_id: str | None = None

    roles: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    token_id: str | None = None

    claims: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
    )

    @property
    def is_authenticated(self) -> bool:
        return bool(self.user_id)

    def has_role(
        self,
        role: str,
    ) -> bool:
        return role in self.roles

    def has_permission(
        self,
        permission: str,
    ) -> bool:
        return permission in self.permissions


# ============================================================================
# Configuration
# ============================================================================


@dataclass(slots=True)
class AuthMiddlewareConfig:
    public_paths: set[str] = field(
        default_factory=lambda: {
            "/",
            "/health",
            "/healthz",
            "/ready",
            "/readyz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
        }
    )

    public_prefixes: tuple[str, ...] = (
        "/docs/",
        "/redoc/",
        "/static/",
    )

    bearer_scheme: str = "Bearer"

    require_access_token_type: bool = True

    # OPTIONS requests are commonly needed for CORS preflight.
    allow_options: bool = True


# ============================================================================
# Authentication Middleware
# ============================================================================


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authenticate API requests using Bearer access tokens.

    Authentication answers:

        "Who is this caller?"

    RBAC / permission middleware should answer:

        "What is this caller allowed to do?"
    """

    def __init__(
        self,
        app: Any,
        *,
        token_verifier: TokenVerifier,
        config: AuthMiddlewareConfig | None = None,
    ) -> None:
        super().__init__(app)

        self.token_verifier = token_verifier
        self.config = (
            config
            or AuthMiddlewareConfig()
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Response],
        ],
    ) -> Response:

        # ---------------------------------------------------------------
        # Initialize safe request state
        # ---------------------------------------------------------------

        self._clear_identity(request)

        # ---------------------------------------------------------------
        # CORS preflight
        # ---------------------------------------------------------------

        if (
            self.config.allow_options
            and request.method.upper() == "OPTIONS"
        ):
            return await call_next(request)

        # ---------------------------------------------------------------
        # Public endpoint
        # ---------------------------------------------------------------

        if self._is_public_path(
            request.url.path
        ):
            return await call_next(request)

        # ---------------------------------------------------------------
        # Extract token
        # ---------------------------------------------------------------

        try:
            token = self._extract_bearer_token(
                request
            )

        except MissingTokenError as exc:
            return self._unauthorized(
                request,
                code="AUTHENTICATION_REQUIRED",
                message=str(exc),
            )

        except InvalidTokenError as exc:
            return self._unauthorized(
                request,
                code="INVALID_AUTHORIZATION_HEADER",
                message=str(exc),
            )

        # ---------------------------------------------------------------
        # Verify token
        # ---------------------------------------------------------------

        try:
            claims = (
                self.token_verifier
                .verify_access_token(token)
            )

            identity = (
                self._identity_from_claims(
                    claims
                )
            )

        except ExpiredTokenError:
            return self._unauthorized(
                request,
                code="TOKEN_EXPIRED",
                message="Access token has expired.",
            )

        except AuthenticationError as exc:
            return self._unauthorized(
                request,
                code="INVALID_TOKEN",
                message=str(exc),
            )

        except Exception:
            # Never expose JWT/parser internals to clients.
            logger.exception(
                "Unexpected authentication failure"
            )

            return self._unauthorized(
                request,
                code="INVALID_TOKEN",
                message=(
                    "Authentication credentials "
                    "could not be verified."
                ),
            )

        # ---------------------------------------------------------------
        # Store verified identity
        # ---------------------------------------------------------------

        request.state.identity = identity
        request.state.user_id = identity.user_id
        request.state.email = identity.email

        request.state.tenant_id = (
            identity.tenant_id
        )

        request.state.organization_id = (
            identity.organization_id
        )

        request.state.roles = identity.roles

        request.state.permissions = (
            identity.permissions
        )

        request.state.is_authenticated = True

        # ---------------------------------------------------------------
        # Continue request
        # ---------------------------------------------------------------

        return await call_next(request)

    # ===================================================================
    # Token extraction
    # ===================================================================

    def _extract_bearer_token(
        self,
        request: Request,
    ) -> str:

        header = request.headers.get(
            "Authorization"
        )

        if not header:
            raise MissingTokenError(
                "Authentication credentials "
                "were not provided."
            )

        try:
            scheme, token = header.split(
                " ",
                1,
            )

        except ValueError as exc:
            raise InvalidTokenError(
                "Malformed Authorization header."
            ) from exc

        if (
            scheme.lower()
            != self.config.bearer_scheme.lower()
        ):
            raise InvalidTokenError(
                "Bearer authentication is required."
            )

        token = token.strip()

        if not token:
            raise InvalidTokenError(
                "Bearer token is empty."
            )

        return token

    # ===================================================================
    # Claims -> identity
    # ===================================================================

    def _identity_from_claims(
        self,
        claims: Mapping[str, Any],
    ) -> AuthenticatedIdentity:

        if not isinstance(
            claims,
            Mapping,
        ):
            raise InvalidTokenError(
                "Invalid token payload."
            )

        user_id = (
            claims.get("sub")
            or claims.get("user_id")
        )

        if not user_id:
            raise InvalidTokenError(
                "Token does not contain "
                "a valid subject."
            )

        # ---------------------------------------------------------------
        # Token type
        # ---------------------------------------------------------------

        token_type = (
            claims.get("token_type")
            or claims.get("type")
        )

        if (
            self.config.require_access_token_type
            and token_type is not None
            and str(token_type).lower()
            != "access"
        ):
            raise InvalidTokenError(
                "An access token is required."
            )

        # ---------------------------------------------------------------
        # Roles / permissions
        # ---------------------------------------------------------------

        roles = self._normalize_collection(
            claims.get("roles")
            or claims.get("role")
        )

        permissions = (
            self._normalize_collection(
                claims.get("permissions")
                or claims.get("scopes")
                or claims.get("scope")
            )
        )

        return AuthenticatedIdentity(
            user_id=str(user_id),
            email=self._optional_string(
                claims.get("email")
            ),
            tenant_id=self._optional_string(
                claims.get("tenant_id")
            ),
            organization_id=(
                self._optional_string(
                    claims.get(
                        "organization_id"
                    )
                    or claims.get("org_id")
                )
            ),
            roles=roles,
            permissions=permissions,
            token_id=self._optional_string(
                claims.get("jti")
            ),
            claims=dict(claims),
        )

    # ===================================================================
    # Public route handling
    # ===================================================================

    def _is_public_path(
        self,
        path: str,
    ) -> bool:

        normalized = (
            path.rstrip("/")
            or "/"
        )

        normalized_public_paths = {
            item.rstrip("/") or "/"
            for item
            in self.config.public_paths
        }

        if normalized in normalized_public_paths:
            return True

        return any(
            path.startswith(prefix)
            for prefix
            in self.config.public_prefixes
        )

    # ===================================================================
    # Request state
    # ===================================================================

    @staticmethod
    def _clear_identity(
        request: Request,
    ) -> None:

        request.state.identity = None
        request.state.user_id = None
        request.state.email = None
        request.state.tenant_id = None
        request.state.organization_id = None

        request.state.roles = ()
        request.state.permissions = ()

        request.state.is_authenticated = False

    # ===================================================================
    # Unauthorized response
    # ===================================================================

    @staticmethod
    def _unauthorized(
        request: Request,
        *,
        code: str,
        message: str,
    ) -> JSONResponse:

        request_id = getattr(
            request.state,
            "request_id",
            None,
        )

        payload: dict[str, Any] = {
            "error": {
                "code": code,
                "message": message,
            }
        }

        if request_id:
            payload["error"][
                "request_id"
            ] = request_id

        return JSONResponse(
            status_code=401,
            content=payload,
            headers={
                "WWW-Authenticate": "Bearer",
                "Cache-Control": "no-store",
            },
        )

    # ===================================================================
    # Helpers
    # ===================================================================

    @staticmethod
    def _normalize_collection(
        value: Any,
    ) -> tuple[str, ...]:

        if value is None:
            return ()

        if isinstance(value, str):
            # Handles:
            # "document:read document:write"
            # "admin"
            values = value.split()

        elif isinstance(
            value,
            (list, tuple, set, frozenset),
        ):
            values = value

        else:
            values = [value]

        result: list[str] = []
        seen: set[str] = set()

        for item in values:

            text = str(item).strip()

            if (
                not text
                or text in seen
            ):
                continue

            seen.add(text)
            result.append(text)

        return tuple(result)

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        value = str(value).strip()

        return value or None


# ============================================================================
# Request helpers
# ============================================================================


def get_identity(
    request: Request,
) -> AuthenticatedIdentity:
    """
    Return the verified identity from request.state.

    Raises AuthenticationError when called for an unauthenticated request.
    """

    identity = getattr(
        request.state,
        "identity",
        None,
    )

    if not isinstance(
        identity,
        AuthenticatedIdentity,
    ):
        raise AuthenticationError(
            "Authenticated identity is unavailable."
        )

    return identity


def get_current_user_id(
    request: Request,
) -> str:

    return get_identity(
        request
    ).user_id


def get_current_tenant_id(
    request: Request,
) -> str | None:

    return get_identity(
        request
    ).tenant_id


def get_current_organization_id(
    request: Request,
) -> str | None:

    return get_identity(
        request
    ).organization_id