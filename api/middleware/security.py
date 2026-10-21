"""
HTTP security middleware.

Responsibilities:
- Apply security-related HTTP headers.
- Enforce request-size limits.
- Restrict unsafe HTTP methods.
- Add basic browser security protections.
- Prevent accidental information leakage.
- Support configurable CORS/security behavior.
- Preserve request IDs for observability.

This module handles transport/API-level security.

Application security such as:
    - authentication
    - RBAC
    - prompt-injection detection
    - PII detection
    - document authorization
    - output guardrails

belongs to the corresponding security/core modules.
"""

from __future__ import annotations

from typing import Callable, Iterable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_BODY_SIZE = 25 * 1024 * 1024  # 25 MB

DEFAULT_ALLOWED_METHODS = frozenset(
    {
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    }
)

DEFAULT_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
        "text/plain",
    }
)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def apply_security_headers(
    response: Response,
    *,
    enable_hsts: bool = True,
    enable_csp: bool = True,
) -> None:
    """
    Apply baseline security headers.

    These headers are deliberately conservative and can be overridden
    at the reverse-proxy layer for production deployments.
    """

    response.headers.setdefault(
        "X-Content-Type-Options",
        "nosniff",
    )

    response.headers.setdefault(
        "X-Frame-Options",
        "DENY",
    )

    response.headers.setdefault(
        "Referrer-Policy",
        "strict-origin-when-cross-origin",
    )

    response.headers.setdefault(
        "Permissions-Policy",
        (
            "camera=(), "
            "microphone=(), "
            "geolocation=(), "
            "payment=()"
        ),
    )

    response.headers.setdefault(
        "Cross-Origin-Opener-Policy",
        "same-origin",
    )

    response.headers.setdefault(
        "Cross-Origin-Resource-Policy",
        "same-origin",
    )

    if enable_hsts:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )

    if enable_csp:
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "object-src 'none';"
            ),
        )


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def validate_http_method(
    method: str,
    allowed_methods: Iterable[str],
) -> None:
    if method.upper() not in {
        value.upper() for value in allowed_methods
    }:
        raise SecurityViolation(
            f"HTTP method '{method}' is not allowed."
        )


def validate_content_length(
    content_length: str | None,
    max_body_size: int,
) -> None:
    """
    Reject obviously oversized requests before application processing.

    Chunked requests without Content-Length are handled separately by
    application/server-level limits.
    """

    if not content_length:
        return

    try:
        size = int(content_length)
    except ValueError as exc:
        raise SecurityViolation(
            "Invalid Content-Length header."
        ) from exc

    if size < 0:
        raise SecurityViolation(
            "Invalid Content-Length header."
        )

    if size > max_body_size:
        raise SecurityViolation(
            "Request body exceeds the configured size limit."
        )


def validate_content_type(
    request: Request,
    allowed_content_types: Iterable[str],
) -> None:
    """
    Validate content type for requests that contain a body.

    GET/HEAD/OPTIONS requests do not require a content type.
    """

    if request.method.upper() in {
        "GET",
        "HEAD",
        "OPTIONS",
    }:
        return

    content_length = request.headers.get("content-length")

    if content_length == "0":
        return

    content_type = request.headers.get("content-type")

    if not content_type:
        return

    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type not in {
        value.lower()
        for value in allowed_content_types
    }:
        raise SecurityViolation(
            f"Unsupported content type: {media_type}"
        )


# ---------------------------------------------------------------------------
# Security exception
# ---------------------------------------------------------------------------


class SecurityViolation(Exception):
    """
    Internal security validation exception.

    Middleware converts this into an HTTP 400/413/405-style response
    without exposing internal implementation details.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Application-level HTTP security middleware.

    Order in the application should generally be:

        Request ID
             ↓
        Security
             ↓
        Rate limiting
             ↓
        Authentication
             ↓
        RBAC / tenant isolation
             ↓
        Route
    """

    def __init__(
        self,
        app,
        *,
        max_body_size: int = DEFAULT_MAX_BODY_SIZE,
        allowed_methods: Iterable[str] = DEFAULT_ALLOWED_METHODS,
        allowed_content_types: Iterable[str] = DEFAULT_ALLOWED_CONTENT_TYPES,
        enable_hsts: bool = True,
        enable_csp: bool = True,
    ) -> None:
        super().__init__(app)

        if max_body_size <= 0:
            raise ValueError(
                "max_body_size must be greater than zero."
            )

        self.max_body_size = max_body_size
        self.allowed_methods = frozenset(allowed_methods)
        self.allowed_content_types = frozenset(
            allowed_content_types
        )
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:

        try:
            validate_http_method(
                request.method,
                self.allowed_methods,
            )

            validate_content_length(
                request.headers.get("content-length"),
                self.max_body_size,
            )

            validate_content_type(
                request,
                self.allowed_content_types,
            )

            response = await call_next(request)

        except SecurityViolation as exc:
            response = Response(
                content='{"detail":"Request rejected by security policy."}',
                status_code=400,
                media_type="application/json",
            )

            # Keep the detailed reason out of the response to avoid
            # exposing security-policy internals.

            request.state.security_violation = exc.message

        apply_security_headers(
            response,
            enable_hsts=self.enable_hsts,
            enable_csp=self.enable_csp,
        )

        # Security headers should not reveal framework/server information.
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)

        return response


# ---------------------------------------------------------------------------
# Helper for programmatic use
# ---------------------------------------------------------------------------


def mark_security_event(
    request: Request,
    event_type: str,
    *,
    severity: str = "medium",
) -> None:
    """
    Attach a security event to the current request.

    The actual persistence/alerting belongs to:
        backend/security/security_events.py
        backend/core/audit.py
    """

    events = getattr(
        request.state,
        "security_events",
        None,
    )

    if events is None:
        events = []
        request.state.security_events = events

    events.append(
        {
            "type": event_type,
            "severity": severity,
        }
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "SecurityMiddleware",
    "SecurityViolation",
    "apply_security_headers",
    "validate_http_method",
    "validate_content_length",
    "validate_content_type",
    "mark_security_event",
]