"""
Request ID middleware.

Responsibilities:
- Generate a unique request ID for every HTTP request.
- Accept a valid incoming request ID for distributed tracing.
- Store the ID in request.state.
- Return the ID in the response headers.
- Make the ID available to logging/observability layers.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_ID_LENGTH = 128

# Safe characters for externally supplied IDs.
REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:/-]+$"
)


# ---------------------------------------------------------------------------
# Request ID helpers
# ---------------------------------------------------------------------------


def generate_request_id() -> str:
    """
    Generate a cryptographically strong UUID-based request ID.
    """
    return str(uuid.uuid4())


def is_valid_request_id(request_id: str) -> bool:
    """
    Validate an externally supplied request ID.

    This prevents arbitrary control characters or excessively large values
    from entering logs and tracing systems.
    """

    if not request_id:
        return False

    if len(request_id) > MAX_REQUEST_ID_LENGTH:
        return False

    return bool(REQUEST_ID_PATTERN.fullmatch(request_id))


def get_request_id(request: Request) -> str:
    """
    Retrieve the request ID from request state.

    Falls back to generating one if middleware has not initialized it.
    """

    request_id = getattr(request.state, "request_id", None)

    if request_id:
        return request_id

    request_id = generate_request_id()
    request.state.request_id = request_id

    return request_id


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add request IDs to every HTTP request.

    Flow:

        Client
          │
          │ X-Request-ID (optional)
          ▼
        Middleware
          │
          ├── validate incoming ID
          │
          ├── generate ID if absent/invalid
          │
          ▼
        request.state.request_id
          │
          ├── application logs
          ├── database audit
          ├── RAG tracing
          ├── agent tracing
          └── response
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:

        incoming_request_id = request.headers.get(
            REQUEST_ID_HEADER
        )

        if is_valid_request_id(incoming_request_id or ""):
            request_id = incoming_request_id
        else:
            request_id = generate_request_id()

        # Make request ID available throughout the request lifecycle.
        request.state.request_id = request_id

        # Useful for downstream middleware/services.
        request.state.correlation_id = request_id

        try:
            response = await call_next(request)

        except Exception:
            # The global error handler is responsible for formatting
            # the exception response. We intentionally re-raise here.
            raise

        # Always expose the request ID to the client.
        response.headers[REQUEST_ID_HEADER] = request_id

        return response


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def request_id_dependency(request: Request) -> str:
    """
    FastAPI dependency for endpoints/services that need the request ID.

    Example:

        @router.get("/health")
        async def health(
            request_id: str = Depends(request_id_dependency),
        ):
            ...
    """

    return get_request_id(request)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "generate_request_id",
    "is_valid_request_id",
    "get_request_id",
    "request_id_dependency",
]