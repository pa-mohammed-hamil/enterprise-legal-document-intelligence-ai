"""
Enterprise Legal Document Intelligence AI
------------------------------------------

Global API Error Handling Middleware

Responsibilities
----------------
- Normalize application exceptions into stable API responses.
- Handle FastAPI validation errors.
- Handle authentication/authorization failures.
- Handle unexpected exceptions without leaking internals.
- Preserve request/correlation IDs.
- Emit structured logs and telemetry.
- Prevent sensitive information from appearing in API responses.
- Support JSON API clients consistently.
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.app.core.exceptions import (
    AppException,
    AuthenticationError,
    AuthorizationError,
    ForbiddenError,
    InvalidCredentialsError,
    InvalidTokenError,
    NotFoundError,
    RateLimitError,
    TenantAccessError,
    ValidationError,
    exception_to_response,
    get_status_code,
)
from backend.app.core.logging import (
    get_logger,
    get_request_id,
    log_exception,
    set_correlation_id,
    set_request_id,
)
from backend.app.core.telemetry import telemetry


logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ERROR_CODE = "INTERNAL_SERVER_ERROR"

DEFAULT_ERROR_MESSAGE = (
    "An unexpected error occurred while processing the request."
)

VALIDATION_ERROR_CODE = "VALIDATION_ERROR"

HTTP_ERROR_CODE = "HTTP_ERROR"

REQUEST_ID_HEADER = "X-Request-ID"

CORRELATION_ID_HEADER = "X-Correlation-ID"


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _safe_string(
    value: Any,
    max_length: int = 500,
) -> Optional[str]:
    """
    Convert a value to a bounded safe string.

    Prevents accidental oversized exception messages from being
    returned to clients or written into telemetry.
    """

    if value is None:
        return None

    try:
        text = str(value).strip()
    except Exception:
        return None

    if not text:
        return None

    return text[:max_length]


def _request_id(
    request: Request,
) -> Optional[str]:
    """
    Resolve the request ID from the logging context or HTTP headers.
    """

    current = get_request_id()

    if current:
        return current

    return (
        request.headers.get(
            REQUEST_ID_HEADER
        )
        or request.headers.get(
            "X-Request-ID".lower()
        )
    )


def _correlation_id(
    request: Request,
) -> Optional[str]:
    """
    Resolve correlation ID from request headers.
    """

    return (
        request.headers.get(
            CORRELATION_ID_HEADER
        )
        or request.headers.get(
            "X-Correlation-ID".lower()
        )
    )


def _client_ip(
    request: Request,
) -> Optional[str]:
    """
    Resolve client IP without trusting arbitrary forwarding headers.

    A trusted reverse proxy can normalize request.client.host before
    the request reaches the application.
    """

    if request.client:
        return request.client.host

    return None


def _safe_path(
    request: Request,
) -> str:
    """
    Return the request path.

    Query parameters are deliberately excluded because they can contain
    tokens, passwords, signed URLs, or other secrets.
    """

    try:
        return request.url.path
    except Exception:
        return "<unknown>"


def _safe_method(
    request: Request,
) -> str:
    try:
        return request.method.upper()
    except Exception:
        return "UNKNOWN"


def _base_error_payload(
    *,
    code: str,
    message: str,
    request_id: Optional[str],
    details: Any = None,
) -> dict[str, Any]:
    """
    Build the stable API error envelope.
    """

    payload: dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }

    if details is not None:
        payload["error"]["details"] = details

    if request_id:
        payload["request_id"] = request_id

    return payload


def _sanitize_validation_errors(
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize Pydantic/FastAPI validation errors.

    Input values are deliberately removed because they can contain
    credentials, PII, or uploaded document content.
    """

    sanitized: list[dict[str, Any]] = []

    for error in errors:
        item: dict[str, Any] = {}

        location = error.get("loc")

        if location:
            item["field"] = [
                str(part)
                for part in location
            ]

        message = error.get("msg")

        if message:
            item["message"] = _safe_string(
                message,
                300,
            )

        error_type = error.get("type")

        if error_type:
            item["type"] = _safe_string(
                error_type,
                150,
            )

        sanitized.append(item)

    return sanitized


def _exception_code(
    exc: AppException,
) -> str:
    """
    Resolve the application's stable error code.
    """

    code = getattr(
        exc,
        "error_code",
        None,
    )

    if code:
        return str(code)

    return DEFAULT_ERROR_CODE


def _exception_message(
    exc: AppException,
) -> str:
    """
    Resolve a safe application exception message.
    """

    message = getattr(
        exc,
        "message",
        None,
    )

    if message:
        return _safe_string(
            message,
            1000,
        ) or DEFAULT_ERROR_MESSAGE

    default_message = getattr(
        exc,
        "default_message",
        None,
    )

    return (
        _safe_string(
            default_message,
            1000,
        )
        or DEFAULT_ERROR_MESSAGE
    )


def _exception_details(
    exc: AppException,
) -> Any:
    """
    Extract structured exception details.

    Details are returned only when the application exception explicitly
    exposes them. Internal traceback/exception objects are never returned.
    """

    details = getattr(
        exc,
        "details",
        None,
    )

    if details is None:
        return None

    if isinstance(
        details,
        (str, int, float, bool),
    ):
        return details

    if isinstance(
        details,
        (list, tuple),
    ):
        return [
            _safe_string(
                item,
                500,
            )
            for item in details
        ]

    if isinstance(
        details,
        dict,
    ):
        sanitized: dict[str, Any] = {}

        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "authorization",
            "cookie",
            "api_key",
            "apikey",
            "private_key",
            "credential",
            "credentials",
        }

        for key, value in details.items():
            normalized_key = str(
                key
            ).lower()

            if normalized_key in sensitive_keys:
                sanitized[str(key)] = "[REDACTED]"
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                sanitized[
                    str(key)
                ] = (
                    _safe_string(
                        value,
                        500,
                    )
                    if isinstance(
                        value,
                        str,
                    )
                    else value
                )
            else:
                sanitized[
                    str(key)
                ] = _safe_string(
                    value,
                    500,
                )

        return sanitized

    return _safe_string(
        details,
        500,
    )


def _is_security_exception(
    exc: BaseException,
) -> bool:
    """
    Determine whether an exception represents a security-sensitive
    authentication, authorization, tenant, or token failure.
    """

    return isinstance(
        exc,
        (
            AuthenticationError,
            AuthorizationError,
            ForbiddenError,
            InvalidCredentialsError,
            InvalidTokenError,
            TenantAccessError,
        ),
    )


def _status_for_app_exception(
    exc: AppException,
) -> int:
    """
    Resolve HTTP status code safely.
    """

    try:
        code = get_status_code(
            exc
        )

        if 400 <= code <= 599:
            return code

    except Exception:
        pass

    try:
        code = int(
            getattr(
                exc,
                "status_code",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        )

        if 400 <= code <= 599:
            return code

    except Exception:
        pass

    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _status_for_http_exception(
    exc: StarletteHTTPException,
) -> int:
    try:
        code = int(
            exc.status_code
        )

        if 400 <= code <= 599:
            return code

    except Exception:
        pass

    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _log_request_context(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    duration_ms: float,
    expected: bool,
) -> None:
    """
    Write structured error telemetry without sensitive request data.
    """

    path = _safe_path(
        request
    )

    method = _safe_method(
        request
    )

    request_id = _request_id(
        request
    )

    client_ip = _client_ip(
        request
    )

    try:
        telemetry.increment(
            "http.errors.total",
            1,
        )

        telemetry.increment(
            f"http.errors.status_{status_code}",
            1,
        )

        telemetry.increment(
            f"http.errors.code_{error_code}",
            1,
        )

        telemetry.observe(
            "http.error.duration_ms",
            duration_ms,
        )

    except Exception:
        logger.debug(
            "error_telemetry_failed",
            exc_info=True,
        )

    logger.warning(
        "api_request_error",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "error_code": error_code,
            "request_id": request_id,
            "client_ip": client_ip,
            "duration_ms": round(
                duration_ms,
                2,
            ),
            "expected": expected,
            "security_exception": _is_security_exception(
                Exception()
            )
            if False
            else False,
        },
    )


# ============================================================================
# EXCEPTION BUILDERS
# ============================================================================


def _handle_app_exception(
    exc: AppException,
    request: Request,
) -> JSONResponse:
    """
    Convert an application exception into a JSON response.
    """

    status_code = _status_for_app_exception(
        exc
    )

    request_id = _request_id(
        request
    )

    error_code = _exception_code(
        exc
    )

    message = _exception_message(
        exc
    )

    details = _exception_details(
        exc
    )

    payload = _base_error_payload(
        code=error_code,
        message=message,
        request_id=request_id,
        details=details,
    )

    headers = getattr(
        exc,
        "headers",
        None,
    )

    response_headers = dict(
        headers
        or {}
    )

    if request_id:
        response_headers.setdefault(
            REQUEST_ID_HEADER,
            request_id,
        )

    if isinstance(
        exc,
        RateLimitError,
    ):
        response_headers.setdefault(
            "Retry-After",
            "60",
        )

    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=response_headers,
    )


def _handle_validation_exception(
    exc: RequestValidationError,
    request: Request,
) -> JSONResponse:
    """
    Convert FastAPI request validation errors into a stable response.
    """

    request_id = _request_id(
        request
    )

    errors = _sanitize_validation_errors(
        exc.errors()
    )

    payload = _base_error_payload(
        code=VALIDATION_ERROR_CODE,
        message=(
            "The request contains invalid or missing fields."
        ),
        request_id=request_id,
        details={
            "fields": errors,
        },
    )

    headers: dict[str, str] = {}

    if request_id:
        headers[
            REQUEST_ID_HEADER
        ] = request_id

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=payload,
        headers=headers,
    )


def _handle_http_exception(
    exc: StarletteHTTPException,
    request: Request,
) -> JSONResponse:
    """
    Normalize Starlette/FastAPI HTTP exceptions.
    """

    request_id = _request_id(
        request
    )

    status_code = _status_for_http_exception(
        exc
    )

    # Do not expose potentially sensitive exception objects.
    if isinstance(
        exc.detail,
        str,
    ):
        message = exc.detail
    else:
        message = (
            "The request could not be completed."
        )

    if status_code == 401:
        code = "UNAUTHORIZED"

    elif status_code == 403:
        code = "FORBIDDEN"

    elif status_code == 404:
        code = "NOT_FOUND"

    elif status_code == 405:
        code = "METHOD_NOT_ALLOWED"

    elif status_code == 409:
        code = "CONFLICT"

    elif status_code == 429:
        code = "RATE_LIMITED"

    elif 400 <= status_code < 500:
        code = "HTTP_CLIENT_ERROR"

    else:
        code = HTTP_ERROR_CODE

    payload = _base_error_payload(
        code=code,
        message=(
            _safe_string(
                message,
                500,
            )
            or "Request failed."
        ),
        request_id=request_id,
    )

    headers = dict(
        getattr(
            exc,
            "headers",
            None,
        )
        or {}
    )

    if request_id:
        headers.setdefault(
            REQUEST_ID_HEADER,
            request_id,
        )

    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=headers,
    )


def _handle_unexpected_exception(
    exc: Exception,
    request: Request,
) -> JSONResponse:
    """
    Handle an unexpected exception.

    Internal exception messages, stack traces, SQL statements,
    provider responses, filesystem paths, and credentials are never
    returned to the client.
    """

    request_id = _request_id(
        request
    )

    logger.exception(
        "unhandled_api_exception",
        extra={
            "request_id": request_id,
            "method": _safe_method(
                request
            ),
            "path": _safe_path(
                request
            ),
        },
    )

    try:
        telemetry.increment(
            "http.errors.unhandled",
            1,
        )
    except Exception:
        logger.debug(
            "unhandled_error_telemetry_failed",
            exc_info=True,
        )

    payload = _base_error_payload(
        code=DEFAULT_ERROR_CODE,
        message=DEFAULT_ERROR_MESSAGE,
        request_id=request_id,
    )

    headers: dict[str, str] = {}

    if request_id:
        headers[
            REQUEST_ID_HEADER
        ] = request_id

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload,
        headers=headers,
    )


# ============================================================================
# REQUEST ERROR MIDDLEWARE
# ============================================================================


class ErrorHandlerMiddleware(
    BaseHTTPMiddleware
):
    """
    Global exception middleware.

    The middleware catches errors that escape route handlers and
    dependencies, converts them to the application's standard
    response format, and records safe telemetry.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        include_debug_details: bool = False,
    ) -> None:
        super().__init__(
            app
        )

        self.include_debug_details = (
            include_debug_details
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Any],
        ],
    ) -> Any:
        started_at = time.perf_counter()

        request_id = (
            request.headers.get(
                REQUEST_ID_HEADER
            )
            or request.headers.get(
                "X-Request-ID".lower()
            )
        )

        correlation_id = (
            request.headers.get(
                CORRELATION_ID_HEADER
            )
            or request.headers.get(
                "X-Correlation-ID".lower()
            )
        )

        if request_id:
            set_request_id(
                request_id
            )

        if correlation_id:
            set_correlation_id(
                correlation_id
            )

        try:
            response = await call_next(
                request
            )

            return response

        except RequestValidationError as exc:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000

            response = _handle_validation_exception(
                exc,
                request,
            )

            _record_response_error(
                request,
                response.status_code,
                VALIDATION_ERROR_CODE,
                duration_ms,
            )

            return response

        except AppException as exc:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000

            response = _handle_app_exception(
                exc,
                request,
            )

            _record_response_error(
                request,
                response.status_code,
                _exception_code(exc),
                duration_ms,
            )

            return response

        except StarletteHTTPException as exc:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000

            response = _handle_http_exception(
                exc,
                request,
            )

            _record_response_error(
                request,
                response.status_code,
                "HTTP_ERROR",
                duration_ms,
            )

            return response

        except Exception as exc:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000

            response = _handle_unexpected_exception(
                exc,
                request,
            )

            _record_response_error(
                request,
                response.status_code,
                DEFAULT_ERROR_CODE,
                duration_ms,
            )

            return response


def _record_response_error(
    request: Request,
    status_code: int,
    error_code: str,
    duration_ms: float,
) -> None:
    """
    Record safe error telemetry and structured logs.
    """

    request_id = _request_id(
        request
    )

    try:
        telemetry.increment(
            "api.errors.total",
            1,
        )

        telemetry.increment(
            f"api.errors.{error_code.lower()}",
            1,
        )

        telemetry.observe(
            "api.errors.duration_ms",
            duration_ms,
        )

    except Exception:
        logger.debug(
            "api_error_telemetry_failed",
            exc_info=True,
        )

    logger.warning(
        "api_error_response",
        extra={
            "method": _safe_method(
                request
            ),
            "path": _safe_path(
                request
            ),
            "status_code": status_code,
            "error_code": error_code,
            "request_id": request_id,
            "duration_ms": round(
                duration_ms,
                2,
            ),
        },
    )


# ============================================================================
# FASTAPI EXCEPTION HANDLERS
# ============================================================================


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """
    FastAPI handler for application exceptions.
    """

    return _handle_app_exception(
        exc,
        request,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    FastAPI handler for request validation errors.
    """

    return _handle_validation_exception(
        exc,
        request,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """
    FastAPI/Starlette HTTP exception handler.
    """

    return _handle_http_exception(
        exc,
        request,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Final fallback exception handler.
    """

    return _handle_unexpected_exception(
        exc,
        request,
    )


# ============================================================================
# REGISTRATION
# ============================================================================


def register_exception_handlers(
    app: Any,
) -> Any:
    """
    Register global FastAPI exception handlers.

    Usage
    -----
        app = FastAPI()
        register_exception_handlers(app)
    """

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        AppException,
        app_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )

    return app


def add_error_handler_middleware(
    app: Any,
    *,
    include_debug_details: bool = False,
) -> Any:
    """
    Add middleware-based global error handling.

    Use this when you want errors escaping the FastAPI exception
    handling layer to be normalized as well.
    """

    app.add_middleware(
        ErrorHandlerMiddleware,
        include_debug_details=include_debug_details,
    )

    return app


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ErrorHandlerMiddleware",
    "register_exception_handlers",
    "add_error_handler_middleware",
    "app_exception_handler",
    "validation_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",
]