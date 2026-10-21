"""
Enterprise Legal Document Intelligence AI
------------------------------------------

Authentication API Routes

Responsibilities
----------------
- User login
- Token issuance
- Token refresh
- Current-user retrieval
- Logout / token revocation
- Password change
- Authentication health endpoint
- Secure error handling
- Tenant-aware authentication flow

Security principles
-------------------
- Never return passwords or password hashes.
- Never log credentials or access/refresh tokens.
- Authentication failures use generic messages.
- Token operations are delegated to auth_service.
- Authorization remains separate from authentication.
- Tenant context is established after successful authentication.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, Request, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from backend.app.core.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    InvalidTokenError,
    ValidationError,
)
from backend.app.core.logging import get_logger
from backend.app.core.telemetry import telemetry
from backend.app.dependencies import (
    get_auth_service,
    get_current_user,
)
from backend.app.services.auth_service import AuthService

logger = get_logger(__name__)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================================
# SCHEMAS
# ============================================================================


class LoginRequest(BaseModel):
    """Authentication credentials."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    email: str = Field(
        ...,
        min_length=3,
        max_length=320,
    )

    password: SecretStr = Field(
        ...,
        min_length=8,
        max_length=512,
    )

    remember_me: bool = False


class RefreshTokenRequest(BaseModel):
    """Refresh-token request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    refresh_token: SecretStr = Field(
        ...,
        min_length=20,
        max_length=4096,
    )


class LogoutRequest(BaseModel):
    """Optional token revocation request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    refresh_token: Optional[
        SecretStr
    ] = None


class ChangePasswordRequest(BaseModel):
    """Password-change request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    current_password: SecretStr = Field(
        ...,
        min_length=8,
        max_length=512,
    )

    new_password: SecretStr = Field(
        ...,
        min_length=12,
        max_length=512,
    )


class TokenResponse(BaseModel):
    """OAuth-style token response."""

    access_token: str

    refresh_token: Optional[str] = None

    token_type: str = "bearer"

    expires_in: int

    refresh_expires_in: Optional[int] = None

    user: Optional[
        dict[str, Any]
    ] = None


class UserResponse(BaseModel):
    """Safe authenticated-user representation."""

    id: str

    email: str

    name: Optional[str] = None

    role: Optional[str] = None

    organization_id: Optional[str] = None

    tenant_id: Optional[str] = None

    status: Optional[str] = None

    is_active: bool = True

    created_at: Optional[str] = None

    last_login_at: Optional[str] = None


class MessageResponse(BaseModel):
    """Simple API response."""

    message: str

    success: bool = True


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _extract_bearer_token(
    authorization: Optional[str],
) -> Optional[str]:
    """
    Extract a bearer token without logging it.
    """

    if not authorization:
        return None

    scheme, _, token = (
        authorization.partition(" ")
    )

    if (
        scheme.lower() != "bearer"
        or not token.strip()
    ):
        return None

    return token.strip()


def _serialize_user(
    user: Any,
) -> dict[str, Any]:
    """
    Convert an ORM/user object into a safe response.

    Password hashes, authentication secrets, MFA secrets,
    refresh tokens, and internal security fields are deliberately
    excluded.
    """

    if user is None:
        return {}

    def value(
        name: str,
        default: Any = None,
    ) -> Any:
        return getattr(
            user,
            name,
            default,
        )

    def enum_value(
        item: Any,
    ) -> Any:
        if item is None:
            return None

        return getattr(
            item,
            "value",
            item,
        )

    created_at = value(
        "created_at"
    )

    last_login_at = value(
        "last_login_at"
    )

    return {
        "id": str(
            value(
                "id",
                "",
            )
        ),
        "email": str(
            value(
                "email",
                "",
            )
        ),
        "name": (
            value("name")
            or value("full_name")
            or value("display_name")
        ),
        "role": enum_value(
            value("role")
        ),
        "organization_id": (
            _string_or_none(
                value(
                    "organization_id"
                )
            )
        ),
        "tenant_id": (
            _string_or_none(
                value("tenant_id")
                or value(
                    "organization_id"
                )
            )
        ),
        "status": enum_value(
            value("status")
        ),
        "is_active": bool(
            value(
                "is_active",
                True,
            )
        ),
        "created_at": (
            created_at.isoformat()
            if hasattr(
                created_at,
                "isoformat",
            )
            else (
                str(created_at)
                if created_at
                else None
            )
        ),
        "last_login_at": (
            last_login_at.isoformat()
            if hasattr(
                last_login_at,
                "isoformat",
            )
            else (
                str(last_login_at)
                if last_login_at
                else None
            )
        ),
    }


def _string_or_none(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    result = str(
        value
    ).strip()

    return result or None


def _token_response(
    token_data: Any,
    user: Any = None,
) -> TokenResponse:
    """
    Normalize different auth-service token payloads.
    """

    if hasattr(
        token_data,
        "model_dump",
    ):
        data = token_data.model_dump()

    elif hasattr(
        token_data,
        "dict",
    ):
        data = token_data.dict()

    elif isinstance(
        token_data,
        dict,
    ):
        data = dict(
            token_data
        )

    else:
        data = {
            "access_token": getattr(
                token_data,
                "access_token",
                None,
            ),
            "refresh_token": getattr(
                token_data,
                "refresh_token",
                None,
            ),
            "token_type": getattr(
                token_data,
                "token_type",
                "bearer",
            ),
            "expires_in": getattr(
                token_data,
                "expires_in",
                0,
            ),
            "refresh_expires_in": getattr(
                token_data,
                "refresh_expires_in",
                None,
            ),
        }

    access_token = data.get(
        "access_token"
    )

    if not access_token:
        raise AuthenticationError(
            "Authentication service did not return an access token."
        )

    return TokenResponse(
        access_token=str(
            access_token
        ),
        refresh_token=(
            str(
                data["refresh_token"]
            )
            if data.get(
                "refresh_token"
            )
            else None
        ),
        token_type=str(
            data.get(
                "token_type",
                "bearer",
            )
        ),
        expires_in=int(
            data.get(
                "expires_in",
                0,
            )
            or 0
        ),
        refresh_expires_in=(
            int(
                data[
                    "refresh_expires_in"
                ]
            )
            if data.get(
                "refresh_expires_in"
            )
            is not None
            else None
        ),
        user=(
            _serialize_user(user)
            if user is not None
            else None
        ),
    )


def _record_auth_metric(
    name: str,
) -> None:
    try:
        telemetry.increment(
            f"auth.{name}",
            1,
        )
    except Exception:
        logger.debug(
            "auth_telemetry_failed",
            exc_info=True,
        )


# ============================================================================
# LOGIN
# ============================================================================


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user",
    description=(
        "Authenticate a user and issue access and refresh tokens."
    ),
)
async def login(
    payload: LoginRequest,
    request: Request,
    auth_service: AuthService = Depends(
        get_auth_service
    ),
) -> TokenResponse:
    """
    Authenticate a user.

    Credentials are passed directly to the authentication service.
    They are never written to logs or telemetry.
    """

    client_ip = (
        request.client.host
        if request.client
        else None
    )

    try:
        result = await auth_service.authenticate(
            email=payload.email.lower(),
            password=payload.password.get_secret_value(),
            remember_me=payload.remember_me,
            ip_address=client_ip,
            user_agent=request.headers.get(
                "user-agent"
            ),
        )

        user = None

        if isinstance(
            result,
            tuple,
        ):
            token_data = result[0]
            if len(result) > 1:
                user = result[1]
        else:
            token_data = result

            if isinstance(
                result,
                dict,
            ):
                user = result.get(
                    "user"
                )

        _record_auth_metric(
            "login.success"
        )

        logger.info(
            "authentication_success",
            extra={
                "email_domain": (
                    payload.email.rsplit(
                        "@",
                        1,
                    )[-1]
                    if "@"
                    in payload.email
                    else None
                ),
                "client_ip": client_ip,
            },
        )

        return _token_response(
            token_data,
            user,
        )

    except (
        InvalidCredentialsError,
        AuthenticationError,
    ):
        _record_auth_metric(
            "login.failure"
        )

        logger.warning(
            "authentication_failed",
            extra={
                "client_ip": client_ip,
                "reason": "invalid_credentials",
            },
        )

        raise InvalidCredentialsError(
            "Invalid email or password."
        )

    except Exception:
        _record_auth_metric(
            "login.error"
        )

        logger.exception(
            "authentication_service_error"
        )

        raise AuthenticationError(
            "Authentication could not be completed."
        )


# ============================================================================
# REFRESH TOKEN
# ============================================================================


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
)
async def refresh_token(
    payload: RefreshTokenRequest,
    request: Request,
    auth_service: AuthService = Depends(
        get_auth_service
    ),
) -> TokenResponse:
    """
    Rotate/refresh an access token.

    The refresh token is never logged.
    """

    refresh_token_value = (
        payload.refresh_token
        .get_secret_value()
    )

    try:
        result = await auth_service.refresh_token(
            refresh_token=refresh_token_value,
            ip_address=(
                request.client.host
                if request.client
                else None
            ),
            user_agent=request.headers.get(
                "user-agent"
            ),
        )

        user = None

        if isinstance(
            result,
            tuple,
        ):
            token_data = result[0]
            if len(result) > 1:
                user = result[1]
        else:
            token_data = result

            if isinstance(
                result,
                dict,
            ):
                user = result.get(
                    "user"
                )

        _record_auth_metric(
            "refresh.success"
        )

        return _token_response(
            token_data,
            user,
        )

    except (
        InvalidTokenError,
        AuthenticationError,
    ):
        _record_auth_metric(
            "refresh.failure"
        )

        logger.warning(
            "refresh_token_rejected"
        )

        raise InvalidTokenError(
            "Invalid or expired refresh token."
        )

    except Exception:
        logger.exception(
            "refresh_token_service_error"
        )

        raise AuthenticationError(
            "Token refresh could not be completed."
        )


# ============================================================================
# CURRENT USER
# ============================================================================


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get authenticated user",
)
async def get_me(
    current_user: Any = Depends(
        get_current_user
    ),
) -> UserResponse:
    """
    Return the authenticated user's safe profile.
    """

    if current_user is None:
        raise AuthenticationError(
            "Authenticated user not found."
        )

    data = _serialize_user(
        current_user
    )

    return UserResponse(
        **data
    )


# ============================================================================
# VERIFY TOKEN
# ============================================================================


@router.get(
    "/verify",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify authentication",
)
async def verify_authentication(
    current_user: Any = Depends(
        get_current_user
    ),
) -> MessageResponse:
    """
    Lightweight authentication verification endpoint.
    """

    if current_user is None:
        raise AuthenticationError(
            "Authentication required."
        )

    return MessageResponse(
        message="Authentication is valid."
    )


# ============================================================================
# LOGOUT
# ============================================================================


@router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Logout authenticated user",
)
async def logout(
    payload: Optional[
        LogoutRequest
    ] = None,
    authorization: Optional[str] = Header(
        default=None
    ),
    current_user: Any = Depends(
        get_current_user
    ),
    auth_service: AuthService = Depends(
        get_auth_service
    ),
) -> MessageResponse:
    """
    Revoke the current access token and optionally
    revoke the supplied refresh token.

    Logout is intentionally idempotent from the client's perspective.
    """

    if current_user is None:
        raise AuthenticationError(
            "Authentication required."
        )

    access_token = (
        _extract_bearer_token(
            authorization
        )
    )

    refresh_token_value = None

    if payload and payload.refresh_token:
        refresh_token_value = (
            payload.refresh_token
            .get_secret_value()
        )

    try:
        if hasattr(
            auth_service,
            "logout",
        ):
            await auth_service.logout(
                user=current_user,
                access_token=access_token,
                refresh_token=refresh_token_value,
            )

        elif hasattr(
            auth_service,
            "revoke_token",
        ):
            if access_token:
                await auth_service.revoke_token(
                    access_token
                )

            if refresh_token_value:
                await auth_service.revoke_token(
                    refresh_token_value
                )

        _record_auth_metric(
            "logout.success"
        )

        return MessageResponse(
            message="Successfully logged out."
        )

    except Exception:
        logger.exception(
            "logout_service_error"
        )

        raise AuthenticationError(
            "Logout could not be completed."
        )


# ============================================================================
# CHANGE PASSWORD
# ============================================================================


@router.post(
    "/change-password",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Change password",
)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    auth_service: AuthService = Depends(
        get_auth_service
    ),
) -> MessageResponse:
    """
    Change the authenticated user's password.

    A successful password change should invalidate existing sessions
    where supported by the authentication service.
    """

    if current_user is None:
        raise AuthenticationError(
            "Authentication required."
        )

    if (
        payload.current_password.get_secret_value()
        == payload.new_password.get_secret_value()
    ):
        raise ValidationError(
            "New password must differ from current password."
        )

    try:
        await auth_service.change_password(
            user=current_user,
            current_password=(
                payload.current_password
                .get_secret_value()
            ),
            new_password=(
                payload.new_password
                .get_secret_value()
            ),
        )

        _record_auth_metric(
            "password_change.success"
        )

        return MessageResponse(
            message=(
                "Password changed successfully. "
                "Existing sessions may need to authenticate again."
            )
        )

    except (
        InvalidCredentialsError,
        AuthenticationError,
    ):
        _record_auth_metric(
            "password_change.failure"
        )

        raise InvalidCredentialsError(
            "Current password is incorrect."
        )

    except Exception:
        logger.exception(
            "password_change_service_error"
        )

        raise AuthenticationError(
            "Password change could not be completed."
        )


# ============================================================================
# AUTHENTICATION HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Authentication service health",
)
async def authentication_health(
    auth_service: AuthService = Depends(
        get_auth_service
    ),
) -> MessageResponse:
    """
    Authentication dependency health check.

    This endpoint intentionally does not expose provider credentials,
    database details, token configuration, or internal exceptions.
    """

    try:
        if hasattr(
            auth_service,
            "health_check",
        ):
            healthy = await auth_service.health_check()

            if healthy is False:
                raise AuthenticationError(
                    "Authentication service unavailable."
                )

        return MessageResponse(
            message="Authentication service is healthy."
        )

    except AuthenticationError:
        raise

    except Exception:
        logger.exception(
            "authentication_health_check_failed"
        )

        raise AuthenticationError(
            "Authentication service unavailable."
        )


__all__ = [
    "router",
    "LoginRequest",
    "RefreshTokenRequest",
    "LogoutRequest",
    "ChangePasswordRequest",
    "TokenResponse",
    "UserResponse",
    "MessageResponse",
]