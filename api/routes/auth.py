# backend/api/routes/auth.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# DEPENDENCIES
# ============================================================

async def get_auth_service() -> Any:
    """
    Replace with:

        from backend.dependencies import get_auth_service

    Expected AuthService operations:

        authenticate(...)
        register(...)
        refresh_tokens(...)
        logout(...)
        logout_all(...)
        get_current_user(...)
        change_password(...)
        request_password_reset(...)
        reset_password(...)
        request_email_verification(...)
        verify_email(...)
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Auth service dependency is not configured.",
    )


async def get_current_user() -> Any:
    """
    Replace with the application's JWT authentication dependency.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class LoginRequest(BaseModel):
    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=1024,
    )

    organization_id: Optional[str] = None

    remember_me: bool = False


class RegisterRequest(BaseModel):
    email: EmailStr

    password: str = Field(
        min_length=12,
        max_length=1024,
    )

    first_name: str = Field(
        min_length=1,
        max_length=100,
    )

    last_name: str = Field(
        min_length=1,
        max_length=100,
    )

    organization_id: Optional[str] = None

    invitation_token: Optional[str] = Field(
        default=None,
        max_length=2048,
    )

    @field_validator("password")
    @classmethod
    def validate_password(
        cls,
        value: str,
    ) -> str:

        if not any(
            char.isupper()
            for char in value
        ):
            raise ValueError(
                "Password must contain an uppercase letter."
            )

        if not any(
            char.islower()
            for char in value
        ):
            raise ValueError(
                "Password must contain a lowercase letter."
            )

        if not any(
            char.isdigit()
            for char in value
        ):
            raise ValueError(
                "Password must contain a number."
            )

        return value


class RefreshTokenRequest(BaseModel):
    refresh_token: Optional[str] = Field(
        default=None,
        min_length=10,
        max_length=8192,
    )


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = Field(
        default=None,
        max_length=8192,
    )

    all_sessions: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(
        min_length=1,
        max_length=1024,
    )

    new_password: str = Field(
        min_length=12,
        max_length=1024,
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password(
        cls,
        value: str,
    ) -> str:

        if not any(
            char.isupper()
            for char in value
        ):
            raise ValueError(
                "Password must contain an uppercase letter."
            )

        if not any(
            char.islower()
            for char in value
        ):
            raise ValueError(
                "Password must contain a lowercase letter."
            )

        if not any(
            char.isdigit()
            for char in value
        ):
            raise ValueError(
                "Password must contain a number."
            )

        return value


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(
        min_length=10,
        max_length=4096,
    )

    new_password: str = Field(
        min_length=12,
        max_length=1024,
    )


class VerifyEmailRequest(BaseModel):
    token: str = Field(
        min_length=10,
        max_length=4096,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class UserResponse(BaseModel):
    id: str

    tenant_id: str

    organization_id: Optional[str] = None

    email: str

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None

    role: Optional[str] = None
    roles: list[str] = Field(
        default_factory=list
    )

    status: Optional[str] = None

    is_active: bool = True
    is_admin: bool = False

    email_verified: bool = False

    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    access_token: str

    refresh_token: Optional[str] = None

    token_type: str = "bearer"

    expires_in: int

    refresh_expires_in: Optional[int] = None

    session_id: Optional[str] = None

    user: Optional[UserResponse] = None


class MessageResponse(BaseModel):
    message: str


class SessionResponse(BaseModel):
    authenticated: bool

    user: UserResponse


# ============================================================
# HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(
        obj,
        dict,
    ):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def _enum_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


def _roles(
    user: Any,
) -> list[str]:

    result: list[str] = []

    primary = _attr(
        user,
        "role",
    )

    if primary:
        result.append(
            str(
                _enum_value(primary)
            )
        )

    roles = _attr(
        user,
        "roles",
        [],
    )

    for role in roles or []:

        if isinstance(
            role,
            str,
        ):
            name = role

        else:
            name = (
                _attr(
                    role,
                    "name",
                )
                or _attr(
                    role,
                    "role",
                )
            )

        if (
            name
            and str(name)
            not in result
        ):
            result.append(
                str(name)
            )

    return result


def _serialize_user(
    user: Any,
) -> dict[str, Any]:

    if user is None:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="User is unavailable.",
        )

    return {
        "id":
            str(
                _attr(
                    user,
                    "id",
                    "",
                )
            ),

        "tenant_id":
            str(
                _attr(
                    user,
                    "tenant_id",
                    "",
                )
            ),

        "organization_id":
            (
                str(
                    _attr(
                        user,
                        "organization_id",
                    )
                )
                if _attr(
                    user,
                    "organization_id",
                )
                else None
            ),

        "email":
            str(
                _attr(
                    user,
                    "email",
                    "",
                )
            ),

        "first_name":
            _attr(
                user,
                "first_name",
            ),

        "last_name":
            _attr(
                user,
                "last_name",
            ),

        "display_name":
            _attr(
                user,
                "display_name",
            ),

        "role":
            (
                str(
                    _enum_value(
                        _attr(
                            user,
                            "role",
                        )
                    )
                )
                if _attr(
                    user,
                    "role",
                )
                else None
            ),

        "roles":
            _roles(user),

        "status":
            (
                str(
                    _enum_value(
                        _attr(
                            user,
                            "status",
                        )
                    )
                )
                if _attr(
                    user,
                    "status",
                )
                else None
            ),

        "is_active":
            bool(
                _attr(
                    user,
                    "is_active",
                    True,
                )
            ),

        "is_admin":
            bool(
                _attr(
                    user,
                    "is_admin",
                    False,
                )
            ),

        "email_verified":
            bool(
                _attr(
                    user,
                    "email_verified",
                    False,
                )
            ),

        "created_at":
            _attr(
                user,
                "created_at",
            ),

        "last_login_at":
            _attr(
                user,
                "last_login_at",
            ),
    }


def _request_id(
    request: Request,
) -> Optional[str]:

    return (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )


def _client_ip(
    request: Request,
) -> Optional[str]:
    """
    Do not blindly trust X-Forwarded-For.

    Your reverse proxy / trusted-proxy middleware should resolve
    the canonical client IP before this route if proxy support is
    required.
    """

    if request.client is None:
        return None

    return request.client.host


def _user_agent(
    request: Request,
) -> Optional[str]:

    value = request.headers.get(
        "User-Agent"
    )

    if value:
        return value[:1000]

    return None


def _extract_refresh_token(
    request: Request,
    payload_token: Optional[str],
) -> Optional[str]:

    if payload_token:
        return payload_token

    return request.cookies.get(
        "refresh_token"
    )


def _token_response(
    result: Any,
) -> dict[str, Any]:

    access_token = _attr(
        result,
        "access_token",
    )

    if not access_token:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Authentication service returned "
                "an invalid token response."
            ),
        )

    refresh_token = _attr(
        result,
        "refresh_token",
    )

    expires_in = int(
        _attr(
            result,
            "expires_in",
            0,
        )
        or 0
    )

    refresh_expires_in = _attr(
        result,
        "refresh_expires_in",
    )

    session_id = _attr(
        result,
        "session_id",
    )

    user = _attr(
        result,
        "user",
    )

    return {
        "access_token":
            str(access_token),

        "refresh_token":
            (
                str(refresh_token)
                if refresh_token
                else None
            ),

        "token_type":
            str(
                _attr(
                    result,
                    "token_type",
                    "bearer",
                )
            ),

        "expires_in":
            expires_in,

        "refresh_expires_in":
            (
                int(refresh_expires_in)
                if refresh_expires_in
                is not None
                else None
            ),

        "session_id":
            (
                str(session_id)
                if session_id
                else None
            ),

        "user":
            (
                _serialize_user(user)
                if user
                else None
            ),
    }


def _set_refresh_cookie(
    response: Response,
    refresh_token: Optional[str],
    *,
    max_age: Optional[int] = None,
) -> None:

    if not refresh_token:
        return

    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        max_age=max_age,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/api/v1/auth",
    )


def _clear_refresh_cookie(
    response: Response,
) -> None:

    response.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth",
        httponly=True,
        secure=True,
        samesite="lax",
    )


async def _invoke(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=(
                status.HTTP_501_NOT_IMPLEMENTED
            ),
            detail=(
                f"Auth operation "
                f"'{method_name}' is not configured."
            ),
        )

    return await method(**kwargs)


def _authentication_error(
    exc: Exception,
) -> HTTPException:
    """
    Keep login errors intentionally generic.

    Do not reveal whether:
      - email exists
      - password was wrong
      - account was disabled
    """

    return HTTPException(
        status_code=(
            status.HTTP_401_UNAUTHORIZED
        ),
        detail="Invalid credentials.",
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user",
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, Any]:

    try:
        result = await _invoke(
            service,
            "authenticate",

            email=str(
                payload.email
            ),

            password=(
                payload.password
            ),

            organization_id=(
                payload.organization_id
            ),

            remember_me=(
                payload.remember_me
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),

            user_agent=(
                _user_agent(request)
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _authentication_error(
            exc
        ) from exc

    token_data = _token_response(
        result
    )

    refresh_token = (
        token_data.get(
            "refresh_token"
        )
    )

    refresh_max_age = (
        token_data.get(
            "refresh_expires_in"
        )
    )

    _set_refresh_cookie(
        response,
        refresh_token,
        max_age=refresh_max_age,
    )

    return token_data


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=(
        status.HTTP_201_CREATED
    ),
    summary="Register user",
)
async def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, Any]:

    try:
        result = await _invoke(
            service,
            "register",

            email=str(
                payload.email
            ),

            password=(
                payload.password
            ),

            first_name=(
                payload.first_name.strip()
            ),

            last_name=(
                payload.last_name.strip()
            ),

            organization_id=(
                payload.organization_id
            ),

            invitation_token=(
                payload.invitation_token
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),

            user_agent=(
                _user_agent(request)
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    token_data = _token_response(
        result
    )

    _set_refresh_cookie(
        response,
        token_data.get(
            "refresh_token"
        ),
        max_age=(
            token_data.get(
                "refresh_expires_in"
            )
        ),
    )

    return token_data


# ============================================================
# REFRESH TOKEN
# ============================================================

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    request: Request,
    response: Response,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, Any]:

    refresh_token = (
        _extract_refresh_token(
            request,
            payload.refresh_token,
        )
    )

    if not refresh_token:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Refresh token is required."
            ),
        )

    try:
        result = await _invoke(
            service,
            "refresh_tokens",

            refresh_token=(
                refresh_token
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),

            user_agent=(
                _user_agent(request)
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        _clear_refresh_cookie(
            response
        )

        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Invalid or expired "
                "refresh token."
            ),
        ) from exc

    token_data = _token_response(
        result
    )

    _set_refresh_cookie(
        response,
        token_data.get(
            "refresh_token"
        ),
        max_age=(
            token_data.get(
                "refresh_expires_in"
            )
        ),
    )

    return token_data


# ============================================================
# LOGOUT
# ============================================================

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout current session",
)
async def logout(
    payload: LogoutRequest,
    request: Request,
    response: Response,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:

    refresh_token = (
        _extract_refresh_token(
            request,
            payload.refresh_token,
        )
    )

    user_id = _attr(
        current_user,
        "id",
    )

    tenant_id = _attr(
        current_user,
        "tenant_id",
    )

    if payload.all_sessions:

        await _invoke(
            service,
            "logout_all",

            user_id=str(user_id),

            tenant_id=str(
                tenant_id
            ),

            request_id=(
                _request_id(request)
            ),
        )

    else:

        await _invoke(
            service,
            "logout",

            user_id=str(user_id),

            tenant_id=str(
                tenant_id
            ),

            refresh_token=(
                refresh_token
            ),

            request_id=(
                _request_id(request)
            ),
        )

    _clear_refresh_cookie(
        response
    )

    return {
        "message":
            (
                "All sessions logged out."
                if payload.all_sessions
                else "Logged out successfully."
            )
    }


# ============================================================
# CURRENT USER
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
)
async def get_me(
    current_user: Any = Depends(
        get_current_user
    ),
) -> dict[str, Any]:

    return _serialize_user(
        current_user
    )


# ============================================================
# SESSION
# ============================================================

@router.get(
    "/session",
    response_model=SessionResponse,
    summary="Check authentication session",
)
async def get_session(
    current_user: Any = Depends(
        get_current_user
    ),
) -> dict[str, Any]:

    return {
        "authenticated": True,
        "user": _serialize_user(
            current_user
        ),
    }


# ============================================================
# CHANGE PASSWORD
# ============================================================

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:

    if (
        payload.current_password
        == payload.new_password
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "New password must be "
                "different from the current password."
            ),
        )

    await _invoke(
        service,
        "change_password",

        user_id=str(
            _attr(
                current_user,
                "id",
            )
        ),

        tenant_id=str(
            _attr(
                current_user,
                "tenant_id",
            )
        ),

        current_password=(
            payload.current_password
        ),

        new_password=(
            payload.new_password
        ),

        request_id=(
            _request_id(request)
        ),

        ip_address=(
            _client_ip(request)
        ),
    )

    return {
        "message":
            "Password changed successfully."
    }


# ============================================================
# FORGOT PASSWORD
# ============================================================

@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request password reset",
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:
    """
    Always return the same response.

    This prevents email-account enumeration.
    """

    try:
        await _invoke(
            service,
            "request_password_reset",

            email=str(
                payload.email
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),

            user_agent=(
                _user_agent(request)
            ),
        )

    except Exception:
        # Intentionally hide whether the account exists.
        pass

    return {
        "message": (
            "If an account exists for that email, "
            "password reset instructions will be sent."
        )
    }


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password",
)
async def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    response: Response,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:

    try:
        await _invoke(
            service,
            "reset_password",

            token=payload.token,

            new_password=(
                payload.new_password
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Password reset token is "
                "invalid or expired."
            ),
        ) from exc

    _clear_refresh_cookie(
        response
    )

    return {
        "message":
            "Password reset successfully."
    }


# ============================================================
# REQUEST EMAIL VERIFICATION
# ============================================================

@router.post(
    "/email-verification/request",
    response_model=MessageResponse,
    summary="Request email verification",
)
async def request_email_verification(
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:

    if bool(
        _attr(
            current_user,
            "email_verified",
            False,
        )
    ):
        return {
            "message":
                "Email is already verified."
        }

    await _invoke(
        service,
        "request_email_verification",

        user_id=str(
            _attr(
                current_user,
                "id",
            )
        ),

        tenant_id=str(
            _attr(
                current_user,
                "tenant_id",
            )
        ),

        request_id=(
            _request_id(request)
        ),
    )

    return {
        "message":
            "Verification email requested."
    }


# ============================================================
# VERIFY EMAIL
# ============================================================

@router.post(
    "/email-verification/verify",
    response_model=MessageResponse,
    summary="Verify email",
)
async def verify_email(
    payload: VerifyEmailRequest,
    request: Request,

    service: Any = Depends(
        get_auth_service
    ),
) -> dict[str, str]:

    try:
        await _invoke(
            service,
            "verify_email",

            token=payload.token,

            request_id=(
                _request_id(request)
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Email verification token "
                "is invalid or expired."
            ),
        ) from exc

    return {
        "message":
            "Email verified successfully."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    summary="Authentication subsystem health",
)
async def auth_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }