# backend/api/routes/users.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.db.repositories.user_repository import (
    UserCreate,
    UserFilter,
    UserRepository,
    UserUpdate,
)


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies.py
# ============================================================

async def get_user_repository() -> UserRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="User repository dependency is not configured.",
    )


async def get_user_service() -> Any:
    """
    Recommended UserService dependencies:

        UserRepository
        OrganizationRepository
        AuditRepository
        NotificationRepository
        AuthService / IdentityProvider

    UserService should own:
        - authorization
        - organization membership validation
        - role changes
        - activation / suspension
        - invitation workflows
        - audit logging
        - notification dispatch
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="User service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class UserCreateRequest(BaseModel):
    email: EmailStr

    first_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )
    last_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )
    display_name: Optional[str] = Field(
        default=None,
        max_length=300,
    )

    organization_id: Optional[UUID] = None

    status: str = Field(
        default="active",
        max_length=50,
    )

    role: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    job_title: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    department: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    phone: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    locale: Optional[str] = Field(
        default="en",
        max_length=50,
    )

    timezone_name: Optional[str] = Field(
        default="UTC",
        max_length=100,
    )

    email_verified: bool = False

    is_active: bool = True
    is_admin: bool = False

    external_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    identity_provider: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    source: str = Field(
        default="admin",
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("email")
    @classmethod
    def normalize_email(
        cls,
        value: EmailStr,
    ) -> str:
        return str(value).strip().lower()


class UserUpdateRequest(BaseModel):
    """
    model_dump(exclude_unset=True) preserves the distinction
    between:

        field omitted
        field explicitly supplied as null

    This maps naturally to the repository's UNSET approach.
    """

    first_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    last_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    display_name: Optional[str] = Field(
        default=None,
        max_length=300,
    )

    job_title: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    department: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    phone: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    locale: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    timezone_name: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    metadata: Optional[dict[str, Any]] = None


class UserAdminUpdateRequest(UserUpdateRequest):
    organization_id: Optional[UUID] = None

    status: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    role: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


class UserEmailUpdateRequest(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def normalize_email(
        cls,
        value: EmailStr,
    ) -> str:
        return str(value).strip().lower()


class UserRoleRequest(BaseModel):
    role: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class UserSuspendRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class UserDeactivateRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class UserVerifyEmailRequest(BaseModel):
    verified: bool = True


class UserInviteRequest(BaseModel):
    email: EmailStr

    organization_id: Optional[UUID] = None

    role: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    first_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    last_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    @field_validator("email")
    @classmethod
    def normalize_email(
        cls,
        value: EmailStr,
    ) -> str:
        return str(value).strip().lower()


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

    status: str

    role: Optional[str] = None

    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None

    locale: Optional[str] = None
    timezone_name: Optional[str] = None

    email_verified: bool = False

    is_active: bool = True
    is_admin: bool = False

    external_id: Optional[str] = None
    identity_provider: Optional[str] = None

    source: Optional[str] = None

    last_login_at: Optional[datetime] = None

    failed_login_count: int = 0

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserPageResponse(BaseModel):
    items: list[UserResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class UserStatisticsResponse(BaseModel):
    total: int = 0

    active: int = 0
    inactive: int = 0
    suspended: int = 0

    admins: int = 0

    verified: int = 0
    unverified: int = 0

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_role: dict[str, int] = Field(
        default_factory=dict,
    )

    by_department: dict[str, int] = Field(
        default_factory=dict,
    )

    by_source: dict[str, int] = Field(
        default_factory=dict,
    )


class UserSessionSummaryResponse(BaseModel):
    user_id: str
    email: str

    tenant_id: str
    organization_id: Optional[str] = None

    roles: list[str]

    is_active: bool
    is_admin: bool

    email_verified: bool


class MessageResponse(BaseModel):
    message: str


# ============================================================
# HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
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


def _optional_str(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    return str(value)


def _tenant_id(
    user: Any,
) -> str:
    value = _attr(
        user,
        "tenant_id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    return str(value)


def _organization_id(
    user: Any,
) -> Optional[str]:
    return _optional_str(
        _attr(
            user,
            "organization_id",
        )
    )


def _user_id(
    user: Any,
) -> str:
    value = _attr(
        user,
        "id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return str(value)


def _roles(
    user: Any,
) -> set[str]:
    values = (
        _attr(
            user,
            "roles",
            [],
        )
        or []
    )

    if isinstance(values, str):
        values = [values]

    role = _attr(
        user,
        "role",
    )

    if role:
        values = [
            *values,
            role,
        ]

    return {
        str(
            _enum_value(value)
        ).lower()
        for value in values
        if value is not None
    }


def _require_user_admin(
    user: Any,
) -> None:
    allowed = {
        "admin",
        "super_admin",
        "organization_admin",
        "org_admin",
        "user_admin",
        "legal_admin",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User administration permission is required.",
        )


def _require_super_admin(
    user: Any,
) -> None:
    if "super_admin" not in _roles(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super administrator permission is required.",
        )


def _request_id(
    request: Request,
) -> Optional[str]:
    return _optional_str(
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )


def _metadata(
    user: Any,
) -> dict[str, Any]:
    return (
        _attr(
            user,
            "user_metadata",
        )
        or _attr(
            user,
            "metadata_json",
        )
        or {}
    )


def _serialize_user(
    user: Any,
) -> dict[str, Any]:
    return {
        "id": str(
            _attr(
                user,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                user,
                "tenant_id",
                "",
            )
        ),

        "organization_id":
            _optional_str(
                _attr(
                    user,
                    "organization_id",
                )
            ),

        "email": str(
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

        "status": str(
            _enum_value(
                _attr(
                    user,
                    "status",
                    "active",
                )
            )
        ),

        "role":
            _optional_str(
                _enum_value(
                    _attr(
                        user,
                        "role",
                    )
                )
            ),

        "job_title":
            _attr(
                user,
                "job_title",
            ),

        "department":
            _attr(
                user,
                "department",
            ),

        "phone":
            _attr(
                user,
                "phone",
            ),

        "locale":
            _attr(
                user,
                "locale",
            ),

        "timezone_name":
            _attr(
                user,
                "timezone_name",
            ),

        "email_verified":
            bool(
                _attr(
                    user,
                    "email_verified",
                    False,
                )
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

        "external_id":
            _optional_str(
                _attr(
                    user,
                    "external_id",
                )
            ),

        "identity_provider":
            _optional_str(
                _attr(
                    user,
                    "identity_provider",
                )
            ),

        "source":
            _optional_str(
                _attr(
                    user,
                    "source",
                )
            ),

        "last_login_at":
            _attr(
                user,
                "last_login_at",
            ),

        "failed_login_count":
            int(
                _attr(
                    user,
                    "failed_login_count",
                    0,
                )
                or 0
            ),

        "metadata":
            _metadata(user),

        "created_at":
            _attr(
                user,
                "created_at",
            ),

        "updated_at":
            _attr(
                user,
                "updated_at",
            ),
    }


def _serialize_page(
    page: Any,
) -> dict[str, Any]:
    items = list(
        _attr(
            page,
            "items",
            [],
        )
        or []
    )

    total = int(
        _attr(
            page,
            "total",
            len(items),
        )
        or 0
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
        or len(items)
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
        or 0
    )

    return {
        "items": [
            _serialize_user(item)
            for item in items
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": bool(
            _attr(
                page,
                "has_more",
                offset + len(items) < total,
            )
        ),
    }


def _set_filter(
    filters: Any,
    field: str,
    value: Any,
) -> None:
    if value is None:
        return

    if hasattr(filters, field):
        setattr(
            filters,
            field,
            value,
        )


async def _service_call(
    service: Any,
    operation: str,
    **kwargs: Any,
) -> Any:
    method = getattr(
        service,
        operation,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"User operation '{operation}' "
                "is not configured."
            ),
        )

    return await method(**kwargs)


def _ensure_same_organization(
    actor: Any,
    target: Any,
) -> None:
    """
    Defense-in-depth organization boundary.

    Super admins may cross organizations inside the same tenant.
    Normal org admins may not.
    """

    if "super_admin" in _roles(actor):
        return

    actor_org = _organization_id(
        actor
    )

    target_org = _optional_str(
        _attr(
            target,
            "organization_id",
        )
    )

    if (
        actor_org
        and target_org
        and actor_org != target_org
    ):
        # 404 avoids confirming that a user exists in another org.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )


# ============================================================
# CURRENT USER
# Keep static routes before /{user_id}
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_my_profile(
    current_user: Any = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    return _serialize_user(
        current_user
    )


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
)
async def update_my_profile(
    payload: UserUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    values = payload.model_dump(
        exclude_unset=True,
    )

    result = await _service_call(
        service,
        "update_profile",

        user_id=_user_id(
            current_user
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),

        **values,
    )

    return _serialize_user(
        result
    )


@router.get(
    "/me/session",
    response_model=UserSessionSummaryResponse,
    summary="Get current user security context",
)
async def get_my_session(
    current_user: Any = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    return {
        "user_id":
            _user_id(
                current_user
            ),

        "email":
            str(
                _attr(
                    current_user,
                    "email",
                    "",
                )
            ),

        "tenant_id":
            _tenant_id(
                current_user
            ),

        "organization_id":
            _organization_id(
                current_user
            ),

        "roles":
            sorted(
                _roles(
                    current_user
                )
            ),

        "is_active":
            bool(
                _attr(
                    current_user,
                    "is_active",
                    True,
                )
            ),

        "is_admin":
            bool(
                _attr(
                    current_user,
                    "is_admin",
                    False,
                )
            ),

        "email_verified":
            bool(
                _attr(
                    current_user,
                    "email_verified",
                    False,
                )
            ),
    }


# ============================================================
# LIST USERS
# ============================================================

@router.get(
    "",
    response_model=UserPageResponse,
    summary="List users",
)
async def list_users(
    user_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),

    role: Optional[str] = None,
    department: Optional[str] = None,

    is_active: Optional[bool] = None,
    is_admin: Optional[bool] = None,

    email_verified: Optional[bool] = None,

    identity_provider: Optional[str] = None,

    source: Optional[str] = None,

    search: Optional[str] = Query(
        default=None,
        max_length=500,
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    filters = UserFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    # Org admins are automatically restricted to their org.
    # Super admins may inspect the whole tenant.
    if "super_admin" not in _roles(
        current_user
    ):
        _set_filter(
            filters,
            "organization_id",
            _organization_id(
                current_user
            ),
        )

    if user_status:
        if hasattr(
            filters,
            "statuses",
        ):
            filters.statuses = [
                user_status
            ]
        else:
            _set_filter(
                filters,
                "status",
                user_status,
            )

    if role:
        if hasattr(
            filters,
            "roles",
        ):
            filters.roles = [role]
        else:
            _set_filter(
                filters,
                "role",
                role,
            )

    _set_filter(
        filters,
        "department",
        department,
    )

    _set_filter(
        filters,
        "is_active",
        is_active,
    )

    _set_filter(
        filters,
        "is_admin",
        is_admin,
    )

    _set_filter(
        filters,
        "email_verified",
        email_verified,
    )

    _set_filter(
        filters,
        "identity_provider",
        identity_provider,
    )

    _set_filter(
        filters,
        "source",
        source,
    )

    _set_filter(
        filters,
        "search",
        search,
    )

    page = await repository.list(
        filters,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=UserStatisticsResponse,
    summary="Get user statistics",
)
async def user_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    filters = UserFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    if "super_admin" not in _roles(
        current_user
    ):
        _set_filter(
            filters,
            "organization_id",
            _organization_id(
                current_user
            ),
        )

    result = await repository.statistics(
        filters
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {
        "total":
            int(
                _attr(
                    result,
                    "total",
                    0,
                )
                or 0
            ),

        "active":
            int(
                _attr(
                    result,
                    "active",
                    0,
                )
                or 0
            ),

        "inactive":
            int(
                _attr(
                    result,
                    "inactive",
                    0,
                )
                or 0
            ),

        "suspended":
            int(
                _attr(
                    result,
                    "suspended",
                    0,
                )
                or 0
            ),

        "admins":
            int(
                _attr(
                    result,
                    "admins",
                    0,
                )
                or 0
            ),

        "verified":
            int(
                _attr(
                    result,
                    "verified",
                    0,
                )
                or 0
            ),

        "unverified":
            int(
                _attr(
                    result,
                    "unverified",
                    0,
                )
                or 0
            ),

        "by_status":
            _attr(
                result,
                "by_status",
                {},
            ),

        "by_role":
            _attr(
                result,
                "by_role",
                {},
            ),

        "by_department":
            _attr(
                result,
                "by_department",
                {},
            ),

        "by_source":
            _attr(
                result,
                "by_source",
                {},
            ),
    }


# ============================================================
# ACTIVE USERS
# ============================================================

@router.get(
    "/active",
    response_model=UserPageResponse,
    summary="List active users",
)
async def active_users(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    filters = UserFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    _set_filter(
        filters,
        "is_active",
        True,
    )

    if "super_admin" not in _roles(
        current_user
    ):
        _set_filter(
            filters,
            "organization_id",
            _organization_id(
                current_user
            ),
        )

    page = await repository.list(
        filters,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# SEARCH
# ============================================================

@router.get(
    "/search",
    response_model=UserPageResponse,
    summary="Search users",
)
async def search_users(
    q: str = Query(
        min_length=1,
        max_length=500,
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    page = await repository.search(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=(
            None
            if "super_admin"
            in _roles(current_user)
            else _organization_id(
                current_user
            )
        ),

        query=q.strip(),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# INVITE
# ============================================================

@router.post(
    "/invite",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Invite user",
)
async def invite_user(
    payload: UserInviteRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    actor_org = _organization_id(
        current_user
    )

    requested_org = (
        str(payload.organization_id)
        if payload.organization_id
        else actor_org
    )

    if (
        "super_admin"
        not in _roles(current_user)
        and requested_org != actor_org
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Cannot invite users to another organization."
            ),
        )

    result = await _service_call(
        service,
        "invite",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=requested_org,

        email=str(
            payload.email
        ).lower(),

        role=payload.role,

        first_name=payload.first_name,

        last_name=payload.last_name,

        message=payload.message,

        invited_by=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# CREATE USER
# ============================================================

@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
)
async def create_user(
    payload: UserCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    actor_org = _organization_id(
        current_user
    )

    organization_id = (
        str(payload.organization_id)
        if payload.organization_id
        else actor_org
    )

    if (
        "super_admin"
        not in _roles(current_user)
        and organization_id != actor_org
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Cannot create a user in another organization."
            ),
        )

    # Only super admins should normally grant platform-level
    # administrator privileges.
    if (
        payload.is_admin
        and "super_admin"
        not in _roles(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only a super administrator may "
                "grant administrator privileges."
            ),
        )

    result = await _service_call(
        service,
        "create",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=organization_id,

        email=str(
            payload.email
        ).lower(),

        first_name=payload.first_name,
        last_name=payload.last_name,
        display_name=payload.display_name,

        status=payload.status,
        role=payload.role,

        job_title=payload.job_title,
        department=payload.department,
        phone=payload.phone,

        locale=payload.locale,
        timezone_name=(
            payload.timezone_name
        ),

        email_verified=(
            payload.email_verified
        ),

        is_active=payload.is_active,
        is_admin=payload.is_admin,

        external_id=payload.external_id,

        identity_provider=(
            payload.identity_provider
        ),

        source=payload.source,

        metadata=payload.metadata,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# GET USER
# ============================================================

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
)
async def get_user(
    user_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    user = await repository.get_by_id(
        user_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )

    _ensure_same_organization(
        current_user,
        user,
    )

    return _serialize_user(
        user
    )


# ============================================================
# UPDATE USER
# ============================================================

@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
)
async def update_user(
    user_id: UUID,
    payload: UserAdminUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    existing = await repository.get_by_id(
        user_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )

    _ensure_same_organization(
        current_user,
        existing,
    )

    values = payload.model_dump(
        exclude_unset=True,
    )

    if (
        "organization_id" in values
        and values["organization_id"]
        is not None
    ):
        values["organization_id"] = str(
            values["organization_id"]
        )

    if (
        "super_admin"
        not in _roles(current_user)
        and "organization_id" in values
        and values["organization_id"]
        != _organization_id(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Cannot move users to another organization."
            ),
        )

    if (
        values.get("is_admin") is True
        and "super_admin"
        not in _roles(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only a super administrator may "
                "grant administrator privileges."
            ),
        )

    result = await _service_call(
        service,
        "update",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),

        **values,
    )

    return _serialize_user(
        result
    )


# ============================================================
# EMAIL
# ============================================================

@router.patch(
    "/{user_id}/email",
    response_model=UserResponse,
    summary="Update user email",
)
async def update_user_email(
    user_id: UUID,
    payload: UserEmailUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    existing = await repository.get_by_id(
        user_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )

    _ensure_same_organization(
        current_user,
        existing,
    )

    result = await _service_call(
        service,
        "update_email",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        email=str(
            payload.email
        ).lower(),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# ROLE
# ============================================================

@router.patch(
    "/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role",
)
async def update_user_role(
    user_id: UUID,
    payload: UserRoleRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: UserRepository = Depends(
        get_user_repository
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    existing = await repository.get_by_id(
        user_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )

    _ensure_same_organization(
        current_user,
        existing,
    )

    privileged_roles = {
        "super_admin",
        "platform_admin",
    }

    if (
        payload.role
        and payload.role.lower()
        in privileged_roles
        and "super_admin"
        not in _roles(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only a super administrator may "
                "grant this role."
            ),
        )

    result = await _service_call(
        service,
        "update_role",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        role=payload.role,

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# ACTIVATE
# ============================================================

@router.post(
    "/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user",
)
async def activate_user(
    user_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    result = await _service_call(
        service,
        "activate",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# DEACTIVATE
# ============================================================

@router.post(
    "/{user_id}/deactivate",
    response_model=UserResponse,
    summary="Deactivate user",
)
async def deactivate_user(
    user_id: UUID,
    payload: UserDeactivateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    if str(user_id) == _user_id(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Use the account lifecycle workflow "
                "to deactivate your own account."
            ),
        )

    result = await _service_call(
        service,
        "deactivate",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# SUSPEND
# ============================================================

@router.post(
    "/{user_id}/suspend",
    response_model=UserResponse,
    summary="Suspend user",
)
async def suspend_user(
    user_id: UUID,
    payload: UserSuspendRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    if str(user_id) == _user_id(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You cannot suspend your own account.",
        )

    result = await _service_call(
        service,
        "suspend",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# EMAIL VERIFICATION
# ============================================================

@router.patch(
    "/{user_id}/email-verification",
    response_model=UserResponse,
    summary="Update email verification status",
)
async def update_email_verification(
    user_id: UUID,
    payload: UserVerifyEmailRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, Any]:
    _require_user_admin(
        current_user
    )

    operation = (
        "mark_email_verified"
        if payload.verified
        else "mark_email_unverified"
    )

    result = await _service_call(
        service,
        operation,

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_user(
        result
    )


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{user_id}",
    response_model=MessageResponse,
    summary="Delete user",
)
async def delete_user(
    user_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_user_service
    ),
) -> dict[str, str]:
    _require_super_admin(
        current_user
    )

    if str(user_id) == _user_id(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "A super administrator cannot "
                "delete their own account here."
            ),
        )

    deleted = await _service_call(
        service,
        "delete",

        user_id=str(user_id),

        tenant_id=_tenant_id(
            current_user
        ),

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    if deleted is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )

    return {
        "message":
            "User deleted successfully."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="User subsystem health",
)
async def users_health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "users",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }