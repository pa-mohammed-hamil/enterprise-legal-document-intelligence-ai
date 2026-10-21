# backend/api/routes/organizations.py

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
from pydantic import BaseModel, Field

from backend.db.repositories.organization_repository import (
    OrganizationCreate,
    OrganizationFilter,
    OrganizationRepository,
    OrganizationUpdate,
)


router = APIRouter(
    prefix="/organizations",
    tags=["Organizations"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies.py providers.
# ============================================================

async def get_organization_repository() -> OrganizationRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Organization repository dependency is not configured.",
    )


async def get_organization_service() -> Any:
    """
    Recommended OrganizationService dependencies:

        OrganizationRepository
        UserRepository
        DocumentRepository
        ContractRepository
        AuditService
        SecurityService

    Service should own:
        - organization lifecycle
        - tenant consistency
        - quota enforcement
        - dependency checks
        - admin authorization
        - audit events
        - provisioning/deprovisioning
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Organization service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class OrganizationCreateRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=255,
    )

    slug: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=255,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    description: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    status: str = Field(
        default="active",
        max_length=50,
    )

    domain: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    website: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    timezone_name: str = Field(
        default="UTC",
        max_length=100,
    )

    settings: dict[str, Any] = Field(
        default_factory=dict,
    )

    quotas: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class OrganizationUpdateRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    domain: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    website: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    timezone_name: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    metadata: Optional[
        dict[str, Any]
    ] = None


class OrganizationStatusRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class OrganizationSettingsRequest(BaseModel):
    settings: dict[str, Any]

    replace: bool = False


class OrganizationQuotaRequest(BaseModel):
    max_users: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_documents: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_storage_bytes: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_contracts: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_monthly_ai_requests: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_monthly_tokens: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_vector_chunks: Optional[int] = Field(
        default=None,
        ge=1,
    )

    max_concurrent_jobs: Optional[int] = Field(
        default=None,
        ge=1,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class OrganizationResponse(BaseModel):
    id: str

    tenant_id: str

    name: str
    slug: str

    description: Optional[str] = None

    status: str

    domain: Optional[str] = None
    website: Optional[str] = None

    country: Optional[str] = None
    timezone_name: Optional[str] = None

    settings: dict[str, Any] = Field(
        default_factory=dict,
    )

    quotas: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_by: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    deleted_at: Optional[datetime] = None


class OrganizationPageResponse(BaseModel):
    items: list[OrganizationResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class OrganizationStatisticsResponse(BaseModel):
    total: int = 0

    active: int = 0
    inactive: int = 0
    suspended: int = 0
    deleted: int = 0

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_country: dict[str, int] = Field(
        default_factory=dict,
    )


class OrganizationUsageResponse(BaseModel):
    organization_id: str

    users: int = 0
    documents: int = 0
    contracts: int = 0

    storage_bytes: int = 0
    vector_chunks: int = 0

    monthly_ai_requests: int = 0
    monthly_tokens: int = 0

    quotas: dict[str, Any] = Field(
        default_factory=dict,
    )

    utilization: dict[str, float] = Field(
        default_factory=dict,
    )


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

    if isinstance(
        values,
        str,
    ):
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
    }


def _is_super_admin(
    user: Any,
) -> bool:
    return bool(
        _roles(user)
        & {
            "super_admin",
            "platform_admin",
        }
    )


def _require_org_admin(
    user: Any,
) -> None:
    allowed = {
        "admin",
        "super_admin",
        "platform_admin",
        "organization_admin",
        "org_admin",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Organization administration "
                "permission is required."
            ),
        )


def _assert_org_access(
    user: Any,
    organization_id: Any,
) -> None:
    if _is_super_admin(user):
        return

    current_org = _organization_id(
        user
    )

    if (
        current_org is None
        or current_org
        != str(organization_id)
    ):
        # Do not reveal existence of another
        # organization's record.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )


def _request_id(
    request: Request,
) -> Optional[str]:
    value = (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )

    return _optional_str(value)


def _metadata(
    organization: Any,
) -> dict[str, Any]:
    return (
        _attr(
            organization,
            "organization_metadata",
        )
        or _attr(
            organization,
            "metadata_json",
        )
        or {}
    )


def _settings(
    organization: Any,
) -> dict[str, Any]:
    return (
        _attr(
            organization,
            "settings",
        )
        or _attr(
            organization,
            "settings_json",
        )
        or {}
    )


def _quotas(
    organization: Any,
) -> dict[str, Any]:
    return (
        _attr(
            organization,
            "quotas",
        )
        or _attr(
            organization,
            "quotas_json",
        )
        or {}
    )


def _serialize_organization(
    organization: Any,
) -> dict[str, Any]:
    return {
        "id": str(
            _attr(
                organization,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                organization,
                "tenant_id",
                "",
            )
        ),

        "name": str(
            _attr(
                organization,
                "name",
                "",
            )
        ),

        "slug": str(
            _attr(
                organization,
                "slug",
                "",
            )
        ),

        "description": _attr(
            organization,
            "description",
        ),

        "status": str(
            _enum_value(
                _attr(
                    organization,
                    "status",
                    "active",
                )
            )
        ),

        "domain": _attr(
            organization,
            "domain",
        ),

        "website": _attr(
            organization,
            "website",
        ),

        "country": _attr(
            organization,
            "country",
        ),

        "timezone_name": (
            _attr(
                organization,
                "timezone_name",
            )
            or _attr(
                organization,
                "timezone",
            )
        ),

        "settings": _settings(
            organization
        ),

        "quotas": _quotas(
            organization
        ),

        "metadata": _metadata(
            organization
        ),

        "created_by": _optional_str(
            _attr(
                organization,
                "created_by",
            )
        ),

        "created_at": _attr(
            organization,
            "created_at",
        ),

        "updated_at": _attr(
            organization,
            "updated_at",
        ),

        "deleted_at": _attr(
            organization,
            "deleted_at",
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
            _serialize_organization(
                item
            )
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

    if hasattr(
        filters,
        field,
    ):
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
                f"Organization operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(
        **kwargs
    )


# ============================================================
# LIST ORGANIZATIONS
# ============================================================

@router.get(
    "",
    response_model=OrganizationPageResponse,
    summary="List organizations",
)
async def list_organizations(
    organization_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),

    country: Optional[str] = None,

    search: Optional[str] = Query(
        default=None,
        max_length=500,
    ),

    include_deleted: bool = False,

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

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    filters = OrganizationFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    # Normal org admins only see their organization.
    if not _is_super_admin(
        current_user
    ):
        current_org = _organization_id(
            current_user
        )

        if current_org:
            if hasattr(
                filters,
                "organization_ids",
            ):
                filters.organization_ids = [
                    current_org
                ]

            else:
                _set_filter(
                    filters,
                    "organization_id",
                    current_org,
                )

    if organization_status:
        if hasattr(
            filters,
            "statuses",
        ):
            filters.statuses = [
                organization_status
            ]
        else:
            _set_filter(
                filters,
                "status",
                organization_status,
            )

    _set_filter(
        filters,
        "country",
        country,
    )

    _set_filter(
        filters,
        "search",
        search,
    )

    _set_filter(
        filters,
        "include_deleted",
        include_deleted,
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
# Static route before /{organization_id}
# ============================================================

@router.get(
    "/statistics",
    response_model=OrganizationStatisticsResponse,
    summary="Get organization statistics",
)
async def organization_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> Any:
    _require_org_admin(
        current_user
    )

    filters = OrganizationFilter(
        tenant_id=_tenant_id(
            current_user
        )
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
        "total": _attr(
            result,
            "total",
            0,
        ),

        "active": _attr(
            result,
            "active",
            0,
        ),

        "inactive": _attr(
            result,
            "inactive",
            0,
        ),

        "suspended": _attr(
            result,
            "suspended",
            0,
        ),

        "deleted": _attr(
            result,
            "deleted",
            0,
        ),

        "by_status": _attr(
            result,
            "by_status",
            {},
        ),

        "by_country": _attr(
            result,
            "by_country",
            {},
        ),
    }


# ============================================================
# CURRENT USER ORGANIZATION
# ============================================================

@router.get(
    "/me",
    response_model=OrganizationResponse,
    summary="Get current user's organization",
)
async def current_organization(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> dict[str, Any]:
    organization_id = _organization_id(
        current_user
    )

    if not organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    organization = await repository.get_by_id(
        organization_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    return _serialize_organization(
        organization
    )


# ============================================================
# CREATE ORGANIZATION
# ============================================================

@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create organization",
)
async def create_organization(
    payload: OrganizationCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    result = await _service_call(
        service,
        "create",

        tenant_id=_tenant_id(
            current_user
        ),

        created_by=_user_id(
            current_user
        ),

        name=payload.name.strip(),

        slug=payload.slug,

        description=payload.description,

        status=payload.status,

        domain=payload.domain,

        website=payload.website,

        country=payload.country,

        timezone_name=(
            payload.timezone_name
        ),

        settings=payload.settings,

        quotas=payload.quotas,

        metadata=payload.metadata,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_organization(
        result
    )


# ============================================================
# GET ORGANIZATION
# ============================================================

@router.get(
    "/{organization_id}",
    response_model=OrganizationResponse,
    summary="Get organization",
)
async def get_organization(
    organization_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> dict[str, Any]:
    _assert_org_access(
        current_user,
        organization_id,
    )

    organization = await repository.get_by_id(
        organization_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    return _serialize_organization(
        organization
    )


# ============================================================
# UPDATE ORGANIZATION
# ============================================================

@router.patch(
    "/{organization_id}",
    response_model=OrganizationResponse,
    summary="Update organization",
)
async def update_organization(
    organization_id: UUID,
    payload: OrganizationUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    _assert_org_access(
        current_user,
        organization_id,
    )

    values = payload.model_dump(
        exclude_unset=True
    )

    result = await _service_call(
        service,
        "update",

        organization_id=str(
            organization_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        updated_by=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),

        **values,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    return _serialize_organization(
        result
    )


# ============================================================
# SETTINGS
# ============================================================

@router.get(
    "/{organization_id}/settings",
    summary="Get organization settings",
)
async def get_organization_settings(
    organization_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> dict[str, Any]:
    _assert_org_access(
        current_user,
        organization_id,
    )

    organization = await repository.get_by_id(
        organization_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    return {
        "organization_id":
            str(organization_id),

        "settings":
            _settings(
                organization
            ),
    }


@router.patch(
    "/{organization_id}/settings",
    summary="Update organization settings",
)
async def update_organization_settings(
    organization_id: UUID,
    payload: OrganizationSettingsRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    _assert_org_access(
        current_user,
        organization_id,
    )

    result = await _service_call(
        service,
        "update_settings",

        organization_id=str(
            organization_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        settings=payload.settings,

        replace=payload.replace,

        updated_by=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return {
        "organization_id":
            str(organization_id),

        "settings":
            (
                _settings(result)
                if result is not None
                else payload.settings
            ),
    }


# ============================================================
# QUOTAS
# ============================================================

@router.get(
    "/{organization_id}/quotas",
    summary="Get organization quotas",
)
async def get_organization_quotas(
    organization_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
) -> dict[str, Any]:
    _assert_org_access(
        current_user,
        organization_id,
    )

    organization = await repository.get_by_id(
        organization_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization was not found.",
        )

    return {
        "organization_id":
            str(organization_id),

        "quotas":
            _quotas(
                organization
            ),
    }


@router.patch(
    "/{organization_id}/quotas",
    summary="Update organization quotas",
)
async def update_organization_quotas(
    organization_id: UUID,
    payload: OrganizationQuotaRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    # Quotas affect billing/resource boundaries.
    # Restrict this more strongly than normal settings.
    if not _is_super_admin(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Platform administrator "
                "permission is required."
            ),
        )

    values = payload.model_dump(
        exclude_unset=True
    )

    result = await _service_call(
        service,
        "update_quotas",

        organization_id=str(
            organization_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        quotas=values,

        updated_by=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return {
        "organization_id":
            str(organization_id),

        "quotas":
            (
                _quotas(result)
                if result is not None
                else values
            ),
    }


# ============================================================
# USAGE
# ============================================================

@router.get(
    "/{organization_id}/usage",
    response_model=OrganizationUsageResponse,
    summary="Get organization usage",
)
async def organization_usage(
    organization_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> Any:
    _assert_org_access(
        current_user,
        organization_id,
    )

    result = await _service_call(
        service,
        "usage",

        organization_id=str(
            organization_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            "organization_id":
                str(organization_id),
            **result,
        }

    return {
        "organization_id":
            str(organization_id),

        "users":
            _attr(
                result,
                "users",
                0,
            ),

        "documents":
            _attr(
                result,
                "documents",
                0,
            ),

        "contracts":
            _attr(
                result,
                "contracts",
                0,
            ),

        "storage_bytes":
            _attr(
                result,
                "storage_bytes",
                0,
            ),

        "vector_chunks":
            _attr(
                result,
                "vector_chunks",
                0,
            ),

        "monthly_ai_requests":
            _attr(
                result,
                "monthly_ai_requests",
                0,
            ),

        "monthly_tokens":
            _attr(
                result,
                "monthly_tokens",
                0,
            ),

        "quotas":
            _attr(
                result,
                "quotas",
                {},
            ),

        "utilization":
            _attr(
                result,
                "utilization",
                {},
            ),
    }


# ============================================================
# ACTIVATE
# ============================================================

@router.post(
    "/{organization_id}/activate",
    response_model=OrganizationResponse,
    summary="Activate organization",
)
async def activate_organization(
    organization_id: UUID,
    payload: OrganizationStatusRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    _assert_org_access(
        current_user,
        organization_id,
    )

    result = await _service_call(
        service,
        "activate",

        organization_id=str(
            organization_id
        ),

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

    return _serialize_organization(
        result
    )


# ============================================================
# DEACTIVATE
# ============================================================

@router.post(
    "/{organization_id}/deactivate",
    response_model=OrganizationResponse,
    summary="Deactivate organization",
)
async def deactivate_organization(
    organization_id: UUID,
    payload: OrganizationStatusRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    _require_org_admin(
        current_user
    )

    _assert_org_access(
        current_user,
        organization_id,
    )

    result = await _service_call(
        service,
        "deactivate",

        organization_id=str(
            organization_id
        ),

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

    return _serialize_organization(
        result
    )


# ============================================================
# SUSPEND
# ============================================================

@router.post(
    "/{organization_id}/suspend",
    response_model=OrganizationResponse,
    summary="Suspend organization",
)
async def suspend_organization(
    organization_id: UUID,
    payload: OrganizationStatusRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    # Suspension can disable a whole customer.
    if not _is_super_admin(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Platform administrator "
                "permission is required."
            ),
        )

    result = await _service_call(
        service,
        "suspend",

        organization_id=str(
            organization_id
        ),

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

    return _serialize_organization(
        result
    )


# ============================================================
# SOFT DELETE
# ============================================================

@router.delete(
    "/{organization_id}",
    response_model=MessageResponse,
    summary="Soft delete organization",
)
async def delete_organization(
    organization_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, str]:
    if not _is_super_admin(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Platform administrator "
                "permission is required."
            ),
        )

    await _service_call(
        service,
        "delete",

        organization_id=str(
            organization_id
        ),

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

    return {
        "message":
            "Organization deleted successfully."
    }


# ============================================================
# RESTORE
# ============================================================

@router.post(
    "/{organization_id}/restore",
    response_model=OrganizationResponse,
    summary="Restore deleted organization",
)
async def restore_organization(
    organization_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_organization_service
    ),
) -> dict[str, Any]:
    if not _is_super_admin(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Platform administrator "
                "permission is required."
            ),
        )

    result = await _service_call(
        service,
        "restore",

        organization_id=str(
            organization_id
        ),

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

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Deleted organization "
                "was not found."
            ),
        )

    return _serialize_organization(
        result
    )


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Organization subsystem health",
)
async def organization_health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "organizations",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }