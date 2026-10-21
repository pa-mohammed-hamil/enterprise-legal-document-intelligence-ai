# backend/api/routes/audit.py

from __future__ import annotations

from datetime import datetime
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
from pydantic import BaseModel, ConfigDict, Field

from backend.db.repositories.audit_repository import (
    AuditAction,
    AuditCreate,
    AuditFilter,
    AuditOutcome,
    AuditRepository,
    AuditSeverity,
)


router = APIRouter(
    prefix="/audit",
    tags=["Audit"],
)


# ============================================================
# DEPENDENCIES
# Replace these with backend.dependencies imports.
# ============================================================

async def get_audit_repository() -> AuditRepository:
    """
    Replace with the real dependency:

        async def get_audit_repository(
            session: AsyncSession = Depends(get_db),
        ) -> AuditRepository:
            return AuditRepository(session)
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Audit repository dependency is not configured.",
    )


async def get_current_user() -> Any:
    """
    Replace with your real authentication dependency.

    Expected attributes:
        id
        tenant_id
        organization_id
        role / roles
        is_admin
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# SCHEMAS
# ============================================================

class AuditEventResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: UUID | str

    tenant_id: UUID | str

    organization_id: Optional[
        UUID | str
    ] = None

    user_id: Optional[
        UUID | str
    ] = None

    action: str

    severity: str
    outcome: str

    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    ip_address: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    created_at: datetime

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class AuditPageResponse(BaseModel):
    items: list[AuditEventResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class AuditStatisticsResponse(BaseModel):
    total: int = 0

    by_action: dict[str, int] = Field(
        default_factory=dict
    )

    by_severity: dict[str, int] = Field(
        default_factory=dict
    )

    by_outcome: dict[str, int] = Field(
        default_factory=dict
    )

    by_resource_type: dict[str, int] = Field(
        default_factory=dict
    )

    security_events: int = 0
    failed_events: int = 0
    denied_events: int = 0

    model_config = ConfigDict(
        extra="allow"
    )


class AuditEventCreateRequest(BaseModel):
    action: AuditAction

    severity: AuditSeverity = (
        AuditSeverity.INFO
    )

    outcome: AuditOutcome = (
        AuditOutcome.SUCCESS
    )

    resource_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    resource_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    correlation_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# SECURITY HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(
        obj,
        name,
        default,
    )


def _tenant_id(
    current_user: Any,
) -> str:

    tenant_id = _attr(
        current_user,
        "tenant_id",
    )

    if not tenant_id:
        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Tenant context is required."
            ),
        )

    return str(tenant_id)


def _organization_id(
    current_user: Any,
) -> Optional[str]:

    value = _attr(
        current_user,
        "organization_id",
    )

    return (
        str(value)
        if value
        else None
    )


def _user_id(
    current_user: Any,
) -> Optional[str]:

    value = _attr(
        current_user,
        "id",
    )

    return (
        str(value)
        if value
        else None
    )


def _roles(
    current_user: Any,
) -> set[str]:

    result: set[str] = set()

    role = _attr(
        current_user,
        "role",
    )

    if role:
        result.add(
            str(role).lower()
        )

    roles = _attr(
        current_user,
        "roles",
        [],
    )

    if roles:
        for item in roles:

            if isinstance(
                item,
                str,
            ):
                result.add(
                    item.lower()
                )

            else:
                name = _attr(
                    item,
                    "name",
                )

                if name:
                    result.add(
                        str(name).lower()
                    )

    if _attr(
        current_user,
        "is_admin",
        False,
    ):
        result.add("admin")

    return result


def _require_audit_access(
    current_user: Any,
) -> None:

    allowed_roles = {
        "admin",
        "super_admin",
        "compliance",
        "compliance_officer",
        "auditor",
        "security",
        "security_admin",
        "legal_admin",
    }

    if not (
        _roles(current_user)
        & allowed_roles
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "You do not have permission "
                "to access audit records."
            ),
        )


# ============================================================
# SERIALIZATION
# ============================================================

def _serialize_event(
    event: Any,
) -> dict[str, Any]:

    metadata = (
        getattr(
            event,
            "metadata_json",
            None,
        )
        or getattr(
            event,
            "event_metadata",
            None,
        )
        or {}
    )

    return {
        "id":
            event.id,

        "tenant_id":
            event.tenant_id,

        "organization_id":
            getattr(
                event,
                "organization_id",
                None,
            ),

        "user_id":
            getattr(
                event,
                "user_id",
                None,
            ),

        "action":
            _enum_value(
                event.action
            ),

        "severity":
            _enum_value(
                event.severity
            ),

        "outcome":
            _enum_value(
                event.outcome
            ),

        "resource_type":
            getattr(
                event,
                "resource_type",
                None,
            ),

        "resource_id":
            getattr(
                event,
                "resource_id",
                None,
            ),

        "request_id":
            getattr(
                event,
                "request_id",
                None,
            ),

        "correlation_id":
            getattr(
                event,
                "correlation_id",
                None,
            ),

        "ip_address":
            getattr(
                event,
                "ip_address",
                None,
            ),

        "model_name":
            getattr(
                event,
                "model_name",
                None,
            ),

        "model_version":
            getattr(
                event,
                "model_version",
                None,
            ),

        "created_at":
            event.created_at,

        "metadata":
            metadata,
    }


def _enum_value(
    value: Any,
) -> str:

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(enum_value)

    return str(value)


# ============================================================
# LIST AUDIT EVENTS
# ============================================================

@router.get(
    "",
    response_model=AuditPageResponse,
    summary="List audit events",
)
async def list_audit_events(
    action: Optional[
        AuditAction
    ] = Query(default=None),

    severity: Optional[
        AuditSeverity
    ] = Query(default=None),

    outcome: Optional[
        AuditOutcome
    ] = Query(default=None),

    user_id: Optional[str] = Query(
        default=None
    ),

    resource_type: Optional[str] = Query(
        default=None
    ),

    resource_id: Optional[str] = Query(
        default=None
    ),

    request_id: Optional[str] = Query(
        default=None
    ),

    correlation_id: Optional[str] = Query(
        default=None
    ),

    model_name: Optional[str] = Query(
        default=None
    ),

    created_from: Optional[
        datetime
    ] = Query(default=None),

    created_to: Optional[
        datetime
    ] = Query(default=None),

    search: Optional[str] = Query(
        default=None,
        max_length=500,
    ),

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

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    tenant_id = _tenant_id(
        current_user
    )

    filters = AuditFilter(
        tenant_id=tenant_id,
    )

    _set_filter(
        filters,
        "organization_id",
        _organization_id(
            current_user
        ),
    )

    _set_filter(
        filters,
        "action",
        action,
    )

    _set_filter(
        filters,
        "severity",
        severity,
    )

    _set_filter(
        filters,
        "outcome",
        outcome,
    )

    _set_filter(
        filters,
        "user_id",
        user_id,
    )

    _set_filter(
        filters,
        "resource_type",
        resource_type,
    )

    _set_filter(
        filters,
        "resource_id",
        resource_id,
    )

    _set_filter(
        filters,
        "request_id",
        request_id,
    )

    _set_filter(
        filters,
        "correlation_id",
        correlation_id,
    )

    _set_filter(
        filters,
        "model_name",
        model_name,
    )

    _set_filter(
        filters,
        "created_from",
        created_from,
    )

    _set_filter(
        filters,
        "created_to",
        created_to,
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

    return {
        "items": [
            _serialize_event(item)
            for item in page.items
        ],

        "total":
            page.total,

        "limit":
            page.limit,

        "offset":
            page.offset,

        "has_more":
            page.has_more,
    }


# ============================================================
# GET SINGLE EVENT
# ============================================================

@router.get(
    "/events/{audit_id}",
    response_model=AuditEventResponse,
    summary="Get audit event",
)
async def get_audit_event(
    audit_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    event = await repository.get_by_id(
        audit_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if event is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Audit event was not found."
            ),
        )

    return _serialize_event(event)


# ============================================================
# REQUEST TRACE
# ============================================================

@router.get(
    "/requests/{request_id}",
    response_model=list[
        AuditEventResponse
    ],
    summary="Get request audit trail",
)
async def get_request_trace(
    request_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> list[dict[str, Any]]:

    _require_audit_access(
        current_user
    )

    events = (
        await repository.get_by_request_id(
            request_id,
            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    return [
        _serialize_event(event)
        for event in events
    ]


# ============================================================
# CORRELATION TRACE
# ============================================================

@router.get(
    "/correlations/{correlation_id}",
    response_model=list[
        AuditEventResponse
    ],
    summary="Get correlated audit trail",
)
async def get_correlation_trace(
    correlation_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> list[dict[str, Any]]:

    _require_audit_access(
        current_user
    )

    events = (
        await repository.get_by_correlation_id(
            correlation_id,
            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    return [
        _serialize_event(event)
        for event in events
    ]


# ============================================================
# USER AUDIT TRAIL
# ============================================================

@router.get(
    "/users/{target_user_id}",
    response_model=AuditPageResponse,
    summary="Get user audit trail",
)
async def get_user_audit_trail(
    target_user_id: UUID,

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

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    page = await repository.list_for_user(
        tenant_id=_tenant_id(
            current_user
        ),
        user_id=str(target_user_id),
        limit=limit,
        offset=offset,
    )

    return _page_response(page)


# ============================================================
# RESOURCE AUDIT TRAIL
# ============================================================

@router.get(
    "/resources/{resource_type}/{resource_id}",
    response_model=AuditPageResponse,
    summary="Get resource audit trail",
)
async def get_resource_audit_trail(
    resource_type: str,
    resource_id: str,

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

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    page = await repository.list_for_resource(
        tenant_id=_tenant_id(
            current_user
        ),
        resource_type=resource_type,
        resource_id=resource_id,
        limit=limit,
        offset=offset,
    )

    return _page_response(page)


# ============================================================
# SECURITY EVENTS
# ============================================================

@router.get(
    "/security",
    response_model=AuditPageResponse,
    summary="List security audit events",
)
async def list_security_events(
    severity: Optional[
        AuditSeverity
    ] = Query(default=None),

    created_from: Optional[
        datetime
    ] = Query(default=None),

    created_to: Optional[
        datetime
    ] = Query(default=None),

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

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    try:
        page = (
            await repository.list_security_events(
                tenant_id=_tenant_id(
                    current_user
                ),
                severity=severity,
                created_from=created_from,
                created_to=created_to,
                limit=limit,
                offset=offset,
            )
        )

    except TypeError:
        # Compatibility with repositories exposing only
        # tenant + pagination arguments.
        page = (
            await repository.list_security_events(
                tenant_id=_tenant_id(
                    current_user
                ),
                limit=limit,
                offset=offset,
            )
        )

    return _page_response(page)


# ============================================================
# MODEL / LLM AUDIT EVENTS
# ============================================================

@router.get(
    "/models",
    response_model=AuditPageResponse,
    summary="List AI model audit events",
)
async def list_model_events(
    model_name: Optional[str] = Query(
        default=None
    ),

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

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    page = await repository.list_model_events(
        tenant_id=_tenant_id(
            current_user
        ),
        model_name=model_name,
        limit=limit,
        offset=offset,
    )

    return _page_response(page)


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=AuditStatisticsResponse,
    summary="Get audit statistics",
)
async def get_audit_statistics(
    created_from: Optional[
        datetime
    ] = Query(default=None),

    created_to: Optional[
        datetime
    ] = Query(default=None),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    _require_audit_access(
        current_user
    )

    filters = AuditFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    _set_filter(
        filters,
        "organization_id",
        _organization_id(
            current_user
        ),
    )

    _set_filter(
        filters,
        "created_from",
        created_from,
    )

    _set_filter(
        filters,
        "created_to",
        created_to,
    )

    statistics = (
        await repository.statistics(
            filters
        )
    )

    if hasattr(
        statistics,
        "to_dict",
    ):
        return statistics.to_dict()

    if isinstance(
        statistics,
        dict,
    ):
        return statistics

    result: dict[str, Any] = {}

    for key in (
        "total",
        "by_action",
        "by_severity",
        "by_outcome",
        "by_resource_type",
        "security_events",
        "failed_events",
        "denied_events",
    ):
        if hasattr(
            statistics,
            key,
        ):
            result[key] = getattr(
                statistics,
                key,
            )

    return result


# ============================================================
# CREATE INTERNAL AUDIT EVENT
# ============================================================

@router.post(
    "/events",
    response_model=AuditEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create audit event",
)
async def create_audit_event(
    payload: AuditEventCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    tenant_id = _tenant_id(
        current_user
    )

    user_id = _user_id(
        current_user
    )

    organization_id = (
        _organization_id(
            current_user
        )
    )

    request_id = (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )

    ip_address = None

    if request.client:
        ip_address = (
            request.client.host
        )

    # Do not trust arbitrary public clients to create
    # privileged security/admin audit records.
    privileged_actions = {
        AuditAction.ADMIN_ACTION,
        AuditAction.SECURITY_EVENT,
        AuditAction.PERMISSION_DENIED,
        AuditAction.PROMPT_INJECTION,
        AuditAction.PII_DETECTED,
    }

    if (
        payload.action
        in privileged_actions
    ):
        _require_audit_access(
            current_user
        )

    create_data = AuditCreate(
        tenant_id=tenant_id,
        organization_id=organization_id,
        user_id=user_id,

        action=payload.action,
        severity=payload.severity,
        outcome=payload.outcome,

        resource_type=(
            payload.resource_type
        ),

        resource_id=(
            payload.resource_id
        ),

        request_id=request_id,

        correlation_id=(
            payload.correlation_id
        ),

        ip_address=ip_address,

        user_agent=request.headers.get(
            "User-Agent"
        ),

        model_name=(
            payload.model_name
        ),

        model_version=(
            payload.model_version
        ),

        metadata=payload.metadata,
    )

    event = await repository.create(
        create_data
    )

    return _serialize_event(event)


# ============================================================
# CURRENT USER'S OWN EVENTS
# ============================================================

@router.get(
    "/me",
    response_model=AuditPageResponse,
    summary="Get current user's audit events",
)
async def get_my_audit_events(
    limit: int = Query(
        default=50,
        ge=1,
        le=250,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: AuditRepository = Depends(
        get_audit_repository
    ),
) -> dict[str, Any]:

    user_id = _user_id(
        current_user
    )

    if user_id is None:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Authenticated user identity "
                "is required."
            ),
        )

    page = await repository.list_for_user(
        tenant_id=_tenant_id(
            current_user
        ),
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    return _page_response(page)


# ============================================================
# HELPERS
# ============================================================

def _page_response(
    page: Any,
) -> dict[str, Any]:

    items = list(
        getattr(
            page,
            "items",
            [],
        )
    )

    total = int(
        getattr(
            page,
            "total",
            len(items),
        )
    )

    limit = int(
        getattr(
            page,
            "limit",
            len(items),
        )
    )

    offset = int(
        getattr(
            page,
            "offset",
            0,
        )
    )

    has_more = getattr(
        page,
        "has_more",
        (
            offset
            + len(items)
            < total
        ),
    )

    return {
        "items": [
            _serialize_event(item)
            for item in items
        ],

        "total":
            total,

        "limit":
            limit,

        "offset":
            offset,

        "has_more":
            bool(has_more),
    }


def _set_filter(
    filters: Any,
    name: str,
    value: Any,
) -> None:
    """
    Allows the route to remain compatible while the AuditFilter
    dataclass evolves. Unsupported fields are simply ignored.
    """

    if value is None:
        return

    if hasattr(
        filters,
        name,
    ):
        setattr(
            filters,
            name,
            value,
        )