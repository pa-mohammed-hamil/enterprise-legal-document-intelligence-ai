# backend/api/routes/renewals.py

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
from pydantic import BaseModel, Field, model_validator

from backend.db.repositories.renewal_repository import (
    RenewalCreate,
    RenewalFilter,
    RenewalRepository,
    RenewalUpdate,
)


router = APIRouter(
    prefix="/renewals",
    tags=["Renewals"],
)


# ============================================================
# DEPENDENCIES
# Replace these with backend.dependencies.py providers.
# ============================================================

async def get_renewal_repository() -> RenewalRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Renewal repository dependency is not configured.",
    )


async def get_renewal_service() -> Any:
    """
    RenewalService should orchestrate:

        RenewalRepository
        ContractRepository
        NotificationRepository
        ReviewRepository
        RiskRepository
        AuditService
        EscalationEngine
        FireEngine

    Lifecycle-changing operations should go through this service.
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Renewal service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RenewalCreateRequest(BaseModel):
    contract_id: UUID

    renewal_date: datetime

    notice_deadline: Optional[datetime] = None

    renewal_term_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=36500,
    )

    notice_period_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=36500,
    )

    auto_renewal: bool = False

    status: str = Field(
        default="pending",
        max_length=50,
    )

    decision: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    decision_reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    owner_user_id: Optional[UUID] = None

    risk_level: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    escalation_level: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    reminder_at: Optional[datetime] = None

    source: str = Field(
        default="system",
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "RenewalCreateRequest":
        if (
            self.notice_deadline
            and self.notice_deadline > self.renewal_date
        ):
            raise ValueError(
                "notice_deadline cannot be after renewal_date."
            )

        if (
            self.reminder_at
            and self.notice_deadline
            and self.reminder_at > self.notice_deadline
        ):
            raise ValueError(
                "reminder_at should not be after notice_deadline."
            )

        return self


class RenewalUpdateRequest(BaseModel):
    renewal_date: Optional[datetime] = None
    notice_deadline: Optional[datetime] = None

    renewal_term_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=36500,
    )

    notice_period_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=36500,
    )

    auto_renewal: Optional[bool] = None

    owner_user_id: Optional[UUID] = None

    reminder_at: Optional[datetime] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


class RenewalDecisionRequest(BaseModel):
    decision: str = Field(
        min_length=1,
        max_length=100,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class RenewalStatusRequest(BaseModel):
    status: str = Field(
        min_length=1,
        max_length=50,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class RenewalRiskRequest(BaseModel):
    risk_level: str = Field(
        min_length=1,
        max_length=50,
    )

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class RenewalEscalationRequest(BaseModel):
    escalation_level: str = Field(
        min_length=1,
        max_length=50,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class RenewalReminderRequest(BaseModel):
    reminder_at: Optional[datetime] = None

    channel: str = Field(
        default="in_app",
        max_length=50,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class RenewalResponse(BaseModel):
    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    contract_id: str

    renewal_date: Optional[datetime] = None
    notice_deadline: Optional[datetime] = None

    renewal_term_days: Optional[int] = None
    notice_period_days: Optional[int] = None

    auto_renewal: bool = False

    status: str

    decision: Optional[str] = None
    decision_reason: Optional[str] = None

    owner_user_id: Optional[str] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    escalation_level: Optional[str] = None
    escalated_at: Optional[datetime] = None

    reminder_at: Optional[datetime] = None
    reminder_sent_at: Optional[datetime] = None

    source: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RenewalPageResponse(BaseModel):
    items: list[RenewalResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class RenewalStatisticsResponse(BaseModel):
    total: int = 0

    pending: int = 0
    upcoming: int = 0
    overdue: int = 0
    completed: int = 0
    cancelled: int = 0

    auto_renewing: int = 0
    high_risk: int = 0
    escalated: int = 0

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_risk_level: dict[str, int] = Field(
        default_factory=dict,
    )

    by_decision: dict[str, int] = Field(
        default_factory=dict,
    )


class RenewalDecisionResponse(BaseModel):
    renewal: RenewalResponse

    decision: str

    processed_at: datetime


class ReminderResponse(BaseModel):
    renewal_id: str

    scheduled: bool

    reminder_at: Optional[datetime] = None

    message: str


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
        return obj.get(name, default)

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
    }


def _require_renewal_manager(
    user: Any,
) -> None:
    allowed = {
        "admin",
        "super_admin",
        "legal_admin",
        "legal",
        "contract_manager",
        "renewal_manager",
        "compliance",
        "compliance_officer",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Renewal management "
                "permission is required."
            ),
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
    renewal: Any,
) -> dict[str, Any]:
    return (
        _attr(
            renewal,
            "renewal_metadata",
        )
        or _attr(
            renewal,
            "metadata_json",
        )
        or {}
    )


def _serialize_renewal(
    renewal: Any,
) -> dict[str, Any]:
    return {
        "id": str(
            _attr(
                renewal,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                renewal,
                "tenant_id",
                "",
            )
        ),

        "organization_id":
            _optional_str(
                _attr(
                    renewal,
                    "organization_id",
                )
            ),

        "contract_id": str(
            _attr(
                renewal,
                "contract_id",
                "",
            )
        ),

        "renewal_date":
            _attr(
                renewal,
                "renewal_date",
            ),

        "notice_deadline":
            _attr(
                renewal,
                "notice_deadline",
            ),

        "renewal_term_days":
            _attr(
                renewal,
                "renewal_term_days",
            ),

        "notice_period_days":
            _attr(
                renewal,
                "notice_period_days",
            ),

        "auto_renewal": bool(
            _attr(
                renewal,
                "auto_renewal",
                False,
            )
        ),

        "status": str(
            _enum_value(
                _attr(
                    renewal,
                    "status",
                    "pending",
                )
            )
        ),

        "decision":
            _optional_str(
                _attr(
                    renewal,
                    "decision",
                )
            ),

        "decision_reason":
            _attr(
                renewal,
                "decision_reason",
            ),

        "owner_user_id":
            _optional_str(
                _attr(
                    renewal,
                    "owner_user_id",
                )
            ),

        "risk_level":
            _optional_str(
                _enum_value(
                    _attr(
                        renewal,
                        "risk_level",
                    )
                )
            ),

        "risk_score":
            _attr(
                renewal,
                "risk_score",
            ),

        "escalation_level":
            _optional_str(
                _enum_value(
                    _attr(
                        renewal,
                        "escalation_level",
                    )
                )
            ),

        "escalated_at":
            _attr(
                renewal,
                "escalated_at",
            ),

        "reminder_at":
            _attr(
                renewal,
                "reminder_at",
            ),

        "reminder_sent_at":
            _attr(
                renewal,
                "reminder_sent_at",
            ),

        "source":
            _optional_str(
                _attr(
                    renewal,
                    "source",
                )
            ),

        "metadata":
            _metadata(
                renewal
            ),

        "created_at":
            _attr(
                renewal,
                "created_at",
            ),

        "updated_at":
            _attr(
                renewal,
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
            _serialize_renewal(item)
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
                f"Renewal operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# LIST
# ============================================================

@router.get(
    "",
    response_model=RenewalPageResponse,
    summary="List renewals",
)
async def list_renewals(
    contract_id: Optional[UUID] = None,

    renewal_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),

    risk_level: Optional[str] = None,

    auto_renewal: Optional[bool] = None,

    owner_user_id: Optional[UUID] = None,

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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    filters = RenewalFilter(
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
        "contract_id",
        str(contract_id)
        if contract_id
        else None,
    )

    if renewal_status:
        if hasattr(
            filters,
            "statuses",
        ):
            filters.statuses = [
                renewal_status
            ]
        else:
            _set_filter(
                filters,
                "status",
                renewal_status,
            )

    if risk_level:
        if hasattr(
            filters,
            "risk_levels",
        ):
            filters.risk_levels = [
                risk_level
            ]
        else:
            _set_filter(
                filters,
                "risk_level",
                risk_level,
            )

    _set_filter(
        filters,
        "auto_renewal",
        auto_renewal,
    )

    _set_filter(
        filters,
        "owner_user_id",
        str(owner_user_id)
        if owner_user_id
        else None,
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

    return _serialize_page(page)


# ============================================================
# STATIC SPECIALIZED QUERIES
# Keep before /{renewal_id}
# ============================================================

@router.get(
    "/upcoming",
    response_model=RenewalPageResponse,
    summary="List upcoming renewals",
)
async def upcoming_renewals(
    days: int = Query(
        default=90,
        ge=1,
        le=730,
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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    page = await repository.list_upcoming(
        tenant_id=_tenant_id(
            current_user
        ),
        organization_id=_organization_id(
            current_user
        ),
        days=days,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


@router.get(
    "/notice-deadlines",
    response_model=RenewalPageResponse,
    summary="List upcoming notice deadlines",
)
async def notice_deadlines(
    days: int = Query(
        default=30,
        ge=1,
        le=365,
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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    page = await repository.list_notice_deadlines(
        tenant_id=_tenant_id(
            current_user
        ),
        organization_id=_organization_id(
            current_user
        ),
        days=days,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


@router.get(
    "/overdue",
    response_model=RenewalPageResponse,
    summary="List overdue renewals",
)
async def overdue_renewals(
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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    page = await repository.list_overdue(
        tenant_id=_tenant_id(
            current_user
        ),
        organization_id=_organization_id(
            current_user
        ),
        now=datetime.now(
            timezone.utc
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


@router.get(
    "/auto-renewing",
    response_model=RenewalPageResponse,
    summary="List auto-renewing contracts",
)
async def auto_renewing(
    days: int = Query(
        default=90,
        ge=1,
        le=730,
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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    page = await repository.list_auto_renewing(
        tenant_id=_tenant_id(
            current_user
        ),
        organization_id=_organization_id(
            current_user
        ),
        days=days,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


@router.get(
    "/high-risk",
    response_model=RenewalPageResponse,
    summary="List high-risk renewals",
)
async def high_risk_renewals(
    minimum_score: float = Query(
        default=0.75,
        ge=0.0,
        le=1.0,
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

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    page = await repository.list_high_risk(
        tenant_id=_tenant_id(
            current_user
        ),
        organization_id=_organization_id(
            current_user
        ),
        minimum_score=minimum_score,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=RenewalStatisticsResponse,
    summary="Get renewal statistics",
)
async def renewal_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> Any:
    filters = RenewalFilter(
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

    result = await repository.statistics(
        filters
    )

    if isinstance(result, dict):
        return result

    return {
        "total":
            _attr(result, "total", 0),

        "pending":
            _attr(result, "pending", 0),

        "upcoming":
            _attr(result, "upcoming", 0),

        "overdue":
            _attr(result, "overdue", 0),

        "completed":
            _attr(result, "completed", 0),

        "cancelled":
            _attr(result, "cancelled", 0),

        "auto_renewing":
            _attr(
                result,
                "auto_renewing",
                0,
            ),

        "high_risk":
            _attr(
                result,
                "high_risk",
                0,
            ),

        "escalated":
            _attr(
                result,
                "escalated",
                0,
            ),

        "by_status":
            _attr(
                result,
                "by_status",
                {},
            ),

        "by_risk_level":
            _attr(
                result,
                "by_risk_level",
                {},
            ),

        "by_decision":
            _attr(
                result,
                "by_decision",
                {},
            ),
    }


# ============================================================
# CREATE
# ============================================================

@router.post(
    "",
    response_model=RenewalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create renewal record",
)
async def create_renewal(
    payload: RenewalCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "create",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        contract_id=str(
            payload.contract_id
        ),

        renewal_date=payload.renewal_date,

        notice_deadline=(
            payload.notice_deadline
        ),

        renewal_term_days=(
            payload.renewal_term_days
        ),

        notice_period_days=(
            payload.notice_period_days
        ),

        auto_renewal=(
            payload.auto_renewal
        ),

        status=payload.status,

        decision=payload.decision,

        decision_reason=(
            payload.decision_reason
        ),

        owner_user_id=(
            str(payload.owner_user_id)
            if payload.owner_user_id
            else None
        ),

        risk_level=(
            payload.risk_level
        ),

        risk_score=(
            payload.risk_score
        ),

        escalation_level=(
            payload.escalation_level
        ),

        reminder_at=(
            payload.reminder_at
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

    return _serialize_renewal(
        result
    )


# ============================================================
# GET
# ============================================================

@router.get(
    "/{renewal_id}",
    response_model=RenewalResponse,
    summary="Get renewal",
)
async def get_renewal(
    renewal_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: RenewalRepository = Depends(
        get_renewal_repository
    ),
) -> dict[str, Any]:
    renewal = await repository.get_by_id(
        renewal_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if renewal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Renewal was not found.",
        )

    # Defense in depth if repository does not
    # enforce organization scope itself.
    record_org = _optional_str(
        _attr(
            renewal,
            "organization_id",
        )
    )

    current_org = _organization_id(
        current_user
    )

    if (
        record_org
        and current_org
        and record_org != current_org
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Renewal was not found.",
        )

    return _serialize_renewal(
        renewal
    )


# ============================================================
# UPDATE
# ============================================================

@router.patch(
    "/{renewal_id}",
    response_model=RenewalResponse,
    summary="Update renewal",
)
async def update_renewal(
    renewal_id: UUID,
    payload: RenewalUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    values = payload.model_dump(
        exclude_unset=True
    )

    if (
        "owner_user_id"
        in values
        and values["owner_user_id"]
        is not None
    ):
        values["owner_user_id"] = str(
            values["owner_user_id"]
        )

    result = await _service_call(
        service,
        "update",

        renewal_id=str(
            renewal_id
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

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Renewal was not found.",
        )

    return _serialize_renewal(
        result
    )


# ============================================================
# STATUS
# ============================================================

@router.patch(
    "/{renewal_id}/status",
    response_model=RenewalResponse,
    summary="Change renewal status",
)
async def update_renewal_status(
    renewal_id: UUID,
    payload: RenewalStatusRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "update_status",

        renewal_id=str(
            renewal_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        status=payload.status,

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_renewal(
        result
    )


# ============================================================
# DECISION
# ============================================================

@router.post(
    "/{renewal_id}/decision",
    response_model=RenewalDecisionResponse,
    summary="Record renewal decision",
)
async def record_renewal_decision(
    renewal_id: UUID,
    payload: RenewalDecisionRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "record_decision",

        renewal_id=str(
            renewal_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        decision=payload.decision,

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    renewal = (
        _attr(
            result,
            "renewal",
            result,
        )
    )

    return {
        "renewal":
            _serialize_renewal(
                renewal
            ),

        "decision":
            str(
                _attr(
                    result,
                    "decision",
                    payload.decision,
                )
            ),

        "processed_at":
            _attr(
                result,
                "processed_at",
                datetime.now(
                    timezone.utc
                ),
            ),
    }


# ============================================================
# RISK
# ============================================================

@router.patch(
    "/{renewal_id}/risk",
    response_model=RenewalResponse,
    summary="Update renewal risk",
)
async def update_renewal_risk(
    renewal_id: UUID,
    payload: RenewalRiskRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "update_risk",

        renewal_id=str(
            renewal_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        risk_level=(
            payload.risk_level
        ),

        risk_score=(
            payload.risk_score
        ),

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_renewal(
        result
    )


# ============================================================
# ESCALATION
# ============================================================

@router.post(
    "/{renewal_id}/escalate",
    response_model=RenewalResponse,
    summary="Escalate renewal",
)
async def escalate_renewal(
    renewal_id: UUID,
    payload: RenewalEscalationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "escalate",

        renewal_id=str(
            renewal_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        escalation_level=(
            payload.escalation_level
        ),

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_renewal(
        result
    )


@router.post(
    "/{renewal_id}/clear-escalation",
    response_model=RenewalResponse,
    summary="Clear renewal escalation",
)
async def clear_renewal_escalation(
    renewal_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "clear_escalation",

        renewal_id=str(
            renewal_id
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

    return _serialize_renewal(
        result
    )


# ============================================================
# REMINDER
# ============================================================

@router.post(
    "/{renewal_id}/reminder",
    response_model=ReminderResponse,
    summary="Schedule renewal reminder",
)
async def schedule_renewal_reminder(
    renewal_id: UUID,
    payload: RenewalReminderRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "schedule_reminder",

        renewal_id=str(
            renewal_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        reminder_at=(
            payload.reminder_at
        ),

        channel=payload.channel,

        message=payload.message,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return {
        "renewal_id":
            str(renewal_id),

        "scheduled":
            bool(
                _attr(
                    result,
                    "scheduled",
                    True,
                )
            ),

        "reminder_at":
            _attr(
                result,
                "reminder_at",
                payload.reminder_at,
            ),

        "message":
            str(
                _attr(
                    result,
                    "message",
                    "Renewal reminder scheduled.",
                )
            ),
    }


# ============================================================
# PROCESS
#
# Runs renewal business rules for one record:
# expiry/notice/risk/escalation/reminders.
# ============================================================

@router.post(
    "/{renewal_id}/process",
    response_model=RenewalResponse,
    summary="Process renewal lifecycle",
)
async def process_renewal(
    renewal_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, Any]:
    _require_renewal_manager(
        current_user
    )

    result = await _service_call(
        service,
        "process",

        renewal_id=str(
            renewal_id
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

    return _serialize_renewal(
        result
    )


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{renewal_id}",
    response_model=MessageResponse,
    summary="Delete renewal",
)
async def delete_renewal(
    renewal_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_renewal_service
    ),
) -> dict[str, str]:
    _require_renewal_manager(
        current_user
    )

    deleted = await _service_call(
        service,
        "delete",

        renewal_id=str(
            renewal_id
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

    if deleted is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Renewal was not found.",
        )

    return {
        "message":
            "Renewal deleted successfully."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Renewal subsystem health",
)
async def renewal_health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "renewals",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }