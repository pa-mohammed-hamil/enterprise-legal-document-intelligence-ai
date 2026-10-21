# backend/api/routes/reviews.py

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

from backend.db.repositories.review_repository import (
    ReviewCreate,
    ReviewFilter,
    ReviewRepository,
    ReviewUpdate,
)


router = APIRouter(
    prefix="/reviews",
    tags=["Reviews"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies.py
# ============================================================

async def get_review_repository() -> ReviewRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Review repository dependency is not configured.",
    )


async def get_review_service() -> Any:
    """
    Recommended ReviewService dependencies:

        ReviewRepository
        DocumentRepository
        ContractRepository
        ClauseRepository
        RiskRepository
        UserRepository
        NotificationRepository
        AuditRepository
        EscalationEngine

    ReviewService should own workflow transitions,
    authorization, notifications and audit history.
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Review service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ReviewCreateRequest(BaseModel):
    review_type: str = Field(
        default="legal",
        min_length=1,
        max_length=100,
    )

    priority: str = Field(
        default="normal",
        min_length=1,
        max_length=50,
    )

    document_id: Optional[UUID] = None
    contract_id: Optional[UUID] = None
    clause_id: Optional[UUID] = None
    risk_id: Optional[UUID] = None

    assigned_to: Optional[UUID] = None

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    due_at: Optional[datetime] = None

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

    source: str = Field(
        default="manual",
        max_length=100,
    )

    correlation_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ReviewUpdateRequest(BaseModel):
    review_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    priority: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    due_at: Optional[datetime] = None

    risk_level: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: Optional[
        dict[str, Any]
    ] = None


class ReviewAssignRequest(BaseModel):
    reviewer_id: UUID

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ReviewDecisionRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    notes: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )


class ReviewChangesRequest(BaseModel):
    reason: str = Field(
        min_length=1,
        max_length=20_000,
    )

    notes: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )


class ReviewRiskRequest(BaseModel):
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


class ReviewEscalationRequest(BaseModel):
    escalation_level: str = Field(
        min_length=1,
        max_length=50,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ReviewCancelRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ReviewResponse(BaseModel):
    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    review_type: str
    status: str
    priority: str

    requested_by: str
    assigned_to: Optional[str] = None

    document_id: Optional[str] = None
    contract_id: Optional[str] = None
    clause_id: Optional[str] = None
    risk_id: Optional[str] = None

    title: str
    description: Optional[str] = None

    due_at: Optional[datetime] = None

    decision: Optional[str] = None
    decision_reason: Optional[str] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    escalation_level: Optional[str] = None
    escalated_at: Optional[datetime] = None

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    source: Optional[str] = None
    correlation_id: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReviewPageResponse(BaseModel):
    items: list[ReviewResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class ReviewStatisticsResponse(BaseModel):
    total: int = 0

    pending: int = 0
    assigned: int = 0
    in_review: int = 0
    completed: int = 0
    changes_requested: int = 0
    cancelled: int = 0

    overdue: int = 0
    unassigned: int = 0
    escalated: int = 0

    high_priority: int = 0
    high_risk: int = 0

    approved: int = 0
    rejected: int = 0

    approval_rate: float = 0.0
    completion_rate: float = 0.0

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_priority: dict[str, int] = Field(
        default_factory=dict,
    )

    by_risk_level: dict[str, int] = Field(
        default_factory=dict,
    )


class ReviewDecisionResponse(BaseModel):
    review: ReviewResponse

    decision: str

    reviewer_id: str

    processed_at: datetime


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


def _enum_value(value: Any) -> Any:
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


def _tenant_id(user: Any) -> str:
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


def _user_id(user: Any) -> str:
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


def _roles(user: Any) -> set[str]:
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


def _require_review_manager(
    user: Any,
) -> None:
    allowed = {
        "admin",
        "super_admin",
        "legal_admin",
        "legal",
        "reviewer",
        "review_manager",
        "contract_manager",
        "compliance",
        "compliance_officer",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Review management permission is required.",
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
    review: Any,
) -> dict[str, Any]:
    return (
        _attr(
            review,
            "review_metadata",
        )
        or _attr(
            review,
            "metadata_json",
        )
        or {}
    )


def _serialize_review(
    review: Any,
) -> dict[str, Any]:
    return {
        "id": str(
            _attr(
                review,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                review,
                "tenant_id",
                "",
            )
        ),

        "organization_id":
            _optional_str(
                _attr(
                    review,
                    "organization_id",
                )
            ),

        "review_type":
            str(
                _enum_value(
                    _attr(
                        review,
                        "review_type",
                        "legal",
                    )
                )
            ),

        "status":
            str(
                _enum_value(
                    _attr(
                        review,
                        "status",
                        "pending",
                    )
                )
            ),

        "priority":
            str(
                _enum_value(
                    _attr(
                        review,
                        "priority",
                        "normal",
                    )
                )
            ),

        "requested_by":
            str(
                _attr(
                    review,
                    "requested_by",
                    "",
                )
            ),

        "assigned_to":
            _optional_str(
                _attr(
                    review,
                    "assigned_to",
                )
            ),

        "document_id":
            _optional_str(
                _attr(
                    review,
                    "document_id",
                )
            ),

        "contract_id":
            _optional_str(
                _attr(
                    review,
                    "contract_id",
                )
            ),

        "clause_id":
            _optional_str(
                _attr(
                    review,
                    "clause_id",
                )
            ),

        "risk_id":
            _optional_str(
                _attr(
                    review,
                    "risk_id",
                )
            ),

        "title":
            str(
                _attr(
                    review,
                    "title",
                    "",
                )
            ),

        "description":
            _attr(
                review,
                "description",
            ),

        "due_at":
            _attr(
                review,
                "due_at",
            ),

        "decision":
            _optional_str(
                _enum_value(
                    _attr(
                        review,
                        "decision",
                    )
                )
            ),

        "decision_reason":
            _attr(
                review,
                "decision_reason",
            ),

        "risk_level":
            _optional_str(
                _enum_value(
                    _attr(
                        review,
                        "risk_level",
                    )
                )
            ),

        "risk_score":
            _attr(
                review,
                "risk_score",
            ),

        "escalation_level":
            _optional_str(
                _enum_value(
                    _attr(
                        review,
                        "escalation_level",
                    )
                )
            ),

        "escalated_at":
            _attr(
                review,
                "escalated_at",
            ),

        "started_at":
            _attr(
                review,
                "started_at",
            ),

        "completed_at":
            _attr(
                review,
                "completed_at",
            ),

        "cancelled_at":
            _attr(
                review,
                "cancelled_at",
            ),

        "source":
            _optional_str(
                _attr(
                    review,
                    "source",
                )
            ),

        "correlation_id":
            _optional_str(
                _attr(
                    review,
                    "correlation_id",
                )
            ),

        "metadata":
            _metadata(review),

        "created_at":
            _attr(
                review,
                "created_at",
            ),

        "updated_at":
            _attr(
                review,
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
            _serialize_review(item)
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
                f"Review operation '{operation}' "
                "is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# LIST
# ============================================================

@router.get(
    "",
    response_model=ReviewPageResponse,
    summary="List reviews",
)
async def list_reviews(
    review_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),

    review_type: Optional[str] = None,
    priority: Optional[str] = None,

    document_id: Optional[UUID] = None,
    contract_id: Optional[UUID] = None,
    clause_id: Optional[UUID] = None,
    risk_id: Optional[UUID] = None,

    assigned_to: Optional[UUID] = None,
    requested_by: Optional[UUID] = None,

    risk_level: Optional[str] = None,

    overdue_only: bool = False,

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

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    filters = ReviewFilter(
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

    if review_status:
        if hasattr(filters, "statuses"):
            filters.statuses = [
                review_status
            ]
        else:
            _set_filter(
                filters,
                "status",
                review_status,
            )

    if review_type:
        if hasattr(filters, "review_types"):
            filters.review_types = [
                review_type
            ]
        else:
            _set_filter(
                filters,
                "review_type",
                review_type,
            )

    if priority:
        if hasattr(filters, "priorities"):
            filters.priorities = [
                priority
            ]
        else:
            _set_filter(
                filters,
                "priority",
                priority,
            )

    for field, value in (
        ("document_id", document_id),
        ("contract_id", contract_id),
        ("clause_id", clause_id),
        ("risk_id", risk_id),
        ("assigned_to", assigned_to),
        ("requested_by", requested_by),
    ):
        _set_filter(
            filters,
            field,
            str(value)
            if value
            else None,
        )

    _set_filter(
        filters,
        "risk_level",
        risk_level,
    )

    _set_filter(
        filters,
        "overdue_only",
        overdue_only,
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
# MY REVIEWS
# Static routes must stay above /{review_id}
# ============================================================

@router.get(
    "/mine",
    response_model=ReviewPageResponse,
    summary="List reviews assigned to current user",
)
async def my_reviews(
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

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_for_reviewer(
        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=_user_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# REQUESTED BY ME
# ============================================================

@router.get(
    "/requested-by-me",
    response_model=ReviewPageResponse,
    summary="List reviews requested by current user",
)
async def requested_by_me(
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

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_for_requester(
        tenant_id=_tenant_id(
            current_user
        ),

        requester_id=_user_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# UNASSIGNED
# ============================================================

@router.get(
    "/unassigned",
    response_model=ReviewPageResponse,
    summary="List unassigned reviews",
)
async def unassigned_reviews(
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

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    page = await repository.list_unassigned(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# OVERDUE
# ============================================================

@router.get(
    "/overdue",
    response_model=ReviewPageResponse,
    summary="List overdue reviews",
)
async def overdue_reviews(
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

    repository: ReviewRepository = Depends(
        get_review_repository
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


# ============================================================
# HIGH PRIORITY
# ============================================================

@router.get(
    "/high-priority",
    response_model=ReviewPageResponse,
    summary="List high-priority reviews",
)
async def high_priority_reviews(
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

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_high_priority(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# HIGH RISK
# ============================================================

@router.get(
    "/high-risk",
    response_model=ReviewPageResponse,
    summary="List high-risk reviews",
)
async def high_risk_reviews(
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

    repository: ReviewRepository = Depends(
        get_review_repository
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
    response_model=ReviewStatisticsResponse,
    summary="Get review statistics",
)
async def review_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> Any:

    filters = ReviewFilter(
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

    total = int(
        _attr(result, "total", 0)
        or 0
    )

    completed = int(
        _attr(
            result,
            "completed",
            0,
        )
        or 0
    )

    approved = int(
        _attr(
            result,
            "approved",
            0,
        )
        or 0
    )

    return {
        "total": total,

        "pending":
            _attr(result, "pending", 0),

        "assigned":
            _attr(result, "assigned", 0),

        "in_review":
            _attr(result, "in_review", 0),

        "completed": completed,

        "changes_requested":
            _attr(
                result,
                "changes_requested",
                0,
            ),

        "cancelled":
            _attr(
                result,
                "cancelled",
                0,
            ),

        "overdue":
            _attr(
                result,
                "overdue",
                0,
            ),

        "unassigned":
            _attr(
                result,
                "unassigned",
                0,
            ),

        "escalated":
            _attr(
                result,
                "escalated",
                0,
            ),

        "high_priority":
            _attr(
                result,
                "high_priority",
                0,
            ),

        "high_risk":
            _attr(
                result,
                "high_risk",
                0,
            ),

        "approved": approved,

        "rejected":
            _attr(
                result,
                "rejected",
                0,
            ),

        "approval_rate":
            _attr(
                result,
                "approval_rate",
                (
                    approved / completed
                    if completed
                    else 0.0
                ),
            ),

        "completion_rate":
            _attr(
                result,
                "completion_rate",
                (
                    completed / total
                    if total
                    else 0.0
                ),
            ),

        "by_status":
            _attr(
                result,
                "by_status",
                {},
            ),

        "by_type":
            _attr(
                result,
                "by_type",
                {},
            ),

        "by_priority":
            _attr(
                result,
                "by_priority",
                {},
            ),

        "by_risk_level":
            _attr(
                result,
                "by_risk_level",
                {},
            ),
    }


# ============================================================
# CREATE
# ============================================================

@router.post(
    "",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create review",
)
async def create_review(
    payload: ReviewCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    # A review should point to at least one real resource.
    if not any(
        (
            payload.document_id,
            payload.contract_id,
            payload.clause_id,
            payload.risk_id,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "A review must reference a document, "
                "contract, clause, or risk."
            ),
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

        requested_by=_user_id(
            current_user
        ),

        review_type=payload.review_type,

        priority=payload.priority,

        document_id=(
            str(payload.document_id)
            if payload.document_id
            else None
        ),

        contract_id=(
            str(payload.contract_id)
            if payload.contract_id
            else None
        ),

        clause_id=(
            str(payload.clause_id)
            if payload.clause_id
            else None
        ),

        risk_id=(
            str(payload.risk_id)
            if payload.risk_id
            else None
        ),

        assigned_to=(
            str(payload.assigned_to)
            if payload.assigned_to
            else None
        ),

        title=payload.title.strip(),

        description=payload.description,

        due_at=payload.due_at,

        risk_level=payload.risk_level,

        risk_score=payload.risk_score,

        escalation_level=(
            payload.escalation_level
        ),

        source=payload.source,

        correlation_id=(
            payload.correlation_id
            or _request_id(request)
        ),

        metadata=payload.metadata,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_review(result)


# ============================================================
# GET
# ============================================================

@router.get(
    "/{review_id}",
    response_model=ReviewResponse,
    summary="Get review",
)
async def get_review(
    review_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ReviewRepository = Depends(
        get_review_repository
    ),
) -> dict[str, Any]:

    review = await repository.get_by_id(
        review_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review was not found.",
        )

    record_org = _optional_str(
        _attr(
            review,
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
            detail="Review was not found.",
        )

    return _serialize_review(review)


# ============================================================
# UPDATE
# ============================================================

@router.patch(
    "/{review_id}",
    response_model=ReviewResponse,
    summary="Update review",
)
async def update_review(
    review_id: UUID,
    payload: ReviewUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    result = await _service_call(
        service,
        "update",

        review_id=str(review_id),

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

        **payload.model_dump(
            exclude_unset=True
        ),
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review was not found.",
        )

    return _serialize_review(result)


# ============================================================
# ASSIGN
# ============================================================

@router.post(
    "/{review_id}/assign",
    response_model=ReviewResponse,
    summary="Assign review",
)
async def assign_review(
    review_id: UUID,
    payload: ReviewAssignRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    result = await _service_call(
        service,
        "assign",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=str(
            payload.reviewer_id
        ),

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_review(result)


# ============================================================
# UNASSIGN
# ============================================================

@router.post(
    "/{review_id}/unassign",
    response_model=ReviewResponse,
    summary="Unassign review",
)
async def unassign_review(
    review_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    result = await _service_call(
        service,
        "unassign",

        review_id=str(review_id),

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

    return _serialize_review(result)


# ============================================================
# START REVIEW
# ============================================================

@router.post(
    "/{review_id}/start",
    response_model=ReviewResponse,
    summary="Start review",
)
async def start_review(
    review_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "start_review",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_review(result)


# ============================================================
# APPROVE
# ============================================================

@router.post(
    "/{review_id}/approve",
    response_model=ReviewDecisionResponse,
    summary="Approve review",
)
async def approve_review(
    review_id: UUID,
    payload: ReviewDecisionRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "approve",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=_user_id(
            current_user
        ),

        reason=payload.reason,

        notes=payload.notes,

        request_id=_request_id(
            request
        ),
    )

    review = _attr(
        result,
        "review",
        result,
    )

    return {
        "review":
            _serialize_review(review),

        "decision":
            "approved",

        "reviewer_id":
            _user_id(current_user),

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
# REJECT
# ============================================================

@router.post(
    "/{review_id}/reject",
    response_model=ReviewDecisionResponse,
    summary="Reject review",
)
async def reject_review(
    review_id: UUID,
    payload: ReviewDecisionRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "reject",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=_user_id(
            current_user
        ),

        reason=payload.reason,

        notes=payload.notes,

        request_id=_request_id(
            request
        ),
    )

    review = _attr(
        result,
        "review",
        result,
    )

    return {
        "review":
            _serialize_review(review),

        "decision":
            "rejected",

        "reviewer_id":
            _user_id(current_user),

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
# REQUEST CHANGES
# ============================================================

@router.post(
    "/{review_id}/request-changes",
    response_model=ReviewDecisionResponse,
    summary="Request review changes",
)
async def request_review_changes(
    review_id: UUID,
    payload: ReviewChangesRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "request_changes",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_id=_user_id(
            current_user
        ),

        reason=payload.reason,

        notes=payload.notes,

        request_id=_request_id(
            request
        ),
    )

    review = _attr(
        result,
        "review",
        result,
    )

    return {
        "review":
            _serialize_review(review),

        "decision":
            "changes_requested",

        "reviewer_id":
            _user_id(current_user),

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
    "/{review_id}/risk",
    response_model=ReviewResponse,
    summary="Update review risk",
)
async def update_review_risk(
    review_id: UUID,
    payload: ReviewRiskRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    result = await _service_call(
        service,
        "update_risk",

        review_id=str(review_id),

        tenant_id=_tenant_id(
            current_user
        ),

        risk_level=payload.risk_level,

        risk_score=payload.risk_score,

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_review(result)


# ============================================================
# ESCALATE
# ============================================================

@router.post(
    "/{review_id}/escalate",
    response_model=ReviewResponse,
    summary="Escalate review",
)
async def escalate_review(
    review_id: UUID,
    payload: ReviewEscalationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    _require_review_manager(
        current_user
    )

    result = await _service_call(
        service,
        "escalate",

        review_id=str(review_id),

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

    return _serialize_review(result)


# ============================================================
# CANCEL
# ============================================================

@router.post(
    "/{review_id}/cancel",
    response_model=ReviewResponse,
    summary="Cancel review",
)
async def cancel_review(
    review_id: UUID,
    payload: ReviewCancelRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "cancel",

        review_id=str(review_id),

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

    return _serialize_review(result)


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{review_id}",
    response_model=MessageResponse,
    summary="Delete review",
)
async def delete_review(
    review_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_review_service
    ),
) -> dict[str, str]:

    _require_review_manager(
        current_user
    )

    deleted = await _service_call(
        service,
        "delete",

        review_id=str(review_id),

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
            detail="Review was not found.",
        )

    return {
        "message":
            "Review deleted successfully."
    }


# ============================================================
# HEALTH
#
# IMPORTANT:
# Static route is intentionally declared before any alternate
# catch-all patterns. With UUID review_id the collision risk is
# lower, but keeping static routes grouped first is safer.
# ============================================================

@router.get(
    "/health/service",
    summary="Review subsystem health",
)
async def review_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "reviews",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }