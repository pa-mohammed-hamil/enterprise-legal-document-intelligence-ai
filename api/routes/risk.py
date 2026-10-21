# backend/api/routes/risk.py

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

from backend.db.repositories.risk_repository import (
    RiskCreate,
    RiskFilter,
    RiskRepository,
    RiskUpdate,
)


router = APIRouter(
    prefix="/risk",
    tags=["Risk"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies.py
# ============================================================

async def get_risk_repository() -> RiskRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Risk repository dependency is not configured.",
    )


async def get_risk_service() -> Any:
    """
    Recommended RiskService dependencies:

        RiskRepository
        DocumentRepository
        ContractRepository
        ClauseRepository
        ReviewRepository
        NotificationRepository
        AuditRepository

        RiskClassifier
        RiskAgent
        EscalationEngine
        FireEngine

    Service owns lifecycle transitions, model analysis,
    review creation, escalation, notifications and audit.
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Risk service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RiskCreateRequest(BaseModel):
    category: str = Field(
        min_length=1,
        max_length=100,
    )

    level: str = Field(
        min_length=1,
        max_length=50,
    )

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    document_id: Optional[UUID] = None
    contract_id: Optional[UUID] = None
    clause_id: Optional[UUID] = None

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    status: str = Field(
        default="open",
        max_length=50,
    )

    source: str = Field(
        default="system",
        max_length=100,
    )

    evidence: Optional[str] = Field(
        default=None,
        max_length=30_000,
    )

    recommendation: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    mitigation: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    owner_user_id: Optional[UUID] = None

    requires_review: bool = False

    escalation_level: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_run_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    correlation_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RiskUpdateRequest(BaseModel):
    category: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    level: Optional[str] = Field(
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

    evidence: Optional[str] = Field(
        default=None,
        max_length=30_000,
    )

    recommendation: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    mitigation: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    owner_user_id: Optional[UUID] = None

    requires_review: Optional[bool] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


class RiskScoreRequest(BaseModel):
    level: str = Field(
        min_length=1,
        max_length=50,
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class RiskReviewRequest(BaseModel):
    notes: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    create_review_record: bool = False


class RiskMitigationRequest(BaseModel):
    mitigation: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    notes: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )


class RiskResolutionRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    resolution: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )


class RiskDismissRequest(BaseModel):
    reason: str = Field(
        min_length=1,
        max_length=20_000,
    )


class RiskEscalationRequest(BaseModel):
    escalation_level: str = Field(
        min_length=1,
        max_length=50,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    create_review: bool = True
    notify: bool = True


class RiskAnalyzeRequest(BaseModel):
    force: bool = False

    run_classifier: bool = True
    run_agent: bool = True

    create_risks: bool = True

    create_reviews: bool = True

    trigger_escalation: bool = True

    model: Optional[str] = Field(
        default=None,
        max_length=255,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class RiskResponse(BaseModel):
    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    category: str
    level: str

    title: str
    description: Optional[str] = None

    document_id: Optional[str] = None
    contract_id: Optional[str] = None
    clause_id: Optional[str] = None

    score: Optional[float] = None
    confidence: Optional[float] = None

    status: str
    source: str

    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    mitigation: Optional[str] = None

    owner_user_id: Optional[str] = None

    requires_review: bool = False

    reviewed: bool = False
    reviewer_user_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None

    escalation_level: Optional[str] = None
    escalated_at: Optional[datetime] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_run_id: Optional[str] = None

    correlation_id: Optional[str] = None

    detected_at: Optional[datetime] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RiskPageResponse(BaseModel):
    items: list[RiskResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class RiskStatisticsResponse(BaseModel):
    total: int = 0

    open: int = 0
    acknowledged: int = 0
    mitigated: int = 0
    resolved: int = 0
    dismissed: int = 0

    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0

    requires_review: int = 0
    reviewed: int = 0

    escalated: int = 0

    average_score: float = 0.0
    average_confidence: float = 0.0

    by_level: dict[str, int] = Field(
        default_factory=dict,
    )

    by_category: dict[str, int] = Field(
        default_factory=dict,
    )

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_source: dict[str, int] = Field(
        default_factory=dict,
    )

    by_escalation: dict[str, int] = Field(
        default_factory=dict,
    )


class RiskAnalysisResponse(BaseModel):
    resource_type: str
    resource_id: str

    risks_detected: int = 0
    risks_created: int = 0

    high_risk: int = 0
    critical_risk: int = 0

    reviews_created: int = 0
    escalations_triggered: int = 0

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    model_run_id: Optional[str] = None

    latency_ms: Optional[float] = None

    risks: list[RiskResponse] = Field(
        default_factory=list,
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


def _require_risk_manager(
    user: Any,
) -> None:

    allowed = {
        "admin",
        "super_admin",
        "legal_admin",
        "legal",
        "risk_manager",
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
            detail="Risk management permission is required.",
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
    risk: Any,
) -> dict[str, Any]:

    return (
        _attr(
            risk,
            "risk_metadata",
        )
        or _attr(
            risk,
            "metadata_json",
        )
        or {}
    )


def _serialize_risk(
    risk: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                risk,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                risk,
                "tenant_id",
                "",
            )
        ),

        "organization_id":
            _optional_str(
                _attr(
                    risk,
                    "organization_id",
                )
            ),

        "category": str(
            _enum_value(
                _attr(
                    risk,
                    "category",
                    "other",
                )
            )
        ),

        "level": str(
            _enum_value(
                _attr(
                    risk,
                    "level",
                    "low",
                )
            )
        ),

        "title": str(
            _attr(
                risk,
                "title",
                "",
            )
        ),

        "description":
            _attr(
                risk,
                "description",
            ),

        "document_id":
            _optional_str(
                _attr(
                    risk,
                    "document_id",
                )
            ),

        "contract_id":
            _optional_str(
                _attr(
                    risk,
                    "contract_id",
                )
            ),

        "clause_id":
            _optional_str(
                _attr(
                    risk,
                    "clause_id",
                )
            ),

        "score":
            _attr(
                risk,
                "score",
            ),

        "confidence":
            _attr(
                risk,
                "confidence",
            ),

        "status": str(
            _enum_value(
                _attr(
                    risk,
                    "status",
                    "open",
                )
            )
        ),

        "source": str(
            _enum_value(
                _attr(
                    risk,
                    "source",
                    "system",
                )
            )
        ),

        "evidence":
            _attr(
                risk,
                "evidence",
            ),

        "recommendation":
            _attr(
                risk,
                "recommendation",
            ),

        "mitigation":
            _attr(
                risk,
                "mitigation",
            ),

        "owner_user_id":
            _optional_str(
                _attr(
                    risk,
                    "owner_user_id",
                )
            ),

        "requires_review":
            bool(
                _attr(
                    risk,
                    "requires_review",
                    False,
                )
            ),

        "reviewed":
            bool(
                _attr(
                    risk,
                    "reviewed",
                    False,
                )
            ),

        "reviewer_user_id":
            _optional_str(
                _attr(
                    risk,
                    "reviewer_user_id",
                    _attr(
                        risk,
                        "reviewer_id",
                    ),
                )
            ),

        "reviewed_at":
            _attr(
                risk,
                "reviewed_at",
            ),

        "review_notes":
            _attr(
                risk,
                "review_notes",
            ),

        "escalation_level":
            _optional_str(
                _enum_value(
                    _attr(
                        risk,
                        "escalation_level",
                    )
                )
            ),

        "escalated_at":
            _attr(
                risk,
                "escalated_at",
            ),

        "model_name":
            _attr(
                risk,
                "model_name",
            ),

        "model_version":
            _attr(
                risk,
                "model_version",
            ),

        "model_run_id":
            _optional_str(
                _attr(
                    risk,
                    "model_run_id",
                )
            ),

        "correlation_id":
            _optional_str(
                _attr(
                    risk,
                    "correlation_id",
                )
            ),

        "detected_at":
            _attr(
                risk,
                "detected_at",
            ),

        "metadata":
            _metadata(risk),

        "created_at":
            _attr(
                risk,
                "created_at",
            ),

        "updated_at":
            _attr(
                risk,
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
            _serialize_risk(item)
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
                f"Risk operation '{operation}' "
                "is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# LIST RISKS
# ============================================================

@router.get(
    "",
    response_model=RiskPageResponse,
    summary="List risks",
)
async def list_risks(
    category: Optional[str] = None,
    level: Optional[str] = None,

    risk_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),

    source: Optional[str] = None,

    document_id: Optional[UUID] = None,
    contract_id: Optional[UUID] = None,
    clause_id: Optional[UUID] = None,

    owner_user_id: Optional[UUID] = None,

    requires_review: Optional[bool] = None,
    reviewed: Optional[bool] = None,

    escalation_level: Optional[str] = None,

    minimum_score: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),

    maximum_score: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),

    minimum_confidence: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),

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

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> dict[str, Any]:

    filters = RiskFilter(
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

    if category:
        if hasattr(filters, "categories"):
            filters.categories = [category]
        else:
            _set_filter(
                filters,
                "category",
                category,
            )

    if level:
        if hasattr(filters, "levels"):
            filters.levels = [level]
        else:
            _set_filter(
                filters,
                "level",
                level,
            )

    if risk_status:
        if hasattr(filters, "statuses"):
            filters.statuses = [risk_status]
        else:
            _set_filter(
                filters,
                "status",
                risk_status,
            )

    if source:
        if hasattr(filters, "sources"):
            filters.sources = [source]
        else:
            _set_filter(
                filters,
                "source",
                source,
            )

    for field, value in (
        ("document_id", document_id),
        ("contract_id", contract_id),
        ("clause_id", clause_id),
        ("owner_user_id", owner_user_id),
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
        "requires_review",
        requires_review,
    )

    _set_filter(
        filters,
        "reviewed",
        reviewed,
    )

    _set_filter(
        filters,
        "escalation_level",
        escalation_level,
    )

    _set_filter(
        filters,
        "min_score",
        minimum_score,
    )

    _set_filter(
        filters,
        "max_score",
        maximum_score,
    )

    _set_filter(
        filters,
        "min_confidence",
        minimum_confidence,
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
# CRITICAL
# Static routes before /{risk_id}
# ============================================================

@router.get(
    "/critical",
    response_model=RiskPageResponse,
    summary="List critical risks",
)
async def critical_risks(
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

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_critical(
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
    "/high",
    response_model=RiskPageResponse,
    summary="List high-risk items",
)
async def high_risks(
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

    repository: RiskRepository = Depends(
        get_risk_repository
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
# PENDING REVIEW
# ============================================================

@router.get(
    "/pending-review",
    response_model=RiskPageResponse,
    summary="List risks pending human review",
)
async def pending_review_risks(
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

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_pending_review(
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
# MY RISKS
# ============================================================

@router.get(
    "/mine",
    response_model=RiskPageResponse,
    summary="List risks owned by current user",
)
async def my_risks(
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

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_for_owner(
        tenant_id=_tenant_id(
            current_user
        ),

        owner_user_id=_user_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=RiskStatisticsResponse,
    summary="Get risk statistics",
)
async def risk_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> Any:

    filters = RiskFilter(
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

        "open":
            _attr(result, "open", 0),

        "acknowledged":
            _attr(
                result,
                "acknowledged",
                0,
            ),

        "mitigated":
            _attr(
                result,
                "mitigated",
                0,
            ),

        "resolved":
            _attr(
                result,
                "resolved",
                0,
            ),

        "dismissed":
            _attr(
                result,
                "dismissed",
                0,
            ),

        "low":
            _attr(result, "low", 0),

        "medium":
            _attr(
                result,
                "medium",
                0,
            ),

        "high":
            _attr(result, "high", 0),

        "critical":
            _attr(
                result,
                "critical",
                0,
            ),

        "requires_review":
            _attr(
                result,
                "requires_review",
                0,
            ),

        "reviewed":
            _attr(
                result,
                "reviewed",
                0,
            ),

        "escalated":
            _attr(
                result,
                "escalated",
                0,
            ),

        "average_score":
            _attr(
                result,
                "average_score",
                _attr(
                    result,
                    "avg_score",
                    0.0,
                ),
            ),

        "average_confidence":
            _attr(
                result,
                "average_confidence",
                _attr(
                    result,
                    "avg_confidence",
                    0.0,
                ),
            ),

        "by_level":
            _attr(
                result,
                "by_level",
                {},
            ),

        "by_category":
            _attr(
                result,
                "by_category",
                {},
            ),

        "by_status":
            _attr(
                result,
                "by_status",
                {},
            ),

        "by_source":
            _attr(
                result,
                "by_source",
                {},
            ),

        "by_escalation":
            _attr(
                result,
                "by_escalation",
                {},
            ),
    }


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

@router.post(
    "/analyze/document/{document_id}",
    response_model=RiskAnalysisResponse,
    summary="Analyze document risks",
)
async def analyze_document_risk(
    document_id: UUID,
    payload: RiskAnalyzeRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "analyze_document",

        document_id=str(document_id),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        force=payload.force,

        run_classifier=payload.run_classifier,

        run_agent=payload.run_agent,

        create_risks=payload.create_risks,

        create_reviews=payload.create_reviews,

        trigger_escalation=(
            payload.trigger_escalation
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_analysis(
        result,
        resource_type="document",
        resource_id=str(document_id),
    )


# ============================================================
# ANALYZE CONTRACT
# ============================================================

@router.post(
    "/analyze/contract/{contract_id}",
    response_model=RiskAnalysisResponse,
    summary="Analyze contract risks",
)
async def analyze_contract_risk(
    contract_id: UUID,
    payload: RiskAnalyzeRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "analyze_contract",

        contract_id=str(contract_id),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        force=payload.force,

        run_classifier=payload.run_classifier,

        run_agent=payload.run_agent,

        create_risks=payload.create_risks,

        create_reviews=payload.create_reviews,

        trigger_escalation=(
            payload.trigger_escalation
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_analysis(
        result,
        resource_type="contract",
        resource_id=str(contract_id),
    )


# ============================================================
# ANALYZE CLAUSE
# ============================================================

@router.post(
    "/analyze/clause/{clause_id}",
    response_model=RiskAnalysisResponse,
    summary="Analyze clause risks",
)
async def analyze_clause_risk(
    clause_id: UUID,
    payload: RiskAnalyzeRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "analyze_clause",

        clause_id=str(clause_id),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        force=payload.force,

        run_classifier=payload.run_classifier,

        run_agent=payload.run_agent,

        create_risks=payload.create_risks,

        create_reviews=payload.create_reviews,

        trigger_escalation=(
            payload.trigger_escalation
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_analysis(
        result,
        resource_type="clause",
        resource_id=str(clause_id),
    )


def _serialize_analysis(
    result: Any,
    *,
    resource_type: str,
    resource_id: str,
) -> dict[str, Any]:

    risks = list(
        _attr(
            result,
            "risks",
            [],
        )
        or []
    )

    serialized = [
        _serialize_risk(risk)
        for risk in risks
    ]

    high_count = sum(
        1
        for risk in serialized
        if risk["level"].lower()
        == "high"
    )

    critical_count = sum(
        1
        for risk in serialized
        if risk["level"].lower()
        == "critical"
    )

    return {
        "resource_type":
            resource_type,

        "resource_id":
            resource_id,

        "risks_detected":
            int(
                _attr(
                    result,
                    "risks_detected",
                    len(serialized),
                )
                or 0
            ),

        "risks_created":
            int(
                _attr(
                    result,
                    "risks_created",
                    len(serialized),
                )
                or 0
            ),

        "high_risk":
            int(
                _attr(
                    result,
                    "high_risk",
                    high_count,
                )
                or 0
            ),

        "critical_risk":
            int(
                _attr(
                    result,
                    "critical_risk",
                    critical_count,
                )
                or 0
            ),

        "reviews_created":
            int(
                _attr(
                    result,
                    "reviews_created",
                    0,
                )
                or 0
            ),

        "escalations_triggered":
            int(
                _attr(
                    result,
                    "escalations_triggered",
                    0,
                )
                or 0
            ),

        "model_name":
            _attr(
                result,
                "model_name",
            ),

        "model_version":
            _attr(
                result,
                "model_version",
            ),

        "model_run_id":
            _optional_str(
                _attr(
                    result,
                    "model_run_id",
                )
            ),

        "latency_ms":
            _attr(
                result,
                "latency_ms",
            ),

        "risks":
            serialized,
    }


# ============================================================
# CREATE
# ============================================================

@router.post(
    "",
    response_model=RiskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create risk",
)
async def create_risk(
    payload: RiskCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    if not any(
        (
            payload.document_id,
            payload.contract_id,
            payload.clause_id,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Risk must reference a document, "
                "contract, or clause."
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

        category=payload.category,

        level=payload.level,

        title=payload.title.strip(),

        description=payload.description,

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

        score=payload.score,

        confidence=payload.confidence,

        status=payload.status,

        source=payload.source,

        evidence=payload.evidence,

        recommendation=(
            payload.recommendation
        ),

        mitigation=payload.mitigation,

        owner_user_id=(
            str(payload.owner_user_id)
            if payload.owner_user_id
            else None
        ),

        requires_review=(
            payload.requires_review
        ),

        escalation_level=(
            payload.escalation_level
        ),

        model_name=payload.model_name,

        model_version=payload.model_version,

        model_run_id=payload.model_run_id,

        correlation_id=(
            payload.correlation_id
            or _request_id(request)
        ),

        metadata=payload.metadata,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# GET RISK
# ============================================================

@router.get(
    "/{risk_id}",
    response_model=RiskResponse,
    summary="Get risk",
)
async def get_risk(
    risk_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: RiskRepository = Depends(
        get_risk_repository
    ),
) -> dict[str, Any]:

    risk = await repository.get_by_id(
        risk_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if risk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk was not found.",
        )

    record_org = _optional_str(
        _attr(
            risk,
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
            detail="Risk was not found.",
        )

    return _serialize_risk(risk)


# ============================================================
# UPDATE
# ============================================================

@router.patch(
    "/{risk_id}",
    response_model=RiskResponse,
    summary="Update risk",
)
async def update_risk(
    risk_id: UUID,
    payload: RiskUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    values = payload.model_dump(
        exclude_unset=True
    )

    if (
        "owner_user_id" in values
        and values["owner_user_id"]
        is not None
    ):
        values["owner_user_id"] = str(
            values["owner_user_id"]
        )

    result = await _service_call(
        service,
        "update",

        risk_id=str(risk_id),

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

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk was not found.",
        )

    return _serialize_risk(result)


# ============================================================
# UPDATE SCORE
# ============================================================

@router.patch(
    "/{risk_id}/score",
    response_model=RiskResponse,
    summary="Update risk score",
)
async def update_risk_score(
    risk_id: UUID,
    payload: RiskScoreRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "update_score",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        level=payload.level,

        score=payload.score,

        confidence=payload.confidence,

        reason=payload.reason,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# ACKNOWLEDGE
# ============================================================

@router.post(
    "/{risk_id}/acknowledge",
    response_model=RiskResponse,
    summary="Acknowledge risk",
)
async def acknowledge_risk(
    risk_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "acknowledge",

        risk_id=str(risk_id),

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

    return _serialize_risk(result)


# ============================================================
# START MITIGATION
# ============================================================

@router.post(
    "/{risk_id}/mitigation/start",
    response_model=RiskResponse,
    summary="Start risk mitigation",
)
async def start_mitigation(
    risk_id: UUID,
    payload: RiskMitigationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "start_mitigation",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        mitigation=payload.mitigation,

        notes=payload.notes,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# MARK MITIGATED
# ============================================================

@router.post(
    "/{risk_id}/mitigation/complete",
    response_model=RiskResponse,
    summary="Mark risk as mitigated",
)
async def complete_mitigation(
    risk_id: UUID,
    payload: RiskMitigationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "mark_mitigated",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        mitigation=payload.mitigation,

        notes=payload.notes,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# RESOLVE
# ============================================================

@router.post(
    "/{risk_id}/resolve",
    response_model=RiskResponse,
    summary="Resolve risk",
)
async def resolve_risk(
    risk_id: UUID,
    payload: RiskResolutionRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "resolve",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reason=payload.reason,

        resolution=payload.resolution,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# DISMISS
# ============================================================

@router.post(
    "/{risk_id}/dismiss",
    response_model=RiskResponse,
    summary="Dismiss risk",
)
async def dismiss_risk(
    risk_id: UUID,
    payload: RiskDismissRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "dismiss",

        risk_id=str(risk_id),

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

    return _serialize_risk(result)


# ============================================================
# REVIEW
# ============================================================

@router.post(
    "/{risk_id}/review",
    response_model=RiskResponse,
    summary="Mark risk reviewed",
)
async def review_risk(
    risk_id: UUID,
    payload: RiskReviewRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "mark_reviewed",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        reviewer_user_id=_user_id(
            current_user
        ),

        notes=payload.notes,

        create_review_record=(
            payload.create_review_record
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


@router.post(
    "/{risk_id}/review/reset",
    response_model=RiskResponse,
    summary="Reset risk review",
)
async def reset_risk_review(
    risk_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "reset_review",

        risk_id=str(risk_id),

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

    return _serialize_risk(result)


# ============================================================
# ESCALATE
# ============================================================

@router.post(
    "/{risk_id}/escalate",
    response_model=RiskResponse,
    summary="Escalate risk",
)
async def escalate_risk(
    risk_id: UUID,
    payload: RiskEscalationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "escalate",

        risk_id=str(risk_id),

        tenant_id=_tenant_id(
            current_user
        ),

        escalation_level=(
            payload.escalation_level
        ),

        reason=payload.reason,

        create_review=(
            payload.create_review
        ),

        notify=payload.notify,

        actor_user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_risk(result)


# ============================================================
# CLEAR ESCALATION
# ============================================================

@router.post(
    "/{risk_id}/clear-escalation",
    response_model=RiskResponse,
    summary="Clear risk escalation",
)
async def clear_risk_escalation(
    risk_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, Any]:

    _require_risk_manager(
        current_user
    )

    result = await _service_call(
        service,
        "clear_escalation",

        risk_id=str(risk_id),

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

    return _serialize_risk(result)


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{risk_id}",
    response_model=MessageResponse,
    summary="Delete risk",
)
async def delete_risk(
    risk_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_risk_service
    ),
) -> dict[str, str]:

    _require_risk_manager(
        current_user
    )

    deleted = await _service_call(
        service,
        "delete",

        risk_id=str(risk_id),

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
            detail="Risk was not found.",
        )

    return {
        "message":
            "Risk deleted successfully."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Risk subsystem health",
)
async def risk_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "risk",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }