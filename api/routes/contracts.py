# backend/api/routes/contracts.py

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.db.repositories.contract_repository import (
    ContractCreate,
    ContractFilter,
    ContractRepository,
    ContractUpdate,
)


router = APIRouter(
    prefix="/contracts",
    tags=["Contracts"],
)


# ============================================================
# DEPENDENCIES
# ============================================================

async def get_contract_repository() -> ContractRepository:
    """
    Replace with the real provider from backend.dependencies:

        async def get_contract_repository(
            session: AsyncSession = Depends(get_db),
        ) -> ContractRepository:
            return ContractRepository(session)
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Contract repository dependency is not configured.",
    )


async def get_contract_service() -> Any:
    """
    Recommended ContractService dependencies:

        ContractRepository
        DocumentRepository
        ClauseRepository
        RiskRepository
        RenewalRepository
        ReviewRepository
        AuditService
        EscalationEngine

    Service-level operations should handle:
        - ACL verification
        - lifecycle validation
        - risk analysis
        - renewal creation
        - expiry processing
        - audit logging
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Contract service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ContractCreateRequest(BaseModel):
    document_id: UUID

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    contract_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    status: str = Field(
        default="draft",
        max_length=100,
    )

    contract_number: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    counterparty_name: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    counterparty_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    owner_user_id: Optional[UUID] = None

    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    signed_date: Optional[date] = None

    renewal_date: Optional[date] = None
    notice_date: Optional[date] = None

    auto_renewal: bool = False

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

    governing_law: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    jurisdiction: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    contract_value: Optional[Decimal] = Field(
        default=None,
        ge=0,
    )

    risk_level: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    review_status: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    source: str = Field(
        default="manual",
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        return value.upper() if value else None


class ContractUpdateRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    contract_type: Optional[str] = None
    status: Optional[str] = None

    contract_number: Optional[str] = None

    counterparty_name: Optional[str] = None
    counterparty_id: Optional[str] = None

    owner_user_id: Optional[UUID] = None

    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    signed_date: Optional[date] = None

    renewal_date: Optional[date] = None
    notice_date: Optional[date] = None

    auto_renewal: Optional[bool] = None

    renewal_term_days: Optional[int] = Field(
        default=None,
        ge=1,
    )

    notice_period_days: Optional[int] = Field(
        default=None,
        ge=0,
    )

    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    contract_value: Optional[Decimal] = Field(
        default=None,
        ge=0,
    )

    risk_level: Optional[str] = None

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    review_status: Optional[str] = None

    metadata: Optional[dict[str, Any]] = None


class ContractStatusRequest(BaseModel):
    status: str = Field(
        min_length=1,
        max_length=100,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ContractRiskRequest(BaseModel):
    risk_level: str = Field(
        min_length=1,
        max_length=50,
    )

    risk_score: float = Field(
        ge=0,
        le=1,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ContractReviewStatusRequest(BaseModel):
    review_status: str = Field(
        min_length=1,
        max_length=100,
    )

    notes: Optional[str] = Field(
        default=None,
        max_length=10000,
    )


class ContractAnalyzeRequest(BaseModel):
    analyze_clauses: bool = True
    analyze_risk: bool = True
    analyze_obligations: bool = True
    analyze_expiry: bool = True

    force: bool = False

    model: Optional[str] = Field(
        default=None,
        max_length=255,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ContractResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    document_id: str

    title: str

    contract_type: Optional[str] = None
    status: str

    contract_number: Optional[str] = None

    counterparty_name: Optional[str] = None
    counterparty_id: Optional[str] = None

    owner_user_id: Optional[str] = None

    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    signed_date: Optional[date] = None

    renewal_date: Optional[date] = None
    notice_date: Optional[date] = None

    auto_renewal: bool = False

    renewal_term_days: Optional[int] = None
    notice_period_days: Optional[int] = None

    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None

    currency: Optional[str] = None
    contract_value: Optional[Decimal] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    review_status: Optional[str] = None

    source: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ContractPageResponse(BaseModel):
    items: list[ContractResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class ContractStatisticsResponse(BaseModel):
    total: int = 0

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_risk_level: dict[str, int] = Field(
        default_factory=dict,
    )

    by_review_status: dict[str, int] = Field(
        default_factory=dict,
    )

    active: int = 0
    expired: int = 0
    auto_renewing: int = 0

    total_contract_value: float = 0

    model_config = ConfigDict(
        extra="allow",
    )


class ContractAnalysisResponse(BaseModel):
    contract_id: str

    clauses_analyzed: int = 0
    obligations_found: int = 0
    risks_found: int = 0

    high_risks: int = 0
    critical_risks: int = 0

    expiry_detected: Optional[date] = None
    renewal_detected: Optional[date] = None
    notice_deadline: Optional[date] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    analyzed_at: datetime


class MessageResponse(BaseModel):
    message: str


# ============================================================
# COMMON HELPERS
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
    contract: Any,
) -> dict[str, Any]:
    return (
        _attr(
            contract,
            "contract_metadata",
        )
        or _attr(
            contract,
            "metadata_json",
        )
        or {}
    )


def _serialize_contract(
    contract: Any,
) -> dict[str, Any]:
    return {
        "id": str(
            _attr(
                contract,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                contract,
                "tenant_id",
                "",
            )
        ),

        "organization_id": _optional_str(
            _attr(
                contract,
                "organization_id",
            )
        ),

        "document_id": str(
            _attr(
                contract,
                "document_id",
                "",
            )
        ),

        "title": str(
            _attr(
                contract,
                "title",
                "",
            )
        ),

        "contract_type": _optional_str(
            _enum_value(
                _attr(
                    contract,
                    "contract_type",
                )
            )
        ),

        "status": str(
            _enum_value(
                _attr(
                    contract,
                    "status",
                    "draft",
                )
            )
        ),

        "contract_number": _attr(
            contract,
            "contract_number",
        ),

        "counterparty_name": _attr(
            contract,
            "counterparty_name",
        ),

        "counterparty_id": _optional_str(
            _attr(
                contract,
                "counterparty_id",
            )
        ),

        "owner_user_id": _optional_str(
            _attr(
                contract,
                "owner_user_id",
            )
        ),

        "effective_date": _attr(
            contract,
            "effective_date",
        ),

        "expiry_date": _attr(
            contract,
            "expiry_date",
        ),

        "signed_date": _attr(
            contract,
            "signed_date",
        ),

        "renewal_date": _attr(
            contract,
            "renewal_date",
        ),

        "notice_date": _attr(
            contract,
            "notice_date",
        ),

        "auto_renewal": bool(
            _attr(
                contract,
                "auto_renewal",
                False,
            )
        ),

        "renewal_term_days": _attr(
            contract,
            "renewal_term_days",
        ),

        "notice_period_days": _attr(
            contract,
            "notice_period_days",
        ),

        "governing_law": _attr(
            contract,
            "governing_law",
        ),

        "jurisdiction": _attr(
            contract,
            "jurisdiction",
        ),

        "currency": _attr(
            contract,
            "currency",
        ),

        "contract_value": _attr(
            contract,
            "contract_value",
        ),

        "risk_level": _optional_str(
            _enum_value(
                _attr(
                    contract,
                    "risk_level",
                )
            )
        ),

        "risk_score": _attr(
            contract,
            "risk_score",
        ),

        "review_status": _optional_str(
            _enum_value(
                _attr(
                    contract,
                    "review_status",
                )
            )
        ),

        "source": _optional_str(
            _enum_value(
                _attr(
                    contract,
                    "source",
                )
            )
        ),

        "metadata": _metadata(
            contract
        ),

        "created_at": _attr(
            contract,
            "created_at",
        ),

        "updated_at": _attr(
            contract,
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
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
    )

    return {
        "items": [
            _serialize_contract(
                contract
            )
            for contract in items
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
                f"Contract operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# LIST / SEARCH
# ============================================================

@router.get(
    "",
    response_model=ContractPageResponse,
    summary="List contracts",
)
async def list_contracts(
    contract_type: Optional[str] = None,
    contract_status: Optional[str] = Query(
        default=None,
        alias="status",
    ),
    counterparty: Optional[str] = None,
    owner_user_id: Optional[UUID] = None,
    risk_level: Optional[str] = None,
    review_status: Optional[str] = None,
    auto_renewal: Optional[bool] = None,
    governing_law: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    currency: Optional[str] = None,

    expiry_from: Optional[date] = None,
    expiry_to: Optional[date] = None,

    min_value: Optional[Decimal] = Query(
        default=None,
        ge=0,
    ),
    max_value: Optional[Decimal] = Query(
        default=None,
        ge=0,
    ),

    search: Optional[str] = Query(
        default=None,
        max_length=1000,
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

    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    filters = ContractFilter(
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

    if contract_type:
        if hasattr(filters, "contract_types"):
            filters.contract_types = [
                contract_type
            ]
        else:
            _set_filter(
                filters,
                "contract_type",
                contract_type,
            )

    if contract_status:
        if hasattr(filters, "statuses"):
            filters.statuses = [
                contract_status
            ]
        else:
            _set_filter(
                filters,
                "status",
                contract_status,
            )

    _set_filter(
        filters,
        "counterparty",
        counterparty,
    )

    _set_filter(
        filters,
        "owner_user_id",
        str(owner_user_id)
        if owner_user_id
        else None,
    )

    if risk_level:
        if hasattr(filters, "risk_levels"):
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
        "review_status",
        review_status,
    )

    _set_filter(
        filters,
        "auto_renewal",
        auto_renewal,
    )

    _set_filter(
        filters,
        "governing_law",
        governing_law,
    )

    _set_filter(
        filters,
        "jurisdiction",
        jurisdiction,
    )

    _set_filter(
        filters,
        "currency",
        currency.upper()
        if currency
        else None,
    )

    _set_filter(
        filters,
        "expiry_from",
        expiry_from,
    )

    _set_filter(
        filters,
        "expiry_to",
        expiry_to,
    )

    _set_filter(
        filters,
        "min_contract_value",
        min_value,
    )

    _set_filter(
        filters,
        "max_contract_value",
        max_value,
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
# Keep static paths above /{contract_id}
# ============================================================

@router.get(
    "/statistics",
    response_model=ContractStatisticsResponse,
    summary="Get contract statistics",
)
async def contract_statistics(
    current_user: Any = Depends(
        get_current_user
    ),
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> Any:

    filters = ContractFilter(
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

    stats = await repository.statistics(
        filters
    )

    if isinstance(stats, dict):
        return stats

    if hasattr(stats, "to_dict"):
        return stats.to_dict()

    return {
        "total": _attr(
            stats,
            "total",
            0,
        ),

        "by_type": _attr(
            stats,
            "by_type",
            {},
        ),

        "by_status": _attr(
            stats,
            "by_status",
            {},
        ),

        "by_risk_level": (
            _attr(
                stats,
                "by_risk_level",
            )
            or _attr(
                stats,
                "by_risk",
                {},
            )
        ),

        "by_review_status": _attr(
            stats,
            "by_review_status",
            {},
        ),

        "active": _attr(
            stats,
            "active",
            0,
        ),

        "expired": _attr(
            stats,
            "expired",
            0,
        ),

        "auto_renewing": _attr(
            stats,
            "auto_renewing",
            0,
        ),

        "total_contract_value": float(
            _attr(
                stats,
                "total_contract_value",
                0,
            )
            or 0
        ),
    }


# ============================================================
# EXPIRING CONTRACTS
# ============================================================

@router.get(
    "/expiring",
    response_model=ContractPageResponse,
    summary="List expiring contracts",
)
async def expiring_contracts(
    days: int = Query(
        default=30,
        ge=1,
        le=3650,
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
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    today = datetime.now(
        timezone.utc
    ).date()

    page = await repository.list_expiring_between(
        tenant_id=_tenant_id(
            current_user
        ),
        start_date=today,
        end_date=(
            today
            + timedelta(days=days)
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# EXPIRED
# ============================================================

@router.get(
    "/expired",
    response_model=ContractPageResponse,
    summary="List expired contracts",
)
async def expired_contracts(
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
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_expired(
        tenant_id=_tenant_id(
            current_user
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# AUTO-RENEWING
# ============================================================

@router.get(
    "/auto-renewing",
    response_model=ContractPageResponse,
    summary="List auto-renewing contracts",
)
async def auto_renewing_contracts(
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
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_auto_renewing(
        tenant_id=_tenant_id(
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
    response_model=ContractPageResponse,
    summary="List high-risk contracts",
)
async def high_risk_contracts(
    min_score: float = Query(
        default=0.70,
        ge=0,
        le=1,
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
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_high_risk(
        tenant_id=_tenant_id(
            current_user
        ),
        min_score=min_score,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# PENDING REVIEW
# ============================================================

@router.get(
    "/pending-review",
    response_model=ContractPageResponse,
    summary="List contracts pending review",
)
async def pending_review_contracts(
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
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_pending_review(
        tenant_id=_tenant_id(
            current_user
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# CREATE
# ============================================================

@router.post(
    "",
    response_model=ContractResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create contract",
)
async def create_contract(
    payload: ContractCreateRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    if (
        payload.effective_date
        and payload.expiry_date
        and payload.expiry_date
        < payload.effective_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "expiry_date cannot be before "
                "effective_date."
            ),
        )

    data = ContractCreate(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        document_id=str(
            payload.document_id
        ),

        title=payload.title.strip(),

        contract_type=(
            payload.contract_type
        ),

        status=payload.status,

        contract_number=(
            payload.contract_number
        ),

        counterparty_name=(
            payload.counterparty_name
        ),

        counterparty_id=(
            payload.counterparty_id
        ),

        owner_user_id=(
            str(payload.owner_user_id)
            if payload.owner_user_id
            else None
        ),

        effective_date=(
            payload.effective_date
        ),

        expiry_date=(
            payload.expiry_date
        ),

        signed_date=(
            payload.signed_date
        ),

        renewal_date=(
            payload.renewal_date
        ),

        notice_date=(
            payload.notice_date
        ),

        auto_renewal=(
            payload.auto_renewal
        ),

        renewal_term_days=(
            payload.renewal_term_days
        ),

        notice_period_days=(
            payload.notice_period_days
        ),

        governing_law=(
            payload.governing_law
        ),

        jurisdiction=(
            payload.jurisdiction
        ),

        currency=payload.currency,

        contract_value=(
            payload.contract_value
        ),

        risk_level=(
            payload.risk_level
        ),

        risk_score=(
            payload.risk_score
        ),

        review_status=(
            payload.review_status
        ),

        source=payload.source,

        metadata=payload.metadata,
    )

    contract = await repository.create(
        data
    )

    return _serialize_contract(
        contract
    )


# ============================================================
# GET BY ID
# ============================================================

@router.get(
    "/{contract_id}",
    response_model=ContractResponse,
    summary="Get contract",
)
async def get_contract(
    contract_id: UUID,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

    contract = await repository.get_by_id(
        contract_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract was not found.",
        )

    return _serialize_contract(
        contract
    )


# ============================================================
# UPDATE
# ============================================================

@router.patch(
    "/{contract_id}",
    response_model=ContractResponse,
    summary="Update contract",
)
async def update_contract(
    contract_id: UUID,
    payload: ContractUpdateRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, Any]:

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

    if (
        values.get("currency")
        is not None
    ):
        values["currency"] = (
            values["currency"].upper()
        )

    contract = await repository.update(
        contract_id=contract_id,
        tenant_id=_tenant_id(
            current_user
        ),
        data=ContractUpdate(
            **values
        ),
    )

    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract was not found.",
        )

    return _serialize_contract(
        contract
    )


# ============================================================
# STATUS / LIFECYCLE
# ============================================================

@router.patch(
    "/{contract_id}/status",
    response_model=ContractResponse,
    summary="Change contract status",
)
async def update_contract_status(
    contract_id: UUID,
    payload: ContractStatusRequest,
    request: Request,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_contract_service
    ),
) -> dict[str, Any]:

    contract = await _service_call(
        service,
        "update_status",

        contract_id=str(
            contract_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        new_status=payload.status,

        reason=payload.reason,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_contract(
        contract
    )


# ============================================================
# RISK
# ============================================================

@router.patch(
    "/{contract_id}/risk",
    response_model=ContractResponse,
    summary="Update contract risk",
)
async def update_contract_risk(
    contract_id: UUID,
    payload: ContractRiskRequest,
    request: Request,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_contract_service
    ),
) -> dict[str, Any]:

    contract = await _service_call(
        service,
        "update_risk",

        contract_id=str(
            contract_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        risk_level=(
            payload.risk_level
        ),

        risk_score=(
            payload.risk_score
        ),

        reason=payload.reason,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_contract(
        contract
    )


# ============================================================
# REVIEW STATUS
# ============================================================

@router.patch(
    "/{contract_id}/review-status",
    response_model=ContractResponse,
    summary="Update contract review status",
)
async def update_review_status(
    contract_id: UUID,
    payload: ContractReviewStatusRequest,
    request: Request,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_contract_service
    ),
) -> dict[str, Any]:

    contract = await _service_call(
        service,
        "update_review_status",

        contract_id=str(
            contract_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        review_status=(
            payload.review_status
        ),

        notes=payload.notes,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_contract(
        contract
    )


# ============================================================
# AI ANALYSIS
# ============================================================

@router.post(
    "/{contract_id}/analyze",
    response_model=ContractAnalysisResponse,
    summary="Analyze contract with legal AI",
)
async def analyze_contract(
    contract_id: UUID,
    payload: ContractAnalyzeRequest,
    request: Request,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_contract_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "analyze",

        contract_id=str(
            contract_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        analyze_clauses=(
            payload.analyze_clauses
        ),

        analyze_risk=(
            payload.analyze_risk
        ),

        analyze_obligations=(
            payload.analyze_obligations
        ),

        analyze_expiry=(
            payload.analyze_expiry
        ),

        model=payload.model,
        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    if isinstance(result, dict):
        return result

    return {
        "contract_id": str(
            contract_id
        ),

        "clauses_analyzed": int(
            _attr(
                result,
                "clauses_analyzed",
                0,
            )
        ),

        "obligations_found": int(
            _attr(
                result,
                "obligations_found",
                0,
            )
        ),

        "risks_found": int(
            _attr(
                result,
                "risks_found",
                0,
            )
        ),

        "high_risks": int(
            _attr(
                result,
                "high_risks",
                0,
            )
        ),

        "critical_risks": int(
            _attr(
                result,
                "critical_risks",
                0,
            )
        ),

        "expiry_detected": _attr(
            result,
            "expiry_detected",
        ),

        "renewal_detected": _attr(
            result,
            "renewal_detected",
        ),

        "notice_deadline": _attr(
            result,
            "notice_deadline",
        ),

        "risk_level": _optional_str(
            _enum_value(
                _attr(
                    result,
                    "risk_level",
                )
            )
        ),

        "risk_score": _attr(
            result,
            "risk_score",
        ),

        "model_name": _attr(
            result,
            "model_name",
        ),

        "model_version": _attr(
            result,
            "model_version",
        ),

        "analyzed_at": (
            _attr(
                result,
                "analyzed_at",
            )
            or datetime.now(
                timezone.utc
            )
        ),
    }


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{contract_id}",
    response_model=MessageResponse,
    summary="Delete contract",
)
async def delete_contract(
    contract_id: UUID,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: ContractRepository = Depends(
        get_contract_repository
    ),
) -> dict[str, str]:

    deleted = await repository.delete(
        contract_id=contract_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract was not found.",
        )

    return {
        "message":
            "Contract deleted successfully."
    }