# backend/api/routes/expiry.py

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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


router = APIRouter(
    prefix="/expiry",
    tags=["Expiry Management"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies providers.
# ============================================================

async def get_expiry_service() -> Any:
    """
    Recommended ExpiryService dependencies:

        ContractRepository
        RenewalRepository
        NotificationRepository
        ReviewRepository
        AuditRepository
        EscalationEngine
        FireEngine
        EmailClient

    Expected operations:
        dashboard(...)
        list_expiring(...)
        list_expired(...)
        list_notice_deadlines(...)
        check_contract(...)
        scan(...)
        update_expiry(...)
        acknowledge(...)
        create_reminder(...)
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Expiry service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ExpiryCheckRequest(BaseModel):
    recalculate: bool = True

    evaluate_notice_period: bool = True
    evaluate_renewal: bool = True
    evaluate_auto_renewal: bool = True

    create_notifications: bool = True
    trigger_escalation: bool = True

    force: bool = False


class ExpiryUpdateRequest(BaseModel):
    expiry_date: date

    notice_date: Optional[date] = None

    renewal_date: Optional[date] = None

    notice_period_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=36500,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ExpiryAcknowledgeRequest(BaseModel):
    notes: Optional[str] = Field(
        default=None,
        max_length=10000,
    )


class ReminderCreateRequest(BaseModel):
    remind_at: datetime

    channel: str = Field(
        default="email",
        max_length=50,
    )

    recipient_user_ids: list[UUID] = Field(
        default_factory=list,
        max_length=100,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=10000,
    )


class ExpiryScanRequest(BaseModel):
    days_ahead: int = Field(
        default=90,
        ge=1,
        le=3650,
    )

    include_expired: bool = True

    create_notifications: bool = True

    trigger_escalations: bool = True

    force: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ExpiryItemResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

    contract_id: str

    contract_title: Optional[str] = None
    contract_number: Optional[str] = None

    counterparty_name: Optional[str] = None

    expiry_date: Optional[date] = None
    effective_date: Optional[date] = None

    notice_date: Optional[date] = None
    renewal_date: Optional[date] = None

    days_until_expiry: Optional[int] = None
    days_until_notice: Optional[int] = None

    auto_renewal: bool = False

    renewal_term_days: Optional[int] = None
    notice_period_days: Optional[int] = None

    status: Optional[str] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    urgency: Optional[str] = None

    requires_action: bool = False
    requires_review: bool = False

    owner_user_id: Optional[str] = None


class ExpiryPageResponse(BaseModel):
    items: list[ExpiryItemResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class ExpiryCheckResponse(BaseModel):
    contract_id: str

    expiry_date: Optional[date] = None

    days_until_expiry: Optional[int] = None

    notice_date: Optional[date] = None
    days_until_notice: Optional[int] = None

    renewal_date: Optional[date] = None

    expired: bool = False
    expiring_soon: bool = False

    notice_due: bool = False
    notice_overdue: bool = False

    auto_renewal: bool = False

    urgency: Optional[str] = None

    escalation_level: Optional[str] = None

    notifications_created: int = 0

    checked_at: datetime


class ExpiryDashboardResponse(BaseModel):
    total_contracts: int = 0

    expired: int = 0

    expiring_7_days: int = 0
    expiring_30_days: int = 0
    expiring_60_days: int = 0
    expiring_90_days: int = 0

    notice_due_7_days: int = 0
    notice_due_30_days: int = 0

    overdue_notices: int = 0

    auto_renewing: int = 0

    high_risk_expiring: int = 0
    critical_expiring: int = 0

    requires_action: int = 0

    generated_at: datetime


class ExpiryScanResponse(BaseModel):
    scanned: int = 0

    expired: int = 0
    expiring: int = 0

    notice_due: int = 0
    notice_overdue: int = 0

    renewals_due: int = 0

    notifications_created: int = 0
    escalations_created: int = 0

    errors: int = 0

    started_at: datetime
    completed_at: datetime


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

    return _optional_str(
        value
    )


def _today() -> date:

    return datetime.now(
        timezone.utc
    ).date()


def _days_until(
    target: Optional[date],
) -> Optional[int]:

    if target is None:
        return None

    return (
        target - _today()
    ).days


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
                f"Expiry operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(
        **kwargs
    )


# ============================================================
# SERIALIZATION
# ============================================================

def _serialize_expiry(
    item: Any,
) -> dict[str, Any]:

    expiry_date = _attr(
        item,
        "expiry_date",
    )

    notice_date = _attr(
        item,
        "notice_date",
    )

    days_until_expiry = _attr(
        item,
        "days_until_expiry",
    )

    if (
        days_until_expiry is None
        and expiry_date
    ):
        days_until_expiry = (
            _days_until(
                expiry_date
            )
        )

    days_until_notice = _attr(
        item,
        "days_until_notice",
    )

    if (
        days_until_notice is None
        and notice_date
    ):
        days_until_notice = (
            _days_until(
                notice_date
            )
        )

    return {
        "contract_id": str(
            _attr(
                item,
                "contract_id",
                _attr(
                    item,
                    "id",
                    "",
                ),
            )
        ),

        "contract_title": (
            _attr(
                item,
                "contract_title",
            )
            or _attr(
                item,
                "title",
            )
        ),

        "contract_number": _attr(
            item,
            "contract_number",
        ),

        "counterparty_name": _attr(
            item,
            "counterparty_name",
        ),

        "expiry_date":
            expiry_date,

        "effective_date": _attr(
            item,
            "effective_date",
        ),

        "notice_date":
            notice_date,

        "renewal_date": _attr(
            item,
            "renewal_date",
        ),

        "days_until_expiry":
            days_until_expiry,

        "days_until_notice":
            days_until_notice,

        "auto_renewal": bool(
            _attr(
                item,
                "auto_renewal",
                False,
            )
        ),

        "renewal_term_days": _attr(
            item,
            "renewal_term_days",
        ),

        "notice_period_days": _attr(
            item,
            "notice_period_days",
        ),

        "status": _optional_str(
            _enum_value(
                _attr(
                    item,
                    "status",
                )
            )
        ),

        "risk_level": _optional_str(
            _enum_value(
                _attr(
                    item,
                    "risk_level",
                )
            )
        ),

        "risk_score": _attr(
            item,
            "risk_score",
        ),

        "urgency": (
            _optional_str(
                _enum_value(
                    _attr(
                        item,
                        "urgency",
                    )
                )
            )
            or _calculate_urgency(
                days_until_expiry
            )
        ),

        "requires_action": bool(
            _attr(
                item,
                "requires_action",
                (
                    days_until_expiry
                    is not None
                    and days_until_expiry
                    <= 30
                ),
            )
        ),

        "requires_review": bool(
            _attr(
                item,
                "requires_review",
                False,
            )
        ),

        "owner_user_id": _optional_str(
            _attr(
                item,
                "owner_user_id",
            )
        ),
    }


def _calculate_urgency(
    days_remaining: Optional[int],
) -> Optional[str]:

    if days_remaining is None:
        return None

    if days_remaining < 0:
        return "expired"

    if days_remaining <= 7:
        return "critical"

    if days_remaining <= 30:
        return "high"

    if days_remaining <= 60:
        return "medium"

    if days_remaining <= 90:
        return "low"

    return "normal"


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
            _serialize_expiry(
                item
            )
            for item in items
        ],

        "total":
            total,

        "limit":
            limit,

        "offset":
            offset,

        "has_more":
            bool(
                _attr(
                    page,
                    "has_more",
                    (
                        offset
                        + len(items)
                        < total
                    ),
                )
            ),
    }


# ============================================================
# DASHBOARD
# Static route must appear before /{contract_id}
# ============================================================

@router.get(
    "/dashboard",
    response_model=ExpiryDashboardResponse,
    summary="Get expiry dashboard",
)
async def expiry_dashboard(
    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "dashboard",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            **result,
            "generated_at":
                result.get(
                    "generated_at"
                )
                or datetime.now(
                    timezone.utc
                ),
        }

    return {
        "total_contracts":
            _attr(
                result,
                "total_contracts",
                0,
            ),

        "expired":
            _attr(
                result,
                "expired",
                0,
            ),

        "expiring_7_days":
            _attr(
                result,
                "expiring_7_days",
                0,
            ),

        "expiring_30_days":
            _attr(
                result,
                "expiring_30_days",
                0,
            ),

        "expiring_60_days":
            _attr(
                result,
                "expiring_60_days",
                0,
            ),

        "expiring_90_days":
            _attr(
                result,
                "expiring_90_days",
                0,
            ),

        "notice_due_7_days":
            _attr(
                result,
                "notice_due_7_days",
                0,
            ),

        "notice_due_30_days":
            _attr(
                result,
                "notice_due_30_days",
                0,
            ),

        "overdue_notices":
            _attr(
                result,
                "overdue_notices",
                0,
            ),

        "auto_renewing":
            _attr(
                result,
                "auto_renewing",
                0,
            ),

        "high_risk_expiring":
            _attr(
                result,
                "high_risk_expiring",
                0,
            ),

        "critical_expiring":
            _attr(
                result,
                "critical_expiring",
                0,
            ),

        "requires_action":
            _attr(
                result,
                "requires_action",
                0,
            ),

        "generated_at":
            datetime.now(
                timezone.utc
            ),
    }


# ============================================================
# EXPIRING
# ============================================================

@router.get(
    "/expiring",
    response_model=ExpiryPageResponse,
    summary="List expiring contracts",
)
async def list_expiring(
    days: int = Query(
        default=90,
        ge=1,
        le=3650,
    ),

    risk_level: Optional[str] = Query(
        default=None
    ),

    owner_user_id: Optional[UUID] = Query(
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

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    page = await _service_call(
        service,
        "list_expiring",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        start_date=_today(),

        end_date=(
            _today()
            + timedelta(
                days=days
            )
        ),

        risk_level=risk_level,

        owner_user_id=(
            str(owner_user_id)
            if owner_user_id
            else None
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# EXPIRED
# ============================================================

@router.get(
    "/expired",
    response_model=ExpiryPageResponse,
    summary="List expired contracts",
)
async def list_expired(
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

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    page = await _service_call(
        service,
        "list_expired",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        as_of=_today(),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# NOTICE DEADLINES
# ============================================================

@router.get(
    "/notice-deadlines",
    response_model=ExpiryPageResponse,
    summary="List notice deadlines",
)
async def notice_deadlines(
    days: int = Query(
        default=30,
        ge=1,
        le=3650,
    ),

    include_overdue: bool = Query(
        default=True
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

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    page = await _service_call(
        service,
        "list_notice_deadlines",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        start_date=(
            None
            if include_overdue
            else _today()
        ),

        end_date=(
            _today()
            + timedelta(
                days=days
            )
        ),

        include_overdue=(
            include_overdue
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# AUTO RENEWALS
# ============================================================

@router.get(
    "/auto-renewals",
    response_model=ExpiryPageResponse,
    summary="List upcoming auto renewals",
)
async def auto_renewals(
    days: int = Query(
        default=90,
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

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    page = await _service_call(
        service,
        "list_auto_renewals",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        start_date=_today(),

        end_date=(
            _today()
            + timedelta(
                days=days
            )
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# GLOBAL EXPIRY SCAN
# ============================================================

@router.post(
    "/scan",
    response_model=ExpiryScanResponse,
    summary="Run expiry scan",
)
async def scan_expiries(
    payload: ExpiryScanRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "scan",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        days_ahead=(
            payload.days_ahead
        ),

        include_expired=(
            payload.include_expired
        ),

        create_notifications=(
            payload.create_notifications
        ),

        trigger_escalations=(
            payload.trigger_escalations
        ),

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {
        "scanned":
            _attr(
                result,
                "scanned",
                0,
            ),

        "expired":
            _attr(
                result,
                "expired",
                0,
            ),

        "expiring":
            _attr(
                result,
                "expiring",
                0,
            ),

        "notice_due":
            _attr(
                result,
                "notice_due",
                0,
            ),

        "notice_overdue":
            _attr(
                result,
                "notice_overdue",
                0,
            ),

        "renewals_due":
            _attr(
                result,
                "renewals_due",
                0,
            ),

        "notifications_created":
            _attr(
                result,
                "notifications_created",
                0,
            ),

        "escalations_created":
            _attr(
                result,
                "escalations_created",
                0,
            ),

        "errors":
            _attr(
                result,
                "errors",
                0,
            ),

        "started_at":
            _attr(
                result,
                "started_at",
                datetime.now(
                    timezone.utc
                ),
            ),

        "completed_at":
            _attr(
                result,
                "completed_at",
                datetime.now(
                    timezone.utc
                ),
            ),
    }


# ============================================================
# GET CONTRACT EXPIRY
# ============================================================

@router.get(
    "/{contract_id}",
    response_model=ExpiryItemResponse,
    summary="Get contract expiry details",
)
async def get_contract_expiry(
    contract_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "get_contract_expiry",

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
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Contract expiry information "
                "was not found."
            ),
        )

    return _serialize_expiry(
        result
    )


# ============================================================
# CHECK SINGLE CONTRACT
# ============================================================

@router.post(
    "/{contract_id}/check",
    response_model=ExpiryCheckResponse,
    summary="Evaluate contract expiry",
)
async def check_contract_expiry(
    contract_id: UUID,
    payload: ExpiryCheckRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "check_contract",

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

        recalculate=(
            payload.recalculate
        ),

        evaluate_notice_period=(
            payload.evaluate_notice_period
        ),

        evaluate_renewal=(
            payload.evaluate_renewal
        ),

        evaluate_auto_renewal=(
            payload.evaluate_auto_renewal
        ),

        create_notifications=(
            payload.create_notifications
        ),

        trigger_escalation=(
            payload.trigger_escalation
        ),

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            **result,
            "contract_id":
                result.get(
                    "contract_id",
                    str(contract_id),
                ),
            "checked_at":
                result.get(
                    "checked_at"
                )
                or datetime.now(
                    timezone.utc
                ),
        }

    return {
        "contract_id":
            str(contract_id),

        "expiry_date":
            _attr(
                result,
                "expiry_date",
            ),

        "days_until_expiry":
            _attr(
                result,
                "days_until_expiry",
            ),

        "notice_date":
            _attr(
                result,
                "notice_date",
            ),

        "days_until_notice":
            _attr(
                result,
                "days_until_notice",
            ),

        "renewal_date":
            _attr(
                result,
                "renewal_date",
            ),

        "expired":
            bool(
                _attr(
                    result,
                    "expired",
                    False,
                )
            ),

        "expiring_soon":
            bool(
                _attr(
                    result,
                    "expiring_soon",
                    False,
                )
            ),

        "notice_due":
            bool(
                _attr(
                    result,
                    "notice_due",
                    False,
                )
            ),

        "notice_overdue":
            bool(
                _attr(
                    result,
                    "notice_overdue",
                    False,
                )
            ),

        "auto_renewal":
            bool(
                _attr(
                    result,
                    "auto_renewal",
                    False,
                )
            ),

        "urgency":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "urgency",
                    )
                )
            ),

        "escalation_level":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "escalation_level",
                    )
                )
            ),

        "notifications_created":
            int(
                _attr(
                    result,
                    "notifications_created",
                    0,
                )
                or 0
            ),

        "checked_at":
            _attr(
                result,
                "checked_at",
                datetime.now(
                    timezone.utc
                ),
            ),
    }


# ============================================================
# UPDATE EXPIRY
# ============================================================

@router.patch(
    "/{contract_id}",
    response_model=ExpiryItemResponse,
    summary="Update contract expiry",
)
async def update_expiry(
    contract_id: UUID,
    payload: ExpiryUpdateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, Any]:

    if (
        payload.notice_date
        and payload.notice_date
        > payload.expiry_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Notice date cannot be after "
                "the expiry date."
            ),
        )

    result = await _service_call(
        service,
        "update_expiry",

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

        expiry_date=(
            payload.expiry_date
        ),

        notice_date=(
            payload.notice_date
        ),

        renewal_date=(
            payload.renewal_date
        ),

        notice_period_days=(
            payload.notice_period_days
        ),

        reason=payload.reason,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_expiry(
        result
    )


# ============================================================
# ACKNOWLEDGE EXPIRY
# ============================================================

@router.post(
    "/{contract_id}/acknowledge",
    response_model=MessageResponse,
    summary="Acknowledge expiry alert",
)
async def acknowledge_expiry(
    contract_id: UUID,
    payload: ExpiryAcknowledgeRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, str]:

    await _service_call(
        service,
        "acknowledge",

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

        notes=payload.notes,

        request_id=_request_id(
            request
        ),
    )

    return {
        "message":
            "Expiry alert acknowledged."
    }


# ============================================================
# CREATE REMINDER
# ============================================================

@router.post(
    "/{contract_id}/reminders",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create expiry reminder",
)
async def create_expiry_reminder(
    contract_id: UUID,
    payload: ReminderCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_expiry_service
    ),
) -> dict[str, str]:

    if (
        payload.remind_at.tzinfo
        is None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "remind_at must include "
                "timezone information."
            ),
        )

    await _service_call(
        service,
        "create_reminder",

        contract_id=str(
            contract_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        created_by=_user_id(
            current_user
        ),

        remind_at=(
            payload.remind_at
        ),

        channel=(
            payload.channel
        ),

        recipient_user_ids=[
            str(user_id)
            for user_id
            in payload.recipient_user_ids
        ],

        message=(
            payload.message
        ),

        request_id=_request_id(
            request
        ),
    )

    return {
        "message":
            "Expiry reminder created."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Expiry subsystem health",
)
async def expiry_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "expiry-management",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }