# backend/app/schemas/expiry.py

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# Enums
# ============================================================


class ExpiryStatus(str, Enum):
    ACTIVE = "active"
    UPCOMING = "upcoming"
    DUE_SOON = "due_soon"
    EXPIRED = "expired"
    RENEWED = "renewed"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class ExpiryUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeadlineType(str, Enum):
    EXPIRY = "expiry"
    RENEWAL = "renewal"
    NOTICE = "notice"
    TERMINATION = "termination"
    REVIEW = "review"
    OBLIGATION = "obligation"
    OTHER = "other"


class ReminderChannel(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


# ============================================================
# Expiry calculation
# ============================================================


class ExpiryCalculationRequest(BaseModel):
    contract_id: UUID

    reference_date: date | None = None

    warning_days: list[int] = Field(
        default_factory=lambda: [90, 60, 30, 14, 7, 1]
    )

    include_notice_deadline: bool = True

    include_renewal: bool = True


class ExpiryCalculationResponse(BaseModel):
    contract_id: UUID

    reference_date: date

    expiry_date: date | None = None

    renewal_date: date | None = None

    notice_deadline: date | None = None

    days_until_expiry: int | None = None

    days_until_renewal: int | None = None

    days_until_notice_deadline: int | None = None

    status: ExpiryStatus

    urgency: ExpiryUrgency

    is_expired: bool = False

    is_notice_deadline_missed: bool = False

    auto_renewal: bool = False

    action_required: bool = False

    recommended_action: str | None = None


# ============================================================
# Expiry item
# ============================================================


class ExpiryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    contract_id: UUID

    organization_id: UUID

    title: str

    contract_number: str | None = None

    contract_type: str | None = None

    counterparty: str | None = None

    status: ExpiryStatus

    urgency: ExpiryUrgency

    effective_date: date | None = None

    expiry_date: date | None = None

    renewal_date: date | None = None

    notice_deadline: date | None = None

    days_until_expiry: int | None = None

    days_until_renewal: int | None = None

    days_until_notice_deadline: int | None = None

    auto_renewal: bool = False

    notice_period_days: int | None = None

    renewal_term_months: int | None = None

    contract_value: float | None = None

    currency: str | None = None

    risk_level: str | None = None

    requires_action: bool = False

    recommended_action: str | None = None

    tags: list[str] = Field(default_factory=list)


# ============================================================
# Upcoming expiry
# ============================================================


class UpcomingExpiryRequest(BaseModel):
    organization_id: UUID | None = None

    days: int = Field(
        default=90,
        ge=1,
        le=3650,
    )

    include_expired: bool = False

    include_auto_renewal: bool = True

    risk_levels: list[str] | None = None

    contract_types: list[str] | None = None

    counterparties: list[str] | None = None


class UpcomingExpiryResponse(BaseModel):
    items: list[ExpiryItem] = Field(default_factory=list)

    total: int = Field(default=0, ge=0)

    window_days: int = Field(ge=1)

    generated_at: datetime


# ============================================================
# Deadline
# ============================================================


class ContractDeadline(BaseModel):
    id: UUID | None = None

    contract_id: UUID

    contract_title: str | None = None

    deadline_type: DeadlineType

    deadline_date: date

    title: str

    description: str | None = None

    days_remaining: int | None = None

    urgency: ExpiryUrgency = ExpiryUrgency.LOW

    completed: bool = False

    action_required: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Calendar
# ============================================================


class ExpiryCalendarRequest(BaseModel):
    start_date: date

    end_date: date

    organization_id: UUID | None = None

    include_expiry: bool = True

    include_renewals: bool = True

    include_notice_deadlines: bool = True

    include_termination: bool = True

    @model_validator(mode="after")
    def validate_range(self) -> "ExpiryCalendarRequest":
        if self.end_date < self.start_date:
            raise ValueError(
                "end_date must be greater than or equal to start_date"
            )

        return self


class ExpiryCalendarEvent(BaseModel):
    id: str

    contract_id: UUID

    title: str

    event_type: DeadlineType

    date: date

    status: ExpiryStatus | None = None

    urgency: ExpiryUrgency

    counterparty: str | None = None

    risk_level: str | None = None

    action_required: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpiryCalendarResponse(BaseModel):
    start_date: date

    end_date: date

    events: list[ExpiryCalendarEvent] = Field(
        default_factory=list
    )

    total_events: int = Field(default=0, ge=0)


# ============================================================
# Reminder
# ============================================================


class ExpiryReminderCreate(BaseModel):
    contract_id: UUID

    deadline_type: DeadlineType = DeadlineType.EXPIRY

    reminder_date: datetime

    channels: list[ReminderChannel] = Field(
        default_factory=lambda: [
            ReminderChannel.IN_APP,
            ReminderChannel.EMAIL,
        ]
    )

    recipient_user_ids: list[UUID] = Field(
        default_factory=list
    )

    message: str | None = Field(
        default=None,
        max_length=5000,
    )


class ExpiryReminderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    contract_id: UUID

    reminder_date: datetime

    deadline_type: DeadlineType

    channels: list[ReminderChannel]

    sent: bool = False

    sent_at: datetime | None = None

    created_at: datetime


# ============================================================
# Expiry action
# ============================================================


class ExpiryActionRequest(BaseModel):
    contract_id: UUID

    action: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )


class ExpiryActionResponse(BaseModel):
    contract_id: UUID

    action: str

    success: bool

    message: str

    processed_at: datetime


# ============================================================
# Statistics
# ============================================================


class ExpiryStatistics(BaseModel):
    total_contracts: int = 0

    active_contracts: int = 0

    expired_contracts: int = 0

    expiring_7_days: int = 0

    expiring_30_days: int = 0

    expiring_60_days: int = 0

    expiring_90_days: int = 0

    auto_renewing_contracts: int = 0

    notice_deadlines_due: int = 0

    missed_notice_deadlines: int = 0

    high_risk_expiring: int = 0

    critical_expiring: int = 0


class ExpiryValueStatistics(BaseModel):
    currency: str

    total_contract_value: float = 0.0

    expiring_30_days_value: float = 0.0

    expiring_90_days_value: float = 0.0

    expired_value: float = 0.0

    auto_renewal_value: float = 0.0


class ExpiryDashboardResponse(BaseModel):
    statistics: ExpiryStatistics

    value_statistics: list[ExpiryValueStatistics] = Field(
        default_factory=list
    )

    critical_items: list[ExpiryItem] = Field(
        default_factory=list
    )

    upcoming_items: list[ExpiryItem] = Field(
        default_factory=list
    )

    generated_at: datetime


# ============================================================
# Expiry filter
# ============================================================


class ExpiryFilter(BaseModel):
    status: list[ExpiryStatus] | None = None

    urgency: list[ExpiryUrgency] | None = None

    contract_types: list[str] | None = None

    risk_levels: list[str] | None = None

    start_date: date | None = None

    end_date: date | None = None

    auto_renewal: bool | None = None

    action_required: bool | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# Pagination
# ============================================================


class ExpiryListResponse(BaseModel):
    items: list[ExpiryItem] = Field(default_factory=list)

    total: int = Field(ge=0)

    page: int = Field(default=1, ge=1)

    page_size: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    pages: int = Field(default=0, ge=0)


# ============================================================
# Bulk expiry operation
# ============================================================


class BulkExpiryRequest(BaseModel):
    contract_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    action: str

    notes: str | None = None


class BulkExpiryResult(BaseModel):
    contract_id: UUID

    success: bool

    message: str | None = None


class BulkExpiryResponse(BaseModel):
    processed: int = 0

    successful: int = 0

    failed: int = 0

    results: list[BulkExpiryResult] = Field(
        default_factory=list
    )