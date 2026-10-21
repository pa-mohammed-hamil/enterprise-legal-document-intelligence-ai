# backend/app/schemas/renewal.py

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RenewalStatus(str, Enum):
    PENDING = "pending"
    UPCOMING = "upcoming"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    RENEWED = "renewed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RenewalType(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    OPTIONAL = "optional"


class RenewalDecision(str, Enum):
    RENEW = "renew"
    DO_NOT_RENEW = "do_not_renew"
    NEGOTIATE = "negotiate"
    EXTEND = "extend"
    PENDING = "pending"


class RenewalPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RenewalBase(BaseModel):
    renewal_type: RenewalType = RenewalType.MANUAL
    status: RenewalStatus = RenewalStatus.PENDING
    priority: RenewalPriority = RenewalPriority.MEDIUM

    current_expiry_date: date | None = None
    renewal_date: date | None = None
    notice_deadline: date | None = None

    renewal_term_months: int | None = Field(
        default=None,
        ge=1,
        le=1200,
    )

    auto_renewal: bool = False

    notes: str | None = Field(
        default=None,
        max_length=10_000,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class RenewalCreate(RenewalBase):
    organization_id: UUID
    contract_id: UUID

    assigned_to: UUID | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "RenewalCreate":
        if (
            self.notice_deadline
            and self.renewal_date
            and self.notice_deadline > self.renewal_date
        ):
            raise ValueError(
                "notice_deadline cannot be after renewal_date"
            )

        return self


class RenewalUpdate(BaseModel):
    renewal_type: RenewalType | None = None
    status: RenewalStatus | None = None
    priority: RenewalPriority | None = None

    current_expiry_date: date | None = None
    renewal_date: date | None = None
    notice_deadline: date | None = None

    renewal_term_months: int | None = Field(
        default=None,
        ge=1,
        le=1200,
    )

    auto_renewal: bool | None = None
    assigned_to: UUID | None = None

    notes: str | None = Field(
        default=None,
        max_length=10_000,
    )

    metadata: dict[str, Any] | None = None


class RenewalResponse(RenewalBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    contract_id: UUID

    assigned_to: UUID | None = None

    decision: RenewalDecision = RenewalDecision.PENDING

    reviewed_by: UUID | None = None
    reviewed_at: datetime | None = None

    approved_by: UUID | None = None
    approved_at: datetime | None = None

    completed_at: datetime | None = None

    created_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class RenewalDetailResponse(RenewalResponse):
    contract_title: str | None = None
    contract_number: str | None = None
    contract_type: str | None = None

    counterparty: str | None = None

    contract_value: float | None = None
    currency: str | None = None

    risk_level: str | None = None
    risk_score: float | None = None

    days_until_renewal: int | None = None
    days_until_notice_deadline: int | None = None

    notice_deadline_missed: bool = False
    action_required: bool = False

    recommended_action: str | None = None


class RenewalDecisionRequest(BaseModel):
    decision: RenewalDecision

    notes: str | None = Field(
        default=None,
        max_length=10_000,
    )

    proposed_renewal_date: date | None = None
    proposed_term_months: int | None = Field(
        default=None,
        ge=1,
        le=1200,
    )


class RenewalDecisionResponse(BaseModel):
    renewal_id: UUID
    contract_id: UUID

    decision: RenewalDecision
    status: RenewalStatus

    decided_by: UUID | None = None
    decided_at: datetime

    message: str


class RenewalApproveRequest(BaseModel):
    notes: str | None = Field(
        default=None,
        max_length=5000,
    )


class RenewalApproveResponse(BaseModel):
    renewal_id: UUID

    approved: bool
    approved_by: UUID
    approved_at: datetime

    status: RenewalStatus


class RenewalCompleteRequest(BaseModel):
    new_expiry_date: date

    new_contract_value: float | None = Field(
        default=None,
        ge=0,
    )

    notes: str | None = None


class RenewalCompleteResponse(BaseModel):
    renewal_id: UUID
    contract_id: UUID

    previous_expiry_date: date | None = None
    new_expiry_date: date

    status: RenewalStatus = RenewalStatus.RENEWED

    completed_at: datetime


class UpcomingRenewalRequest(BaseModel):
    days: int = Field(
        default=90,
        ge=1,
        le=3650,
    )

    organization_id: UUID | None = None
    assigned_to: UUID | None = None

    priorities: list[RenewalPriority] | None = None

    include_auto_renewal: bool = True


class UpcomingRenewalResponse(BaseModel):
    items: list[RenewalDetailResponse] = Field(
        default_factory=list
    )

    total: int = Field(default=0, ge=0)

    window_days: int = 90
    generated_at: datetime


class RenewalFilter(BaseModel):
    organization_id: UUID | None = None
    contract_id: UUID | None = None
    assigned_to: UUID | None = None

    statuses: list[RenewalStatus] | None = None
    renewal_types: list[RenewalType] | None = None
    priorities: list[RenewalPriority] | None = None
    decisions: list[RenewalDecision] | None = None

    auto_renewal: bool | None = None
    action_required: bool | None = None

    renewal_from: date | None = None
    renewal_to: date | None = None

    notice_from: date | None = None
    notice_to: date | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


class RenewalListResponse(BaseModel):
    items: list[RenewalResponse] = Field(
        default_factory=list
    )

    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)

    page_size: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    pages: int = Field(default=0, ge=0)


class RenewalStatistics(BaseModel):
    total: int = 0

    pending: int = 0
    upcoming: int = 0
    under_review: int = 0

    approved: int = 0
    declined: int = 0
    negotiating: int = 0

    renewed: int = 0
    cancelled: int = 0
    expired: int = 0

    auto_renewals: int = 0

    due_7_days: int = 0
    due_30_days: int = 0
    due_60_days: int = 0
    due_90_days: int = 0

    missed_notice_deadlines: int = 0

    high_priority: int = 0
    critical: int = 0


class RenewalValueStatistics(BaseModel):
    currency: str

    total_value: float = 0.0
    upcoming_value: float = 0.0
    renewed_value: float = 0.0
    declined_value: float = 0.0


class RenewalDashboardResponse(BaseModel):
    statistics: RenewalStatistics

    value_statistics: list[RenewalValueStatistics] = Field(
        default_factory=list
    )

    urgent: list[RenewalDetailResponse] = Field(
        default_factory=list
    )

    upcoming: list[RenewalDetailResponse] = Field(
        default_factory=list
    )

    generated_at: datetime


class RenewalReminderCreate(BaseModel):
    renewal_id: UUID

    reminder_at: datetime

    recipient_user_ids: list[UUID] = Field(
        default_factory=list
    )

    message: str | None = Field(
        default=None,
        max_length=5000,
    )


class RenewalReminderResponse(BaseModel):
    renewal_id: UUID

    reminder_id: UUID | None = None
    reminder_at: datetime

    scheduled: bool = True

    created_at: datetime | None = None


class BulkRenewalDecisionRequest(BaseModel):
    renewal_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    decision: RenewalDecision

    notes: str | None = None


class BulkRenewalResult(BaseModel):
    renewal_id: UUID
    success: bool

    error: str | None = None


class BulkRenewalResponse(BaseModel):
    requested: int = 0
    successful: int = 0
    failed: int = 0

    results: list[BulkRenewalResult] = Field(
        default_factory=list
    )