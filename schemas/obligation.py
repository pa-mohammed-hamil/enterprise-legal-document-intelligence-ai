# backend/app/schemas/obligation.py

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# Enums
# ============================================================


class ObligationType(str, Enum):
    PAYMENT = "payment"
    DELIVERY = "delivery"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    CONFIDENTIALITY = "confidentiality"
    NOTICE = "notice"
    RENEWAL = "renewal"
    TERMINATION = "termination"
    INSURANCE = "insurance"
    AUDIT = "audit"
    MAINTENANCE = "maintenance"
    SERVICE_LEVEL = "service_level"
    DATA_PROTECTION = "data_protection"
    INDEMNIFICATION = "indemnification"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ObligationStatus(str, Enum):
    PENDING = "pending"
    UPCOMING = "upcoming"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    WAIVED = "waived"
    CANCELLED = "cancelled"
    NOT_APPLICABLE = "not_applicable"


class ObligationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ObligationFrequency(str, Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    CUSTOM = "custom"


class ObligationSource(str, Enum):
    MANUAL = "manual"
    AI_EXTRACTED = "ai_extracted"
    CLAUSE_EXTRACTION = "clause_extraction"
    IMPORTED = "imported"


# ============================================================
# Base
# ============================================================


class ObligationBase(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    obligation_type: ObligationType = ObligationType.OTHER

    priority: ObligationPriority = ObligationPriority.MEDIUM

    responsible_party: str | None = Field(
        default=None,
        max_length=500,
    )

    beneficiary_party: str | None = Field(
        default=None,
        max_length=500,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Create
# ============================================================


class ObligationCreate(ObligationBase):
    organization_id: UUID

    contract_id: UUID

    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    due_date: date | None = None

    start_date: date | None = None

    frequency: ObligationFrequency = ObligationFrequency.ONCE

    recurrence_interval: int | None = Field(
        default=None,
        ge=1,
    )

    source: ObligationSource = ObligationSource.MANUAL

    source_text: str | None = None

    extraction_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "ObligationCreate":
        if (
            self.start_date is not None
            and self.due_date is not None
            and self.due_date < self.start_date
        ):
            raise ValueError(
                "due_date cannot be earlier than start_date"
            )

        return self


# ============================================================
# Update
# ============================================================


class ObligationUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    obligation_type: ObligationType | None = None

    status: ObligationStatus | None = None

    priority: ObligationPriority | None = None

    responsible_party: str | None = None

    beneficiary_party: str | None = None

    assigned_to: UUID | None = None

    start_date: date | None = None

    due_date: date | None = None

    frequency: ObligationFrequency | None = None

    recurrence_interval: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, Any] | None = None


# ============================================================
# Response
# ============================================================


class ObligationResponse(ObligationBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID

    contract_id: UUID

    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    status: ObligationStatus

    start_date: date | None = None

    due_date: date | None = None

    completed_at: datetime | None = None

    frequency: ObligationFrequency

    recurrence_interval: int | None = None

    source: ObligationSource

    source_text: str | None = None

    extraction_confidence: float | None = None

    is_active: bool = True

    created_by: UUID | None = None

    created_at: datetime

    updated_at: datetime


# ============================================================
# Detailed response
# ============================================================


class ObligationDetailResponse(ObligationResponse):
    contract_title: str | None = None

    contract_number: str | None = None

    clause_title: str | None = None

    clause_number: str | None = None

    clause_text: str | None = None

    assigned_user_name: str | None = None

    days_until_due: int | None = None

    days_overdue: int | None = None

    is_overdue: bool = False

    action_required: bool = False

    recommended_action: str | None = None


# ============================================================
# Completion
# ============================================================


class ObligationCompleteRequest(BaseModel):
    completion_date: datetime | None = None

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )

    evidence: list[str] = Field(default_factory=list)


class ObligationCompleteResponse(BaseModel):
    obligation_id: UUID

    status: ObligationStatus

    completed_at: datetime

    completed_by: UUID | None = None

    message: str = "Obligation completed successfully"


# ============================================================
# Assignment
# ============================================================


class ObligationAssignRequest(BaseModel):
    assigned_to: UUID

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )


class ObligationAssignResponse(BaseModel):
    obligation_id: UUID

    assigned_to: UUID

    assigned_at: datetime

    success: bool = True


# ============================================================
# AI extraction
# ============================================================


class ExtractedObligation(BaseModel):
    title: str

    description: str | None = None

    obligation_type: ObligationType

    responsible_party: str | None = None

    beneficiary_party: str | None = None

    due_date: date | None = None

    frequency: ObligationFrequency = ObligationFrequency.ONCE

    source_text: str | None = None

    page_number: int | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class ObligationExtractionRequest(BaseModel):
    contract_id: UUID

    clause_ids: list[UUID] | None = None

    minimum_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    force_reprocess: bool = False


class ObligationExtractionResponse(BaseModel):
    contract_id: UUID

    obligations: list[ExtractedObligation] = Field(
        default_factory=list
    )

    extracted_count: int = 0

    skipped_count: int = 0

    model_name: str | None = None

    processing_time_ms: float | None = None


# ============================================================
# Filter / search
# ============================================================


class ObligationFilter(BaseModel):
    organization_id: UUID | None = None

    contract_id: UUID | None = None

    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    obligation_types: list[ObligationType] | None = None

    statuses: list[ObligationStatus] | None = None

    priorities: list[ObligationPriority] | None = None

    due_from: date | None = None

    due_to: date | None = None

    overdue_only: bool = False

    search: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# List / pagination
# ============================================================


class ObligationListResponse(BaseModel):
    items: list[ObligationResponse] = Field(
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


# ============================================================
# Upcoming / overdue
# ============================================================


class UpcomingObligationRequest(BaseModel):
    days: int = Field(
        default=30,
        ge=1,
        le=3650,
    )

    contract_id: UUID | None = None

    assigned_to: UUID | None = None

    priorities: list[ObligationPriority] | None = None


class UpcomingObligationResponse(BaseModel):
    items: list[ObligationDetailResponse] = Field(
        default_factory=list
    )

    total: int = 0

    window_days: int = 30

    generated_at: datetime


class OverdueObligationResponse(BaseModel):
    items: list[ObligationDetailResponse] = Field(
        default_factory=list
    )

    total: int = 0

    critical: int = 0

    high_priority: int = 0

    generated_at: datetime


# ============================================================
# Statistics / dashboard
# ============================================================


class ObligationStatistics(BaseModel):
    total: int = 0

    pending: int = 0

    upcoming: int = 0

    in_progress: int = 0

    completed: int = 0

    overdue: int = 0

    waived: int = 0

    critical: int = 0

    high_priority: int = 0

    due_7_days: int = 0

    due_30_days: int = 0


class ObligationTypeStatistics(BaseModel):
    obligation_type: ObligationType

    total: int = 0

    completed: int = 0

    overdue: int = 0

    completion_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )


class ObligationDashboardResponse(BaseModel):
    statistics: ObligationStatistics

    by_type: list[ObligationTypeStatistics] = Field(
        default_factory=list
    )

    upcoming: list[ObligationDetailResponse] = Field(
        default_factory=list
    )

    overdue: list[ObligationDetailResponse] = Field(
        default_factory=list
    )

    generated_at: datetime


# ============================================================
# Bulk operations
# ============================================================


class BulkObligationStatusRequest(BaseModel):
    obligation_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    status: ObligationStatus

    notes: str | None = None


class BulkObligationResult(BaseModel):
    obligation_id: UUID

    success: bool

    error: str | None = None


class BulkObligationResponse(BaseModel):
    requested: int = 0

    successful: int = 0

    failed: int = 0

    results: list[BulkObligationResult] = Field(
        default_factory=list
    )


# ============================================================
# Reminder
# ============================================================


class ObligationReminderRequest(BaseModel):
    obligation_id: UUID

    reminder_at: datetime

    user_ids: list[UUID] = Field(default_factory=list)

    message: str | None = Field(
        default=None,
        max_length=5000,
    )


class ObligationReminderResponse(BaseModel):
    obligation_id: UUID

    reminder_id: UUID | None = None

    reminder_at: datetime

    scheduled: bool

    message: str | None = None