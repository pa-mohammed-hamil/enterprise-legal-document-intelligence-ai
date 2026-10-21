# backend/app/schemas/review.py

from __future__ import annotations

from datetime import datetime
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


class ReviewType(str, Enum):
    CONTRACT = "contract"
    DOCUMENT = "document"
    CLAUSE = "clause"
    RISK = "risk"
    COMPLIANCE = "compliance"
    RENEWAL = "renewal"
    OBLIGATION = "obligation"
    LEGAL = "legal"
    AI_VALIDATION = "ai_validation"
    OTHER = "other"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingType(str, Enum):
    LEGAL_RISK = "legal_risk"
    CLAUSE_RISK = "clause_risk"
    COMPLIANCE = "compliance"
    MISSING_CLAUSE = "missing_clause"
    NON_STANDARD_CLAUSE = "non_standard_clause"
    OBLIGATION = "obligation"
    EXPIRY = "expiry"
    RENEWAL = "renewal"
    DATA_QUALITY = "data_quality"
    AI_UNCERTAINTY = "ai_uncertainty"
    OTHER = "other"


# ============================================================
# Base
# ============================================================


class ReviewBase(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    review_type: ReviewType = ReviewType.LEGAL

    priority: ReviewPriority = ReviewPriority.MEDIUM

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Create
# ============================================================


class ReviewCreate(ReviewBase):
    organization_id: UUID

    contract_id: UUID | None = None
    document_id: UUID | None = None
    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    due_at: datetime | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "ReviewCreate":
        if not any(
            (
                self.contract_id,
                self.document_id,
                self.clause_id,
            )
        ):
            raise ValueError(
                "A review must reference a contract, "
                "document, or clause"
            )

        return self


# ============================================================
# Update
# ============================================================


class ReviewUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    review_type: ReviewType | None = None
    priority: ReviewPriority | None = None
    status: ReviewStatus | None = None

    assigned_to: UUID | None = None

    due_at: datetime | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# Finding
# ============================================================


class ReviewFindingCreate(BaseModel):
    finding_type: FindingType

    severity: FindingSeverity = FindingSeverity.MEDIUM

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )

    clause_id: UUID | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    source_text: str | None = None

    recommendation: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewFindingResponse(ReviewFindingCreate):
    id: UUID

    review_id: UUID

    resolved: bool = False

    resolution: str | None = None

    resolved_by: UUID | None = None

    resolved_at: datetime | None = None

    created_at: datetime


# ============================================================
# Comments
# ============================================================


class ReviewCommentCreate(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )

    internal: bool = True


class ReviewCommentResponse(BaseModel):
    id: UUID

    review_id: UUID

    user_id: UUID

    content: str

    internal: bool

    created_at: datetime

    updated_at: datetime | None = None


# ============================================================
# Response
# ============================================================


class ReviewResponse(ReviewBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID

    contract_id: UUID | None = None
    document_id: UUID | None = None
    clause_id: UUID | None = None

    status: ReviewStatus

    assigned_to: UUID | None = None

    decision: ReviewDecision | None = None

    due_at: datetime | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    reviewed_by: UUID | None = None

    created_by: UUID | None = None

    created_at: datetime
    updated_at: datetime


# ============================================================
# Detailed response
# ============================================================


class ReviewDetailResponse(ReviewResponse):
    contract_title: str | None = None
    document_title: str | None = None
    clause_title: str | None = None

    assigned_user_name: str | None = None
    reviewer_name: str | None = None

    findings: list[ReviewFindingResponse] = Field(
        default_factory=list
    )

    comments: list[ReviewCommentResponse] = Field(
        default_factory=list
    )

    finding_count: int = 0
    unresolved_findings: int = 0

    critical_findings: int = 0
    high_findings: int = 0

    is_overdue: bool = False
    action_required: bool = False


# ============================================================
# Assignment
# ============================================================


class ReviewAssignRequest(BaseModel):
    assigned_to: UUID

    due_at: datetime | None = None

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )


class ReviewAssignResponse(BaseModel):
    review_id: UUID

    assigned_to: UUID

    status: ReviewStatus = ReviewStatus.ASSIGNED

    assigned_at: datetime

    success: bool = True


# ============================================================
# Start review
# ============================================================


class ReviewStartResponse(BaseModel):
    review_id: UUID

    status: ReviewStatus = ReviewStatus.IN_PROGRESS

    started_at: datetime

    started_by: UUID


# ============================================================
# Decision
# ============================================================


class ReviewDecisionRequest(BaseModel):
    decision: ReviewDecision

    notes: str | None = Field(
        default=None,
        max_length=10_000,
    )

    @model_validator(mode="after")
    def validate_notes(self) -> "ReviewDecisionRequest":
        if (
            self.decision
            in {
                ReviewDecision.REJECT,
                ReviewDecision.REQUEST_CHANGES,
                ReviewDecision.ESCALATE,
            }
            and not self.notes
        ):
            raise ValueError(
                "notes are required for rejection, "
                "change requests, or escalation"
            )

        return self


class ReviewDecisionResponse(BaseModel):
    review_id: UUID

    decision: ReviewDecision

    status: ReviewStatus

    reviewed_by: UUID

    reviewed_at: datetime

    message: str


# ============================================================
# Finding resolution
# ============================================================


class FindingResolveRequest(BaseModel):
    resolution: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )


class FindingResolveResponse(BaseModel):
    finding_id: UUID

    review_id: UUID

    resolved: bool = True

    resolution: str

    resolved_by: UUID

    resolved_at: datetime


# ============================================================
# AI-assisted review
# ============================================================


class AIReviewRequest(BaseModel):
    review_id: UUID

    analyze_clauses: bool = True
    analyze_risk: bool = True
    analyze_compliance: bool = True
    detect_missing_clauses: bool = True
    detect_non_standard_clauses: bool = True
    analyze_obligations: bool = True

    minimum_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )


class AIReviewFinding(BaseModel):
    finding_type: FindingType

    severity: FindingSeverity

    title: str

    description: str

    clause_id: UUID | None = None

    source_text: str | None = None

    recommendation: str | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class AIReviewResponse(BaseModel):
    review_id: UUID

    findings: list[AIReviewFinding] = Field(
        default_factory=list
    )

    risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    summary: str | None = None

    recommendation: str | None = None

    model_name: str | None = None

    processing_time_ms: float | None = None

    generated_at: datetime


# ============================================================
# Filters
# ============================================================


class ReviewFilter(BaseModel):
    organization_id: UUID | None = None

    contract_id: UUID | None = None
    document_id: UUID | None = None
    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    review_types: list[ReviewType] | None = None
    statuses: list[ReviewStatus] | None = None
    priorities: list[ReviewPriority] | None = None

    overdue_only: bool = False

    created_from: datetime | None = None
    created_to: datetime | None = None

    due_from: datetime | None = None
    due_to: datetime | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# Pagination
# ============================================================


class ReviewListResponse(BaseModel):
    items: list[ReviewResponse] = Field(
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
# Statistics
# ============================================================


class ReviewStatistics(BaseModel):
    total: int = 0

    pending: int = 0
    assigned: int = 0
    in_progress: int = 0

    changes_requested: int = 0

    approved: int = 0
    rejected: int = 0
    completed: int = 0

    overdue: int = 0

    critical_priority: int = 0
    high_priority: int = 0

    unresolved_findings: int = 0

    critical_findings: int = 0
    high_findings: int = 0


class ReviewDashboardResponse(BaseModel):
    statistics: ReviewStatistics

    pending_reviews: list[ReviewDetailResponse] = Field(
        default_factory=list
    )

    overdue_reviews: list[ReviewDetailResponse] = Field(
        default_factory=list
    )

    critical_reviews: list[ReviewDetailResponse] = Field(
        default_factory=list
    )

    generated_at: datetime


# ============================================================
# Bulk assignment
# ============================================================


class BulkReviewAssignRequest(BaseModel):
    review_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    assigned_to: UUID

    due_at: datetime | None = None


class BulkReviewStatusRequest(BaseModel):
    review_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    status: ReviewStatus


class BulkReviewResult(BaseModel):
    review_id: UUID

    success: bool

    error: str | None = None


class BulkReviewResponse(BaseModel):
    requested: int = 0
    successful: int = 0
    failed: int = 0

    results: list[BulkReviewResult] = Field(
        default_factory=list
    )