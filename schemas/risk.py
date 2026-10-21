# backend/app/schemas/risk.py

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


class RiskLevel(str, Enum):
    UNKNOWN = "unknown"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    MITIGATING = "mitigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    WAIVED = "waived"
    CLOSED = "closed"


class RiskType(str, Enum):
    LEGAL = "legal"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    CONTRACTUAL = "contractual"
    REGULATORY = "regulatory"
    PRIVACY = "privacy"
    SECURITY = "security"

    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    RENEWAL = "renewal"
    PAYMENT = "payment"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PROTECTION = "data_protection"
    JURISDICTION = "jurisdiction"
    SERVICE_LEVEL = "service_level"

    MISSING_CLAUSE = "missing_clause"
    NON_STANDARD_CLAUSE = "non_standard_clause"
    AMBIGUOUS_LANGUAGE = "ambiguous_language"

    OTHER = "other"


class RiskSource(str, Enum):
    MANUAL = "manual"
    RULE_ENGINE = "rule_engine"
    ML_CLASSIFIER = "ml_classifier"
    LLM = "llm"
    CLAUSE_ANALYSIS = "clause_analysis"
    CONTRACT_ANALYSIS = "contract_analysis"
    COMPLIANCE_CHECK = "compliance_check"


class RiskDecision(str, Enum):
    ACCEPT = "accept"
    MITIGATE = "mitigate"
    ESCALATE = "escalate"
    WAIVE = "waive"
    RESOLVE = "resolve"


# ============================================================
# Base schema
# ============================================================


class RiskBase(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    risk_type: RiskType = RiskType.OTHER

    risk_level: RiskLevel = RiskLevel.UNKNOWN

    risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    impact: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    recommendation: str | None = Field(
        default=None,
        max_length=10_000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# Create
# ============================================================


class RiskCreate(RiskBase):
    organization_id: UUID

    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    source: RiskSource = RiskSource.MANUAL

    source_text: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    assigned_to: UUID | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "RiskCreate":
        if not any(
            (
                self.contract_id,
                self.document_id,
                self.clause_id,
            )
        ):
            raise ValueError(
                "Risk must reference a contract, "
                "document, or clause"
            )

        return self


# ============================================================
# Update
# ============================================================


class RiskUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=10_000,
    )

    risk_type: RiskType | None = None

    risk_level: RiskLevel | None = None

    status: RiskStatus | None = None

    risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    impact: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    recommendation: str | None = None

    assigned_to: UUID | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# Response
# ============================================================


class RiskResponse(RiskBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID

    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    status: RiskStatus = RiskStatus.OPEN

    source: RiskSource

    source_text: str | None = None

    page_number: int | None = None

    confidence: float | None = None

    assigned_to: UUID | None = None

    reviewed_by: UUID | None = None

    reviewed_at: datetime | None = None

    resolved_by: UUID | None = None

    resolved_at: datetime | None = None

    created_by: UUID | None = None

    created_at: datetime

    updated_at: datetime


# ============================================================
# Detailed response
# ============================================================


class RiskDetailResponse(RiskResponse):
    contract_title: str | None = None

    contract_number: str | None = None

    document_title: str | None = None

    clause_title: str | None = None

    clause_number: str | None = None

    clause_text: str | None = None

    counterparty: str | None = None

    assigned_user_name: str | None = None

    action_required: bool = False

    mitigation_status: str | None = None


# ============================================================
# Risk scoring
# ============================================================


class RiskScoreRequest(BaseModel):
    likelihood: float = Field(
        ge=0.0,
        le=1.0,
    )

    impact: float = Field(
        ge=0.0,
        le=1.0,
    )


class RiskScoreResponse(BaseModel):
    likelihood: float

    impact: float

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    risk_level: RiskLevel


# ============================================================
# AI risk analysis
# ============================================================


class RiskAnalysisRequest(BaseModel):
    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    analyze_clauses: bool = True

    detect_missing_clauses: bool = True

    detect_non_standard_clauses: bool = True

    include_compliance: bool = True

    include_financial_risk: bool = True

    minimum_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    force_reprocess: bool = False

    @model_validator(mode="after")
    def validate_target(self) -> "RiskAnalysisRequest":
        if not any(
            (
                self.contract_id,
                self.document_id,
                self.clause_id,
            )
        ):
            raise ValueError(
                "Provide contract_id, document_id, "
                "or clause_id"
            )

        return self


class DetectedRisk(BaseModel):
    risk_type: RiskType

    risk_level: RiskLevel

    title: str

    description: str

    risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    impact: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    clause_id: UUID | None = None

    page_number: int | None = None

    source_text: str | None = None

    recommendation: str | None = None

    evidence: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RiskAnalysisResponse(BaseModel):
    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    risks: list[DetectedRisk] = Field(
        default_factory=list
    )

    overall_risk_level: RiskLevel

    overall_risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    summary: str | None = None

    recommendation: str | None = None

    analyzed_clauses: int = 0

    model_name: str | None = None

    model_version: str | None = None

    processing_time_ms: float | None = None

    generated_at: datetime


# ============================================================
# Risk decision
# ============================================================


class RiskDecisionRequest(BaseModel):
    decision: RiskDecision

    notes: str | None = Field(
        default=None,
        max_length=10_000,
    )

    @model_validator(mode="after")
    def validate_decision(self) -> "RiskDecisionRequest":
        if (
            self.decision
            in {
                RiskDecision.ACCEPT,
                RiskDecision.WAIVE,
            }
            and not self.notes
        ):
            raise ValueError(
                "notes are required when accepting "
                "or waiving a risk"
            )

        return self


class RiskDecisionResponse(BaseModel):
    risk_id: UUID

    decision: RiskDecision

    status: RiskStatus

    decided_by: UUID

    decided_at: datetime

    message: str


# ============================================================
# Mitigation
# ============================================================


class RiskMitigationCreate(BaseModel):
    risk_id: UUID

    action: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    owner_id: UUID | None = None

    due_at: datetime | None = None

    notes: str | None = None


class RiskMitigationResponse(BaseModel):
    id: UUID

    risk_id: UUID

    action: str

    owner_id: UUID | None = None

    status: str

    due_at: datetime | None = None

    completed_at: datetime | None = None

    created_at: datetime


# ============================================================
# Resolve
# ============================================================


class RiskResolveRequest(BaseModel):
    resolution: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )


class RiskResolveResponse(BaseModel):
    risk_id: UUID

    status: RiskStatus = RiskStatus.RESOLVED

    resolution: str

    resolved_by: UUID

    resolved_at: datetime


# ============================================================
# Filters
# ============================================================


class RiskFilter(BaseModel):
    organization_id: UUID | None = None

    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    assigned_to: UUID | None = None

    risk_types: list[RiskType] | None = None

    risk_levels: list[RiskLevel] | None = None

    statuses: list[RiskStatus] | None = None

    sources: list[RiskSource] | None = None

    minimum_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    maximum_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    unresolved_only: bool = False

    created_from: datetime | None = None

    created_to: datetime | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# Pagination
# ============================================================


class RiskListResponse(BaseModel):
    items: list[RiskResponse] = Field(
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


class RiskStatistics(BaseModel):
    total: int = 0

    unknown: int = 0

    low: int = 0

    medium: int = 0

    high: int = 0

    critical: int = 0

    open: int = 0

    under_review: int = 0

    accepted: int = 0

    mitigating: int = 0

    mitigated: int = 0

    resolved: int = 0

    waived: int = 0

    average_risk_score: float = 0.0

    unresolved_high_risk: int = 0

    unresolved_critical_risk: int = 0


class RiskTypeStatistics(BaseModel):
    risk_type: RiskType

    total: int = 0

    low: int = 0

    medium: int = 0

    high: int = 0

    critical: int = 0

    average_score: float = 0.0


# ============================================================
# Dashboard
# ============================================================


class RiskDashboardResponse(BaseModel):
    statistics: RiskStatistics

    by_type: list[RiskTypeStatistics] = Field(
        default_factory=list
    )

    critical_risks: list[RiskDetailResponse] = Field(
        default_factory=list
    )

    high_risks: list[RiskDetailResponse] = Field(
        default_factory=list
    )

    recently_detected: list[RiskDetailResponse] = Field(
        default_factory=list
    )

    generated_at: datetime


# ============================================================
# Contract risk summary
# ============================================================


class ContractRiskSummary(BaseModel):
    contract_id: UUID

    overall_risk_level: RiskLevel

    overall_risk_score: float | None = None

    total_risks: int = 0

    low_risks: int = 0

    medium_risks: int = 0

    high_risks: int = 0

    critical_risks: int = 0

    unresolved_risks: int = 0

    requires_legal_review: bool = False

    key_risks: list[RiskResponse] = Field(
        default_factory=list
    )


# ============================================================
# Clause risk summary
# ============================================================


class ClauseRiskSummary(BaseModel):
    clause_id: UUID

    clause_type: str | None = None

    risk_level: RiskLevel

    risk_score: float | None = None

    non_standard: bool = False

    risk_reasons: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )


# ============================================================
# Bulk operations
# ============================================================


class BulkRiskStatusRequest(BaseModel):
    risk_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    status: RiskStatus

    notes: str | None = None


class BulkRiskAssignRequest(BaseModel):
    risk_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    assigned_to: UUID


class BulkRiskResult(BaseModel):
    risk_id: UUID

    success: bool

    error: str | None = None


class BulkRiskResponse(BaseModel):
    requested: int = 0

    successful: int = 0

    failed: int = 0

    results: list[BulkRiskResult] = Field(
        default_factory=list
    )