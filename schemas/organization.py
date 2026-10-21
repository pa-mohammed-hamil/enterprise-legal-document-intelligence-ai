# backend/app/schemas/organization.py

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# Enums
# ============================================================


class OrganizationStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


class OrganizationPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class OrganizationRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    LEGAL_ADMIN = "legal_admin"
    LEGAL_REVIEWER = "legal_reviewer"
    MEMBER = "member"
    VIEWER = "viewer"


# ============================================================
# Shared base
# ============================================================


class OrganizationBase(BaseModel):
    name: str = Field(
        ...,
        min_length=2,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    website: str | None = Field(
        default=None,
        max_length=500,
    )

    industry: str | None = Field(
        default=None,
        max_length=255,
    )

    country: str | None = Field(
        default=None,
        max_length=100,
    )

    timezone: str = Field(
        default="UTC",
        max_length=100,
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("organization name cannot be empty")

        return value


# ============================================================
# Creation
# ============================================================


class OrganizationCreate(OrganizationBase):
    slug: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    plan: OrganizationPlan = OrganizationPlan.FREE

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Update
# ============================================================


class OrganizationUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    website: str | None = Field(
        default=None,
        max_length=500,
    )

    industry: str | None = Field(
        default=None,
        max_length=255,
    )

    country: str | None = Field(
        default=None,
        max_length=100,
    )

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    status: OrganizationStatus | None = None

    plan: OrganizationPlan | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# Organization settings
# ============================================================


class OrganizationSettings(BaseModel):
    default_language: str = "en"

    default_currency: str = "USD"

    timezone: str = "UTC"

    date_format: str = "YYYY-MM-DD"

    enable_ocr: bool = True

    enable_ai_extraction: bool = True

    enable_rag: bool = True

    enable_agents: bool = True

    enable_notifications: bool = True

    enable_email_notifications: bool = True

    enable_expiry_tracking: bool = True

    enable_risk_analysis: bool = True

    enable_audit_logging: bool = True

    require_legal_review_high_risk: bool = True

    default_expiry_warning_days: list[int] = Field(
        default_factory=lambda: [90, 60, 30, 14, 7, 1]
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class OrganizationSettingsUpdate(BaseModel):
    default_language: str | None = None

    default_currency: str | None = None

    timezone: str | None = None

    date_format: str | None = None

    enable_ocr: bool | None = None

    enable_ai_extraction: bool | None = None

    enable_rag: bool | None = None

    enable_agents: bool | None = None

    enable_notifications: bool | None = None

    enable_email_notifications: bool | None = None

    enable_expiry_tracking: bool | None = None

    enable_risk_analysis: bool | None = None

    enable_audit_logging: bool | None = None

    require_legal_review_high_risk: bool | None = None

    default_expiry_warning_days: list[int] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# Tenant limits
# ============================================================


class OrganizationLimits(BaseModel):
    max_users: int = Field(default=5, ge=1)

    max_documents: int = Field(default=100, ge=1)

    max_contracts: int = Field(default=100, ge=1)

    max_storage_bytes: int = Field(
        default=1_073_741_824,
        ge=0,
    )

    max_file_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
    )

    monthly_ai_requests: int = Field(
        default=1000,
        ge=0,
    )

    monthly_embedding_tokens: int = Field(
        default=1_000_000,
        ge=0,
    )


class OrganizationUsage(BaseModel):
    users: int = Field(default=0, ge=0)

    documents: int = Field(default=0, ge=0)

    contracts: int = Field(default=0, ge=0)

    storage_bytes: int = Field(default=0, ge=0)

    ai_requests_this_month: int = Field(default=0, ge=0)

    embedding_tokens_this_month: int = Field(
        default=0,
        ge=0,
    )


# ============================================================
# Response
# ============================================================


class OrganizationResponse(OrganizationBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    slug: str

    status: OrganizationStatus

    plan: OrganizationPlan

    owner_id: UUID | None = None

    is_active: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime

    updated_at: datetime


# ============================================================
# Detailed response
# ============================================================


class OrganizationDetailResponse(OrganizationResponse):
    settings: OrganizationSettings = Field(
        default_factory=OrganizationSettings
    )

    limits: OrganizationLimits = Field(
        default_factory=OrganizationLimits
    )

    usage: OrganizationUsage = Field(
        default_factory=OrganizationUsage
    )

    member_count: int = Field(default=0, ge=0)

    document_count: int = Field(default=0, ge=0)

    contract_count: int = Field(default=0, ge=0)


# ============================================================
# Membership
# ============================================================


class OrganizationMemberCreate(BaseModel):
    user_id: UUID

    role: OrganizationRole = OrganizationRole.MEMBER


class OrganizationMemberUpdate(BaseModel):
    role: OrganizationRole


class OrganizationMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID

    organization_id: UUID

    role: OrganizationRole

    email: str | None = None

    full_name: str | None = None

    is_active: bool = True

    joined_at: datetime | None = None


# ============================================================
# Invitation
# ============================================================


class OrganizationInvitationCreate(BaseModel):
    email: str = Field(
        ...,
        min_length=3,
        max_length=320,
    )

    role: OrganizationRole = OrganizationRole.MEMBER


class OrganizationInvitationResponse(BaseModel):
    id: UUID

    organization_id: UUID

    email: str

    role: OrganizationRole

    status: str

    invited_by: UUID | None = None

    expires_at: datetime | None = None

    created_at: datetime


# ============================================================
# Statistics
# ============================================================


class OrganizationStatistics(BaseModel):
    total_users: int = 0

    active_users: int = 0

    total_documents: int = 0

    processed_documents: int = 0

    failed_documents: int = 0

    total_contracts: int = 0

    active_contracts: int = 0

    expired_contracts: int = 0

    upcoming_expiries: int = 0

    total_clauses: int = 0

    high_risk_clauses: int = 0

    critical_risk_clauses: int = 0

    pending_reviews: int = 0

    pending_obligations: int = 0

    overdue_obligations: int = 0

    unread_notifications: int = 0


class OrganizationDashboardResponse(BaseModel):
    organization: OrganizationResponse

    statistics: OrganizationStatistics

    usage: OrganizationUsage

    limits: OrganizationLimits

    generated_at: datetime


# ============================================================
# List / pagination
# ============================================================


class OrganizationListResponse(BaseModel):
    items: list[OrganizationResponse] = Field(
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
# Switch organization / tenant
# ============================================================


class OrganizationSwitchRequest(BaseModel):
    organization_id: UUID


class OrganizationSwitchResponse(BaseModel):
    organization_id: UUID

    organization_name: str

    role: OrganizationRole

    switched: bool = True


# ============================================================
# Member list
# ============================================================


class OrganizationMemberListResponse(BaseModel):
    items: list[OrganizationMemberResponse] = Field(
        default_factory=list
    )

    total: int = Field(default=0, ge=0)


# ============================================================
# Remove member
# ============================================================


class OrganizationMemberRemoveResponse(BaseModel):
    organization_id: UUID

    user_id: UUID

    removed: bool

    message: str


# ============================================================
# Archive organization
# ============================================================


class OrganizationArchiveResponse(BaseModel):
    organization_id: UUID

    archived: bool

    archived_at: datetime | None = None

    message: str


# ============================================================
# Plan
# ============================================================


class OrganizationPlanUpdate(BaseModel):
    plan: OrganizationPlan


class OrganizationPlanResponse(BaseModel):
    organization_id: UUID

    plan: OrganizationPlan

    limits: OrganizationLimits

    updated_at: datetime