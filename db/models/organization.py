# backend/db/models/organization.py

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    """
    Tenant-aware enterprise organization.

    Organization is the business boundary used by:
        - users
        - documents
        - contracts
        - clauses
        - obligations
        - risks
        - renewals
        - reviews
        - notifications
        - model runs
        - audit events

    tenant_id remains the primary security boundary.

    organization_id is an additional business/data partition and
    must never replace tenant isolation.
    """

    __tablename__ = "organizations"

    # ========================================================
    # IDENTITY
    # ========================================================

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # ========================================================
    # ORGANIZATION DETAILS
    # ========================================================

    name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    legal_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    slug: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    organization_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="company",
        index=True,
    )

    industry: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # CONTACT
    # ========================================================

    primary_email: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    primary_phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    website: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    # ========================================================
    # ADDRESS
    # ========================================================

    address_line1: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    address_line2: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    city: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    state: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    postal_code: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    country_code: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    # ========================================================
    # LOCALE
    # ========================================================

    timezone_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="UTC",
    )

    locale: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
    )

    default_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
    )

    default_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="USD",
    )

    # ========================================================
    # STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # PLAN / SUBSCRIPTION
    # ========================================================

    plan: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="free",
        index=True,
    )

    subscription_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        index=True,
    )

    subscription_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    subscription_ends_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    billing_customer_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # RESOURCE QUOTAS
    # ========================================================

    max_users: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
    )

    max_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1000,
    )

    max_storage_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10 * 1024 * 1024 * 1024,
    )

    max_monthly_ai_requests: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10000,
    )

    max_monthly_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10_000_000,
    )

    max_vector_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1_000_000,
    )

    # ========================================================
    # CURRENT USAGE
    # ========================================================

    user_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    storage_used_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    monthly_ai_requests: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    monthly_tokens_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    vector_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    usage_period_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # LEGAL / COMPLIANCE SETTINGS
    # ========================================================

    legal_jurisdictions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    compliance_frameworks: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    default_governing_law: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    require_legal_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    high_risk_review_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ========================================================
    # SECURITY SETTINGS
    # ========================================================

    mfa_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    sso_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    sso_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    sso_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    allowed_email_domains: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_ip_ranges: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    session_timeout_minutes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=60,
    )

    # ========================================================
    # AI / RAG SETTINGS
    # ========================================================

    ai_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    rag_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    default_llm_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    default_embedding_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    default_reranker_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    vector_namespace: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    ai_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    rag_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # RETENTION
    # ========================================================

    default_retention_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    audit_retention_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    model_run_retention_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # NOTIFICATION SETTINGS
    # ========================================================

    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    email_notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    notification_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # BRANDING / UI
    # ========================================================

    logo_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
    )

    branding: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # FEATURE FLAGS
    # ========================================================

    feature_flags: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # EXTENSION METADATA
    # ========================================================

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # SOFT DELETE
    # ========================================================

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )

    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    # Keep these relationships one-way until counterpart models
    # intentionally define matching back_populates relationships.

    users = relationship(
        "User",
        foreign_keys="User.organization_id",
        lazy="selectin",
    )

    documents = relationship(
        "Document",
        primaryjoin=(
            "Organization.id == "
            "foreign(Document.organization_id)"
        ),
        lazy="selectin",
        viewonly=True,
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "slug",
            name="uq_organizations_tenant_slug",
        ),

        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_organizations_tenant_external_id",
        ),

        CheckConstraint(
            "max_users >= 0",
            name="ck_organizations_max_users",
        ),

        CheckConstraint(
            "max_documents >= 0",
            name="ck_organizations_max_documents",
        ),

        CheckConstraint(
            "max_storage_bytes >= 0",
            name="ck_organizations_max_storage",
        ),

        CheckConstraint(
            "max_monthly_ai_requests >= 0",
            name="ck_organizations_max_ai_requests",
        ),

        CheckConstraint(
            "max_monthly_tokens >= 0",
            name="ck_organizations_max_tokens",
        ),

        Index(
            "ix_organizations_tenant_status",
            "tenant_id",
            "status",
        ),

        Index(
            "ix_organizations_tenant_active",
            "tenant_id",
            "is_active",
            "is_deleted",
        ),

        Index(
            "ix_organizations_tenant_name",
            "tenant_id",
            "name",
        ),

        Index(
            "ix_organizations_tenant_industry",
            "tenant_id",
            "industry",
        ),

        Index(
            "ix_organizations_tenant_plan",
            "tenant_id",
            "plan",
            "subscription_status",
        ),

        Index(
            "ix_organizations_subscription_end",
            "tenant_id",
            "subscription_ends_at",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @property
    def display_name(self) -> str:
        return self.legal_name or self.name

    @property
    def enabled(self) -> bool:
        return bool(
            self.is_active
            and not self.is_deleted
            and self.status.lower() == "active"
        )

    @property
    def subscription_active(self) -> bool:
        if (
            self.subscription_status
            or ""
        ).lower() not in {
            "active",
            "trial",
        }:
            return False

        if self.subscription_ends_at is None:
            return True

        return (
            self.subscription_ends_at
            > utcnow()
        )

    # ========================================================
    # QUOTA HELPERS
    # ========================================================

    @property
    def user_quota_exceeded(self) -> bool:
        return (
            self.max_users > 0
            and self.user_count
            >= self.max_users
        )

    @property
    def document_quota_exceeded(self) -> bool:
        return (
            self.max_documents > 0
            and self.document_count
            >= self.max_documents
        )

    @property
    def storage_quota_exceeded(self) -> bool:
        return (
            self.max_storage_bytes > 0
            and self.storage_used_bytes
            >= self.max_storage_bytes
        )

    @property
    def ai_request_quota_exceeded(self) -> bool:
        return (
            self.max_monthly_ai_requests > 0
            and self.monthly_ai_requests
            >= self.max_monthly_ai_requests
        )

    @property
    def token_quota_exceeded(self) -> bool:
        return (
            self.max_monthly_tokens > 0
            and self.monthly_tokens_used
            >= self.max_monthly_tokens
        )

    @property
    def vector_quota_exceeded(self) -> bool:
        return (
            self.max_vector_count > 0
            and self.vector_count
            >= self.max_vector_count
        )

    @property
    def quota_summary(self) -> dict[str, Any]:
        return {
            "users": {
                "used": self.user_count,
                "limit": self.max_users,
                "exceeded":
                    self.user_quota_exceeded,
            },
            "documents": {
                "used": self.document_count,
                "limit": self.max_documents,
                "exceeded":
                    self.document_quota_exceeded,
            },
            "storage": {
                "used": self.storage_used_bytes,
                "limit": self.max_storage_bytes,
                "exceeded":
                    self.storage_quota_exceeded,
            },
            "ai_requests": {
                "used": self.monthly_ai_requests,
                "limit":
                    self.max_monthly_ai_requests,
                "exceeded":
                    self.ai_request_quota_exceeded,
            },
            "tokens": {
                "used": self.monthly_tokens_used,
                "limit": self.max_monthly_tokens,
                "exceeded":
                    self.token_quota_exceeded,
            },
            "vectors": {
                "used": self.vector_count,
                "limit": self.max_vector_count,
                "exceeded":
                    self.vector_quota_exceeded,
            },
        }

    def can_add_users(
        self,
        count: int = 1,
    ) -> bool:
        if count < 0:
            return False

        if self.max_users == 0:
            return True

        return (
            self.user_count + count
            <= self.max_users
        )

    def can_add_documents(
        self,
        count: int = 1,
    ) -> bool:
        if count < 0:
            return False

        if self.max_documents == 0:
            return True

        return (
            self.document_count + count
            <= self.max_documents
        )

    def can_use_storage(
        self,
        bytes_required: int,
    ) -> bool:
        if bytes_required < 0:
            return False

        if self.max_storage_bytes == 0:
            return True

        return (
            self.storage_used_bytes
            + bytes_required
            <= self.max_storage_bytes
        )

    def can_consume_tokens(
        self,
        tokens: int,
    ) -> bool:
        if tokens < 0:
            return False

        if self.max_monthly_tokens == 0:
            return True

        return (
            self.monthly_tokens_used
            + tokens
            <= self.max_monthly_tokens
        )

    # ========================================================
    # USAGE
    # ========================================================

    def add_ai_usage(
        self,
        *,
        requests: int = 1,
        tokens: int = 0,
    ) -> None:
        self.monthly_ai_requests += max(
            0,
            requests,
        )

        self.monthly_tokens_used += max(
            0,
            tokens,
        )

    def add_storage_usage(
        self,
        bytes_used: int,
    ) -> None:
        self.storage_used_bytes = max(
            0,
            self.storage_used_bytes
            + bytes_used,
        )

    def add_vectors(
        self,
        count: int,
    ) -> None:
        self.vector_count = max(
            0,
            self.vector_count + count,
        )

    def reset_monthly_usage(self) -> None:
        self.monthly_ai_requests = 0
        self.monthly_tokens_used = 0
        self.usage_period_started_at = (
            utcnow()
        )

    # ========================================================
    # FEATURE FLAGS
    # ========================================================

    def feature_enabled(
        self,
        feature: str,
        *,
        default: bool = False,
    ) -> bool:
        value = (
            self.feature_flags or {}
        ).get(
            feature,
            default,
        )

        return bool(value)

    def set_feature(
        self,
        feature: str,
        enabled: bool,
    ) -> None:
        flags = dict(
            self.feature_flags or {}
        )

        flags[feature] = bool(enabled)

        # Reassignment ensures JSONB mutation is detected.
        self.feature_flags = flags

    # ========================================================
    # SETTINGS
    # ========================================================

    def update_setting(
        self,
        key: str,
        value: Any,
    ) -> None:
        current = dict(
            self.settings or {}
        )

        current[key] = value
        self.settings = current

    # ========================================================
    # STATUS
    # ========================================================

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted organization cannot "
                "be activated."
            )

        self.status = "active"
        self.is_active = True

    def suspend(self) -> None:
        self.status = "suspended"
        self.is_active = False

    def deactivate(self) -> None:
        self.status = "inactive"
        self.is_active = False

    def verify(self) -> None:
        self.verified = True
        self.verified_at = utcnow()

    # ========================================================
    # SOFT DELETE
    # ========================================================

    def soft_delete(
        self,
        *,
        deleted_by: Optional[
            uuid.UUID
        ] = None,
    ) -> None:
        self.is_deleted = True
        self.is_active = False
        self.status = "deleted"

        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

    def restore(self) -> None:
        self.is_deleted = False
        self.is_active = True
        self.status = "active"

        self.deleted_at = None
        self.deleted_by = None

    # ========================================================
    # SLUG
    # ========================================================

    @staticmethod
    def generate_slug(
        name: str,
    ) -> str:
        slug = name.strip().lower()

        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            slug,
        )

        return slug.strip("-")

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<Organization("
            f"id={self.id}, "
            f"name={self.name!r}, "
            f"slug={self.slug!r}, "
            f"status={self.status!r}, "
            f"plan={self.plan!r}"
            ")>"
        )