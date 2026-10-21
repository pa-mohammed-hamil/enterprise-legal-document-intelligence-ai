# backend/db/models/contract.py

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Contract(Base):
    """
    Enterprise legal contract record.

    The Contract model stores contract lifecycle information and
    links the source document to clauses, obligations, risks,
    renewals, reviews and downstream legal intelligence.

    Important:
        Extracted dates, notice periods, renewal conditions and
        risk values may originate from AI extraction. They should
        not automatically become legally authoritative until
        validated by the appropriate workflow.
    """

    __tablename__ = "contracts"

    # ========================================================
    # IDENTITY / TENANCY
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

    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # SOURCE DOCUMENT
    # ========================================================

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    document_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # CONTRACT IDENTITY
    # ========================================================

    contract_number: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    external_contract_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    contract_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
        index=True,
    )

    contract_subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    # ========================================================
    # LIFECYCLE STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft",
        index=True,
    )

    lifecycle_stage: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # PARTIES
    # ========================================================

    parties: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    counterparty_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    counterparty_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    internal_entity_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    owner_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    department: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    business_unit: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # IMPORTANT DATES
    # ========================================================

    execution_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    effective_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    commencement_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    expiry_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    termination_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    signed_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    # ========================================================
    # TERM
    # ========================================================

    term_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    term_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    evergreen: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # RENEWAL
    # ========================================================

    auto_renewal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    renewal_term_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    renewal_term_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    renewal_notice_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    renewal_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    renewal_status: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    # ========================================================
    # TERMINATION / NOTICE
    # ========================================================

    termination_notice_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    notice_deadline: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    notice_method: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    termination_for_convenience: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    termination_for_cause: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    # ========================================================
    # FINANCIAL
    # ========================================================

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    contract_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(
            precision=20,
            scale=4,
        ),
        nullable=True,
    )

    annual_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(
            precision=20,
            scale=4,
        ),
        nullable=True,
    )

    monthly_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(
            precision=20,
            scale=4,
        ),
        nullable=True,
    )

    payment_terms: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    payment_frequency: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ========================================================
    # JURISDICTION
    # ========================================================

    governing_law: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    jurisdiction: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    dispute_resolution: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # RISK
    # ========================================================

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unknown",
        index=True,
    )

    risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
    )

    risk_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    risk_categories: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    risk_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    high_risk_clause_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    critical_risk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    requires_risk_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # REVIEW
    # ========================================================

    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    reviewed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    approved: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    # ========================================================
    # EXTRACTION / AI PROVENANCE
    # ========================================================

    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    extraction_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    extraction_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    classification_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # LEGAL INTELLIGENCE
    # ========================================================

    extracted_entities: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    key_terms: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    obligations_summary: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    rights_summary: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    important_dates: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # RAG / SEARCH
    # ========================================================

    rag_indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    rag_indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    vector_namespace: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # VERSIONING
    # ========================================================

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    supersedes_contract_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    content_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
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
    # METADATA
    # ========================================================

    # Python attribute is metadata_json because Base.metadata
    # is reserved by SQLAlchemy.
    #
    # Physical PostgreSQL column remains "metadata".
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
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

    document = relationship(
        "Document",
        back_populates="contracts",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    document_version = relationship(
        "DocumentVersion",
        foreign_keys=[document_version_id],
        lazy="selectin",
    )

    clauses = relationship(
        "Clause",
        back_populates="contract",
        foreign_keys="Clause.contract_id",
        lazy="selectin",
    )

    obligations = relationship(
        "Obligation",
        back_populates="contract",
        lazy="selectin",
    )

    risks = relationship(
        "Risk",
        back_populates="contract",
        lazy="selectin",
    )

    renewals = relationship(
        "Renewal",
        back_populates="contract",
        lazy="selectin",
    )

    reviews = relationship(
        "Review",
        back_populates="contract",
        lazy="selectin",
    )

    supersedes_contract = relationship(
        "Contract",
        remote_side=[id],
        foreign_keys=[supersedes_contract_id],
        uselist=False,
    )

    # ========================================================
    # INDEXES / CONSTRAINTS
    # ========================================================

    __table_args__ = (
        Index(
            "ix_contracts_tenant_org",
            "tenant_id",
            "organization_id",
        ),
        Index(
            "ix_contracts_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_contracts_tenant_type",
            "tenant_id",
            "contract_type",
        ),
        Index(
            "ix_contracts_tenant_expiry",
            "tenant_id",
            "expiry_date",
        ),
        Index(
            "ix_contracts_tenant_notice_deadline",
            "tenant_id",
            "notice_deadline",
        ),
        Index(
            "ix_contracts_tenant_renewal_date",
            "tenant_id",
            "renewal_date",
        ),
        Index(
            "ix_contracts_tenant_risk",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_contracts_tenant_review",
            "tenant_id",
            "review_status",
        ),
        Index(
            "ix_contracts_tenant_current",
            "tenant_id",
            "is_current",
            "is_deleted",
        ),
        Index(
            "ix_contracts_counterparty",
            "tenant_id",
            "counterparty_name",
        ),
        Index(
            "ix_contracts_owner",
            "tenant_id",
            "owner_user_id",
        ),
        Index(
            "ix_contracts_model_run",
            "tenant_id",
            "model_run_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "document_version_id",
            "content_hash",
            name="uq_contract_document_version_hash",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @property
    def is_expired(self) -> bool:
        if self.expiry_date is None:
            return False

        return self.expiry_date < date.today()

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.expiry_date is None:
            return None

        return (
            self.expiry_date
            - date.today()
        ).days

    @property
    def days_until_notice_deadline(self) -> Optional[int]:
        if self.notice_deadline is None:
            return None

        return (
            self.notice_deadline
            - date.today()
        ).days

    @property
    def notice_deadline_passed(self) -> bool:
        if self.notice_deadline is None:
            return False

        return (
            self.notice_deadline
            < date.today()
        )

    @property
    def renewal_due(self) -> bool:
        if self.renewal_date is None:
            return False

        return (
            self.renewal_date
            <= date.today()
        )

    @property
    def is_high_risk(self) -> bool:
        level = (
            self.risk_level or ""
        ).lower()

        return (
            level in {
                "high",
                "critical",
            }
            or (
                self.risk_score is not None
                and self.risk_score >= 0.75
            )
        )

    @property
    def is_critical_risk(self) -> bool:
        return (
            (
                self.risk_level or ""
            ).lower()
            == "critical"
            or (
                self.risk_score is not None
                and self.risk_score >= 0.90
            )
        )

    @property
    def needs_review(self) -> bool:
        return bool(
            self.requires_review
            or self.requires_risk_review
            or self.is_high_risk
            or (
                self.review_status
                or ""
            ).lower()
            in {
                "pending",
                "changes_requested",
                "escalated",
            }
        )

    @property
    def display_name(self) -> str:
        if self.contract_number:
            return (
                f"{self.contract_number} — "
                f"{self.title}"
            )

        return self.title

    @property
    def lifecycle_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.lifecycle_stage,
            "effective_date": (
                self.effective_date.isoformat()
                if self.effective_date
                else None
            ),
            "expiry_date": (
                self.expiry_date.isoformat()
                if self.expiry_date
                else None
            ),
            "renewal_date": (
                self.renewal_date.isoformat()
                if self.renewal_date
                else None
            ),
            "notice_deadline": (
                self.notice_deadline.isoformat()
                if self.notice_deadline
                else None
            ),
            "auto_renewal": self.auto_renewal,
            "evergreen": self.evergreen,
            "days_until_expiry":
                self.days_until_expiry,
            "days_until_notice_deadline":
                self.days_until_notice_deadline,
        }

    def mark_reviewed(
        self,
        *,
        reviewer_id: uuid.UUID,
        approved: bool,
        notes: Optional[str] = None,
    ) -> None:
        self.reviewed_by = reviewer_id
        self.reviewed_at = utcnow()
        self.approved = approved
        self.review_notes = notes

        self.review_status = (
            "approved"
            if approved
            else "changes_requested"
        )

    def mark_rag_indexed(
        self,
        *,
        chunk_count: int,
        namespace: Optional[str] = None,
    ) -> None:
        self.rag_indexed = True
        self.rag_indexed_at = utcnow()
        self.chunk_count = max(
            0,
            chunk_count,
        )

        if namespace is not None:
            self.vector_namespace = namespace

    def soft_delete(
        self,
        *,
        deleted_by: Optional[uuid.UUID] = None,
    ) -> None:
        self.is_deleted = True
        self.is_active = False
        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

    def restore(self) -> None:
        self.is_deleted = False
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None

    def __repr__(self) -> str:
        return (
            "<Contract("
            f"id={self.id}, "
            f"title={self.title!r}, "
            f"status={self.status!r}, "
            f"expiry_date={self.expiry_date!r}, "
            f"risk={self.risk_level!r}"
            ")>"
        )