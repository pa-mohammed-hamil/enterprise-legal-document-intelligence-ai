# backend/app/db/models/contract.py

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base

if TYPE_CHECKING:
    from app.db.models.clause import Clause
    from app.db.models.document import Document
    from app.db.models.organization import Organization


class Contract(Base):
    """
    Core legal contract entity.

    A contract links an uploaded legal document with structured contract
    intelligence including:

    - contract type and lifecycle state
    - parties and counterparties
    - effective / expiry / renewal dates
    - financial terms
    - extracted clauses
    - obligations
    - legal risks
    - review state
    - AI extraction provenance
    """

    __tablename__ = "contracts"

    __table_args__ = (
        Index("ix_contracts_organization_id", "organization_id"),
        Index("ix_contracts_document_id", "document_id"),
        Index("ix_contracts_status", "status"),
        Index("ix_contracts_contract_type", "contract_type"),
        Index("ix_contracts_expiry_date", "expiry_date"),
        Index("ix_contracts_renewal_date", "renewal_date"),
        Index("ix_contracts_risk_level", "risk_level"),
        Index(
            "ix_contracts_org_status",
            "organization_id",
            "status",
        ),
        Index(
            "ix_contracts_org_expiry",
            "organization_id",
            "expiry_date",
        ),
    )

    # ================================================================
    # Identity
    # ================================================================

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    contract_number: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    external_reference: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ================================================================
    # Basic contract information
    # ================================================================

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    contract_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
    )

    subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft",
    )

    lifecycle_stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft",
    )

    # ================================================================
    # Parties
    # ================================================================

    primary_party: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    counterparty: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    parties: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    signatories: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ================================================================
    # Important dates
    # ================================================================

    execution_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    effective_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    start_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    expiry_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    termination_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    renewal_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    notice_deadline: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    # ================================================================
    # Renewal / termination
    # ================================================================

    auto_renewal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    renewal_term_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    notice_period_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    termination_notice_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    termination_for_convenience: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ================================================================
    # Financial information
    # ================================================================

    contract_value: Mapped[Optional[float]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    payment_terms: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    billing_frequency: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ================================================================
    # Legal / jurisdiction
    # ================================================================

    governing_law: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    jurisdiction: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    dispute_resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    confidentiality_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ================================================================
    # AI-generated summary
    # ================================================================

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    executive_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    key_terms: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    extracted_entities: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ================================================================
    # Risk intelligence
    # ================================================================

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unknown",
    )

    risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
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

    requires_legal_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ================================================================
    # AI extraction / model provenance
    # ================================================================

    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    extraction_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    classification_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    processing_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
    )

    processing_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ================================================================
    # Review
    # ================================================================

    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
    )

    reviewed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    approved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ================================================================
    # Metadata
    # ================================================================

    tags: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    custom_fields: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # ================================================================
    # Lifecycle
    # ================================================================

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ================================================================
    # Relationships
    # ================================================================

    organization: Mapped["Organization"] = relationship(
        "Organization",
        lazy="selectin",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        lazy="selectin",
    )

    clauses: Mapped[list["Clause"]] = relationship(
        "Clause",
        back_populates="contract",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Clause.position",
    )

    # ================================================================
    # Computed properties
    # ================================================================

    @property
    def is_expired(self) -> bool:
        return (
            self.expiry_date is not None
            and self.expiry_date < date.today()
        )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level.lower() in {
            "high",
            "critical",
        }

    @property
    def needs_review(self) -> bool:
        return (
            self.requires_legal_review
            or self.review_status == "pending"
            or self.is_high_risk
        )

    @property
    def days_until_expiry(self) -> Optional[int]:
        if self.expiry_date is None:
            return None

        return (self.expiry_date - date.today()).days

    @property
    def days_until_notice_deadline(self) -> Optional[int]:
        if self.notice_deadline is None:
            return None

        return (self.notice_deadline - date.today()).days

    @property
    def active_clause_count(self) -> int:
        return sum(
            1
            for clause in self.clauses
            if clause.is_active
        )

    # ================================================================
    # Domain operations
    # ================================================================

    def mark_processing_started(self) -> None:
        self.processing_status = "processing"
        self.processing_error = None

    def mark_processing_complete(self) -> None:
        self.processing_status = "completed"
        self.processing_error = None
        self.processed_at = datetime.utcnow()

    def mark_processing_failed(
        self,
        error: str,
    ) -> None:
        self.processing_status = "failed"
        self.processing_error = error

    def update_risk_summary(self) -> None:
        """
        Recalculate contract-level risk from clauses.
        """

        active_clauses = [
            clause
            for clause in self.clauses
            if clause.is_active
        ]

        self.high_risk_clause_count = sum(
            clause.risk_level.lower() == "high"
            for clause in active_clauses
        )

        self.critical_risk_count = sum(
            clause.risk_level.lower() == "critical"
            for clause in active_clauses
        )

        scores = [
            clause.risk_score
            for clause in active_clauses
            if clause.risk_score is not None
        ]

        if scores:
            self.risk_score = max(scores)

        if self.critical_risk_count:
            self.risk_level = "critical"
        elif self.high_risk_clause_count:
            self.risk_level = "high"
        elif scores and max(scores) >= 0.5:
            self.risk_level = "medium"
        elif scores:
            self.risk_level = "low"
        else:
            self.risk_level = "unknown"

        self.requires_legal_review = (
            self.risk_level in {"high", "critical"}
        )

    def mark_reviewed(
        self,
        reviewer_id: uuid.UUID,
        notes: Optional[str] = None,
    ) -> None:
        self.reviewed_by = reviewer_id
        self.reviewed_at = datetime.utcnow()
        self.review_status = "reviewed"

        if notes is not None:
            self.review_notes = notes

    def approve(
        self,
        approver_id: uuid.UUID,
    ) -> None:
        self.approved_by = approver_id
        self.approved_at = datetime.utcnow()
        self.review_status = "approved"

    def archive(self) -> None:
        self.is_archived = True
        self.is_active = False
        self.status = "archived"

    # ================================================================
    # Serialization
    # ================================================================

    def to_dict(
        self,
        include_clauses: bool = False,
    ) -> dict[str, Any]:

        data: dict[str, Any] = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "document_id": str(self.document_id),
            "contract_number": self.contract_number,
            "title": self.title,
            "contract_type": self.contract_type,
            "subtype": self.subtype,
            "status": self.status,
            "lifecycle_stage": self.lifecycle_stage,
            "primary_party": self.primary_party,
            "counterparty": self.counterparty,
            "parties": self.parties,
            "execution_date": (
                self.execution_date.isoformat()
                if self.execution_date
                else None
            ),
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
            "notice_period_days": self.notice_period_days,
            "contract_value": (
                float(self.contract_value)
                if self.contract_value is not None
                else None
            ),
            "currency": self.currency,
            "governing_law": self.governing_law,
            "jurisdiction": self.jurisdiction,
            "summary": self.summary,
            "executive_summary": self.executive_summary,
            "key_terms": self.key_terms,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "risk_summary": self.risk_summary,
            "high_risk_clause_count": (
                self.high_risk_clause_count
            ),
            "critical_risk_count": (
                self.critical_risk_count
            ),
            "requires_legal_review": (
                self.requires_legal_review
            ),
            "processing_status": self.processing_status,
            "review_status": self.review_status,
            "is_expired": self.is_expired,
            "days_until_expiry": self.days_until_expiry,
            "days_until_notice_deadline": (
                self.days_until_notice_deadline
            ),
            "active_clause_count": self.active_clause_count,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at
                else None
            ),
        }

        if include_clauses:
            data["clauses"] = [
                clause.to_dict()
                for clause in self.clauses
            ]

        return data

    def __repr__(self) -> str:
        return (
            f"<Contract("
            f"id={self.id}, "
            f"title={self.title!r}, "
            f"type={self.contract_type!r}, "
            f"status={self.status!r}, "
            f"risk={self.risk_level!r}"
            f")>"
        )