# backend/db/models/obligation.py

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
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Obligation(Base):
    """
    Contractual obligation extracted from or attached to a legal
    document, contract or clause.

    Examples:
        - payment obligations
        - reporting obligations
        - notice obligations
        - confidentiality duties
        - insurance requirements
        - audit requirements
        - data protection obligations
        - service-level commitments
        - renewal/termination actions
        - compliance requirements

    Important:
        AI-extracted obligations are analytical records and should
        preserve source evidence and confidence. High-impact legal
        obligations should pass through human review before being
        treated as authoritative.
    """

    __tablename__ = "obligations"

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
    # SOURCE REFERENCES
    # ========================================================

    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="SET NULL",
        ),
        nullable=True,
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

    contract_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # OBLIGATION IDENTITY
    # ========================================================

    obligation_number: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    obligation_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
        index=True,
    )

    obligation_subtype: Mapped[Optional[str]] = mapped_column(
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
    # PARTIES / RESPONSIBILITY
    # ========================================================

    obligor: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    obligee: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    responsible_party: Mapped[Optional[str]] = mapped_column(
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
    )

    stakeholders: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    priority: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="normal",
        index=True,
    )

    mandatory: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    conditional: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    condition_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # DATES
    # ========================================================

    start_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    due_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    end_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    grace_period_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    notice_period_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    notice_deadline: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    # ========================================================
    # RECURRENCE
    # ========================================================

    recurring: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    recurrence_frequency: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    recurrence_interval: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    recurrence_rule: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    next_due_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # FINANCIAL OBLIGATION
    # ========================================================

    monetary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(
            precision=20,
            scale=4,
        ),
        nullable=True,
    )

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    payment_frequency: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ========================================================
    # SOURCE EVIDENCE / CITATION
    # ========================================================

    source_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    source_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    parent_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    page_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    page_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    char_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    char_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    section_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    section_title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    # ========================================================
    # AI EXTRACTION PROVENANCE
    # ========================================================

    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
    )

    extraction_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    extraction_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
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

    risk_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    risk_categories: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    breach_risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # COMPLETION / COMPLIANCE
    # ========================================================

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    completed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    completion_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    completion_evidence: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    evidence_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    evidence_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    evidence_verified_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    evidence_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # BREACH / NON-COMPLIANCE
    # ========================================================

    breached: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    breach_detected_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    breach_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    breach_resolved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    breach_resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # REMINDERS
    # ========================================================

    reminders_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    reminder_days: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    next_reminder_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    last_reminder_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    reminder_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # ESCALATION
    # ========================================================

    escalated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    escalation_level: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    escalation_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    escalated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    escalation_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # REVIEW
    # ========================================================

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    reviewed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
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

    approved: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
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

    supersedes_obligation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "obligations.id",
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
    # METADATA
    # ========================================================

    tags: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # SOFT DELETE
    # ========================================================

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

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

    # One-way relationships avoid requiring back_populates
    # on every source model.

    document = relationship(
        "Document",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    document_version = relationship(
        "DocumentVersion",
        foreign_keys=[document_version_id],
        lazy="selectin",
    )

    contract = relationship(
        "Contract",
        foreign_keys=[contract_id],
        lazy="selectin",
    )

    clause = relationship(
        "Clause",
        foreign_keys=[clause_id],
        lazy="selectin",
    )

    supersedes_obligation = relationship(
        "Obligation",
        remote_side=[id],
        foreign_keys=[supersedes_obligation_id],
        uselist=False,
    )

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_obligations_tenant_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_obligations_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_obligations_tenant_clause",
            "tenant_id",
            "clause_id",
        ),
        Index(
            "ix_obligations_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_obligations_tenant_due",
            "tenant_id",
            "due_date",
            "status",
        ),
        Index(
            "ix_obligations_tenant_next_due",
            "tenant_id",
            "next_due_date",
            "status",
        ),
        Index(
            "ix_obligations_tenant_notice",
            "tenant_id",
            "notice_deadline",
        ),
        Index(
            "ix_obligations_tenant_owner",
            "tenant_id",
            "owner_user_id",
            "status",
        ),
        Index(
            "ix_obligations_tenant_type",
            "tenant_id",
            "obligation_type",
        ),
        Index(
            "ix_obligations_tenant_priority",
            "tenant_id",
            "priority",
            "status",
        ),
        Index(
            "ix_obligations_tenant_risk",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_obligations_tenant_review",
            "tenant_id",
            "review_status",
        ),
        Index(
            "ix_obligations_tenant_escalation",
            "tenant_id",
            "escalated",
            "escalation_level",
        ),
        Index(
            "ix_obligations_reminders",
            "tenant_id",
            "reminders_enabled",
            "next_reminder_at",
        ),
        Index(
            "ix_obligations_current",
            "tenant_id",
            "is_current",
            "is_deleted",
        ),
        Index(
            "ix_obligations_model_run",
            "tenant_id",
            "model_run_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def effective_due_date(self) -> Optional[date]:
        if self.recurring and self.next_due_date:
            return self.next_due_date
        return self.due_date

    @property
    def days_until_due(self) -> Optional[int]:
        due = self.effective_due_date

        if due is None:
            return None

        return (due - date.today()).days

    @property
    def is_due_today(self) -> bool:
        due = self.effective_due_date
        return bool(
            due is not None
            and due == date.today()
        )

    @property
    def is_overdue(self) -> bool:
        if self.is_completed:
            return False

        due = self.effective_due_date

        if due is None:
            return False

        grace = max(
            0,
            self.grace_period_days,
        )

        return (
            date.today().toordinal()
            > due.toordinal() + grace
        )

    @property
    def is_completed(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "completed",
            "fulfilled",
            "waived",
            "cancelled",
        }

    @property
    def is_high_risk(self) -> bool:
        return bool(
            (
                self.risk_level
                or ""
            ).lower()
            in {
                "high",
                "critical",
            }
            or (
                self.risk_score is not None
                and self.risk_score >= 0.75
            )
        )

    @property
    def is_critical(self) -> bool:
        return bool(
            (
                self.risk_level
                or ""
            ).lower() == "critical"
            or (
                self.risk_score is not None
                and self.risk_score >= 0.90
            )
        )

    @property
    def needs_attention(self) -> bool:
        days = self.days_until_due

        approaching = (
            days is not None
            and 0 <= days <= 30
        )

        return bool(
            self.is_overdue
            or self.breached
            or self.is_high_risk
            or self.escalated
            or self.requires_review
            or approaching
        )

    @property
    def page_reference(self) -> Optional[str]:
        if self.page_start is None:
            return None

        if (
            self.page_end is None
            or self.page_end == self.page_start
        ):
            return f"page {self.page_start}"

        return (
            f"pages {self.page_start}"
            f"-{self.page_end}"
        )

    @property
    def source_reference(self) -> dict[str, Any]:
        return {
            "document_id": (
                str(self.document_id)
                if self.document_id
                else None
            ),
            "document_version_id": (
                str(self.document_version_id)
                if self.document_version_id
                else None
            ),
            "contract_id": (
                str(self.contract_id)
                if self.contract_id
                else None
            ),
            "clause_id": (
                str(self.clause_id)
                if self.clause_id
                else None
            ),
            "chunk_id":
                self.source_chunk_id,
            "page_start":
                self.page_start,
            "page_end":
                self.page_end,
            "section_number":
                self.section_number,
            "section_title":
                self.section_title,
        }

    # ========================================================
    # WORKFLOW
    # ========================================================

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted obligation cannot be activated."
            )

        self.status = "active"
        self.is_active = True

    def mark_due(self) -> None:
        if not self.is_completed:
            self.status = "due"

    def mark_overdue(self) -> None:
        if not self.is_completed:
            self.status = "overdue"

    def complete(
        self,
        *,
        completed_by: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
        evidence: Optional[list[Any]] = None,
    ) -> None:
        if (
            self.evidence_required
            and not evidence
            and not self.completion_evidence
        ):
            raise ValueError(
                "Completion evidence is required."
            )

        self.status = "completed"
        self.completed_at = utcnow()
        self.completed_by = completed_by
        self.completion_notes = notes

        if evidence is not None:
            self.completion_evidence = evidence

        self.next_reminder_at = None

    def waive(
        self,
        *,
        notes: Optional[str] = None,
    ) -> None:
        self.status = "waived"
        self.completed_at = utcnow()
        self.completion_notes = notes
        self.next_reminder_at = None

    def cancel(
        self,
        *,
        notes: Optional[str] = None,
    ) -> None:
        self.status = "cancelled"
        self.completed_at = utcnow()
        self.completion_notes = notes
        self.next_reminder_at = None

    # ========================================================
    # EVIDENCE
    # ========================================================

    def add_evidence(
        self,
        evidence: dict[str, Any],
    ) -> None:
        current = list(
            self.completion_evidence or []
        )

        current.append(evidence)

        # Reassignment is safer for JSONB mutation tracking.
        self.completion_evidence = current

    def verify_evidence(
        self,
        *,
        verified_by: uuid.UUID,
    ) -> None:
        if not self.completion_evidence:
            raise ValueError(
                "No completion evidence exists."
            )

        self.evidence_verified = True
        self.evidence_verified_by = verified_by
        self.evidence_verified_at = utcnow()

    # ========================================================
    # BREACH
    # ========================================================

    def mark_breached(
        self,
        *,
        reason: str,
    ) -> None:
        self.breached = True
        self.breach_detected_at = utcnow()
        self.breach_reason = reason
        self.breach_resolved = False

        if not self.is_completed:
            self.status = "breached"

    def resolve_breach(self) -> None:
        if not self.breached:
            raise ValueError(
                "Obligation is not marked as breached."
            )

        self.breach_resolved = True
        self.breach_resolved_at = utcnow()

    # ========================================================
    # REMINDERS
    # ========================================================

    def record_reminder(
        self,
        *,
        next_reminder_at: Optional[
            datetime
        ] = None,
    ) -> None:
        self.last_reminder_at = utcnow()

        self.reminder_count += 1
        self.next_reminder_at = (
            next_reminder_at
        )

    # ========================================================
    # ESCALATION
    # ========================================================

    def escalate(
        self,
        *,
        level: str,
        reason: Optional[str] = None,
    ) -> None:
        self.escalated = True
        self.escalation_level = level
        self.escalation_reason = reason
        self.escalated_at = utcnow()

        self.escalation_count += 1

    def clear_escalation(self) -> None:
        self.escalated = False
        self.escalation_level = None
        self.escalation_reason = None

    # ========================================================
    # REVIEW
    # ========================================================

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

    # ========================================================
    # VERSIONING
    # ========================================================

    def mark_superseded(self) -> None:
        self.is_current = False
        self.is_active = False

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
        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

        self.next_reminder_at = None

    def restore(self) -> None:
        self.is_deleted = False
        self.is_active = True

        self.deleted_at = None
        self.deleted_by = None

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<Obligation("
            f"id={self.id}, "
            f"title={self.title!r}, "
            f"type={self.obligation_type!r}, "
            f"status={self.status!r}, "
            f"due_date={self.effective_due_date!r}"
            ")>"
        )