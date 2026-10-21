# backend/db/models/renewal.py

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


class Renewal(Base):
    """
    Contract renewal lifecycle record.

    Supports:
        - automatic/manual renewals
        - renewal windows
        - notice deadlines
        - termination/non-renewal notices
        - reminder scheduling
        - owner assignment
        - renewal decisions
        - financial impact
        - risk scoring
        - escalation
        - legal review
        - source-clause provenance
        - AI extraction provenance

    Important:
        AI-extracted renewal dates and notice periods are analytical
        signals. Source evidence and review status should be retained
        before a high-impact renewal/termination decision is made.
    """

    __tablename__ = "renewals"

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
    # RESOURCE REFERENCES
    # ========================================================

    contract_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

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
    # RENEWAL IDENTITY
    # ========================================================

    renewal_number: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    renewal_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="manual",
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    # ========================================================
    # RENEWAL TERMS
    # ========================================================

    auto_renewal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    evergreen: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    renewal_term_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    renewal_term_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    maximum_renewals: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    renewal_sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    # ========================================================
    # IMPORTANT DATES
    # ========================================================

    current_term_start: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    current_term_end: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    renewal_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    renewal_window_start: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    renewal_window_end: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        index=True,
    )

    # ========================================================
    # NOTICE
    # ========================================================

    notice_period_days: Mapped[Optional[int]] = mapped_column(
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

    notice_address: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    notice_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    notice_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    notice_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    notice_sent_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    notice_reference: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    # ========================================================
    # OWNER / RESPONSIBILITY
    # ========================================================

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

    responsible_party: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    stakeholders: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # DECISION
    # ========================================================

    decision: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    decision_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    decision_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    decided_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    decision_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # FINANCIAL IMPACT
    # ========================================================

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    current_contract_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    proposed_renewal_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    expected_renewal_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    renewal_cost_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    renewal_cost_change_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    financial_notes: Mapped[Optional[str]] = mapped_column(
        Text,
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

    risk_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    risk_categories: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    missed_deadline_risk: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    business_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    # ========================================================
    # SOURCE / CITATION
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
    # AI EXTRACTION / MODEL PROVENANCE
    # ========================================================

    ai_extracted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

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
        default=True,
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
    # WORKFLOW COMPLETION
    # ========================================================

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    completion_notes: Mapped[Optional[str]] = mapped_column(
        Text,
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

    supersedes_renewal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "renewals.id",
            ondelete="SET NULL",
        ),
        nullable=True,
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

    # Matches Contract.renewals from the existing contract model.
    contract = relationship(
        "Contract",
        back_populates="renewals",
        foreign_keys=[contract_id],
        lazy="selectin",
    )

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

    clause = relationship(
        "Clause",
        foreign_keys=[clause_id],
        lazy="selectin",
    )

    supersedes_renewal = relationship(
        "Renewal",
        remote_side=[id],
        foreign_keys=[supersedes_renewal_id],
        uselist=False,
    )

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_renewals_tenant_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_renewals_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_renewals_tenant_date",
            "tenant_id",
            "renewal_date",
            "status",
        ),
        Index(
            "ix_renewals_tenant_notice",
            "tenant_id",
            "notice_deadline",
            "status",
        ),
        Index(
            "ix_renewals_tenant_term_end",
            "tenant_id",
            "current_term_end",
        ),
        Index(
            "ix_renewals_tenant_auto",
            "tenant_id",
            "auto_renewal",
            "renewal_date",
        ),
        Index(
            "ix_renewals_tenant_owner",
            "tenant_id",
            "owner_user_id",
            "status",
        ),
        Index(
            "ix_renewals_tenant_risk",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_renewals_tenant_review",
            "tenant_id",
            "review_status",
        ),
        Index(
            "ix_renewals_tenant_escalation",
            "tenant_id",
            "escalated",
            "escalation_level",
        ),
        Index(
            "ix_renewals_reminder_due",
            "tenant_id",
            "reminders_enabled",
            "next_reminder_at",
        ),
        Index(
            "ix_renewals_tenant_current",
            "tenant_id",
            "is_current",
            "is_deleted",
        ),
        Index(
            "ix_renewals_document",
            "tenant_id",
            "document_id",
            "document_version_id",
        ),
        Index(
            "ix_renewals_model_run",
            "tenant_id",
            "model_run_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def days_until_renewal(self) -> Optional[int]:
        if self.renewal_date is None:
            return None

        return (
            self.renewal_date
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
        return bool(
            self.notice_deadline is not None
            and date.today()
            > self.notice_deadline
        )

    @property
    def renewal_due(self) -> bool:
        if self.renewal_date is None:
            return False

        return (
            self.renewal_date
            <= date.today()
            and not self.is_completed
        )

    @property
    def renewal_window_open(self) -> bool:
        today = date.today()

        if self.renewal_window_start is not None:
            if today < self.renewal_window_start:
                return False

        if self.renewal_window_end is not None:
            if today > self.renewal_window_end:
                return False

        return (
            self.renewal_window_start is not None
            or self.renewal_window_end is not None
        )

    @property
    def is_completed(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "renewed",
            "terminated",
            "non_renewed",
            "cancelled",
            "completed",
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
    def requires_attention(self) -> bool:
        renewal_days = self.days_until_renewal
        notice_days = self.days_until_notice_deadline

        approaching_renewal = bool(
            renewal_days is not None
            and 0 <= renewal_days <= 90
        )

        approaching_notice = bool(
            notice_days is not None
            and 0 <= notice_days <= 30
        )

        return bool(
            self.notice_deadline_passed
            or approaching_renewal
            or approaching_notice
            or self.is_high_risk
            or self.escalated
            or self.requires_review
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
            f"pages {self.page_start}-"
            f"{self.page_end}"
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
            "contract_id": str(
                self.contract_id
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
    # DECISION WORKFLOW
    # ========================================================

    def decide(
        self,
        *,
        decision: str,
        decided_by: uuid.UUID,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        allowed = {
            "renew",
            "non_renew",
            "terminate",
            "renegotiate",
            "extend",
            "defer",
        }

        normalized = (
            decision.strip().lower()
        )

        if normalized not in allowed:
            raise ValueError(
                f"Invalid renewal decision: "
                f"{decision}"
            )

        self.decision = normalized
        self.decision_reason = reason
        self.decision_notes = notes

        self.decided_by = decided_by
        self.decided_at = utcnow()

        self.status = "decided"

    # ========================================================
    # NOTICE
    # ========================================================

    def mark_notice_sent(
        self,
        *,
        sent_by: Optional[uuid.UUID] = None,
        reference: Optional[str] = None,
    ) -> None:
        self.notice_sent = True
        self.notice_sent_at = utcnow()
        self.notice_sent_by = sent_by
        self.notice_reference = reference

    # ========================================================
    # RENEWAL COMPLETION
    # ========================================================

    def mark_renewed(
        self,
        *,
        completed_by: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.status = "renewed"
        self.decision = (
            self.decision or "renew"
        )

        self.completed_at = utcnow()
        self.completed_by = completed_by
        self.completion_notes = notes

        self.next_reminder_at = None

    def mark_non_renewed(
        self,
        *,
        completed_by: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.status = "non_renewed"
        self.decision = (
            self.decision or "non_renew"
        )

        self.completed_at = utcnow()
        self.completed_by = completed_by
        self.completion_notes = notes

        self.next_reminder_at = None

    def mark_terminated(
        self,
        *,
        completed_by: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.status = "terminated"
        self.decision = (
            self.decision or "terminate"
        )

        self.completed_at = utcnow()
        self.completed_by = completed_by
        self.completion_notes = notes

        self.next_reminder_at = None

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
    # RISK
    # ========================================================

    def update_risk(
        self,
        *,
        level: str,
        score: Optional[float] = None,
        reason: Optional[str] = None,
        categories: Optional[list[Any]] = None,
    ) -> None:
        self.risk_level = level

        if score is not None:
            self.risk_score = min(
                1.0,
                max(0.0, float(score)),
            )

        if reason is not None:
            self.risk_reason = reason

        if categories is not None:
            self.risk_categories = list(
                categories
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
            "<Renewal("
            f"id={self.id}, "
            f"contract_id={self.contract_id}, "
            f"renewal_date={self.renewal_date!r}, "
            f"status={self.status!r}, "
            f"auto_renewal={self.auto_renewal}"
            ")>"
        )