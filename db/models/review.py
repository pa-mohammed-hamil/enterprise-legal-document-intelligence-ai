

# backend/db/models/review.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Review(Base):
    """
    Human/legal review workflow.

    Typical lifecycle:

        pending
            -> assigned
            -> in_review
            -> completed

    Alternative outcomes:

        changes_requested
        rejected
        cancelled

    AI recommendations and human decisions are deliberately stored
    separately. AI-generated legal analysis must not silently become
    the authoritative human review decision.
    """

    __tablename__ = "reviews"

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
    # RESOURCE LINKS
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

    risk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "risks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    obligation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "obligations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    renewal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "renewals.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # REVIEW IDENTITY
    # ========================================================

    review_number: Mapped[Optional[str]] = mapped_column(
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

    review_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="legal",
        index=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # ========================================================
    # WORKFLOW STATUS
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

    required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ========================================================
    # REQUESTER
    # ========================================================

    requested_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    request_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # ASSIGNMENT
    # ========================================================

    assigned_to: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    assigned_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    assigned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    reviewer_role: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    reviewer_team: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # SCHEDULING
    # ========================================================

    due_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # HUMAN DECISION
    # ========================================================

    decision: Mapped[Optional[str]] = mapped_column(
        String(100),
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

    decision_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    decision_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    approved: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        index=True,
    )

    # ========================================================
    # REVIEW FINDINGS
    # ========================================================

    findings: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    recommendations: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    required_changes: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    reviewer_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # AI RECOMMENDATION
    # ========================================================

    ai_reviewed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    ai_recommendation: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    ai_reasoning_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    ai_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    ai_findings: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    ai_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    ai_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # HUMAN / AI COMPARISON
    # ========================================================

    ai_human_agreement: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    ai_override: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    ai_override_reason: Mapped[Optional[str]] = mapped_column(
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

    risk_categories: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    risk_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # SOURCE EVIDENCE
    # ========================================================

    source_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    source_chunk_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    citation_references: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    page_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    page_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
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

    escalated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    escalation_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # CHANGE REQUEST
    # ========================================================

    changes_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    changes_requested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    changes_requested_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    change_request_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    changes_resolved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    changes_resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # DECISION HISTORY
    # ========================================================

    # For maximum audit integrity, move this to an append-only
    # ReviewDecision table later. This JSON history is useful as
    # an immediate self-contained implementation.

    decision_history: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # REQUEST / TRACE
    # ========================================================

    request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # VERSIONING
    # ========================================================

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    # Optimistic concurrency value.
    lock_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
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

    # These match the previously defined Document and Contract
    # back_populates relationships.

    document = relationship(
        "Document",
        back_populates="reviews",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    contract = relationship(
        "Contract",
        back_populates="reviews",
        foreign_keys=[contract_id],
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

    risk = relationship(
        "Risk",
        foreign_keys=[risk_id],
        lazy="selectin",
    )

    obligation = relationship(
        "Obligation",
        foreign_keys=[obligation_id],
        lazy="selectin",
    )

    renewal = relationship(
        "Renewal",
        foreign_keys=[renewal_id],
        lazy="selectin",
    )

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_reviews_tenant_status",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_reviews_tenant_assignee",
            "tenant_id",
            "assigned_to",
            "status",
        ),
        Index(
            "ix_reviews_tenant_priority",
            "tenant_id",
            "priority",
            "status",
        ),
        Index(
            "ix_reviews_tenant_due",
            "tenant_id",
            "due_at",
            "status",
        ),
        Index(
            "ix_reviews_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_reviews_tenant_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_reviews_tenant_clause",
            "tenant_id",
            "clause_id",
        ),
        Index(
            "ix_reviews_tenant_risk",
            "tenant_id",
            "risk_id",
        ),
        Index(
            "ix_reviews_tenant_obligation",
            "tenant_id",
            "obligation_id",
        ),
        Index(
            "ix_reviews_tenant_renewal",
            "tenant_id",
            "renewal_id",
        ),
        Index(
            "ix_reviews_tenant_risk_level",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_reviews_tenant_escalated",
            "tenant_id",
            "escalated",
            "escalation_level",
        ),
        Index(
            "ix_reviews_tenant_decision",
            "tenant_id",
            "decision",
            "decision_at",
        ),
        Index(
            "ix_reviews_model_run",
            "tenant_id",
            "model_run_id",
        ),
        Index(
            "ix_reviews_trace",
            "tenant_id",
            "trace_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def is_pending(self) -> bool:
        return (
            self.status or ""
        ).lower() == "pending"

    @property
    def is_assigned(self) -> bool:
        return self.assigned_to is not None

    @property
    def is_in_review(self) -> bool:
        return (
            self.status or ""
        ).lower() == "in_review"

    @property
    def is_completed(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "completed",
            "approved",
            "rejected",
            "cancelled",
        }

    @property
    def is_overdue(self) -> bool:
        if self.is_completed:
            return False

        return bool(
            self.due_at is not None
            and self.due_at < utcnow()
        )

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
        return bool(
            self.is_overdue
            or self.is_high_risk
            or self.escalated
            or self.changes_requested
        )

    @property
    def days_until_due(self) -> Optional[int]:
        if self.due_at is None:
            return None

        delta = self.due_at - utcnow()

        return int(
            delta.total_seconds() // 86400
        )

    @property
    def resource_reference(self) -> dict[str, Optional[str]]:
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
            "risk_id": (
                str(self.risk_id)
                if self.risk_id
                else None
            ),
            "obligation_id": (
                str(self.obligation_id)
                if self.obligation_id
                else None
            ),
            "renewal_id": (
                str(self.renewal_id)
                if self.renewal_id
                else None
            ),
        }

    # ========================================================
    # INTERNAL HISTORY
    # ========================================================

    def _append_history(
        self,
        *,
        event: str,
        actor_id: Optional[uuid.UUID] = None,
        decision: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        history = list(
            self.decision_history or []
        )

        history.append(
            {
                "event": event,
                "actor_id": (
                    str(actor_id)
                    if actor_id
                    else None
                ),
                "decision": decision,
                "notes": notes,
                "metadata": metadata or {},
                "timestamp": utcnow().isoformat(),
            }
        )

        # Reassignment ensures JSONB change tracking.
        self.decision_history = history
        self.lock_version += 1

    # ========================================================
    # ASSIGNMENT
    # ========================================================

    def assign(
        self,
        *,
        reviewer_id: uuid.UUID,
        assigned_by: Optional[uuid.UUID] = None,
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Completed review cannot be reassigned."
            )

        self.assigned_to = reviewer_id
        self.assigned_by = assigned_by
        self.assigned_at = utcnow()
        self.status = "assigned"

        self._append_history(
            event="assigned",
            actor_id=assigned_by,
            metadata={
                "reviewer_id": str(
                    reviewer_id
                )
            },
        )

    # ========================================================
    # REVIEW LIFECYCLE
    # ========================================================

    def start_review(
        self,
        *,
        reviewer_id: uuid.UUID,
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Completed review cannot be started."
            )

        if (
            self.assigned_to is not None
            and self.assigned_to
            != reviewer_id
        ):
            raise ValueError(
                "Review is assigned to another reviewer."
            )

        if self.assigned_to is None:
            self.assigned_to = reviewer_id
            self.assigned_at = utcnow()

        self.status = "in_review"

        if self.started_at is None:
            self.started_at = utcnow()

        self._append_history(
            event="review_started",
            actor_id=reviewer_id,
        )

    # ========================================================
    # DECISION
    # ========================================================

    def approve(
        self,
        *,
        reviewer_id: uuid.UUID,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._complete_decision(
            decision="approved",
            approved=True,
            reviewer_id=reviewer_id,
            reason=reason,
            notes=notes,
        )

    def reject(
        self,
        *,
        reviewer_id: uuid.UUID,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._complete_decision(
            decision="rejected",
            approved=False,
            reviewer_id=reviewer_id,
            reason=reason,
            notes=notes,
        )

    def _complete_decision(
        self,
        *,
        decision: str,
        approved: bool,
        reviewer_id: uuid.UUID,
        reason: Optional[str],
        notes: Optional[str],
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Review already has a final decision."
            )

        if (
            self.assigned_to is not None
            and self.assigned_to
            != reviewer_id
        ):
            raise ValueError(
                "Reviewer is not assigned to this review."
            )

        now = utcnow()

        self.decision = decision
        self.decision_reason = reason
        self.decision_notes = notes

        self.decision_by = reviewer_id
        self.decision_at = now

        self.approved = approved

        self.status = (
            "approved"
            if approved
            else "rejected"
        )

        self.completed_at = now

        if self.ai_recommendation:
            ai_decision = (
                self.ai_recommendation
                .strip()
                .lower()
            )

            aliases = {
                "approve": "approved",
                "approved": "approved",
                "reject": "rejected",
                "rejected": "rejected",
            }

            normalized_ai = aliases.get(
                ai_decision,
                ai_decision,
            )

            self.ai_human_agreement = (
                normalized_ai == decision
            )

            self.ai_override = (
                not self.ai_human_agreement
            )

        self._append_history(
            event="decision",
            actor_id=reviewer_id,
            decision=decision,
            notes=notes,
            metadata={
                "approved": approved,
                "reason": reason,
            },
        )

    # ========================================================
    # CHANGE REQUEST
    # ========================================================

    def request_changes(
        self,
        *,
        reviewer_id: uuid.UUID,
        notes: str,
        changes: Optional[list[Any]] = None,
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Finalized review cannot request changes."
            )

        self.status = "changes_requested"

        self.changes_requested = True
        self.changes_requested_at = utcnow()
        self.changes_requested_by = reviewer_id
        self.change_request_notes = notes
        self.changes_resolved = False

        if changes is not None:
            self.required_changes = list(
                changes
            )

        self._append_history(
            event="changes_requested",
            actor_id=reviewer_id,
            notes=notes,
        )

    def resolve_changes(
        self,
        *,
        actor_id: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
    ) -> None:
        if not self.changes_requested:
            raise ValueError(
                "No outstanding change request."
            )

        self.changes_resolved = True
        self.changes_resolved_at = utcnow()

        self.status = "in_review"

        self._append_history(
            event="changes_resolved",
            actor_id=actor_id,
            notes=notes,
        )

    # ========================================================
    # AI REVIEW
    # ========================================================

    def set_ai_recommendation(
        self,
        *,
        recommendation: str,
        confidence: Optional[float] = None,
        reasoning_summary: Optional[str] = None,
        findings: Optional[list[Any]] = None,
        model: Optional[str] = None,
        model_version: Optional[str] = None,
        model_run_id: Optional[str] = None,
    ) -> None:
        self.ai_reviewed = True
        self.ai_recommendation = recommendation
        self.ai_reasoning_summary = reasoning_summary

        if confidence is not None:
            self.ai_confidence = min(
                1.0,
                max(
                    0.0,
                    float(confidence),
                ),
            )

        if findings is not None:
            self.ai_findings = list(
                findings
            )

        self.ai_model = model
        self.ai_model_version = (
            model_version
        )
        self.model_run_id = model_run_id

    # ========================================================
    # FINDINGS
    # ========================================================

    def add_finding(
        self,
        finding: dict[str, Any],
    ) -> None:
        findings = list(
            self.findings or []
        )

        findings.append(finding)
        self.findings = findings

    def add_recommendation(
        self,
        recommendation: dict[str, Any],
    ) -> None:
        recommendations = list(
            self.recommendations or []
        )

        recommendations.append(
            recommendation
        )

        self.recommendations = (
            recommendations
        )

    # ========================================================
    # RISK
    # ========================================================

    def update_risk(
        self,
        *,
        level: str,
        score: Optional[float] = None,
        summary: Optional[str] = None,
        categories: Optional[list[Any]] = None,
    ) -> None:
        self.risk_level = level

        if score is not None:
            self.risk_score = min(
                1.0,
                max(
                    0.0,
                    float(score),
                ),
            )

        if summary is not None:
            self.risk_summary = summary

        if categories is not None:
            self.risk_categories = list(
                categories
            )

    # ========================================================
    # ESCALATION
    # ========================================================

    def escalate(
        self,
        *,
        level: str,
        reason: str,
        actor_id: Optional[uuid.UUID] = None,
    ) -> None:
        self.escalated = True
        self.escalation_level = level
        self.escalation_reason = reason
        self.escalated_at = utcnow()
        self.escalated_by = actor_id
        self.escalation_count += 1

        self._append_history(
            event="escalated",
            actor_id=actor_id,
            notes=reason,
            metadata={
                "level": level,
            },
        )

    def clear_escalation(
        self,
        *,
        actor_id: Optional[uuid.UUID] = None,
    ) -> None:
        self.escalated = False
        self.escalation_level = None
        self.escalation_reason = None

        self._append_history(
            event="escalation_cleared",
            actor_id=actor_id,
        )

    # ========================================================
    # CANCELLATION
    # ========================================================

    def cancel(
        self,
        *,
        actor_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None,
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Completed review cannot be cancelled."
            )

        self.status = "cancelled"
        self.completed_at = utcnow()

        self._append_history(
            event="cancelled",
            actor_id=actor_id,
            notes=reason,
        )

    # ========================================================
    # SOFT DELETE
    # ========================================================

    def soft_delete(
        self,
        *,
        deleted_by: Optional[uuid.UUID] = None,
    ) -> None:
        if self.is_completed:
            raise ValueError(
                "Completed legal review should not normally "
                "be deleted. Archive it instead."
            )

        self.is_deleted = True
        self.is_active = False
        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

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
            "<Review("
            f"id={self.id}, "
            f"type={self.review_type!r}, "
            f"status={self.status!r}, "
            f"priority={self.priority!r}, "
            f"decision={self.decision!r}"
            ")>"
        )