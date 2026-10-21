# backend/db/models/risk.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
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


class Risk(Base):
    """
    Enterprise legal / contractual risk record.

    A Risk may be associated with:
        - Document
        - DocumentVersion
        - Contract
        - Clause
        - Obligation
        - Renewal

    Typical lifecycle:

        detected
            -> open
            -> acknowledged
            -> under_review
            -> mitigated
            -> resolved

    Alternative terminal state:
        dismissed

    AI-generated risk assessments remain analytical signals.
    Evidence, model provenance, confidence and human review are
    retained separately.
    """

    __tablename__ = "risks"

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
    # SOURCE RESOURCES
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
    # RISK IDENTITY
    # ========================================================

    risk_number: Mapped[Optional[str]] = mapped_column(
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

    risk_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="legal",
        index=True,
    )

    risk_subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    category: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
        index=True,
    )

    subcategory: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    # ========================================================
    # RISK SCORING
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

    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    likelihood: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    impact: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    severity: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    # ========================================================
    # BUSINESS / LEGAL IMPACT
    # ========================================================

    legal_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    financial_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    operational_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    compliance_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    reputational_impact: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    impact_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # FINANCIAL EXPOSURE
    # ========================================================

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    estimated_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    minimum_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    maximum_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    expected_loss: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 4),
        nullable=True,
    )

    # ========================================================
    # STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="detected",
        index=True,
    )

    priority: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="normal",
        index=True,
    )

    # ========================================================
    # OWNER
    # ========================================================

    owner_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
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
    # EVIDENCE / SOURCE
    # ========================================================

    evidence: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    evidence_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

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

    citation_references: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # AI / MODEL PROVENANCE
    # ========================================================

    ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    detection_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    model_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    model_reasoning_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    risk_signals: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # ACKNOWLEDGEMENT
    # ========================================================

    acknowledged: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    acknowledged_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    acknowledgement_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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

    review_approved: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    # ========================================================
    # MITIGATION
    # ========================================================

    mitigation_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    mitigation_plan: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    mitigation_actions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    mitigation_owner_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    mitigation_due_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    mitigation_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    mitigated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    mitigated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    mitigation_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # RESIDUAL RISK
    # ========================================================

    residual_risk_level: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    residual_risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    residual_risk_reason: Mapped[Optional[str]] = mapped_column(
        Text,
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
    # RESOLUTION
    # ========================================================

    resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    resolution_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    resolved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # DISMISSAL
    # ========================================================

    dismissed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    dismissed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    dismissal_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # DEADLINES
    # ========================================================

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    due_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # REQUEST / OBSERVABILITY
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

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    supersedes_risk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "risks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    # ========================================================
    # TAGS / METADATA
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

    # Matches Document.risks
    document = relationship(
        "Document",
        back_populates="risks",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    # Matches Contract.risks
    contract = relationship(
        "Contract",
        back_populates="risks",
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

    supersedes_risk = relationship(
        "Risk",
        remote_side=[id],
        foreign_keys=[supersedes_risk_id],
        uselist=False,
    )

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_risks_tenant_status",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_risks_tenant_level",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_risks_tenant_category",
            "tenant_id",
            "category",
            "status",
        ),
        Index(
            "ix_risks_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_risks_tenant_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_risks_tenant_clause",
            "tenant_id",
            "clause_id",
        ),
        Index(
            "ix_risks_tenant_obligation",
            "tenant_id",
            "obligation_id",
        ),
        Index(
            "ix_risks_tenant_renewal",
            "tenant_id",
            "renewal_id",
        ),
        Index(
            "ix_risks_tenant_owner",
            "tenant_id",
            "owner_user_id",
            "status",
        ),
        Index(
            "ix_risks_tenant_review",
            "tenant_id",
            "requires_review",
            "review_status",
        ),
        Index(
            "ix_risks_tenant_mitigation",
            "tenant_id",
            "mitigation_required",
            "mitigation_due_at",
        ),
        Index(
            "ix_risks_tenant_escalation",
            "tenant_id",
            "escalated",
            "escalation_level",
        ),
        Index(
            "ix_risks_tenant_acknowledged",
            "tenant_id",
            "acknowledged",
            "status",
        ),
        Index(
            "ix_risks_tenant_current",
            "tenant_id",
            "is_current",
            "is_deleted",
        ),
        Index(
            "ix_risks_model_run",
            "tenant_id",
            "model_run_id",
        ),
        Index(
            "ix_risks_trace",
            "tenant_id",
            "trace_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def is_high_risk(self) -> bool:
        return bool(
            (self.risk_level or "").lower()
            in {"high", "critical"}
            or (
                self.risk_score is not None
                and self.risk_score >= 0.75
            )
        )

    @property
    def is_critical(self) -> bool:
        return bool(
            (self.risk_level or "").lower()
            == "critical"
            or (
                self.risk_score is not None
                and self.risk_score >= 0.90
            )
        )

    @property
    def is_closed(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "resolved",
            "dismissed",
            "cancelled",
        }

    @property
    def is_overdue(self) -> bool:
        return bool(
            not self.is_closed
            and self.due_at is not None
            and self.due_at < utcnow()
        )

    @property
    def mitigation_overdue(self) -> bool:
        return bool(
            self.mitigation_required
            and self.mitigated_at is None
            and self.mitigation_due_at is not None
            and self.mitigation_due_at < utcnow()
        )

    @property
    def requires_attention(self) -> bool:
        return bool(
            not self.is_closed
            and (
                self.is_high_risk
                or self.is_overdue
                or self.mitigation_overdue
                or self.escalated
                or (
                    self.requires_review
                    and self.review_status
                    not in {
                        "approved",
                        "completed",
                    }
                )
            )
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
            "chunk_id": self.source_chunk_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_number": self.section_number,
            "section_title": self.section_title,
        }

    # ========================================================
    # SCORE
    # ========================================================

    def set_score(
        self,
        *,
        score: float,
        level: Optional[str] = None,
        confidence: Optional[float] = None,
        likelihood: Optional[float] = None,
        impact: Optional[float] = None,
    ) -> None:
        self.risk_score = min(
            1.0,
            max(0.0, float(score)),
        )

        if level is not None:
            self.risk_level = level

        if confidence is not None:
            self.confidence = min(
                1.0,
                max(0.0, float(confidence)),
            )

        if likelihood is not None:
            self.likelihood = min(
                1.0,
                max(0.0, float(likelihood)),
            )

        if impact is not None:
            self.impact = min(
                1.0,
                max(0.0, float(impact)),
            )

    # ========================================================
    # OWNER
    # ========================================================

    def assign(
        self,
        *,
        owner_user_id: uuid.UUID,
        assigned_by: Optional[uuid.UUID] = None,
    ) -> None:
        if self.is_closed:
            raise ValueError(
                "Closed risk cannot be assigned."
            )

        self.owner_user_id = owner_user_id
        self.assigned_by = assigned_by
        self.assigned_at = utcnow()

        if self.status == "detected":
            self.status = "open"

    # ========================================================
    # ACKNOWLEDGEMENT
    # ========================================================

    def acknowledge(
        self,
        *,
        user_id: uuid.UUID,
        notes: Optional[str] = None,
    ) -> None:
        if self.is_closed:
            raise ValueError(
                "Closed risk cannot be acknowledged."
            )

        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = utcnow()
        self.acknowledgement_notes = notes

        self.status = "acknowledged"

    # ========================================================
    # REVIEW
    # ========================================================

    def start_review(self) -> None:
        if self.is_closed:
            raise ValueError(
                "Closed risk cannot enter review."
            )

        self.requires_review = True
        self.review_status = "in_review"
        self.status = "under_review"

    def mark_reviewed(
        self,
        *,
        reviewer_id: uuid.UUID,
        approved: bool,
        notes: Optional[str] = None,
    ) -> None:
        self.reviewed_by = reviewer_id
        self.reviewed_at = utcnow()
        self.review_approved = approved
        self.review_notes = notes

        self.review_status = (
            "approved"
            if approved
            else "changes_requested"
        )

    # ========================================================
    # MITIGATION
    # ========================================================

    def start_mitigation(
        self,
        *,
        owner_id: Optional[uuid.UUID] = None,
        plan: Optional[str] = None,
    ) -> None:
        if self.is_closed:
            raise ValueError(
                "Closed risk cannot be mitigated."
            )

        self.mitigation_required = True
        self.mitigation_owner_id = (
            owner_id
            or self.mitigation_owner_id
        )

        if plan is not None:
            self.mitigation_plan = plan

        self.mitigation_started_at = utcnow()
        self.status = "mitigating"

    def add_mitigation_action(
        self,
        action: dict[str, Any],
    ) -> None:
        actions = list(
            self.mitigation_actions or []
        )

        actions.append(action)
        self.mitigation_actions = actions

    def mark_mitigated(
        self,
        *,
        user_id: uuid.UUID,
        notes: Optional[str] = None,
        residual_level: Optional[str] = None,
        residual_score: Optional[float] = None,
        residual_reason: Optional[str] = None,
    ) -> None:
        self.status = "mitigated"

        self.mitigated_at = utcnow()
        self.mitigated_by = user_id
        self.mitigation_notes = notes

        self.residual_risk_level = (
            residual_level
        )

        if residual_score is not None:
            self.residual_risk_score = min(
                1.0,
                max(
                    0.0,
                    float(residual_score),
                ),
            )

        self.residual_risk_reason = (
            residual_reason
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
        if self.is_closed:
            raise ValueError(
                "Closed risk cannot be escalated."
            )

        self.escalated = True
        self.escalation_level = level
        self.escalation_reason = reason

        self.escalated_at = utcnow()
        self.escalated_by = actor_id
        self.escalation_count += 1

    def clear_escalation(self) -> None:
        self.escalated = False
        self.escalation_level = None
        self.escalation_reason = None

    # ========================================================
    # RESOLUTION
    # ========================================================

    def resolve(
        self,
        *,
        user_id: uuid.UUID,
        resolution: str,
        resolution_type: Optional[str] = None,
    ) -> None:
        self.status = "resolved"

        self.resolution = resolution
        self.resolution_type = (
            resolution_type
        )

        self.resolved_at = utcnow()
        self.resolved_by = user_id

        self.escalated = False
        self.escalation_level = None

    # ========================================================
    # DISMISSAL
    # ========================================================

    def dismiss(
        self,
        *,
        user_id: uuid.UUID,
        reason: str,
    ) -> None:
        self.status = "dismissed"

        self.dismissed_at = utcnow()
        self.dismissed_by = user_id
        self.dismissal_reason = reason

        self.escalated = False
        self.escalation_level = None

    # ========================================================
    # EVIDENCE
    # ========================================================

    def add_evidence(
        self,
        item: dict[str, Any],
    ) -> None:
        evidence = list(
            self.evidence or []
        )

        evidence.append(item)
        self.evidence = evidence

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
        deleted_by: Optional[uuid.UUID] = None,
    ) -> None:
        if self.is_critical and not self.is_closed:
            raise ValueError(
                "Open critical risk should be resolved or "
                "dismissed before deletion."
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
            "<Risk("
            f"id={self.id}, "
            f"title={self.title!r}, "
            f"category={self.category!r}, "
            f"level={self.risk_level!r}, "
            f"score={self.risk_score!r}, "
            f"status={self.status!r}"
            ")>"
        )