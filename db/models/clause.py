# backend/db/models/clause.py

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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Clause(Base):
    """
    Extracted or manually-created legal clause.

    A Clause represents a logical legal provision linked to a
    document and optionally a contract.

    Examples:
        confidentiality
        indemnification
        limitation of liability
        termination
        payment
        renewal
        governing law
        data protection
        intellectual property
        SLA
        insurance
        warranty
        audit rights
        force majeure

    Security:
        tenant_id is mandatory and must participate in every
        repository query.

    AI:
        Classification, risk score and extracted attributes are
        analytical signals only. They must not be treated as
        authoritative legal conclusions.
    """

    __tablename__ = "clauses"

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
    # DOCUMENT / CONTRACT
    # ========================================================

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
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

    contract_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # CLAUSE HIERARCHY
    # ========================================================

    parent_clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    clause_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    section_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    section_title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    heading: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    hierarchy_path: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
    )

    hierarchy_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    sequence_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # CLAUSE CONTENT
    # ========================================================

    title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
        index=True,
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    clause_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
        index=True,
    )

    clause_subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    classification_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    classifier_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    classifier_version: Mapped[Optional[str]] = mapped_column(
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

    risk_category: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    risk_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    risk_signals: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    risk_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    risk_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    risk_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    requires_legal_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # LEGAL ATTRIBUTES
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

    currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )

    monetary_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(
            precision=20,
            scale=4,
        ),
        nullable=True,
    )

    notice_period_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    term_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    effective_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    expiry_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    auto_renewal: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        index=True,
    )

    # ========================================================
    # EXTRACTED ENTITIES / OBLIGATIONS
    # ========================================================

    parties: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    entities: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    obligations: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    rights: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    exceptions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    conditions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    defined_terms: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # SOURCE / PROVENANCE
    # ========================================================

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

    source_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    parent_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

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

    # ========================================================
    # REVIEW
    # ========================================================

    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
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

    supersedes_clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
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
    # EXTENSION METADATA
    # ========================================================

    # Do not name the ORM attribute "metadata".
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
        back_populates="clauses",
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
        back_populates="clauses",
        foreign_keys=[contract_id],
        lazy="selectin",
    )

    parent_clause = relationship(
        "Clause",
        remote_side=[id],
        foreign_keys=[parent_clause_id],
        back_populates="child_clauses",
    )

    child_clauses = relationship(
        "Clause",
        foreign_keys=[parent_clause_id],
        back_populates="parent_clause",
        lazy="selectin",
    )

    supersedes_clause = relationship(
        "Clause",
        remote_side=[id],
        foreign_keys=[supersedes_clause_id],
        uselist=False,
    )

    # ========================================================
    # INDEXES / CONSTRAINTS
    # ========================================================

    __table_args__ = (
        Index(
            "ix_clauses_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_clauses_tenant_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_clauses_tenant_type",
            "tenant_id",
            "clause_type",
        ),
        Index(
            "ix_clauses_tenant_risk",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),
        Index(
            "ix_clauses_tenant_review",
            "tenant_id",
            "review_status",
        ),
        Index(
            "ix_clauses_tenant_current",
            "tenant_id",
            "is_current",
            "is_deleted",
        ),
        Index(
            "ix_clauses_document_sequence",
            "document_id",
            "sequence_number",
        ),
        Index(
            "ix_clauses_document_pages",
            "document_id",
            "page_start",
            "page_end",
        ),
        Index(
            "ix_clauses_parent",
            "tenant_id",
            "parent_clause_id",
        ),
        Index(
            "ix_clauses_source_chunk",
            "tenant_id",
            "source_chunk_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "document_version_id",
            "content_hash",
            name="uq_clause_document_version_hash",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

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
    def is_critical(self) -> bool:
        return (
            (self.risk_level or "").lower()
            == "critical"
            or (
                self.risk_score is not None
                and self.risk_score >= 0.90
            )
        )

    @property
    def requires_review(self) -> bool:
        return bool(
            self.requires_legal_review
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
    def display_name(self) -> str:
        parts: list[str] = []

        if self.clause_number:
            parts.append(
                self.clause_number
            )

        if self.heading:
            parts.append(
                self.heading
            )

        elif self.title:
            parts.append(
                self.title
            )

        elif self.clause_type:
            parts.append(
                self.clause_type.replace(
                    "_",
                    " ",
                ).title()
            )

        return " — ".join(parts) or str(
            self.id
        )

    @property
    def source_reference(self) -> dict[str, Any]:
        return {
            "document_id": str(
                self.document_id
            ),
            "document_version_id": (
                str(
                    self.document_version_id
                )
                if self.document_version_id
                else None
            ),
            "contract_id": (
                str(self.contract_id)
                if self.contract_id
                else None
            ),
            "clause_id": str(
                self.id
            ),
            "clause_number":
                self.clause_number,
            "heading":
                self.heading,
            "page_start":
                self.page_start,
            "page_end":
                self.page_end,
            "char_start":
                self.char_start,
            "char_end":
                self.char_end,
            "source_chunk_id":
                self.source_chunk_id,
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

    def soft_delete(
        self,
        *,
        deleted_by: Optional[uuid.UUID] = None,
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

    def __repr__(self) -> str:
        return (
            "<Clause("
            f"id={self.id}, "
            f"document_id={self.document_id}, "
            f"type={self.clause_type!r}, "
            f"risk={self.risk_level!r}, "
            f"review={self.review_status!r}"
            ")>"
        )