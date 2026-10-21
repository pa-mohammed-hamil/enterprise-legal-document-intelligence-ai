# backend/app/db/models/clause.py

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base

if TYPE_CHECKING:
    from app.db.models.contract import Contract
    from app.db.models.document import Document
    from app.db.models.organization import Organization


class Clause(Base):
    """
    Represents an individual legal clause extracted from a contract/document.

    A Clause stores:
    - Original clause text
    - Normalized/extracted text
    - Clause classification
    - Location within the source document
    - AI extraction/classification confidence
    - Risk information
    - Review state
    - Citation/source metadata
    - Model provenance
    """

    __tablename__ = "clauses"

    __table_args__ = (
        UniqueConstraint(
            "contract_id",
            "clause_key",
            name="uq_clause_contract_key",
        ),
        Index("ix_clauses_organization_id", "organization_id"),
        Index("ix_clauses_document_id", "document_id"),
        Index("ix_clauses_contract_id", "contract_id"),
        Index("ix_clauses_clause_type", "clause_type"),
        Index("ix_clauses_risk_level", "risk_level"),
        Index("ix_clauses_review_status", "review_status"),
        Index(
            "ix_clauses_contract_type",
            "contract_id",
            "clause_type",
        ),
    )

    # ------------------------------------------------------------------
    # Primary identity
    # ------------------------------------------------------------------

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
    )

    contract_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Clause identity / hierarchy
    # ------------------------------------------------------------------

    clause_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    clause_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    parent_clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    hierarchy_level: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    position: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Clause content
    # ------------------------------------------------------------------

    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    clause_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="other",
    )

    subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    original_text: Mapped[str] = mapped_column(
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

    # ------------------------------------------------------------------
    # Document source location / citation
    # ------------------------------------------------------------------

    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    start_page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    end_page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    start_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    end_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    source_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_chunks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    citation: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # AI classification
    # ------------------------------------------------------------------

    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    classifier_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    extraction_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk analysis
    # ------------------------------------------------------------------

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unknown",
    )

    risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    risk_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_non_standard: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    deviation_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Legal entities / structured extraction
    # ------------------------------------------------------------------

    parties: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    dates: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    monetary_values: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    jurisdictions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    obligations: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    extracted_entities: Mapped[dict[str, Any]] = mapped_column(
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

    # ------------------------------------------------------------------
    # Review workflow
    # ------------------------------------------------------------------

    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
    )

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    reviewer_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Processing / AI provenance
    # ------------------------------------------------------------------

    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    processing_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    needs_reprocessing: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
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

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    organization: Mapped["Organization"] = relationship(
        "Organization",
        lazy="selectin",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        lazy="selectin",
    )

    contract: Mapped["Contract"] = relationship(
        "Contract",
        back_populates="clauses",
        lazy="selectin",
    )

    parent_clause: Mapped[Optional["Clause"]] = relationship(
        "Clause",
        remote_side="Clause.id",
        back_populates="child_clauses",
        foreign_keys=[parent_clause_id],
    )

    child_clauses: Mapped[list["Clause"]] = relationship(
        "Clause",
        back_populates="parent_clause",
        foreign_keys=[parent_clause_id],
    )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level.lower() in {
            "high",
            "critical",
        }

    @property
    def confidence(self) -> float:
        """
        Combined confidence estimate.

        Extraction and classification confidence are averaged when both
        values are available.
        """
        scores = [
            score
            for score in (
                self.extraction_confidence,
                self.classification_confidence,
            )
            if score is not None
        ]

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    @property
    def source_pages(self) -> list[int]:
        if self.start_page is not None and self.end_page is not None:
            return list(range(self.start_page, self.end_page + 1))

        if self.page_number is not None:
            return [self.page_number]

        return []

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    def mark_for_review(
        self,
        reason: Optional[str] = None,
    ) -> None:
        self.requires_review = True
        self.review_status = "pending"

        if reason:
            self.reviewer_notes = reason

    def mark_reviewed(
        self,
        reviewer_id: uuid.UUID,
        notes: Optional[str] = None,
        verified: bool = True,
    ) -> None:
        self.reviewed_by = reviewer_id
        self.reviewed_at = datetime.utcnow()
        self.review_status = "reviewed"
        self.requires_review = False
        self.is_verified = verified

        if notes is not None:
            self.reviewer_notes = notes

    def update_risk(
        self,
        *,
        level: str,
        score: Optional[float] = None,
        reason: Optional[str] = None,
        recommended_action: Optional[str] = None,
    ) -> None:
        self.risk_level = level
        self.risk_score = score
        self.risk_reason = reason
        self.recommended_action = recommended_action

        if level.lower() in {"high", "critical"}:
            self.requires_review = True

    def mark_for_reprocessing(self) -> None:
        self.needs_reprocessing = True

    def mark_reprocessed(self) -> None:
        self.needs_reprocessing = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "document_id": str(self.document_id),
            "contract_id": str(self.contract_id),
            "clause_key": self.clause_key,
            "clause_number": self.clause_number,
            "title": self.title,
            "clause_type": self.clause_type,
            "subtype": self.subtype,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "summary": self.summary,
            "page_number": self.page_number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "citation": self.citation,
            "classification_confidence": self.classification_confidence,
            "extraction_confidence": self.extraction_confidence,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "risk_reason": self.risk_reason,
            "recommended_action": self.recommended_action,
            "is_non_standard": self.is_non_standard,
            "requires_review": self.requires_review,
            "review_status": self.review_status,
            "is_verified": self.is_verified,
            "source_pages": self.source_pages,
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

    def __repr__(self) -> str:
        return (
            f"<Clause("
            f"id={self.id}, "
            f"type={self.clause_type!r}, "
            f"risk={self.risk_level!r}"
            f")>"
        )