# backend/db/models/document.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
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


class Document(Base):
    """
    Top-level enterprise legal document entity.

    A Document is the stable logical identity of a file/document.

    The actual uploaded revisions live in DocumentVersion.

    Example:

        Document
            ├── DocumentVersion v1
            ├── DocumentVersion v2
            └── DocumentVersion v3 (current)

    Downstream objects may include:

        DocumentVersion
        DocumentChunk
        Contract
        Clause
        Obligation
        Risk
        Review
        Renewal
        RAG indexes
        Audit events

    Security:
        All repository and retrieval operations must be tenant
        scoped. Organization/document ACL checks should happen
        before document content enters a RAG context.
    """

    __tablename__ = "documents"

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

    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # DOCUMENT IDENTITY
    # ========================================================

    name: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    reference_number: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    document_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="unknown",
        index=True,
    )

    document_subtype: Mapped[Optional[str]] = mapped_column(
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
    # CLASSIFICATION
    # ========================================================

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
    # CURRENT VERSION
    # ========================================================

    current_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    current_version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    version_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # SOURCE / FILE SUMMARY
    # ========================================================

    original_filename: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    mime_type: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    file_extension: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    sha256: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # ========================================================
    # LANGUAGE
    # ========================================================

    language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
        index=True,
    )

    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    multilingual: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # DOCUMENT STATISTICS
    # ========================================================

    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    table_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    image_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    clause_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    contract_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # PROCESSING
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="uploaded",
        index=True,
    )

    processing_stage: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    processing_progress: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    processing_error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    processing_error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # INDEX / RAG STATUS
    # ========================================================

    indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    vector_indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    lexical_indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    rag_ready: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    citation_ready: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    vector_namespace: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # LEGAL / CONTRACT INTELLIGENCE
    # ========================================================

    contains_contract: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    contains_obligations: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    contains_expiry_terms: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    contains_renewal_terms: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

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

    risk_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    high_risk_clause_count: Mapped[int] = mapped_column(
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
    # OWNERSHIP
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

    # ========================================================
    # SECURITY / ACL
    # ========================================================

    visibility: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="private",
        index=True,
    )

    access_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="restricted",
        index=True,
    )

    allowed_user_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_role_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_group_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    contains_pii: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    contains_sensitive_data: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    security_labels: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    security_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    quarantined: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    quarantine_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # RETENTION
    # ========================================================

    retention_policy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    legal_hold: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    legal_hold_reason: Mapped[Optional[str]] = mapped_column(
        Text,
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

    versions = relationship(
        "DocumentVersion",
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        foreign_keys="DocumentChunk.document_id",
        lazy="selectin",
    )

    clauses = relationship(
        "Clause",
        back_populates="document",
        foreign_keys="Clause.document_id",
        lazy="selectin",
    )

    contracts = relationship(
        "Contract",
        back_populates="document",
        foreign_keys="Contract.document_id",
        lazy="selectin",
    )

    risks = relationship(
        "Risk",
        back_populates="document",
        foreign_keys="Risk.document_id",
        lazy="selectin",
    )

    reviews = relationship(
        "Review",
        back_populates="document",
        foreign_keys="Review.document_id",
        lazy="selectin",
    )

    # ========================================================
    # INDEXES / CONSTRAINTS
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_documents_tenant_external_id",
        ),

        Index(
            "ix_documents_tenant_org",
            "tenant_id",
            "organization_id",
        ),

        Index(
            "ix_documents_tenant_status",
            "tenant_id",
            "status",
        ),

        Index(
            "ix_documents_tenant_type",
            "tenant_id",
            "document_type",
        ),

        Index(
            "ix_documents_tenant_owner",
            "tenant_id",
            "owner_user_id",
        ),

        Index(
            "ix_documents_tenant_risk",
            "tenant_id",
            "risk_level",
            "risk_score",
        ),

        Index(
            "ix_documents_tenant_review",
            "tenant_id",
            "review_status",
        ),

        Index(
            "ix_documents_tenant_rag",
            "tenant_id",
            "rag_ready",
            "indexed",
            "is_deleted",
        ),

        Index(
            "ix_documents_tenant_security",
            "tenant_id",
            "security_status",
            "quarantined",
        ),

        Index(
            "ix_documents_tenant_acl",
            "tenant_id",
            "organization_id",
            "visibility",
            "access_level",
        ),

        Index(
            "ix_documents_sha256",
            "tenant_id",
            "sha256",
        ),

        Index(
            "ix_documents_retention",
            "tenant_id",
            "retention_until",
            "legal_hold",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def display_name(self) -> str:
        return (
            self.title
            or self.name
            or self.original_filename
            or str(self.id)
        )

    @property
    def fully_indexed(self) -> bool:
        return bool(
            self.vector_indexed
            and self.lexical_indexed
        )

    @property
    def safe_for_rag(self) -> bool:
        return bool(
            self.is_active
            and not self.is_deleted
            and not self.quarantined
            and (
                self.security_status
                or ""
            ).lower()
            in {
                "passed",
                "approved",
                "safe",
            }
        )

    @property
    def searchable(self) -> bool:
        return bool(
            self.rag_ready
            and self.fully_indexed
            and self.safe_for_rag
        )

    @property
    def is_high_risk(self) -> bool:
        level = (
            self.risk_level
            or ""
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
                self.risk_level
                or ""
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
    def can_be_deleted(self) -> bool:
        """
        Legal-hold documents must not be deleted by the normal
        document deletion workflow.
        """

        return not self.legal_hold

    @property
    def statistics(self) -> dict[str, Any]:
        return {
            "pages": self.page_count,
            "words": self.word_count,
            "characters":
                self.character_count,
            "tables": self.table_count,
            "images": self.image_count,
            "chunks": self.chunk_count,
            "clauses": self.clause_count,
            "contracts": self.contract_count,
            "versions": self.version_count,
        }

    # ========================================================
    # PROCESSING
    # ========================================================

    def start_processing(
        self,
        *,
        stage: str = "validation",
    ) -> None:
        self.status = "processing"
        self.processing_stage = stage
        self.processing_progress = 0.0

        self.processing_error_code = None
        self.processing_error_message = None

    def update_processing(
        self,
        *,
        stage: str,
        progress: float,
    ) -> None:
        self.processing_stage = stage

        self.processing_progress = min(
            100.0,
            max(
                0.0,
                float(progress),
            ),
        )

    def mark_processing_failed(
        self,
        *,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.status = "failed"
        self.processing_stage = "failed"

        self.processing_error_code = (
            error_code
        )

        self.processing_error_message = (
            error_message
        )

        self.rag_ready = False
        self.indexed = False

    def mark_processing_complete(
        self,
    ) -> None:
        self.status = "ready"
        self.processing_stage = "completed"
        self.processing_progress = 100.0
        self.processed_at = utcnow()

    # ========================================================
    # INDEXING
    # ========================================================

    def refresh_index_status(
        self,
    ) -> None:
        self.indexed = bool(
            self.vector_indexed
            and self.lexical_indexed
        )

        if self.indexed:
            self.indexed_at = (
                self.indexed_at
                or utcnow()
            )
        else:
            self.indexed_at = None

        self.rag_ready = bool(
            self.indexed
            and self.citation_ready
            and self.safe_for_rag
        )

    def mark_vector_indexed(
        self,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        self.vector_indexed = True

        if namespace is not None:
            self.vector_namespace = namespace

        self.refresh_index_status()

    def mark_lexical_indexed(
        self,
    ) -> None:
        self.lexical_indexed = True
        self.refresh_index_status()

    def mark_citation_ready(
        self,
    ) -> None:
        self.citation_ready = True
        self.refresh_index_status()

    def invalidate_indexes(
        self,
    ) -> None:
        self.indexed = False
        self.indexed_at = None

        self.vector_indexed = False
        self.lexical_indexed = False

        self.rag_ready = False

    # ========================================================
    # VERSION MANAGEMENT
    # ========================================================

    def set_current_version(
        self,
        *,
        version_id: uuid.UUID,
        version_number: int,
    ) -> None:
        self.current_version_id = (
            version_id
        )

        self.current_version_number = (
            version_number
        )

        self.version_count = max(
            self.version_count,
            version_number,
        )

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
    # SECURITY
    # ========================================================

    def quarantine(
        self,
        *,
        reason: str,
    ) -> None:
        self.quarantined = True
        self.quarantine_reason = reason

        self.security_status = (
            "quarantined"
        )

        self.rag_ready = False

    def release_quarantine(
        self,
    ) -> None:
        self.quarantined = False
        self.quarantine_reason = None

        self.security_status = "passed"

        self.refresh_index_status()

    # ========================================================
    # LEGAL HOLD
    # ========================================================

    def place_legal_hold(
        self,
        *,
        reason: str,
    ) -> None:
        self.legal_hold = True
        self.legal_hold_reason = reason

    def release_legal_hold(
        self,
    ) -> None:
        self.legal_hold = False
        self.legal_hold_reason = None

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
        if self.legal_hold:
            raise ValueError(
                "Document is under legal hold "
                "and cannot be deleted."
            )

        self.is_deleted = True
        self.is_active = False

        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

        self.rag_ready = False

    def restore(
        self,
    ) -> None:
        self.is_deleted = False
        self.is_active = True

        self.deleted_at = None
        self.deleted_by = None

        self.refresh_index_status()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:
        return (
            "<Document("
            f"id={self.id}, "
            f"name={self.name!r}, "
            f"type={self.document_type!r}, "
            f"status={self.status!r}, "
            f"version={self.current_version_number}"
            ")>"
        )