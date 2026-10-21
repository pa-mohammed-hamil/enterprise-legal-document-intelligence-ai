# backend/db/models/document_version.py

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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentVersion(Base):
    """
    Immutable logical version of an uploaded legal document.

    A new DocumentVersion should normally be created whenever the
    source file or authoritative normalized content changes.

    Typical lifecycle:

        upload
          -> validation
          -> malware/security scan
          -> parsing
          -> OCR
          -> normalization
          -> metadata extraction
          -> chunking
          -> embedding
          -> BM25/vector indexing
          -> clause/contract extraction
          -> ready

    Important:
        Historical versions should not be overwritten. They provide
        provenance for citations, legal review, model evaluation,
        contract comparison and audit reconstruction.
    """

    __tablename__ = "document_versions"

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

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # ========================================================
    # VERSION
    # ========================================================

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    version_label: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    revision: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    is_latest: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    supersedes_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # SOURCE FILE
    # ========================================================

    original_filename: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    storage_bucket: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    storage_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
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

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # INTEGRITY / IDEMPOTENCY
    # ========================================================

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    content_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    normalized_content_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    # ========================================================
    # PROCESSING STATUS
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

    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    processing_error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # DOCUMENT STRUCTURE
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

    clause_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # EXTRACTED CONTENT
    # ========================================================

    extracted_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    normalized_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    extracted_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    page_metadata: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    extracted_tables: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    extracted_entities: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # LANGUAGE
    # ========================================================

    language: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    script: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    multilingual: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    detected_languages: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # PARSER
    # ========================================================

    parser_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    parser_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    parser_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # OCR
    # ========================================================

    ocr_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    ocr_used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    ocr_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    ocr_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    ocr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    ocr_page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    scanned_page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    document_type: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    document_subtype: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    classification_method: Mapped[Optional[str]] = mapped_column(
        String(100),
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
    # CHUNKING
    # ========================================================

    chunking_strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    chunker_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    chunker_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    chunking_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    chunked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_dimension: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    embeddings_created: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    embeddings_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # VECTOR INDEX
    # ========================================================

    vector_indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    vector_indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    vector_collection: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    vector_namespace: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    vector_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # LEXICAL / BM25 INDEX
    # ========================================================

    lexical_indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    lexical_indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    lexical_index_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # RAG READINESS
    # ========================================================

    rag_ready: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    rag_ready_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    citation_ready: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # QUALITY
    # ========================================================

    extraction_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    chunk_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    overall_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
    )

    requires_manual_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    quality_issues: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # SECURITY
    # ========================================================

    security_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    malware_scanned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    malware_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    prompt_injection_scanned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    prompt_injection_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    pii_scanned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    pii_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    security_findings: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
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
    # MODEL / PIPELINE PROVENANCE
    # ========================================================

    ingestion_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    pipeline_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_runs: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # APPROVAL / REVIEW
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
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # STATUS
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
    # EXTENSION METADATA
    # ========================================================

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

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    document = relationship(
        "Document",
        back_populates="versions",
        foreign_keys=[document_id],
    )

    supersedes_version = relationship(
        "DocumentVersion",
        remote_side=[id],
        foreign_keys=[supersedes_version_id],
        uselist=False,
    )

    chunks = relationship(
        "DocumentChunk",
        foreign_keys="DocumentChunk.document_version_id",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    clauses = relationship(
        "Clause",
        foreign_keys="Clause.document_version_id",
        lazy="selectin",
    )

    contracts = relationship(
        "Contract",
        foreign_keys="Contract.document_version_id",
        lazy="selectin",
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "version_number",
            name="uq_document_version_number",
        ),

        Index(
            "ix_document_versions_tenant_document",
            "tenant_id",
            "document_id",
        ),

        Index(
            "ix_document_versions_document_current",
            "document_id",
            "is_current",
        ),

        Index(
            "ix_document_versions_document_latest",
            "document_id",
            "is_latest",
        ),

        Index(
            "ix_document_versions_tenant_status",
            "tenant_id",
            "status",
        ),

        Index(
            "ix_document_versions_tenant_type",
            "tenant_id",
            "document_type",
        ),

        Index(
            "ix_document_versions_sha256",
            "tenant_id",
            "sha256",
        ),

        Index(
            "ix_document_versions_rag_ready",
            "tenant_id",
            "rag_ready",
            "is_current",
            "is_deleted",
        ),

        Index(
            "ix_document_versions_security",
            "tenant_id",
            "security_status",
            "quarantined",
        ),

        Index(
            "ix_document_versions_ingestion_run",
            "tenant_id",
            "ingestion_run_id",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @property
    def fully_indexed(self) -> bool:
        return bool(
            self.vector_indexed
            and self.lexical_indexed
        )

    @property
    def safe_for_rag(self) -> bool:
        return bool(
            not self.is_deleted
            and self.is_active
            and not self.quarantined
            and not self.malware_detected
            and not self.prompt_injection_detected
            and self.security_status.lower()
            in {
                "passed",
                "approved",
                "safe",
            }
        )

    @property
    def ready_for_rag(self) -> bool:
        return bool(
            self.rag_ready
            and self.fully_indexed
            and self.citation_ready
            and self.safe_for_rag
        )

    @property
    def processing_failed(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "failed",
            "error",
        }

    @property
    def processing_complete(self) -> bool:
        return (
            self.status or ""
        ).lower() in {
            "ready",
            "completed",
            "indexed",
        }

    @property
    def requires_review(self) -> bool:
        return bool(
            self.requires_manual_review
            or self.quality_issues
            or self.quarantined
            or self.malware_detected
            or self.prompt_injection_detected
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
    def indexing_summary(self) -> dict[str, Any]:
        return {
            "chunks": self.chunk_count,
            "embeddings_created":
                self.embeddings_created,
            "vectors": self.vector_count,
            "vector_indexed":
                self.vector_indexed,
            "lexical_indexed":
                self.lexical_indexed,
            "rag_ready":
                self.rag_ready,
            "citation_ready":
                self.citation_ready,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "document_id": str(
                self.document_id
            ),
            "document_version_id": str(
                self.id
            ),
            "version_number":
                self.version_number,
            "sha256":
                self.sha256,
            "content_hash":
                self.content_hash,
            "parser":
                self.parser_name,
            "parser_version":
                self.parser_version,
            "ocr_used":
                self.ocr_used,
            "ocr_provider":
                self.ocr_provider,
            "chunker":
                self.chunker_name,
            "chunker_version":
                self.chunker_version,
            "embedding_model":
                self.embedding_model,
            "embedding_model_version":
                self.embedding_model_version,
            "pipeline_version":
                self.pipeline_version,
            "ingestion_run_id":
                self.ingestion_run_id,
        }

    # ========================================================
    # PROCESSING METHODS
    # ========================================================

    def start_processing(
        self,
        *,
        stage: str = "validation",
    ) -> None:
        self.status = "processing"
        self.processing_stage = stage
        self.processing_progress = 0.0
        self.processing_started_at = utcnow()

        self.processing_completed_at = None
        self.processing_error_code = None
        self.processing_error_message = None

    def update_processing_progress(
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

        self.processing_completed_at = (
            utcnow()
        )

    def mark_processing_complete(
        self,
    ) -> None:
        self.status = "ready"
        self.processing_stage = "completed"
        self.processing_progress = 100.0
        self.processing_completed_at = (
            utcnow()
        )

    # ========================================================
    # INDEX METHODS
    # ========================================================

    def mark_embeddings_created(
        self,
        *,
        model: str,
        model_version: Optional[str],
        dimension: int,
    ) -> None:
        self.embedding_model = model
        self.embedding_model_version = (
            model_version
        )
        self.embedding_dimension = (
            dimension
        )

        self.embeddings_created = True
        self.embeddings_created_at = (
            utcnow()
        )

    def mark_vector_indexed(
        self,
        *,
        collection: str,
        namespace: Optional[str],
        vector_count: int,
    ) -> None:
        self.vector_collection = collection
        self.vector_namespace = namespace
        self.vector_count = max(
            0,
            vector_count,
        )

        self.vector_indexed = True
        self.vector_indexed_at = utcnow()

        self._refresh_rag_status()

    def mark_lexical_indexed(
        self,
        *,
        index_name: Optional[str] = None,
    ) -> None:
        self.lexical_index_name = index_name
        self.lexical_indexed = True
        self.lexical_indexed_at = utcnow()

        self._refresh_rag_status()

    def mark_citation_ready(
        self,
    ) -> None:
        self.citation_ready = True

        self._refresh_rag_status()

    def invalidate_indexes(
        self,
    ) -> None:
        self.embeddings_created = False
        self.embeddings_created_at = None

        self.vector_indexed = False
        self.vector_indexed_at = None
        self.vector_count = 0

        self.lexical_indexed = False
        self.lexical_indexed_at = None

        self.rag_ready = False
        self.rag_ready_at = None

    def _refresh_rag_status(
        self,
    ) -> None:
        ready = bool(
            self.vector_indexed
            and self.lexical_indexed
            and self.citation_ready
            and not self.quarantined
            and not self.malware_detected
            and not self.prompt_injection_detected
        )

        self.rag_ready = ready

        if ready:
            self.rag_ready_at = (
                self.rag_ready_at
                or utcnow()
            )
        else:
            self.rag_ready_at = None

    # ========================================================
    # SECURITY METHODS
    # ========================================================

    def quarantine(
        self,
        *,
        reason: str,
    ) -> None:
        self.quarantined = True
        self.quarantine_reason = reason
        self.security_status = "quarantined"

        self.rag_ready = False
        self.rag_ready_at = None

    def release_quarantine(
        self,
    ) -> None:
        self.quarantined = False
        self.quarantine_reason = None

        if (
            not self.malware_detected
            and not self.prompt_injection_detected
        ):
            self.security_status = "passed"

        self._refresh_rag_status()

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
        self.review_notes = notes

        self.review_status = (
            "approved"
            if approved
            else "changes_requested"
        )

    # ========================================================
    # VERSION METHODS
    # ========================================================

    def mark_superseded(
        self,
    ) -> None:
        self.is_current = False
        self.is_latest = False

        # Historical chunks should not normally appear in
        # ordinary "current document" retrieval.
        self.rag_ready = False
        self.rag_ready_at = None

    # ========================================================
    # DELETE / RESTORE
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

        self.rag_ready = False
        self.rag_ready_at = None

    def restore(
        self,
    ) -> None:
        self.is_deleted = False
        self.is_active = True

        self.deleted_at = None
        self.deleted_by = None

        self._refresh_rag_status()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:
        return (
            "<DocumentVersion("
            f"id={self.id}, "
            f"document_id={self.document_id}, "
            f"version={self.version_number}, "
            f"status={self.status!r}, "
            f"rag_ready={self.rag_ready}"
            ")>"
        )