# backend/db/models/document_chunk.py

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


class DocumentChunk(Base):
    """
    Searchable RAG unit derived from a document version.

    Supports:
        - semantic chunking
        - legal chunking
        - parent-child chunking
        - table chunking
        - BM25 retrieval
        - dense vector retrieval
        - hybrid retrieval
        - reranking
        - citations
        - tenant / organization isolation
        - document ACL filtering
        - embedding lifecycle
        - extraction provenance

    Security:
        Every retrieval path must enforce tenant_id and access
        control before the chunk enters LLM context.
    """

    __tablename__ = "document_chunks"

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
    # DOCUMENT
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
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # CHUNK IDENTITY
    # ========================================================

    chunk_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    chunk_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="text",
        index=True,
    )

    chunk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="child",
        index=True,
    )

    chunking_strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # ========================================================
    # PARENT / CHILD
    # ========================================================

    parent_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_chunks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    root_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_chunks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    hierarchy_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    hierarchy_path: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # CONTENT
    # ========================================================

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    retrieval_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    heading: Mapped[Optional[str]] = mapped_column(
        String(1000),
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

    language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
        index=True,
    )

    # ========================================================
    # SIZE
    # ========================================================

    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # SOURCE LOCATION / CITATION
    # ========================================================

    page_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
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

    source_block_index: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    source_element_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    source_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    citation_label: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    citation_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # LEGAL STRUCTURE
    # ========================================================

    clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
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

    clause_type: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    legal_section_type: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    # ========================================================
    # TABLE INFORMATION
    # ========================================================

    is_table: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    table_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    table_title: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    table_headers: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    table_rows: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    embedding_model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_dimension: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_normalized: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    embedding_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    embedding_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    # Vector itself normally lives in Qdrant/Pinecone/etc.
    vector_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
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

    # ========================================================
    # BM25 / LEXICAL INDEX
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

    lexical_document_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # RETRIEVAL METADATA
    # ========================================================

    searchable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    retrieval_priority: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )

    tags: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    keywords: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    entities: Mapped[list[Any]] = mapped_column(
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

    # ========================================================
    # EXTRACTION PROVENANCE
    # ========================================================

    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    extraction_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    ocr_used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    ocr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    parser_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    parser_version: Mapped[Optional[str]] = mapped_column(
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

    # ========================================================
    # QUALITY
    # ========================================================

    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
    )

    semantic_coherence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    completeness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    citation_ready: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    # ========================================================
    # CONTENT HASH / IDEMPOTENCY
    # ========================================================

    content_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    source_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )

    # ========================================================
    # STATUS / VERSION
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

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    document = relationship(
        "Document",
        back_populates="chunks",
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

    contract = relationship(
        "Contract",
        foreign_keys=[contract_id],
        lazy="selectin",
    )

    parent_chunk = relationship(
        "DocumentChunk",
        remote_side=[id],
        foreign_keys=[parent_chunk_id],
        back_populates="child_chunks",
    )

    child_chunks = relationship(
        "DocumentChunk",
        foreign_keys=[parent_chunk_id],
        back_populates="parent_chunk",
        lazy="selectin",
    )

    root_chunk = relationship(
        "DocumentChunk",
        remote_side=[id],
        foreign_keys=[root_chunk_id],
        uselist=False,
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "document_version_id",
            "chunk_id",
            name="uq_document_chunk_identity",
        ),

        Index(
            "ix_chunks_tenant_document",
            "tenant_id",
            "document_id",
        ),

        Index(
            "ix_chunks_tenant_document_version",
            "tenant_id",
            "document_id",
            "document_version_id",
        ),

        Index(
            "ix_chunks_tenant_org",
            "tenant_id",
            "organization_id",
        ),

        Index(
            "ix_chunks_document_order",
            "document_id",
            "document_version_id",
            "chunk_index",
        ),

        Index(
            "ix_chunks_tenant_type",
            "tenant_id",
            "chunk_type",
        ),

        Index(
            "ix_chunks_parent",
            "tenant_id",
            "parent_chunk_id",
        ),

        Index(
            "ix_chunks_contract",
            "tenant_id",
            "contract_id",
        ),

        Index(
            "ix_chunks_clause",
            "tenant_id",
            "clause_id",
        ),

        Index(
            "ix_chunks_pages",
            "document_id",
            "page_start",
            "page_end",
        ),

        Index(
            "ix_chunks_vector_searchable",
            "tenant_id",
            "vector_indexed",
            "searchable",
            "is_current",
            "is_deleted",
        ),

        Index(
            "ix_chunks_lexical_searchable",
            "tenant_id",
            "lexical_indexed",
            "searchable",
            "is_current",
            "is_deleted",
        ),

        Index(
            "ix_chunks_acl",
            "tenant_id",
            "organization_id",
            "visibility",
            "access_level",
        ),

        Index(
            "ix_chunks_content_hash",
            "tenant_id",
            "document_id",
            "content_hash",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @property
    def is_parent(self) -> bool:
        return (
            self.chunk_level or ""
        ).lower() == "parent"

    @property
    def is_child(self) -> bool:
        return (
            self.chunk_level or ""
        ).lower() == "child"

    @property
    def is_searchable(self) -> bool:
        return bool(
            self.searchable
            and self.is_active
            and self.is_current
            and not self.is_deleted
        )

    @property
    def fully_indexed(self) -> bool:
        return bool(
            self.vector_indexed
            and self.lexical_indexed
        )

    @property
    def page_reference(self) -> Optional[str]:
        if self.page_start is None:
            return None

        if (
            self.page_end is None
            or self.page_start == self.page_end
        ):
            return f"page {self.page_start}"

        return (
            f"pages {self.page_start}"
            f"-{self.page_end}"
        )

    @property
    def citation_reference(self) -> dict[str, Any]:
        return {
            "document_id": str(
                self.document_id
            ),
            "document_version_id": (
                str(self.document_version_id)
                if self.document_version_id
                else None
            ),
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "heading": self.heading,
            "clause_id": (
                str(self.clause_id)
                if self.clause_id
                else None
            ),
        }

    @property
    def vector_payload(self) -> dict[str, Any]:
        """
        Metadata to persist alongside the vector.

        Permission-aware vector search should apply these
        filters before candidates are returned.
        """

        return {
            "chunk_id": self.chunk_id,
            "db_chunk_id": str(self.id),

            "tenant_id": str(
                self.tenant_id
            ),

            "organization_id": (
                str(self.organization_id)
                if self.organization_id
                else None
            ),

            "document_id": str(
                self.document_id
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

            "chunk_type":
                self.chunk_type,

            "chunk_level":
                self.chunk_level,

            "parent_chunk_id": (
                str(self.parent_chunk_id)
                if self.parent_chunk_id
                else None
            ),

            "page_start":
                self.page_start,

            "page_end":
                self.page_end,

            "language":
                self.language,

            "clause_type":
                self.clause_type,

            "visibility":
                self.visibility,

            "access_level":
                self.access_level,

            "allowed_user_ids":
                self.allowed_user_ids,

            "allowed_role_ids":
                self.allowed_role_ids,

            "allowed_group_ids":
                self.allowed_group_ids,

            "contains_pii":
                self.contains_pii,

            "contains_sensitive_data":
                self.contains_sensitive_data,

            "security_labels":
                self.security_labels,

            "is_current":
                self.is_current,

            "is_active":
                self.is_active,

            "is_deleted":
                self.is_deleted,

            "namespace":
                self.vector_namespace,
        }

    def mark_vector_indexed(
        self,
        *,
        vector_id: str,
        collection: str,
        namespace: Optional[str],
        model: str,
        model_version: Optional[str],
        dimension: int,
        embedding_hash: Optional[str] = None,
    ) -> None:
        self.vector_id = vector_id
        self.vector_collection = collection
        self.vector_namespace = namespace

        self.embedding_model = model
        self.embedding_model_version = model_version
        self.embedding_dimension = dimension
        self.embedding_hash = embedding_hash
        self.embedding_created_at = utcnow()

        self.vector_indexed = True
        self.vector_indexed_at = utcnow()

    def mark_lexical_indexed(
        self,
        *,
        lexical_document_id: Optional[str] = None,
    ) -> None:
        self.lexical_document_id = (
            lexical_document_id
            or self.chunk_id
        )

        self.lexical_indexed = True
        self.lexical_indexed_at = utcnow()

    def invalidate_indexes(self) -> None:
        """
        Call when chunk content changes.
        """

        self.vector_indexed = False
        self.vector_indexed_at = None
        self.vector_id = None

        self.lexical_indexed = False
        self.lexical_indexed_at = None
        self.lexical_document_id = None

    def soft_delete(self) -> None:
        self.is_deleted = True
        self.is_active = False
        self.searchable = False
        self.is_current = False
        self.deleted_at = utcnow()

    def restore(self) -> None:
        self.is_deleted = False
        self.is_active = True
        self.searchable = True
        self.is_current = True
        self.deleted_at = None

    def __repr__(self) -> str:
        return (
            "<DocumentChunk("
            f"id={self.id}, "
            f"chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id}, "
            f"type={self.chunk_type!r}, "
            f"level={self.chunk_level!r}"
            ")>"
        )