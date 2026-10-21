# backend/db/repositories/document_repository.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import asc, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.document import Document


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class DocumentCreate:
    tenant_id: str
    organization_id: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: str = "unknown"
    document_subtype: Optional[str] = None
    category: Optional[str] = None
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    file_extension: Optional[str] = None
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    language: str = "en"
    status: str = "pending"
    external_id: Optional[str] = None
    reference_number: Optional[str] = None
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentUpdate:
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    external_id: Optional[str] = None
    reference_number: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class DocumentFilter:
    tenant_id: str
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    document_type: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    language: Optional[str] = None
    search_query: Optional[str] = None
    min_file_size: Optional[int] = None
    max_file_size: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    is_deleted: bool = False
    is_quarantined: Optional[bool] = None


@dataclass(slots=True)
class DocumentPage:
    items: list[Document]
    total: int
    offset: int
    limit: int


@dataclass(slots=True)
class DocumentVersionCreate:
    document_id: str
    version_number: int
    file_path: str
    file_size_bytes: int
    mime_type: str
    sha256: str
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# EXCEPTIONS
# ============================================================

class DocumentRepositoryError(RuntimeError):
    pass


class DocumentNotFoundError(DocumentRepositoryError):
    def __init__(self, document_id: str | UUID):
        super().__init__(f"Document not found: {document_id}")
        self.document_id = str(document_id)


class DuplicateDocumentError(DocumentRepositoryError):
    def __init__(self, sha256: str):
        super().__init__(f"Document already exists with SHA256: {sha256}")
        self.sha256 = sha256


# ============================================================
# REPOSITORY
# ============================================================

class DocumentRepository:
    """
    Async persistence layer for enterprise documents.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: DocumentCreate) -> Document:
        """
        Create a new document.
        """
        document = Document(
            tenant_id=UUID(data.tenant_id) if isinstance(data.tenant_id, str) else data.tenant_id,
            organization_id=UUID(data.organization_id) if data.organization_id and isinstance(data.organization_id, str) else data.organization_id,
            name=data.name,
            title=data.title,
            description=data.description,
            document_type=data.document_type,
            document_subtype=data.document_subtype,
            category=data.category,
            original_filename=data.original_filename,
            mime_type=data.mime_type,
            file_extension=data.file_extension,
            file_size_bytes=data.file_size_bytes,
            sha256=data.sha256,
            language=data.language,
            status=data.status,
            external_id=data.external_id,
            reference_number=data.reference_number,
            uploaded_by=UUID(data.user_id) if data.user_id and isinstance(data.user_id, str) else data.user_id,
            metadata_=data.metadata or {},
        )

        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)

        return document

    async def get_by_id(
        self,
        document_id: UUID | str,
        tenant_id: UUID | str,
    ) -> Document:
        """
        Get a document by ID.
        """
        if isinstance(document_id, str):
            document_id = UUID(document_id)
        if isinstance(tenant_id, str):
            tenant_id = UUID(tenant_id)

        stmt = select(Document).where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.deleted_at.is_(None),
        )

        result = await self.session.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            raise DocumentNotFoundError(document_id)

        return document

    async def update(
        self,
        document_id: UUID | str,
        tenant_id: UUID | str,
        data: DocumentUpdate,
    ) -> Document:
        """
        Update a document.
        """
        document = await self.get_by_id(document_id, tenant_id)

        if data.title is not None:
            document.title = data.title
        if data.description is not None:
            document.description = data.description
        if data.document_type is not None:
            document.document_type = data.document_type
        if data.document_subtype is not None:
            document.document_subtype = data.document_subtype
        if data.category is not None:
            document.category = data.category
        if data.status is not None:
            document.status = data.status
        if data.external_id is not None:
            document.external_id = data.external_id
        if data.reference_number is not None:
            document.reference_number = data.reference_number
        if data.metadata is not None:
            document.metadata_ = {**document.metadata_, **data.metadata}

        document.updated_at = datetime.now(timezone.utc)

        await self.session.flush()
        await self.session.refresh(document)

        return document

    async def list(
        self,
        filter_: DocumentFilter,
        offset: int = 0,
        limit: int = 100,
        order_by: str = "created_at",
        order_dir: str = "desc",
    ) -> DocumentPage:
        """
        List documents with filtering and pagination.
        """
        tenant_id = UUID(filter_.tenant_id) if isinstance(filter_.tenant_id, str) else filter_.tenant_id

        # Base query
        stmt = select(Document).where(
            Document.tenant_id == tenant_id,
        )

        # Deleted filter
        if not filter_.is_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))

        # Organization filter
        if filter_.organization_id:
            org_id = UUID(filter_.organization_id) if isinstance(filter_.organization_id, str) else filter_.organization_id
            stmt = stmt.where(Document.organization_id == org_id)

        # Type filters
        if filter_.document_type:
            stmt = stmt.where(Document.document_type == filter_.document_type)
        if filter_.category:
            stmt = stmt.where(Document.category == filter_.category)
        if filter_.status:
            stmt = stmt.where(Document.status == filter_.status)
        if filter_.language:
            stmt = stmt.where(Document.language == filter_.language)

        # Search query
        if filter_.search_query:
            search = f"%{filter_.search_query}%"
            stmt = stmt.where(
                or_(
                    Document.name.ilike(search),
                    Document.title.ilike(search),
                    Document.description.ilike(search),
                    Document.reference_number.ilike(search),
                )
            )

        # File size filters
        if filter_.min_file_size is not None:
            stmt = stmt.where(Document.file_size_bytes >= filter_.min_file_size)
        if filter_.max_file_size is not None:
            stmt = stmt.where(Document.file_size_bytes <= filter_.max_file_size)

        # Date filters
        if filter_.created_after:
            stmt = stmt.where(Document.created_at >= filter_.created_after)
        if filter_.created_before:
            stmt = stmt.where(Document.created_at <= filter_.created_before)
        if filter_.updated_after:
            stmt = stmt.where(Document.updated_at >= filter_.updated_after)
        if filter_.updated_before:
            stmt = stmt.where(Document.updated_at <= filter_.updated_before)

        # Quarantine filter
        if filter_.is_quarantined is not None:
            if filter_.is_quarantined:
                stmt = stmt.where(Document.quarantined_at.is_not(None))
            else:
                stmt = stmt.where(Document.quarantined_at.is_(None))

        # Count total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Order
        order_column = getattr(Document, order_by, Document.created_at)
        if order_dir == "asc":
            stmt = stmt.order_by(asc(order_column))
        else:
            stmt = stmt.order_by(desc(order_column))

        # Paginate
        stmt = stmt.offset(offset).limit(limit)

        # Execute
        result = await self.session.execute(stmt)
        documents = list(result.scalars().all())

        return DocumentPage(
            items=documents,
            total=total,
            offset=offset,
            limit=limit,
        )

    async def delete(
        self,
        document_id: UUID | str,
        tenant_id: UUID | str,
        hard_delete: bool = False,
    ) -> None:
        """
        Delete a document (soft delete by default).
        """
        document = await self.get_by_id(document_id, tenant_id)

        if hard_delete:
            await self.session.delete(document)
        else:
            document.deleted_at = datetime.now(timezone.utc)

        await self.session.flush()

    async def check_duplicate(
        self,
        tenant_id: UUID | str,
        sha256: str,
    ) -> Optional[Document]:
        """
        Check if a document with the same SHA256 already exists.
        """
        if isinstance(tenant_id, str):
            tenant_id = UUID(tenant_id)

        stmt = select(Document).where(
            Document.tenant_id == tenant_id,
            Document.sha256 == sha256,
            Document.deleted_at.is_(None),
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
