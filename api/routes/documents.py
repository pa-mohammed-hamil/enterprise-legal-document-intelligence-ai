# backend/api/routes/documents.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.db.repositories.document_repository import (
    DocumentCreate,
    DocumentFilter,
    DocumentNotFoundError,
    DocumentRepository,
    DocumentUpdate,
    DocumentVersionCreate,
    DuplicateDocumentError,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ============================================================
# DEPENDENCIES
# Replace these with backend.dependencies providers.
# ============================================================

async def get_document_repository() -> DocumentRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Document repository dependency is not configured.",
    )


async def get_document_service() -> Any:
    """
    Recommended DocumentService dependencies:

        DocumentRepository
        UploadService
        IngestionPipeline
        ObjectStorage
        DocumentParser
        OCRProcessor
        MetadataExtractor
        LanguageDetector
        DocumentNormalizer
        LegalChunker
        EmbeddingService
        VectorStore
        AuditService
        SecurityService
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Document service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class DocumentCreateRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=500,
    )

    original_filename: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    storage_key: str = Field(
        min_length=1,
        max_length=2000,
    )

    mime_type: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    file_size: Optional[int] = Field(
        default=None,
        ge=0,
    )

    sha256: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    document_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    language: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    source: str = Field(
        default="api",
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class DocumentUpdateRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    document_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    language: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    metadata: Optional[
        dict[str, Any]
    ] = None


class ClassificationUpdateRequest(BaseModel):
    document_type: str = Field(
        min_length=1,
        max_length=100,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=255,
    )


class ReprocessRequest(BaseModel):
    run_ocr: bool = True
    run_classification: bool = True
    run_extraction: bool = True
    run_chunking: bool = True
    run_embeddings: bool = True
    rebuild_index: bool = True

    force: bool = False


class ReindexRequest(BaseModel):
    rebuild_chunks: bool = False
    rebuild_embeddings: bool = True

    delete_existing_vectors: bool = True

    force: bool = False


class DocumentVersionRequest(BaseModel):
    storage_key: str = Field(
        min_length=1,
        max_length=2000,
    )

    filename: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    mime_type: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    file_size: Optional[int] = Field(
        default=None,
        ge=0,
    )

    sha256: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    change_summary: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    uploaded_by: Optional[str] = None

    name: str

    original_filename: Optional[str] = None

    document_type: Optional[str] = None
    language: Optional[str] = None

    mime_type: Optional[str] = None
    file_size: Optional[int] = None

    sha256: Optional[str] = None

    storage_key: Optional[str] = None

    processing_status: Optional[str] = None
    ocr_status: Optional[str] = None
    indexing_status: Optional[str] = None

    classification_confidence: Optional[float] = None

    page_count: Optional[int] = None
    chunk_count: Optional[int] = None

    current_version: Optional[int] = None

    source: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    processed_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None


class DocumentPageResponse(BaseModel):
    items: list[DocumentResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class DocumentVersionResponse(BaseModel):
    id: str
    document_id: str

    version_number: int

    storage_key: str

    filename: Optional[str] = None
    mime_type: Optional[str] = None

    file_size: Optional[int] = None
    sha256: Optional[str] = None

    change_summary: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None


class DocumentStatisticsResponse(BaseModel):
    total: int = 0

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_language: dict[str, int] = Field(
        default_factory=dict,
    )

    by_ocr_status: dict[str, int] = Field(
        default_factory=dict,
    )

    by_indexing_status: dict[str, int] = Field(
        default_factory=dict,
    )

    ready_for_indexing: int = 0
    indexed: int = 0
    failed: int = 0

    model_config = ConfigDict(
        extra="allow",
    )


class ProcessingResponse(BaseModel):
    document_id: str

    processing_status: Optional[str] = None
    ocr_status: Optional[str] = None
    indexing_status: Optional[str] = None

    page_count: Optional[int] = None
    chunk_count: Optional[int] = None

    error: Optional[str] = None

    processed_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None


class UploadResponse(BaseModel):
    document: DocumentResponse

    duplicate: bool = False

    ingestion_started: bool = False

    message: str


class MessageResponse(BaseModel):
    message: str


# ============================================================
# HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(
        obj,
        name,
        default,
    )


def _enum_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


def _optional_str(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    return str(value)


def _tenant_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "tenant_id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    return str(value)


def _organization_id(
    user: Any,
) -> Optional[str]:

    return _optional_str(
        _attr(
            user,
            "organization_id",
        )
    )


def _user_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return str(value)


def _request_id(
    request: Request,
) -> Optional[str]:

    value = (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )

    return _optional_str(value)


def _metadata(
    obj: Any,
) -> dict[str, Any]:

    return (
        _attr(
            obj,
            "document_metadata",
        )
        or _attr(
            obj,
            "metadata_json",
        )
        or {}
    )


def _serialize_document(
    document: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                document,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                document,
                "tenant_id",
                "",
            )
        ),

        "organization_id": _optional_str(
            _attr(
                document,
                "organization_id",
            )
        ),

        "uploaded_by": _optional_str(
            _attr(
                document,
                "uploaded_by",
            )
            or _attr(
                document,
                "created_by",
            )
        ),

        "name": str(
            _attr(
                document,
                "name",
                "",
            )
        ),

        "original_filename": _attr(
            document,
            "original_filename",
        ),

        "document_type": _optional_str(
            _enum_value(
                _attr(
                    document,
                    "document_type",
                )
            )
        ),

        "language": _attr(
            document,
            "language",
        ),

        "mime_type": _attr(
            document,
            "mime_type",
        ),

        "file_size": _attr(
            document,
            "file_size",
        ),

        "sha256": _attr(
            document,
            "sha256",
        ),

        "storage_key": _attr(
            document,
            "storage_key",
        ),

        "processing_status": _optional_str(
            _enum_value(
                _attr(
                    document,
                    "processing_status",
                )
            )
        ),

        "ocr_status": _optional_str(
            _enum_value(
                _attr(
                    document,
                    "ocr_status",
                )
            )
        ),

        "indexing_status": _optional_str(
            _enum_value(
                _attr(
                    document,
                    "indexing_status",
                )
            )
        ),

        "classification_confidence": _attr(
            document,
            "classification_confidence",
        ),

        "page_count": _attr(
            document,
            "page_count",
        ),

        "chunk_count": _attr(
            document,
            "chunk_count",
        ),

        "current_version": (
            _attr(
                document,
                "current_version",
            )
            or _attr(
                document,
                "version_number",
            )
        ),

        "source": _optional_str(
            _enum_value(
                _attr(
                    document,
                    "source",
                )
            )
        ),

        "metadata": _metadata(
            document
        ),

        "created_at": _attr(
            document,
            "created_at",
        ),

        "updated_at": _attr(
            document,
            "updated_at",
        ),

        "processed_at": _attr(
            document,
            "processed_at",
        ),

        "indexed_at": _attr(
            document,
            "indexed_at",
        ),
    }


def _serialize_page(
    page: Any,
) -> dict[str, Any]:

    items = list(
        _attr(
            page,
            "items",
            [],
        )
        or []
    )

    total = int(
        _attr(
            page,
            "total",
            len(items),
        )
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
    )

    return {
        "items": [
            _serialize_document(
                item
            )
            for item in items
        ],

        "total": total,
        "limit": limit,
        "offset": offset,

        "has_more": bool(
            _attr(
                page,
                "has_more",
                offset + len(items) < total,
            )
        ),
    }


def _serialize_version(
    version: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                version,
                "id",
                "",
            )
        ),

        "document_id": str(
            _attr(
                version,
                "document_id",
                "",
            )
        ),

        "version_number": int(
            _attr(
                version,
                "version_number",
                1,
            )
        ),

        "storage_key": str(
            _attr(
                version,
                "storage_key",
                "",
            )
        ),

        "filename": (
            _attr(
                version,
                "filename",
            )
            or _attr(
                version,
                "original_filename",
            )
        ),

        "mime_type": _attr(
            version,
            "mime_type",
        ),

        "file_size": _attr(
            version,
            "file_size",
        ),

        "sha256": _attr(
            version,
            "sha256",
        ),

        "change_summary": _attr(
            version,
            "change_summary",
        ),

        "metadata": (
            _attr(
                version,
                "version_metadata",
            )
            or _attr(
                version,
                "metadata_json",
            )
            or {}
        ),

        "created_at": _attr(
            version,
            "created_at",
        ),
    }


def _set_filter(
    filters: Any,
    name: str,
    value: Any,
) -> None:

    if value is None:
        return

    if hasattr(
        filters,
        name,
    ):
        setattr(
            filters,
            name,
            value,
        )


async def _service_call(
    service: Any,
    operation: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        operation,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Document operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(
        **kwargs
    )


# ============================================================
# LIST / SEARCH
# ============================================================

@router.get(
    "",
    response_model=DocumentPageResponse,
    summary="List documents",
)
async def list_documents(
    document_type: Optional[str] = None,

    processing_status: Optional[str] = None,
    ocr_status: Optional[str] = None,
    indexing_status: Optional[str] = None,

    language: Optional[str] = None,

    uploaded_by: Optional[UUID] = None,

    search: Optional[str] = Query(
        default=None,
        max_length=1000,
    ),

    created_from: Optional[
        datetime
    ] = None,

    created_to: Optional[
        datetime
    ] = None,

    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    filters = DocumentFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    _set_filter(
        filters,
        "organization_id",
        _organization_id(
            current_user
        ),
    )

    if document_type:

        if hasattr(
            filters,
            "document_types",
        ):
            filters.document_types = [
                document_type
            ]

        else:
            _set_filter(
                filters,
                "document_type",
                document_type,
            )

    if processing_status:

        if hasattr(
            filters,
            "processing_statuses",
        ):
            filters.processing_statuses = [
                processing_status
            ]

        else:
            _set_filter(
                filters,
                "processing_status",
                processing_status,
            )

    _set_filter(
        filters,
        "ocr_status",
        ocr_status,
    )

    _set_filter(
        filters,
        "indexing_status",
        indexing_status,
    )

    _set_filter(
        filters,
        "language",
        language,
    )

    _set_filter(
        filters,
        "uploaded_by",
        (
            str(uploaded_by)
            if uploaded_by
            else None
        ),
    )

    _set_filter(
        filters,
        "search",
        search,
    )

    _set_filter(
        filters,
        "created_from",
        created_from,
    )

    _set_filter(
        filters,
        "created_to",
        created_to,
    )

    page = await repository.list(
        filters,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# STATISTICS
# Keep static routes before /{document_id}
# ============================================================

@router.get(
    "/statistics",
    response_model=DocumentStatisticsResponse,
    summary="Get document statistics",
)
async def document_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> Any:

    filters = DocumentFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    _set_filter(
        filters,
        "organization_id",
        _organization_id(
            current_user
        ),
    )

    result = await repository.statistics(
        filters
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    if hasattr(
        result,
        "to_dict",
    ):
        return result.to_dict()

    return {
        "total": _attr(
            result,
            "total",
            0,
        ),

        "by_type": _attr(
            result,
            "by_type",
            {},
        ),

        "by_status": (
            _attr(
                result,
                "by_status",
            )
            or _attr(
                result,
                "by_processing_status",
                {},
            )
        ),

        "by_language": _attr(
            result,
            "by_language",
            {},
        ),

        "by_ocr_status": _attr(
            result,
            "by_ocr_status",
            {},
        ),

        "by_indexing_status": _attr(
            result,
            "by_indexing_status",
            {},
        ),

        "ready_for_indexing": _attr(
            result,
            "ready_for_indexing",
            0,
        ),

        "indexed": _attr(
            result,
            "indexed",
            0,
        ),

        "failed": _attr(
            result,
            "failed",
            0,
        ),
    }


# ============================================================
# READY FOR INDEXING
# ============================================================

@router.get(
    "/ready-for-indexing",
    response_model=DocumentPageResponse,
    summary="List documents ready for indexing",
)
async def ready_for_indexing(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_ready_for_indexing(
        tenant_id=_tenant_id(
            current_user
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# INDEXED DOCUMENTS
# ============================================================

@router.get(
    "/indexed",
    response_model=DocumentPageResponse,
    summary="List indexed documents",
)
async def indexed_documents(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_indexed(
        tenant_id=_tenant_id(
            current_user
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# UPLOAD
# ============================================================

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload legal document",
)
async def upload_document(
    request: Request,

    file: UploadFile = File(...),

    document_type: Optional[str] = Form(
        default=None
    ),

    language: Optional[str] = Form(
        default=None
    ),

    run_ocr: bool = Form(
        default=True
    ),

    auto_process: bool = Form(
        default=True
    ),

    auto_index: bool = Form(
        default=True
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_document_service
    ),
) -> Any:

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file requires a filename.",
        )

    result = await _service_call(
        service,
        "upload",

        file=file,

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        document_type=document_type,

        language=language,

        run_ocr=run_ocr,

        auto_process=auto_process,

        auto_index=auto_index,

        request_id=_request_id(
            request
        ),
    )

    document = _attr(
        result,
        "document",
        result,
    )

    return {
        "document":
            _serialize_document(
                document
            ),

        "duplicate":
            bool(
                _attr(
                    result,
                    "duplicate",
                    False,
                )
            ),

        "ingestion_started":
            bool(
                _attr(
                    result,
                    "ingestion_started",
                    auto_process,
                )
            ),

        "message":
            str(
                _attr(
                    result,
                    "message",
                    "Document uploaded successfully.",
                )
            ),
    }


# ============================================================
# CREATE DOCUMENT METADATA RECORD
# ============================================================

@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create document record",
)
async def create_document(
    payload: DocumentCreateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    try:
        data = DocumentCreate(
            tenant_id=_tenant_id(
                current_user
            ),

            organization_id=_organization_id(
                current_user
            ),

            uploaded_by=_user_id(
                current_user
            ),

            name=payload.name.strip(),

            original_filename=(
                payload.original_filename
            ),

            storage_key=(
                payload.storage_key
            ),

            mime_type=(
                payload.mime_type
            ),

            file_size=(
                payload.file_size
            ),

            sha256=payload.sha256,

            document_type=(
                payload.document_type
            ),

            language=(
                payload.language
            ),

            source=payload.source,

            metadata=(
                payload.metadata
            ),
        )

        document = (
            await repository.create(
                data
            )
        )

    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document already exists.",
        ) from exc

    return _serialize_document(
        document
    )


# ============================================================
# GET DOCUMENT
# ============================================================

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document",
)
async def get_document(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    document = await repository.get_by_id(
        document_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        )

    return _serialize_document(
        document
    )


# ============================================================
# UPDATE DOCUMENT
# ============================================================

@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document",
)
async def update_document(
    document_id: UUID,
    payload: DocumentUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    values = payload.model_dump(
        exclude_unset=True
    )

    try:
        document = await repository.update(
            document_id=document_id,

            tenant_id=_tenant_id(
                current_user
            ),

            data=DocumentUpdate(
                **values
            ),
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        ) from exc

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        )

    return _serialize_document(
        document
    )


# ============================================================
# PROCESSING STATUS
# ============================================================

@router.get(
    "/{document_id}/status",
    response_model=ProcessingResponse,
    summary="Get document processing status",
)
async def document_status(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    document = await repository.get_by_id(
        document_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        )

    return {
        "document_id":
            str(document_id),

        "processing_status":
            _optional_str(
                _enum_value(
                    _attr(
                        document,
                        "processing_status",
                    )
                )
            ),

        "ocr_status":
            _optional_str(
                _enum_value(
                    _attr(
                        document,
                        "ocr_status",
                    )
                )
            ),

        "indexing_status":
            _optional_str(
                _enum_value(
                    _attr(
                        document,
                        "indexing_status",
                    )
                )
            ),

        "page_count":
            _attr(
                document,
                "page_count",
            ),

        "chunk_count":
            _attr(
                document,
                "chunk_count",
            ),

        "error":
            (
                _attr(
                    document,
                    "processing_error",
                )
                or _attr(
                    document,
                    "error_message",
                )
            ),

        "processed_at":
            _attr(
                document,
                "processed_at",
            ),

        "indexed_at":
            _attr(
                document,
                "indexed_at",
            ),
    }


# ============================================================
# CLASSIFICATION
# ============================================================

@router.patch(
    "/{document_id}/classification",
    response_model=DocumentResponse,
    summary="Update document classification",
)
async def update_classification(
    document_id: UUID,
    payload: ClassificationUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    document = (
        await repository.update_classification(
            document_id=document_id,

            tenant_id=_tenant_id(
                current_user
            ),

            document_type=(
                payload.document_type
            ),

            confidence=(
                payload.confidence
            ),

            model_name=(
                payload.model_name
            ),

            model_version=(
                payload.model_version
            ),
        )
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        )

    return _serialize_document(
        document
    )


# ============================================================
# REPROCESS
# ============================================================

@router.post(
    "/{document_id}/reprocess",
    response_model=MessageResponse,
    summary="Reprocess document",
)
async def reprocess_document(
    document_id: UUID,
    payload: ReprocessRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_document_service
    ),
) -> dict[str, str]:

    await _service_call(
        service,
        "reprocess",

        document_id=str(
            document_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        run_ocr=(
            payload.run_ocr
        ),

        run_classification=(
            payload.run_classification
        ),

        run_extraction=(
            payload.run_extraction
        ),

        run_chunking=(
            payload.run_chunking
        ),

        run_embeddings=(
            payload.run_embeddings
        ),

        rebuild_index=(
            payload.rebuild_index
        ),

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    return {
        "message":
            "Document reprocessing started."
    }


# ============================================================
# INDEX / RAG
# ============================================================

@router.post(
    "/{document_id}/index",
    response_model=MessageResponse,
    summary="Index document for RAG",
)
async def index_document(
    document_id: UUID,
    payload: ReindexRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_document_service
    ),
) -> dict[str, str]:

    await _service_call(
        service,
        "index_document",

        document_id=str(
            document_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        rebuild_chunks=(
            payload.rebuild_chunks
        ),

        rebuild_embeddings=(
            payload.rebuild_embeddings
        ),

        delete_existing_vectors=(
            payload.delete_existing_vectors
        ),

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    return {
        "message":
            "Document indexing started."
    }


# ============================================================
# DOWNLOAD
# ============================================================

@router.get(
    "/{document_id}/download",
    summary="Download document",
)
async def download_document(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_document_service
    ),
) -> StreamingResponse:

    result = await _service_call(
        service,
        "download",

        document_id=str(
            document_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),
    )

    stream = _attr(
        result,
        "stream",
        result,
    )

    filename = (
        _attr(
            result,
            "filename",
            "document",
        )
        or "document"
    )

    mime_type = (
        _attr(
            result,
            "mime_type",
            "application/octet-stream",
        )
        or "application/octet-stream"
    )

    safe_filename = (
        str(filename)
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    return StreamingResponse(
        stream,
        media_type=mime_type,
        headers={
            "Content-Disposition":
                (
                    f'attachment; '
                    f'filename="{safe_filename}"'
                ),
            "Cache-Control":
                "private, no-store",
        },
    )


# ============================================================
# DOCUMENT VERSIONS
# ============================================================

@router.get(
    "/{document_id}/versions",
    response_model=list[
        DocumentVersionResponse
    ],
    summary="List document versions",
)
async def list_document_versions(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> list[dict[str, Any]]:

    versions = (
        await repository.list_versions(
            document_id=document_id,

            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    return [
        _serialize_version(
            version
        )
        for version in versions
    ]


@router.get(
    "/{document_id}/versions/latest",
    response_model=DocumentVersionResponse,
    summary="Get latest document version",
)
async def latest_document_version(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    version = (
        await repository.latest_version(
            document_id=document_id,

            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document version was not found.",
        )

    return _serialize_version(
        version
    )


@router.post(
    "/{document_id}/versions",
    response_model=DocumentVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create document version",
)
async def create_document_version(
    document_id: UUID,
    payload: DocumentVersionRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    # Ensure document belongs to this tenant.
    document = await repository.get_by_id(
        document_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document was not found.",
        )

    version_number = (
        await repository.next_version_number(
            document_id=document_id,

            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    data = DocumentVersionCreate(
        tenant_id=_tenant_id(
            current_user
        ),

        document_id=str(
            document_id
        ),

        version_number=(
            version_number
        ),

        storage_key=(
            payload.storage_key
        ),

        filename=(
            payload.filename
        ),

        mime_type=(
            payload.mime_type
        ),

        file_size=(
            payload.file_size
        ),

        sha256=(
            payload.sha256
        ),

        change_summary=(
            payload.change_summary
        ),

        metadata=(
            payload.metadata
        ),
    )

    version = (
        await repository.create_version(
            data
        )
    )

    return _serialize_version(
        version
    )


# ============================================================
# SOFT DELETE
# ============================================================

@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
    summary="Soft delete document",
)
async def delete_document(
    document_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_document_service
    ),
) -> dict[str, str]:

    await _service_call(
        service,
        "delete",

        document_id=str(
            document_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return {
        "message":
            "Document deleted successfully."
    }


# ============================================================
# RESTORE
# ============================================================

@router.post(
    "/{document_id}/restore",
    response_model=DocumentResponse,
    summary="Restore deleted document",
)
async def restore_document(
    document_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: DocumentRepository = Depends(
        get_document_repository
    ),
) -> dict[str, Any]:

    document = await repository.restore(
        document_id=document_id,

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deleted document was not found.",
        )

    return _serialize_document(
        document
    )


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Document subsystem health",
)
async def document_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "documents",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }