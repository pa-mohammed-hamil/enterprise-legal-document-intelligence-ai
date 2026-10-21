# backend/app/schemas/document.py

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# Enums
# ============================================================


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ProcessingStage(str, Enum):
    UPLOAD = "upload"
    PARSING = "parsing"
    OCR = "ocr"
    NORMALIZATION = "normalization"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    CONTRACT = "contract"
    NDA = "nda"
    VENDOR_AGREEMENT = "vendor_agreement"
    LEASE = "lease"
    LICENSE = "license"
    SERVICE_AGREEMENT = "service_agreement"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    PURCHASE_AGREEMENT = "purchase_agreement"
    POLICY = "policy"
    AMENDMENT = "amendment"
    LEGAL_NOTICE = "legal_notice"
    OTHER = "other"


# ============================================================
# Shared base
# ============================================================


class DocumentBase(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    document_type: DocumentType = DocumentType.OTHER

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    language: str | None = Field(
        default=None,
        max_length=50,
    )

    tags: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Creation
# ============================================================


class DocumentCreate(DocumentBase):
    """
    Metadata submitted when a document is created/uploaded.

    Binary file content is normally handled by FastAPI UploadFile.
    """

    organization_id: UUID

    filename: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    content_type: str | None = Field(
        default=None,
        max_length=255,
    )

    folder: str | None = Field(
        default=None,
        max_length=500,
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("filename cannot be empty")

        if "/" in value or "\\" in value:
            raise ValueError(
                "filename must not contain path separators"
            )

        return value


class DocumentUploadRequest(BaseModel):
    """
    Form metadata accompanying UploadFile.
    """

    title: str | None = Field(
        default=None,
        max_length=500,
    )

    document_type: DocumentType = DocumentType.OTHER

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    tags: list[str] = Field(default_factory=list)

    auto_process: bool = True

    run_ocr: bool = True

    create_embeddings: bool = True

    extract_contract: bool = True


# ============================================================
# Updates
# ============================================================


class DocumentUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    document_type: DocumentType | None = None

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    language: str | None = Field(
        default=None,
        max_length=50,
    )

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None

    is_active: bool | None = None


# ============================================================
# Processing
# ============================================================


class DocumentProcessingRequest(BaseModel):
    run_ocr: bool = True

    detect_tables: bool = True

    detect_language: bool = True

    extract_metadata: bool = True

    normalize_text: bool = True

    create_chunks: bool = True

    create_embeddings: bool = True

    classify_document: bool = True

    extract_contract: bool = True

    force_reprocess: bool = False


class DocumentProcessingStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID

    status: DocumentStatus

    stage: ProcessingStage | None = None

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    message: str | None = None

    error: str | None = None

    started_at: datetime | None = None

    completed_at: datetime | None = None


# ============================================================
# OCR
# ============================================================


class OCRMetadata(BaseModel):
    provider: str | None = None

    model: str | None = None

    pages_processed: int = Field(
        default=0,
        ge=0,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    detected_language: str | None = None

    duration_ms: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# Document chunk
# ============================================================


class DocumentChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    document_id: UUID

    parent_chunk_id: UUID | None = None

    chunk_index: int

    text: str

    page_number: int | None = None

    start_page: int | None = None

    end_page: int | None = None

    start_offset: int | None = None

    end_offset: int | None = None

    token_count: int | None = None

    chunk_type: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime | None = None


# ============================================================
# Document version
# ============================================================


class DocumentVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    document_id: UUID

    version_number: int

    storage_key: str | None = None

    filename: str | None = None

    file_size: int | None = None

    checksum: str | None = None

    created_by: UUID | None = None

    created_at: datetime

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# Basic response
# ============================================================


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID

    title: str

    filename: str

    document_type: str

    description: str | None = None

    content_type: str | None = None

    file_size: int | None = None

    extension: str | None = None

    language: str | None = None

    status: str

    processing_stage: str | None = None

    processing_progress: float | None = None

    page_count: int | None = None

    word_count: int | None = None

    chunk_count: int | None = None

    tags: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    is_active: bool = True

    created_by: UUID | None = None

    created_at: datetime

    updated_at: datetime


# ============================================================
# Detailed response
# ============================================================


class DocumentDetailResponse(DocumentResponse):
    storage_key: str | None = None

    checksum: str | None = None

    extracted_text: str | None = None

    detected_language: str | None = None

    classification_confidence: float | None = None

    extraction_confidence: float | None = None

    processing_error: str | None = None

    processed_at: datetime | None = None

    ocr_metadata: OCRMetadata | None = None

    versions: list[DocumentVersionResponse] = Field(
        default_factory=list
    )

    chunks: list[DocumentChunkResponse] = Field(
        default_factory=list
    )


# ============================================================
# Lightweight summary
# ============================================================


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    title: str

    filename: str

    document_type: str

    status: str

    page_count: int | None = None

    created_at: datetime


# ============================================================
# Pagination
# ============================================================


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]

    total: int = Field(ge=0)

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
    )

    pages: int = Field(
        default=0,
        ge=0,
    )


# ============================================================
# Upload response
# ============================================================


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse

    upload_successful: bool = True

    processing_queued: bool = False

    task_id: str | None = None

    message: str = "Document uploaded successfully"


# ============================================================
# Delete/archive response
# ============================================================


class DocumentDeleteResponse(BaseModel):
    document_id: UUID

    deleted: bool = False

    archived: bool = False

    message: str


# ============================================================
# Reprocessing
# ============================================================


class DocumentReprocessResponse(BaseModel):
    document_id: UUID

    task_id: str | None = None

    status: str = "queued"

    message: str = "Document reprocessing queued"