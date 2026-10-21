
"""
tests/unit/test_documents.py

Lexora AI — Document Management Unit Tests

Expected module:
    backend.app.services.document_service

Expected class:
    DocumentService

Expected constructor:
    DocumentService(
        document_repository=...,
        storage_service=...,
        permission_service=...,
        audit_service=...,
    )

Expected async methods:
    create_document(
        tenant_id: str,
        user_id: str,
        filename: str,
        content: bytes,
        metadata: dict | None = None,
    ) -> dict

    get_document(
        document_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict

    list_documents(
        tenant_id: str,
        user_id: str,
        filters: dict | None = None,
    ) -> list[dict]

    update_metadata(
        document_id: str,
        tenant_id: str,
        user_id: str,
        metadata: dict,
    ) -> dict

    update_processing_status(
        document_id: str,
        tenant_id: str,
        status: str,
    ) -> dict

    create_version(
        document_id: str,
        tenant_id: str,
        user_id: str,
        content: bytes,
    ) -> dict

    delete_document(
        document_id: str,
        tenant_id: str,
        user_id: str,
    ) -> bool

Expected document:
    {
        "id": str,
        "tenant_id": str,
        "owner_id": str,
        "filename": str,
        "content_type": str,
        "storage_key": str,
        "file_size": int,
        "version": int,
        "processing_status": str,
        "metadata": dict,
        "deleted_at": datetime | None,
    }

Security requirements:
    - Validate tenant and user identities.
    - Check permissions before document operations.
    - Enforce tenant isolation.
    - Validate file extensions and size limits.
    - Reject empty and malformed uploads.
    - Prevent path traversal.
    - Never trust caller-provided storage paths.
    - Prevent unauthorized metadata changes.
    - Preserve document version history.
    - Audit document modifications.
    - Soft-delete documents.
    - Prevent cross-tenant data leakage.

These tests define a reference contract.
They have not been executed against the actual backend.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

USER_A = "user_alpha"
USER_B = "user_beta"
VIEWER_A = "viewer_alpha"

DOCUMENT_A = "document_alpha"
DOCUMENT_B = "document_beta"

MAX_FILE_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
    ".odt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
}

VALID_PROCESSING_STATUSES = {
    "pending",
    "processing",
    "completed",
    "failed",
}

PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog >>\n"
    b"endobj\n"
    b"%%EOF\n"
)

TEXT_CONTENT = (
    b"Master Service Agreement\n"
    b"Termination requires thirty days notice.\n"
)


# ============================================================
# IMPORT
# ============================================================

def load_document_service_class():
    candidates = (
        "backend.app.services.document_service",
        "backend.app.services.documents",
        "backend.app.documents.service",
        "app.services.document_service",
        "app.services.documents",
        "app.documents.service",
    )

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(str(exc))
            continue

        service_class = getattr(
            module,
            "DocumentService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "DocumentService could not be imported. "
        "Update the import path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# DOCUMENT FACTORY
# ============================================================

def make_document(
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    owner_id=USER_A,
    filename="contract.pdf",
    content_type="application/pdf",
    storage_key=None,
    file_size=len(PDF_CONTENT),
    version=1,
    processing_status="pending",
    metadata=None,
    deleted_at=None,
):
    return {
        "id": document_id,
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "filename": filename,
        "content_type": content_type,
        "storage_key": (
            storage_key
            or f"{tenant_id}/documents/{document_id}/v{version}"
        ),
        "file_size": file_size,
        "version": version,
        "processing_status": processing_status,
        "metadata": (
            deepcopy(metadata)
            if metadata is not None
            else {
                "department": "legal",
                "classification": "confidential",
                "source": filename,
            }
        ),
        "deleted_at": deleted_at,
    }


# ============================================================
# FAKE DOCUMENT REPOSITORY
# ============================================================

class FakeDocumentRepository:

    def __init__(self):
        self.documents = {
            DOCUMENT_A: make_document(),
            DOCUMENT_B: make_document(
                document_id=DOCUMENT_B,
                tenant_id=TENANT_B,
                owner_id=USER_B,
                filename="private_contract.pdf",
            ),
        }

        self.versions = {
            DOCUMENT_A: [deepcopy(self.documents[DOCUMENT_A])],
            DOCUMENT_B: [deepcopy(self.documents[DOCUMENT_B])],
        }

    async def create(self, data):
        document_id = data.get(
            "id",
            f"document_{uuid.uuid4().hex}",
        )

        if document_id in self.documents:
            raise ValueError("Duplicate document ID")

        document = {
            **deepcopy(data),
            "id": document_id,
            "deleted_at": None,
        }

        self.documents[document_id] = document
        self.versions[document_id] = [
            deepcopy(document)
        ]

        return deepcopy(document)

    async def get_by_id(
        self,
        document_id,
        tenant_id,
    ):
        document = self.documents.get(document_id)

        if document is None:
            return None

        if document["tenant_id"] != tenant_id:
            return None

        if document["deleted_at"] is not None:
            return None

        return deepcopy(document)

    async def list_by_tenant(self, tenant_id):
        return [
            deepcopy(document)
            for document in self.documents.values()
            if (
                document["tenant_id"] == tenant_id
                and document["deleted_at"] is None
            )
        ]

    async def update(
        self,
        document_id,
        tenant_id,
        changes,
    ):
        document = self.documents.get(document_id)

        if (
            document is None
            or document["tenant_id"] != tenant_id
            or document["deleted_at"] is not None
        ):
            raise LookupError("Document not found")

        document.update(deepcopy(changes))

        return deepcopy(document)

    async def add_version(
        self,
        document_id,
        tenant_id,
        version_data,
    ):
        document = self.documents.get(document_id)

        if (
            document is None
            or document["tenant_id"] != tenant_id
            or document["deleted_at"] is not None
        ):
            raise LookupError("Document not found")

        new_version = deepcopy(version_data)

        self.versions[document_id].append(
            new_version
        )

        document.update(new_version)

        return deepcopy(document)

    async def soft_delete(
        self,
        document_id,
        tenant_id,
    ):
        document = self.documents.get(document_id)

        if (
            document is None
            or document["tenant_id"] != tenant_id
            or document["deleted_at"] is not None
        ):
            return False

        document["deleted_at"] = datetime.now(
            timezone.utc
        )

        return True


# ============================================================
# FAKE STORAGE SERVICE
# ============================================================

class FakeStorageService:

    def __init__(self):
        self.files = {}

    async def upload(
        self,
        tenant_id,
        document_id,
        content,
        filename,
        version=1,
    ):
        key = (
            f"{tenant_id}/documents/"
            f"{document_id}/v{version}/"
            f"{Path(filename).name}"
        )

        self.files[key] = bytes(content)

        return {
            "storage_key": key,
            "file_size": len(content),
        }

    async def download(self, storage_key):
        if storage_key not in self.files:
            raise FileNotFoundError(storage_key)

        return self.files[storage_key]

    async def delete(self, storage_key):
        return self.files.pop(
            storage_key,
            None,
        ) is not None


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def document_repository():
    return FakeDocumentRepository()


@pytest.fixture
def storage_service():
    return FakeStorageService()


@pytest.fixture
def permission_service():
    service = MagicMock()

    permissions = {
        (USER_A, TENANT_A): {
            "create",
            "read",
            "update",
            "delete",
            "version",
        },
        (USER_B, TENANT_B): {
            "create",
            "read",
            "update",
            "delete",
            "version",
        },
        (VIEWER_A, TENANT_A): {
            "read",
        },
    }

    async def has_permission(
        user_id,
        tenant_id,
        action,
        resource_id=None,
    ):
        return action in permissions.get(
            (user_id, tenant_id),
            set(),
        )

    service.has_permission = AsyncMock(
        side_effect=has_permission
    )

    return service


@pytest.fixture
def audit_service():
    service = MagicMock()

    service.record_event = AsyncMock(
        return_value=None
    )

    return service


@pytest.fixture
def document_service(
    document_repository,
    storage_service,
    permission_service,
    audit_service,
):
    service_class = load_document_service_class()

    return service_class(
        document_repository=document_repository,
        storage_service=storage_service,
        permission_service=permission_service,
        audit_service=audit_service,
    )


# ============================================================
# HELPERS
# ============================================================

async def invoke(method, **kwargs):
    result = method(**kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


async def create_document(
    service,
    filename="contract.pdf",
    content=PDF_CONTENT,
    tenant_id=TENANT_A,
    user_id=USER_A,
    metadata=None,
):
    return await invoke(
        service.create_document,
        tenant_id=tenant_id,
        user_id=user_id,
        filename=filename,
        content=content,
        metadata=metadata,
    )


async def get_document(
    service,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.get_document,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


async def list_documents(
    service,
    tenant_id=TENANT_A,
    user_id=USER_A,
    filters=None,
):
    return await invoke(
        service.list_documents,
        tenant_id=tenant_id,
        user_id=user_id,
        filters=filters,
    )


async def update_metadata(
    service,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
    metadata=None,
):
    return await invoke(
        service.update_metadata,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata=metadata or {
            "department": "compliance",
        },
    )


async def update_status(
    service,
    status,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
):
    return await invoke(
        service.update_processing_status,
        document_id=document_id,
        tenant_id=tenant_id,
        status=status,
    )


async def create_version(
    service,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
    content=TEXT_CONTENT,
):
    return await invoke(
        service.create_version,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        content=content,
    )


async def delete_document(
    service,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.delete_document,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


def assert_valid_document(document):
    assert isinstance(document, dict)

    required = {
        "id",
        "tenant_id",
        "owner_id",
        "filename",
        "content_type",
        "storage_key",
        "file_size",
        "version",
        "processing_status",
        "metadata",
        "deleted_at",
    }

    assert required.issubset(document.keys())

    assert document["id"]
    assert document["tenant_id"]
    assert document["filename"]
    assert document["storage_key"]

    assert document["file_size"] > 0
    assert document["version"] >= 1

    assert document["processing_status"] in (
        VALID_PROCESSING_STATUSES
    )

    assert isinstance(document["metadata"], dict)


# ============================================================
# INITIALIZATION
# ============================================================

async def test_document_service_initialization(
    document_service,
):
    assert document_service is not None

    assert callable(document_service.create_document)
    assert callable(document_service.get_document)
    assert callable(document_service.list_documents)
    assert callable(document_service.update_metadata)
    assert callable(
        document_service.update_processing_status
    )
    assert callable(document_service.create_version)
    assert callable(document_service.delete_document)


# ============================================================
# DOCUMENT CREATION
# ============================================================

async def test_create_pdf_document(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert_valid_document(result)


async def test_create_text_document(
    document_service,
):
    result = await create_document(
        document_service,
        filename="agreement.txt",
        content=TEXT_CONTENT,
    )

    assert result["filename"] == "agreement.txt"


async def test_document_owner_preserved(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert result["owner_id"] == USER_A


async def test_document_tenant_preserved(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert result["tenant_id"] == TENANT_A


async def test_document_default_version(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert result["version"] == 1


async def test_document_default_processing_status(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert result["processing_status"] == "pending"


async def test_document_file_size(
    document_service,
):
    result = await create_document(
        document_service
    )

    assert result["file_size"] == len(PDF_CONTENT)


async def test_unique_document_ids(
    document_service,
):
    first = await create_document(
        document_service
    )

    second = await create_document(
        document_service
    )

    assert first["id"] != second["id"]


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

@pytest.mark.parametrize(
    "filename,content",
    [
        ("contract.pdf", PDF_CONTENT),
        ("agreement.txt", TEXT_CONTENT),
        ("contract.docx", b"PK\x03\x04test-docx"),
        ("contract.doc", b"\xd0\xcf\x11\xe0test-doc"),
        ("contract.rtf", b"{\\rtf1 test}"),
        ("contract.odt", b"PK\x03\x04test-odt"),
        ("scan.png", b"\x89PNG\r\n\x1a\nfake"),
        ("scan.jpg", b"\xff\xd8\xfffake"),
        ("scan.jpeg", b"\xff\xd8\xfffake"),
        ("scan.tiff", b"II*\x00fake"),
    ],
)
async def test_supported_extensions(
    document_service,
    filename,
    content,
):
    # These byte strings are format signatures, not
    # complete valid files. This test specifies
    # extension acceptance only; real parser tests
    # must use valid format fixtures.
    result = await create_document(
        document_service,
        filename=filename,
        content=content,
    )

    assert result["filename"] == filename


@pytest.mark.parametrize(
    "filename",
    [
        "malware.exe",
        "script.js",
        "payload.sh",
        "archive.zip",
        "unknown.xyz",
    ],
)
async def test_unsupported_extensions_rejected(
    document_service,
    filename,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_document(
            document_service,
            filename=filename,
            content=b"test content",
        )


# ============================================================
# UPLOAD VALIDATION
# ============================================================

async def test_empty_file_rejected(
    document_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_document(
            document_service,
            content=b"",
        )


async def test_none_content_rejected(
    document_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_document(
            document_service,
            content=None,
        )


async def test_file_size_limit(
    document_service,
):
    oversized = b"x" * (MAX_FILE_SIZE + 1)

    with pytest.raises(
        (ValueError, OverflowError)
    ):
        await create_document(
            document_service,
            content=oversized,
        )


@pytest.mark.parametrize(
    "filename",
    [
        "../private.pdf",
        "../../secrets.pdf",
        "/etc/passwd",
        r"..\..\private.pdf",
    ],
)
async def test_path_traversal_rejected(
    document_service,
    filename,
):
    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await create_document(
            document_service,
            filename=filename,
        )


async def test_empty_filename_rejected(
    document_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_document(
            document_service,
            filename="",
        )


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

async def test_get_document(
    document_service,
):
    result = await get_document(
        document_service
    )

    assert_valid_document(result)
    assert result["id"] == DOCUMENT_A


async def test_created_document_persisted(
    document_service,
):
    created = await create_document(
        document_service
    )

    retrieved = await get_document(
        document_service,
        document_id=created["id"],
    )

    assert retrieved["id"] == created["id"]


async def test_nonexistent_document(
    document_service,
):
    try:
        result = await get_document(
            document_service,
            document_id="missing_document",
        )
    except (LookupError, ValueError):
        return

    assert result is None


# ============================================================
# DOCUMENT LISTING
# ============================================================

async def test_list_documents(
    document_service,
):
    results = await list_documents(
        document_service
    )

    assert isinstance(results, list)
    assert len(results) >= 1


async def test_list_documents_tenant_scoped(
    document_service,
):
    results = await list_documents(
        document_service
    )

    assert all(
        item["tenant_id"] == TENANT_A
        for item in results
    )


async def test_document_filter_by_status(
    document_service,
):
    results = await list_documents(
        document_service,
        filters={
            "processing_status": "pending",
        },
    )

    assert all(
        item["processing_status"] == "pending"
        for item in results
    )


async def test_document_filter_by_department(
    document_service,
):
    results = await list_documents(
        document_service,
        filters={
            "department": "legal",
        },
    )

    assert all(
        item["metadata"].get("department") == "legal"
        for item in results
    )


# ============================================================
# METADATA
# ============================================================

async def test_create_with_metadata(
    document_service,
):
    metadata = {
        "department": "finance",
        "classification": "restricted",
    }

    result = await create_document(
        document_service,
        metadata=metadata,
    )

    assert result["metadata"] == metadata


async def test_update_metadata(
    document_service,
):
    result = await update_metadata(
        document_service,
        metadata={
            "department": "compliance",
        },
    )

    assert result["metadata"]["department"] == (
        "compliance"
    )


async def test_metadata_update_persisted(
    document_service,
):
    await update_metadata(
        document_service,
        metadata={
            "department": "legal_operations",
        },
    )

    result = await get_document(
        document_service
    )

    assert result["metadata"]["department"] == (
        "legal_operations"
    )


async def test_metadata_cannot_override_tenant(
    document_service,
):
    try:
        result = await update_metadata(
            document_service,
            metadata={
                "tenant_id": TENANT_B,
            },
        )
    except (ValueError, PermissionError):
        return

    assert result["tenant_id"] == TENANT_A


# ============================================================
# PROCESSING STATUS
# ============================================================

async def test_status_pending_to_processing(
    document_service,
):
    result = await update_status(
        document_service,
        status="processing",
    )

    assert result["processing_status"] == "processing"


async def test_status_processing_to_completed(
    document_service,
    document_repository,
):
    document_repository.documents[
        DOCUMENT_A
    ]["processing_status"] = "processing"

    result = await update_status(
        document_service,
        status="completed",
    )

    assert result["processing_status"] == "completed"


async def test_status_processing_to_failed(
    document_service,
    document_repository,
):
    document_repository.documents[
        DOCUMENT_A
    ]["processing_status"] = "processing"

    result = await update_status(
        document_service,
        status="failed",
    )

    assert result["processing_status"] == "failed"


async def test_invalid_processing_status(
    document_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await update_status(
            document_service,
            status="unknown_status",
        )


# ============================================================
# VERSIONING
# ============================================================

async def test_create_document_version(
    document_service,
):
    result = await create_version(
        document_service
    )

    assert result["version"] == 2


async def test_version_increment(
    document_service,
):
    first = await create_version(
        document_service
    )

    second = await create_version(
        document_service,
        content=b"Third document version",
    )

    assert first["version"] == 2
    assert second["version"] == 3


async def test_version_history_preserved(
    document_service,
    document_repository,
):
    await create_version(
        document_service
    )

    versions = document_repository.versions[
        DOCUMENT_A
    ]

    assert len(versions) >= 2


async def test_version_preserves_document_identity(
    document_service,
):
    result = await create_version(
        document_service
    )

    assert result["id"] == DOCUMENT_A
    assert result["tenant_id"] == TENANT_A


async def test_version_storage_key_changes(
    document_service,
):
    before = await get_document(
        document_service
    )

    after = await create_version(
        document_service
    )

    assert (
        after["storage_key"]
        != before["storage_key"]
    )


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_cross_tenant_document_read_rejected(
    document_service,
):
    try:
        result = await get_document(
            document_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result is None


async def test_cross_tenant_metadata_update_rejected(
    document_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await update_metadata(
            document_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )


async def test_cross_tenant_version_rejected(
    document_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await create_version(
            document_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )


async def test_cross_tenant_delete_rejected(
    document_service,
):
    try:
        result = await delete_document(
            document_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result is False


async def test_tenant_b_can_read_own_document(
    document_service,
):
    result = await get_document(
        document_service,
        document_id=DOCUMENT_B,
        tenant_id=TENANT_B,
        user_id=USER_B,
    )

    assert result["tenant_id"] == TENANT_B


# ============================================================
# PERMISSIONS
# ============================================================

async def test_permission_checked_on_create(
    document_service,
    permission_service,
):
    await create_document(
        document_service
    )

    permission_service.has_permission.assert_awaited()


async def test_viewer_can_read(
    document_service,
):
    result = await get_document(
        document_service,
        user_id=VIEWER_A,
    )

    assert result["id"] == DOCUMENT_A


async def test_viewer_cannot_create(
    document_service,
):
    with pytest.raises(PermissionError):
        await create_document(
            document_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_update_metadata(
    document_service,
):
    with pytest.raises(PermissionError):
        await update_metadata(
            document_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_create_version(
    document_service,
):
    with pytest.raises(PermissionError):
        await create_version(
            document_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_delete(
    document_service,
):
    with pytest.raises(PermissionError):
        await delete_document(
            document_service,
            user_id=VIEWER_A,
        )


async def test_permission_failure_fails_closed(
    document_service,
    permission_service,
):
    permission_service.has_permission = AsyncMock(
        side_effect=RuntimeError(
            "Permission backend unavailable"
        )
    )

    with pytest.raises(
        (RuntimeError, PermissionError)
    ):
        await create_document(
            document_service
        )


# ============================================================
# SOFT DELETION
# ============================================================

async def test_delete_document(
    document_service,
):
    result = await delete_document(
        document_service
    )

    assert result is True


async def test_deleted_document_hidden(
    document_service,
):
    await delete_document(
        document_service
    )

    try:
        result = await get_document(
            document_service
        )
    except (LookupError, ValueError):
        return

    assert result is None


async def test_deleted_document_not_listed(
    document_service,
):
    await delete_document(
        document_service
    )

    results = await list_documents(
        document_service
    )

    assert DOCUMENT_A not in {
        item["id"]
        for item in results
    }


async def test_soft_delete_preserves_record(
    document_service,
    document_repository,
):
    await delete_document(
        document_service
    )

    assert DOCUMENT_A in (
        document_repository.documents
    )

    assert (
        document_repository.documents[
            DOCUMENT_A
        ]["deleted_at"]
        is not None
    )


# ============================================================
# AUDIT
# ============================================================

async def test_document_creation_audited(
    document_service,
    audit_service,
):
    await create_document(
        document_service
    )

    audit_service.record_event.assert_awaited()


async def test_metadata_update_audited(
    document_service,
    audit_service,
):
    await update_metadata(
        document_service
    )

    audit_service.record_event.assert_awaited()


async def test_document_deletion_audited(
    document_service,
    audit_service,
):
    await delete_document(
        document_service
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant_context(
    document_service,
    audit_service,
):
    await create_document(
        document_service
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_document_creation(
    document_service,
):
    results = await asyncio.gather(
        *[
            create_document(
                document_service,
                filename=f"contract_{index}.txt",
                content=(
                    f"Contract document {index}"
                ).encode(),
            )
            for index in range(20)
        ]
    )

    assert len(results) == 20

    ids = {
        item["id"]
        for item in results
    }

    assert len(ids) == 20


async def test_concurrent_tenant_reads(
    document_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        get_document(
            document_service,
            document_id=DOCUMENT_A,
            tenant_id=TENANT_A,
            user_id=USER_A,
        ),
        get_document(
            document_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_B,
            user_id=USER_B,
        ),
    )

    assert tenant_a["tenant_id"] == TENANT_A
    assert tenant_b["tenant_id"] == TENANT_B


async def test_concurrent_document_listing(
    document_service,
):
    results = await asyncio.gather(
        *[
            list_documents(
                document_service,
                tenant_id=TENANT_A,
            )
            for _ in range(10)
        ]
    )

    assert len(results) == 10

    for documents in results:
        assert all(
            document["tenant_id"] == TENANT_A
            for document in documents
        )


# ============================================================
# END
# ============================================================
