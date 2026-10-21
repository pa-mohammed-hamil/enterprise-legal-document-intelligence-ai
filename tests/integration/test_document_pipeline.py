
"""
tests/integration/test_document_pipeline.py

Enterprise Legal Document Intelligence AI
=========================================

Integration tests for the document processing pipeline.

Reference interface
-------------------

DocumentPipeline(
    document_repository=...,
    file_validator=...,
    malware_scanner=...,
    document_processor=...,
    ocr_service=...,
    document_classifier=...,
    chunker=...,
    embedding_service=...,
    vector_store=...,
    audit_service=...,
)

await pipeline.process_document(
    document_id="doc_001",
    tenant_id="tenant_alpha",
    user_id="user_001",
)

Expected response:
{
    "document_id": "doc_001",
    "tenant_id": "tenant_alpha",
    "status": "completed",
    "document_type": "contract",
    "page_count": 4,
    "chunk_count": 2,
    "indexed": True,
    "metadata": {...},
}

The tests define a reference contract. Adapt names and
schemas to your production implementation.

Run:
    pytest tests/integration/test_document_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

USER_A = "user_001"
USER_B = "user_002"

DOCUMENT_A = "doc_001"
DOCUMENT_B = "doc_002"

SUPPORTED_EXTENSIONS = (
    "pdf",
    "docx",
    "doc",
    "txt",
    "rtf",
    "odt",
    "png",
    "jpg",
    "jpeg",
    "tiff",
)


# ============================================================
# PRODUCTION IMPORT
# ============================================================

def load_document_pipeline():
    """Import the actual DocumentPipeline implementation."""

    candidates = (
        "backend.app.pipelines.document_pipeline",
        "app.pipelines.document_pipeline",
    )

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        pipeline_class = getattr(
            module,
            "DocumentPipeline",
            None,
        )

        if pipeline_class is not None:
            return pipeline_class

        errors.append(
            f"{module_name}: DocumentPipeline not found"
        )

    pytest.fail(
        "Cannot import DocumentPipeline.\n"
        + "\n".join(errors)
    )


# ============================================================
# SAMPLE DOCUMENT FIXTURES
# ============================================================

@pytest.fixture
def sample_document():
    return {
        "id": DOCUMENT_A,
        "tenant_id": TENANT_A,
        "uploaded_by": USER_A,
        "filename": "master_services_agreement.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 245760,
        "storage_key": (
            "tenant_alpha/doc_001/"
            "master_services_agreement.pdf"
        ),
        "status": "uploaded",
        "metadata": {
            "source": "user_upload",
            "version": 1,
        },
    }


@pytest.fixture
def sample_text():
    return (
        "MASTER SERVICES AGREEMENT\n\n"
        "Section 1. Confidentiality.\n"
        "Both parties shall protect confidential "
        "information.\n\n"
        "Section 2. Termination.\n"
        "Either party may terminate this Agreement "
        "with thirty days written notice.\n\n"
        "Section 3. Limitation of Liability.\n"
        "Aggregate liability shall not exceed fees "
        "paid during the preceding twelve months.\n\n"
        "Section 4. Renewal.\n"
        "The Agreement automatically renews unless "
        "notice is provided sixty days before expiry."
    )


@pytest.fixture
def sample_extraction(sample_text):
    return {
        "text": sample_text,
        "page_count": 4,
        "pages": [
            {
                "page": 1,
                "text": "MASTER SERVICES AGREEMENT",
            },
            {
                "page": 2,
                "text": "Confidentiality and termination",
            },
            {
                "page": 3,
                "text": "Limitation of liability",
            },
            {
                "page": 4,
                "text": "Renewal terms",
            },
        ],
        "tables": [],
        "metadata": {
            "language": "en",
            "extraction_method": "native_pdf",
        },
    }


@pytest.fixture
def sample_classification():
    return {
        "document_type": "contract",
        "confidence": 0.98,
        "labels": [
            "legal",
            "commercial_agreement",
        ],
    }


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": DOCUMENT_A,
            "text": (
                "Both parties shall protect confidential "
                "information. Either party may terminate "
                "with thirty days written notice."
            ),
            "page": 2,
            "chunk_index": 0,
            "metadata": {
                "section": "confidentiality_termination",
            },
        },
        {
            "chunk_id": "chunk_002",
            "document_id": DOCUMENT_A,
            "text": (
                "Aggregate liability shall not exceed "
                "fees paid during the preceding twelve "
                "months. The Agreement automatically "
                "renews unless notice is provided."
            ),
            "page": 3,
            "chunk_index": 1,
            "metadata": {
                "section": "liability_renewal",
            },
        },
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]


@pytest.fixture
def expected_result():
    return {
        "document_id": DOCUMENT_A,
        "tenant_id": TENANT_A,
        "status": "completed",
        "document_type": "contract",
        "page_count": 4,
        "chunk_count": 2,
        "indexed": True,
    }


# ============================================================
# MOCK DEPENDENCIES
# ============================================================

@pytest.fixture
def mock_services(
    sample_document,
    sample_extraction,
    sample_classification,
    sample_chunks,
    sample_embeddings,
):
    repository = MagicMock()

    repository.get_document = AsyncMock(
        side_effect=lambda document_id, tenant_id: (
            deepcopy(sample_document)
            if (
                document_id == DOCUMENT_A
                and tenant_id == TENANT_A
            )
            else None
        )
    )

    repository.update_document = AsyncMock(
        return_value=True
    )

    validator = MagicMock()
    validator.validate = AsyncMock(
        return_value={
            "valid": True,
            "mime_type": "application/pdf",
            "size_bytes": 245760,
        }
    )

    scanner = MagicMock()
    scanner.scan = AsyncMock(
        return_value={
            "clean": True,
            "threats": [],
        }
    )

    processor = MagicMock()
    processor.extract = AsyncMock(
        return_value=deepcopy(sample_extraction)
    )

    ocr = MagicMock()
    ocr.extract = AsyncMock(
        return_value=deepcopy(sample_extraction)
    )

    classifier = MagicMock()
    classifier.classify = AsyncMock(
        return_value=deepcopy(sample_classification)
    )

    chunker = MagicMock()
    chunker.chunk = AsyncMock(
        return_value=deepcopy(sample_chunks)
    )

    embedding_service = MagicMock()
    embedding_service.embed = AsyncMock(
        return_value=deepcopy(sample_embeddings)
    )

    vector_store = MagicMock()
    vector_store.upsert = AsyncMock(
        return_value={
            "indexed": True,
            "count": 2,
        }
    )

    audit = MagicMock()
    audit.record_event = AsyncMock(
        return_value=True
    )

    return {
        "document_repository": repository,
        "file_validator": validator,
        "malware_scanner": scanner,
        "document_processor": processor,
        "ocr_service": ocr,
        "document_classifier": classifier,
        "chunker": chunker,
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "audit_service": audit,
    }


# ============================================================
# PIPELINE FIXTURE
# ============================================================

@pytest.fixture
def pipeline(mock_services):
    pipeline_class = load_document_pipeline()

    try:
        return pipeline_class(**mock_services)
    except TypeError as exc:
        pytest.fail(
            "DocumentPipeline constructor does not "
            "match the expected interface: "
            f"{exc}"
        )


# ============================================================
# EXECUTION HELPERS
# ============================================================

async def execute_pipeline(
    pipeline,
    *,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    result = pipeline.process_document(
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    if inspect.isawaitable(result):
        return await result

    return result


def assert_valid_result(result):
    assert isinstance(result, dict)

    assert result["document_id"]
    assert result["tenant_id"]

    assert result["status"] in {
        "completed",
        "failed",
        "partial",
        "rejected",
    }


def get_mock_call_text(mock_method):
    return " ".join(
        str(call)
        for call in mock_method.await_args_list
    )


# ============================================================
# INITIALIZATION
# ============================================================

def test_document_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert hasattr(pipeline, "process_document")
    assert callable(pipeline.process_document)


# ============================================================
# SUCCESSFUL DOCUMENT PROCESSING
# ============================================================

@pytest.mark.asyncio
async def test_document_pipeline_success(
    pipeline,
    expected_result,
):
    result = await execute_pipeline(pipeline)

    assert_valid_result(result)

    for field, value in expected_result.items():
        assert result[field] == value


@pytest.mark.asyncio
async def test_pipeline_preserves_document_identity(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["document_id"] == DOCUMENT_A
    assert result["tenant_id"] == TENANT_A


@pytest.mark.asyncio
async def test_pipeline_returns_document_metadata(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert isinstance(result["metadata"], dict)


# ============================================================
# REPOSITORY ACCESS
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_loads_document(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "document_repository"
    ].get_document.assert_awaited_once_with(
        DOCUMENT_A,
        TENANT_A,
    )


@pytest.mark.asyncio
async def test_pipeline_updates_document_status(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "document_repository"
    ].update_document.assert_awaited()


# ============================================================
# FILE VALIDATION
# ============================================================

@pytest.mark.asyncio
async def test_file_validator_is_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "file_validator"
    ].validate.assert_awaited()


@pytest.mark.asyncio
async def test_invalid_file_is_rejected(
    pipeline,
    mock_services,
):
    mock_services[
        "file_validator"
    ].validate.return_value = {
        "valid": False,
        "reason": "Unsupported file type",
    }

    try:
        result = await execute_pipeline(pipeline)
    except (ValueError, TypeError):
        pass
    else:
        assert result["status"] in {
            "rejected",
            "failed",
        }

    mock_services[
        "document_processor"
    ].extract.assert_not_awaited()

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_document_rejected(
    pipeline,
    mock_services,
):
    mock_services[
        "file_validator"
    ].validate.return_value = {
        "valid": False,
        "reason": "File size exceeds limit",
    }

    try:
        result = await execute_pipeline(pipeline)
    except (ValueError, TypeError):
        pass
    else:
        assert result["status"] in {
            "rejected",
            "failed",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# MALWARE SCANNING
# ============================================================

@pytest.mark.asyncio
async def test_malware_scanner_is_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "malware_scanner"
    ].scan.assert_awaited()


@pytest.mark.asyncio
async def test_malicious_document_is_blocked(
    pipeline,
    mock_services,
):
    mock_services[
        "malware_scanner"
    ].scan.return_value = {
        "clean": False,
        "threats": [
            "Suspicious embedded executable",
        ],
    }

    try:
        result = await execute_pipeline(pipeline)
    except (ValueError, PermissionError):
        pass
    else:
        assert result["status"] in {
            "rejected",
            "failed",
        }

    mock_services[
        "document_processor"
    ].extract.assert_not_awaited()

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# TEXT EXTRACTION
# ============================================================

@pytest.mark.asyncio
async def test_document_text_extraction(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "document_processor"
    ].extract.assert_awaited()


@pytest.mark.asyncio
async def test_extracted_page_count(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["page_count"] == 4


@pytest.mark.asyncio
async def test_empty_document_handling(
    pipeline,
    mock_services,
    sample_extraction,
):
    extraction = deepcopy(sample_extraction)

    extraction["text"] = ""
    extraction["pages"] = []

    mock_services[
        "document_processor"
    ].extract.return_value = extraction

    try:
        result = await execute_pipeline(pipeline)
    except ValueError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "rejected",
            "partial",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# OCR
# ============================================================

@pytest.mark.asyncio
async def test_scanned_document_uses_ocr(
    pipeline,
    mock_services,
    sample_extraction,
):
    extraction = deepcopy(sample_extraction)

    extraction["text"] = ""
    extraction["metadata"]["extraction_method"] = (
        "scanned_pdf"
    )

    mock_services[
        "document_processor"
    ].extract.return_value = extraction

    await execute_pipeline(pipeline)

    mock_services[
        "ocr_service"
    ].extract.assert_awaited()


@pytest.mark.asyncio
async def test_native_text_does_not_require_ocr(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "ocr_service"
    ].extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocr_failure_is_handled(
    pipeline,
    mock_services,
    sample_extraction,
):
    extraction = deepcopy(sample_extraction)
    extraction["text"] = ""

    mock_services[
        "document_processor"
    ].extract.return_value = extraction

    mock_services[
        "ocr_service"
    ].extract.side_effect = RuntimeError(
        "OCR engine unavailable"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "partial",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

@pytest.mark.asyncio
async def test_document_classifier_is_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "document_classifier"
    ].classify.assert_awaited()


@pytest.mark.asyncio
async def test_document_classification_result(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["document_type"] == "contract"


@pytest.mark.asyncio
async def test_low_confidence_classification(
    pipeline,
    mock_services,
):
    mock_services[
        "document_classifier"
    ].classify.return_value = {
        "document_type": "unknown",
        "confidence": 0.25,
        "labels": [],
    }

    result = await execute_pipeline(pipeline)

    assert result["document_type"] in {
        "unknown",
        "unclassified",
    }


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

@pytest.mark.asyncio
async def test_chunker_is_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "chunker"
    ].chunk.assert_awaited()


@pytest.mark.asyncio
async def test_chunk_count(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["chunk_count"] == 2


@pytest.mark.asyncio
async def test_empty_chunks_are_not_indexed(
    pipeline,
    mock_services,
):
    mock_services[
        "chunker"
    ].chunk.return_value = []

    try:
        result = await execute_pipeline(pipeline)
    except ValueError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "partial",
            "rejected",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# EMBEDDINGS
# ============================================================

@pytest.mark.asyncio
async def test_embedding_service_is_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "embedding_service"
    ].embed.assert_awaited()


@pytest.mark.asyncio
async def test_embedding_failure_prevents_indexing(
    pipeline,
    mock_services,
):
    mock_services[
        "embedding_service"
    ].embed.side_effect = RuntimeError(
        "Embedding provider unavailable"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "partial",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_count_mismatch(
    pipeline,
    mock_services,
):
    mock_services[
        "embedding_service"
    ].embed.return_value = [
        [0.1, 0.2, 0.3, 0.4]
    ]

    try:
        result = await execute_pipeline(pipeline)
    except ValueError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "partial",
        }

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


# ============================================================
# VECTOR INDEXING
# ============================================================

@pytest.mark.asyncio
async def test_vector_indexing_called(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "vector_store"
    ].upsert.assert_awaited()


@pytest.mark.asyncio
async def test_document_indexed_successfully(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["indexed"] is True


@pytest.mark.asyncio
async def test_vector_index_failure(
    pipeline,
    mock_services,
):
    mock_services[
        "vector_store"
    ].upsert.side_effect = RuntimeError(
        "Vector database unavailable"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        pass
    else:
        assert result["status"] in {
            "failed",
            "partial",
        }

    assert mock_services[
        "vector_store"
    ].upsert.await_count >= 1


# ============================================================
# TENANT ISOLATION
# ============================================================

@pytest.mark.asyncio
async def test_document_lookup_uses_tenant_scope(
    pipeline,
    mock_services,
):
    await execute_pipeline(
        pipeline,
        tenant_id=TENANT_A,
    )

    mock_services[
        "document_repository"
    ].get_document.assert_awaited_once_with(
        DOCUMENT_A,
        TENANT_A,
    )


@pytest.mark.asyncio
async def test_cross_tenant_document_access_denied(
    pipeline,
    mock_services,
):
    try:
        result = await execute_pipeline(
            pipeline,
            document_id=DOCUMENT_A,
            tenant_id=TENANT_B,
            user_id=USER_B,
        )
    except (
        PermissionError,
        ValueError,
        LookupError,
    ):
        pass
    else:
        assert result["status"] in {
            "failed",
            "rejected",
        }

    mock_services[
        "document_processor"
    ].extract.assert_not_awaited()

    mock_services[
        "vector_store"
    ].upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_indexing_contains_tenant_context(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    call_text = get_mock_call_text(
        mock_services["vector_store"].upsert
    )

    assert TENANT_A in call_text


# ============================================================
# AUDIT LOGGING
# ============================================================

@pytest.mark.asyncio
async def test_successful_processing_is_audited(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    mock_services[
        "audit_service"
    ].record_event.assert_awaited()


@pytest.mark.asyncio
async def test_audit_contains_document_context(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    call_text = get_mock_call_text(
        mock_services[
            "audit_service"
        ].record_event
    )

    assert DOCUMENT_A in call_text
    assert TENANT_A in call_text


# ============================================================
# INPUT VALIDATION
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document_id",
    ["", " ", None],
)
async def test_invalid_document_id(
    pipeline,
    document_id,
):
    try:
        result = await execute_pipeline(
            pipeline,
            document_id=document_id,
        )
    except (ValueError, TypeError):
        return

    assert result["status"] in {
        "failed",
        "rejected",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tenant_id",
    ["", " ", None],
)
async def test_invalid_tenant_id(
    pipeline,
    tenant_id,
):
    try:
        result = await execute_pipeline(
            pipeline,
            tenant_id=tenant_id,
        )
    except (ValueError, TypeError):
        return

    assert result["status"] in {
        "failed",
        "rejected",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id",
    ["", " ", None],
)
async def test_invalid_user_id(
    pipeline,
    user_id,
):
    try:
        result = await execute_pipeline(
            pipeline,
            user_id=user_id,
        )
    except (ValueError, TypeError):
        return

    assert result["status"] in {
        "failed",
        "rejected",
    }


# ============================================================
# PROCESSING ORDER
# ============================================================

@pytest.mark.asyncio
async def test_security_checks_precede_extraction(
    pipeline,
    mock_services,
):
    events = []

    async def validate(*args, **kwargs):
        events.append("validate")
        return {"valid": True}

    async def scan(*args, **kwargs):
        events.append("scan")
        return {"clean": True, "threats": []}

    original_extraction = deepcopy(
        mock_services[
            "document_processor"
        ].extract.return_value
    )

    async def extract(*args, **kwargs):
        events.append("extract")
        return original_extraction

    mock_services[
        "file_validator"
    ].validate.side_effect = validate

    mock_services[
        "malware_scanner"
    ].scan.side_effect = scan

    mock_services[
        "document_processor"
    ].extract.side_effect = extract

    await execute_pipeline(pipeline)

    assert "validate" in events
    assert "scan" in events
    assert "extract" in events

    assert events.index("validate") < events.index(
        "extract"
    )

    assert events.index("scan") < events.index(
        "extract"
    )


@pytest.mark.asyncio
async def test_embedding_precedes_vector_indexing(
    pipeline,
    mock_services,
):
    events = []

    original_embeddings = deepcopy(
        mock_services[
            "embedding_service"
        ].embed.return_value
    )

    async def embed(*args, **kwargs):
        events.append("embed")
        return original_embeddings

    async def upsert(*args, **kwargs):
        events.append("upsert")
        return {
            "indexed": True,
            "count": 2,
        }

    mock_services[
        "embedding_service"
    ].embed.side_effect = embed

    mock_services[
        "vector_store"
    ].upsert.side_effect = upsert

    await execute_pipeline(pipeline)

    assert events.index("embed") < events.index(
        "upsert"
    )


# ============================================================
# CONCURRENT EXECUTION
# ============================================================

@pytest.mark.asyncio
async def test_concurrent_document_processing(
    pipeline,
    mock_services,
    sample_document,
):
    async def get_document(
        document_id,
        tenant_id,
    ):
        return {
            **deepcopy(sample_document),
            "id": document_id,
            "tenant_id": tenant_id,
        }

    mock_services[
        "document_repository"
    ].get_document.side_effect = get_document

    tasks = [
        execute_pipeline(
            pipeline,
            document_id=f"doc_{index:03d}",
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
        for index in range(5)
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 5

    for index, result in enumerate(results):
        assert result["document_id"] == (
            f"doc_{index:03d}"
        )

        assert result["tenant_id"] == TENANT_A


# ============================================================
# SERIALIZATION
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_result_is_json_serializable(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    serialized = json.dumps(result)

    decoded = json.loads(serialized)

    assert decoded["document_id"] == DOCUMENT_A
    assert decoded["tenant_id"] == TENANT_A


# ============================================================
# SOURCE TRACEABILITY
# ============================================================

@pytest.mark.asyncio
async def test_chunk_metadata_is_preserved_for_indexing(
    pipeline,
    mock_services,
):
    await execute_pipeline(pipeline)

    call_text = get_mock_call_text(
        mock_services["vector_store"].upsert
    )

    assert (
        "chunk_001" in call_text
        or "chunk_002" in call_text
    )

    assert DOCUMENT_A in call_text


# ============================================================
# PIPELINE FAILURE DOES NOT CLAIM SUCCESS
# ============================================================

@pytest.mark.asyncio
async def test_failed_indexing_not_reported_as_complete(
    pipeline,
    mock_services,
):
    mock_services[
        "vector_store"
    ].upsert.return_value = {
        "indexed": False,
        "count": 0,
    }

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        return

    assert result["status"] != "completed"
    assert result["indexed"] is False


# ============================================================
# END OF TEST SUITE
# ============================================================
