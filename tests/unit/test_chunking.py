
"""
tests/unit/test_chunking.py

Lexora AI — Document Chunking Unit Tests

Expected module:
    backend.app.rag.chunking

Expected class:
    DocumentChunker

Expected constructor:
    DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

Expected method:
    chunk_text(
        text: str,
        metadata: dict | None = None,
    ) -> list[dict]

Expected chunk format:
    {
        "id": "chunk_001",
        "text": "Document content...",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 120,
        "metadata": {
            "document_id": "contract_001",
            "tenant_id": "tenant_alpha",
            "page": 1,
        },
    }

Contract:
    - chunk_size and chunk_overlap are measured in characters.
    - Chunks are nonempty and ordered.
    - Each chunk has valid offsets into the source.
    - Adjacent chunks overlap when required.
    - No source content is silently lost.
    - Tenant/document metadata is preserved.
    - Inputs are not mutated.
    - Chunking is deterministic.

These tests describe a reference implementation contract.
Adapt them if your chunker uses token-based splitting.
"""

from __future__ import annotations

import importlib
from copy import deepcopy

import pytest


# ============================================================
# CONSTANTS
# ============================================================

TENANT_ID = "tenant_alpha"
DOCUMENT_ID = "contract_001"

DEFAULT_CHUNK_SIZE = 100
DEFAULT_OVERLAP = 20


# ============================================================
# IMPORT
# ============================================================

def load_chunker_class():
    candidates = (
        "backend.app.rag.chunking",
        "app.rag.chunking",
    )

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(str(exc))
            continue

        chunker_class = getattr(
            module,
            "DocumentChunker",
            None,
        )

        if chunker_class is not None:
            return chunker_class

    pytest.fail(
        "DocumentChunker could not be imported. "
        "Check the module path and class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def metadata():
    return {
        "document_id": DOCUMENT_ID,
        "tenant_id": TENANT_ID,
        "source": "master_service_agreement.pdf",
        "page": 4,
        "document_type": "contract",
        "classification": "confidential",
    }


@pytest.fixture
def chunker():
    cls = load_chunker_class()

    return cls(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_OVERLAP,
    )


@pytest.fixture
def long_text():
    return (
        "The supplier shall provide the services described "
        "in this agreement. The customer shall pay all "
        "invoices within thirty days. Either party may "
        "terminate the agreement upon material breach. "
        "Confidential information must remain protected. "
    ) * 8


@pytest.fixture
def legal_contract():
    return """
MASTER SERVICE AGREEMENT

1. DEFINITIONS
The term Services means the professional services
described in Schedule A.

2. PAYMENT TERMS
The customer shall pay invoices within thirty days
of receiving a valid invoice.

3. CONFIDENTIALITY
Both parties shall protect confidential information
and shall not disclose it without authorization.

4. TERMINATION
Either party may terminate this agreement by giving
thirty days written notice.

5. GOVERNING LAW
This agreement shall be governed by the laws of India.
""".strip()


@pytest.fixture
def ocr_text():
    return """
CONTRACT NO. 2026-001

CUSTOMER: ACME INDUSTRIES

TERMINATION CLAUSE:
Either party may terminate the agreement
with thirty days written notice.

SIGNATURE:
Authorized Representative
""".strip()


# ============================================================
# HELPERS
# ============================================================

def chunk_text(
    chunker,
    text,
    metadata=None,
):
    return chunker.chunk_text(
        text=text,
        metadata=metadata,
    )


def chunk_contents(chunks):
    return [
        item["text"]
        for item in chunks
    ]


def assert_valid_chunk(chunk):
    assert isinstance(chunk, dict)

    required = {
        "id",
        "text",
        "chunk_index",
        "start_char",
        "end_char",
        "metadata",
    }

    assert required.issubset(chunk.keys())

    assert isinstance(chunk["id"], str)
    assert chunk["id"]

    assert isinstance(chunk["text"], str)
    assert chunk["text"]

    assert isinstance(chunk["chunk_index"], int)

    assert isinstance(chunk["start_char"], int)
    assert isinstance(chunk["end_char"], int)

    assert chunk["start_char"] >= 0
    assert chunk["end_char"] > chunk["start_char"]

    assert isinstance(chunk["metadata"], dict)


def assert_source_alignment(chunks, source):
    for chunk in chunks:
        start = chunk["start_char"]
        end = chunk["end_char"]

        assert end <= len(source)

        assert source[start:end] == chunk["text"]


# ============================================================
# INITIALIZATION
# ============================================================

def test_chunker_initialization(chunker):
    assert chunker is not None
    assert callable(chunker.chunk_text)


def test_chunk_size_configuration(chunker):
    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE


def test_chunk_overlap_configuration(chunker):
    assert chunker.chunk_overlap == DEFAULT_OVERLAP


# ============================================================
# BASIC CHUNKING
# ============================================================

def test_short_text_single_chunk(chunker):
    text = "This agreement is legally binding."

    chunks = chunk_text(chunker, text)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_long_text_multiple_chunks(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    assert len(chunks) > 1


def test_chunk_result_is_list(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    assert isinstance(chunks, list)


def test_chunk_schema(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    for chunk in chunks:
        assert_valid_chunk(chunk)


def test_chunk_indices_are_sequential(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    indices = [
        item["chunk_index"]
        for item in chunks
    ]

    assert indices == list(range(len(chunks)))


# ============================================================
# CHUNK SIZE
# ============================================================

def test_chunks_respect_maximum_size(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    for chunk in chunks:
        assert len(chunk["text"]) <= DEFAULT_CHUNK_SIZE


@pytest.mark.parametrize(
    "size",
    [20, 50, 100, 200],
)
def test_custom_chunk_sizes(size, long_text):
    cls = load_chunker_class()

    chunker = cls(
        chunk_size=size,
        chunk_overlap=min(10, size - 1),
    )

    chunks = chunk_text(chunker, long_text)

    assert chunks

    assert all(
        len(chunk["text"]) <= size
        for chunk in chunks
    )


def test_exact_chunk_size_text(chunker):
    text = "A" * DEFAULT_CHUNK_SIZE

    chunks = chunk_text(chunker, text)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_text_one_character_over_limit(chunker):
    text = "A" * (DEFAULT_CHUNK_SIZE + 1)

    chunks = chunk_text(chunker, text)

    assert len(chunks) >= 2

    assert all(
        len(chunk["text"]) <= DEFAULT_CHUNK_SIZE
        for chunk in chunks
    )


# ============================================================
# CHUNK OVERLAP
# ============================================================

def test_adjacent_chunks_have_overlap(chunker):
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 15

    chunks = chunk_text(chunker, text)

    assert len(chunks) > 1

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        overlap = (
            previous["end_char"]
            - current["start_char"]
        )

        assert overlap == DEFAULT_OVERLAP


def test_zero_overlap():
    cls = load_chunker_class()

    chunker = cls(
        chunk_size=50,
        chunk_overlap=0,
    )

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10

    chunks = chunk_text(chunker, text)

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        assert (
            previous["end_char"]
            == current["start_char"]
        )


def test_overlap_does_not_exceed_chunk_size():
    cls = load_chunker_class()

    with pytest.raises(ValueError):
        cls(
            chunk_size=50,
            chunk_overlap=50,
        )


def test_overlap_greater_than_size_rejected():
    cls = load_chunker_class()

    with pytest.raises(ValueError):
        cls(
            chunk_size=50,
            chunk_overlap=60,
        )


def test_negative_overlap_rejected():
    cls = load_chunker_class()

    with pytest.raises(ValueError):
        cls(
            chunk_size=100,
            chunk_overlap=-1,
        )


# ============================================================
# TEXT COVERAGE AND OFFSETS
# ============================================================

def test_chunk_offsets_match_source(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    assert_source_alignment(chunks, long_text)


def test_first_chunk_starts_at_zero(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    assert chunks[0]["start_char"] == 0


def test_last_chunk_reaches_document_end(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    assert chunks[-1]["end_char"] == len(long_text)


def test_no_gaps_between_chunks(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        assert (
            current["start_char"]
            <= previous["end_char"]
        )


def test_chunk_offsets_are_monotonic(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    starts = [
        chunk["start_char"]
        for chunk in chunks
    ]

    assert starts == sorted(starts)


def test_reconstruct_original_text(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    reconstructed = chunks[0]["text"]

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        overlap = (
            previous["end_char"]
            - current["start_char"]
        )

        reconstructed += current["text"][
            max(0, overlap):
        ]

    assert reconstructed == long_text


# ============================================================
# LEGAL DOCUMENT CHUNKING
# ============================================================

def test_legal_contract_chunking(
    chunker,
    legal_contract,
):
    chunks = chunk_text(
        chunker,
        legal_contract,
    )

    assert chunks
    assert_source_alignment(chunks, legal_contract)


def test_termination_clause_preserved(
    chunker,
    legal_contract,
):
    chunks = chunk_text(
        chunker,
        legal_contract,
    )

    reconstructed = legal_contract[
        chunks[0]["start_char"]:
        chunks[-1]["end_char"]
    ]

    assert "TERMINATION" in reconstructed
    assert "thirty days written notice" in reconstructed


def test_payment_clause_preserved(
    chunker,
    legal_contract,
):
    chunks = chunk_text(
        chunker,
        legal_contract,
    )

    combined = " ".join(
        chunk_contents(chunks)
    )

    assert "PAYMENT TERMS" in combined
    assert "invoices" in combined


def test_governing_law_clause_preserved(
    chunker,
    legal_contract,
):
    chunks = chunk_text(
        chunker,
        legal_contract,
    )

    combined = " ".join(
        chunk_contents(chunks)
    )

    assert "GOVERNING LAW" in combined


# ============================================================
# METADATA PRESERVATION
# ============================================================

def test_document_metadata_preserved(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    for chunk in chunks:
        for key, value in metadata.items():
            assert chunk["metadata"][key] == value


def test_tenant_metadata_preserved(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    assert all(
        item["metadata"]["tenant_id"] == TENANT_ID
        for item in chunks
    )


def test_document_id_preserved(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    assert all(
        item["metadata"]["document_id"] == DOCUMENT_ID
        for item in chunks
    )


def test_page_metadata_preserved(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    assert all(
        item["metadata"]["page"] == 4
        for item in chunks
    )


def test_metadata_not_mutated(
    chunker,
    long_text,
    metadata,
):
    original = deepcopy(metadata)

    chunk_text(
        chunker,
        long_text,
        metadata,
    )

    assert metadata == original


# ============================================================
# UNIQUE IDS AND DETERMINISM
# ============================================================

def test_chunk_ids_unique(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    ids = [
        item["id"]
        for item in chunks
    ]

    assert len(ids) == len(set(ids))


def test_chunking_deterministic(
    chunker,
    long_text,
    metadata,
):
    first = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    second = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    assert first == second


def test_different_documents_have_distinct_chunk_ids(
    chunker,
):
    text = "A contract termination clause." * 10

    first = chunk_text(
        chunker,
        text,
        {
            "document_id": "document_A",
            "tenant_id": TENANT_ID,
        },
    )

    second = chunk_text(
        chunker,
        text,
        {
            "document_id": "document_B",
            "tenant_id": TENANT_ID,
        },
    )

    assert set(
        item["id"] for item in first
    ).isdisjoint(
        item["id"] for item in second
    )


# ============================================================
# OCR AND MULTILINE TEXT
# ============================================================

def test_ocr_document_chunking(
    chunker,
    ocr_text,
):
    chunks = chunk_text(chunker, ocr_text)

    assert chunks
    assert_source_alignment(chunks, ocr_text)


def test_multiline_text_preserved(chunker):
    text = (
        "SECTION ONE\n"
        "Payment terms apply.\n\n"
        "SECTION TWO\n"
        "Termination requires notice.\n"
    ) * 5

    chunks = chunk_text(chunker, text)

    assert_source_alignment(chunks, text)


def test_unicode_document_chunking(chunker):
    text = (
        "കരാർ വ്യവസ്ഥകൾ ബാധകമാണ്. "
        "Confidential agreement. "
    ) * 20

    chunks = chunk_text(chunker, text)

    assert chunks
    assert_source_alignment(chunks, text)


def test_special_characters_preserved(chunker):
    text = (
        "Clause §12 — Payment ₹50,000; "
        "Penalty €1,000; © 2026.\n"
    ) * 15

    chunks = chunk_text(chunker, text)

    assert_source_alignment(chunks, text)


# ============================================================
# EDGE CASES
# ============================================================

def test_empty_text_returns_empty(chunker):
    assert chunk_text(chunker, "") == []


def test_whitespace_only_text_returns_empty(chunker):
    assert chunk_text(chunker, "   \n\t  ") == []


def test_none_text_rejected(chunker):
    with pytest.raises((TypeError, ValueError)):
        chunk_text(chunker, None)


def test_non_string_text_rejected(chunker):
    with pytest.raises((TypeError, ValueError)):
        chunk_text(chunker, 12345)


def test_invalid_chunk_size_zero():
    cls = load_chunker_class()

    with pytest.raises(ValueError):
        cls(
            chunk_size=0,
            chunk_overlap=0,
        )


def test_invalid_chunk_size_negative():
    cls = load_chunker_class()

    with pytest.raises(ValueError):
        cls(
            chunk_size=-10,
            chunk_overlap=0,
        )


def test_single_character_document(chunker):
    chunks = chunk_text(chunker, "A")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "A"


def test_long_unbroken_text(chunker):
    text = "A" * 1000

    chunks = chunk_text(chunker, text)

    assert chunks
    assert_source_alignment(chunks, text)


# ============================================================
# LARGE DOCUMENTS
# ============================================================

def test_large_document_chunking(chunker):
    text = (
        "This contract contains payment, termination, "
        "confidentiality and indemnification clauses. "
    ) * 2000

    chunks = chunk_text(chunker, text)

    assert len(chunks) > 100

    assert chunks[-1]["end_char"] == len(text)


def test_large_document_chunk_indices(chunker):
    text = "Legal document content. " * 1000

    chunks = chunk_text(chunker, text)

    assert [
        chunk["chunk_index"]
        for chunk in chunks
    ] == list(range(len(chunks)))


# ============================================================
# RAG INTEGRATION READINESS
# ============================================================

def test_chunks_contain_embedding_ready_text(
    chunker,
    long_text,
):
    chunks = chunk_text(chunker, long_text)

    for chunk in chunks:
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0


def test_chunks_support_citations(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    for chunk in chunks:
        assert chunk["metadata"]["source"]
        assert chunk["metadata"]["page"] >= 1
        assert chunk["metadata"]["document_id"]


def test_chunks_support_tenant_isolation_metadata(
    chunker,
    long_text,
    metadata,
):
    chunks = chunk_text(
        chunker,
        long_text,
        metadata,
    )

    for chunk in chunks:
        assert chunk["metadata"]["tenant_id"] == TENANT_ID


# ============================================================
# END
# ============================================================
