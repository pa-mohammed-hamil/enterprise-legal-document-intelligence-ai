
"""
tests/unit/test_hybrid_search.py

Unit tests for Lexora AI hybrid retrieval.

Expected production module:
    backend.app.rag.hybrid_search.HybridSearch

Expected interface:
    HybridSearch(
        bm25_retriever=...,
        vector_retriever=...,
        reranker=...,
    )

    await search(
        query: str,
        tenant_id: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]

Expected result fields:
    id, document_id, tenant_id, text, score,
    source, page, metadata

These are contract tests. Adapt the factory and invocation
to your real implementation before running.
"""

import asyncio
import inspect
import math
from unittest.mock import AsyncMock, MagicMock

import pytest


try:
    from backend.app.rag.hybrid_search import HybridSearch
except ImportError:
    from app.rag.hybrid_search import HybridSearch


pytestmark = pytest.mark.asyncio


# ============================================================
# TEST DATA
# ============================================================

TENANT_ID = "tenant_001"
OTHER_TENANT_ID = "tenant_002"

QUERY = "What are the termination conditions?"


@pytest.fixture
def legal_chunks():
    return [
        {
            "id": "chunk_001",
            "document_id": "contract_001",
            "tenant_id": TENANT_ID,
            "text": (
                "Either party may terminate the agreement "
                "with thirty days written notice."
            ),
            "score": 0.95,
            "source": "master_service_agreement.pdf",
            "page": 4,
            "metadata": {
                "document_type": "contract",
                "clause_type": "termination",
                "page": 4,
            },
        },
        {
            "id": "chunk_002",
            "document_id": "contract_001",
            "tenant_id": TENANT_ID,
            "text": (
                "The agreement may be terminated immediately "
                "for a material breach."
            ),
            "score": 0.88,
            "source": "master_service_agreement.pdf",
            "page": 5,
            "metadata": {
                "document_type": "contract",
                "clause_type": "termination",
                "page": 5,
            },
        },
        {
            "id": "chunk_003",
            "document_id": "contract_002",
            "tenant_id": TENANT_ID,
            "text": (
                "All invoices must be paid within "
                "thirty calendar days."
            ),
            "score": 0.72,
            "source": "vendor_agreement.pdf",
            "page": 8,
            "metadata": {
                "document_type": "contract",
                "clause_type": "payment",
                "page": 8,
            },
        },
    ]


@pytest.fixture
def bm25_results(legal_chunks):
    return [
        dict(legal_chunks[0], score=0.93),
        dict(legal_chunks[1], score=0.81),
    ]


@pytest.fixture
def vector_results(legal_chunks):
    return [
        dict(legal_chunks[1], score=0.96),
        dict(legal_chunks[0], score=0.89),
        dict(legal_chunks[2], score=0.65),
    ]


# ============================================================
# MOCK DEPENDENCIES
# ============================================================

@pytest.fixture
def bm25_retriever(bm25_results):
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=bm25_results)
    return retriever


@pytest.fixture
def vector_retriever(vector_results):
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=vector_results)
    return retriever


@pytest.fixture
def reranker():
    service = MagicMock()

    async def rerank_results(query, documents, **kwargs):
        return sorted(
            documents,
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

    service.rerank = AsyncMock(side_effect=rerank_results)
    return service


@pytest.fixture
def hybrid_search(
    bm25_retriever,
    vector_retriever,
    reranker,
):
    return HybridSearch(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        reranker=reranker,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

async def execute_search(
    service,
    query=QUERY,
    tenant_id=TENANT_ID,
    top_k=5,
    filters=None,
):
    result = service.search(
        query=query,
        tenant_id=tenant_id,
        top_k=top_k,
        filters=filters,
    )

    if inspect.isawaitable(result):
        return await result

    return result


def assert_valid_result(result):
    assert isinstance(result, dict)

    required_fields = {
        "id",
        "document_id",
        "tenant_id",
        "text",
        "score",
        "source",
        "page",
        "metadata",
    }

    assert required_fields.issubset(result.keys())

    assert isinstance(result["text"], str)
    assert result["text"].strip()

    assert isinstance(result["score"], (int, float))
    assert math.isfinite(result["score"])

    assert isinstance(result["metadata"], dict)


# ============================================================
# INITIALIZATION
# ============================================================

async def test_hybrid_search_initialization(hybrid_search):
    assert hybrid_search is not None
    assert callable(hybrid_search.search)


# ============================================================
# BASIC RETRIEVAL
# ============================================================

async def test_hybrid_search_returns_results(hybrid_search):
    results = await execute_search(hybrid_search)

    assert isinstance(results, list)
    assert len(results) > 0


async def test_hybrid_search_result_structure(hybrid_search):
    results = await execute_search(hybrid_search)

    for result in results:
        assert_valid_result(result)


async def test_hybrid_search_respects_top_k(hybrid_search):
    results = await execute_search(
        hybrid_search,
        top_k=2,
    )

    assert len(results) <= 2


async def test_hybrid_search_top_k_one(hybrid_search):
    results = await execute_search(
        hybrid_search,
        top_k=1,
    )

    assert len(results) <= 1


# ============================================================
# BM25 + VECTOR RETRIEVAL
# ============================================================

async def test_bm25_retriever_called(
    hybrid_search,
    bm25_retriever,
):
    await execute_search(hybrid_search)

    bm25_retriever.search.assert_awaited()


async def test_vector_retriever_called(
    hybrid_search,
    vector_retriever,
):
    await execute_search(hybrid_search)

    vector_retriever.search.assert_awaited()


async def test_both_retrievers_contribute(
    hybrid_search,
    legal_chunks,
):
    results = await execute_search(hybrid_search)

    ids = {item["id"] for item in results}

    assert "chunk_001" in ids
    assert "chunk_002" in ids


async def test_vector_only_results_are_included(
    hybrid_search,
):
    results = await execute_search(hybrid_search)

    ids = {item["id"] for item in results}

    assert "chunk_003" in ids


# ============================================================
# DEDUPLICATION
# ============================================================

async def test_duplicate_chunks_are_removed(hybrid_search):
    results = await execute_search(hybrid_search)

    ids = [item["id"] for item in results]

    assert len(ids) == len(set(ids))


async def test_duplicate_chunks_from_multiple_retrievers(
    hybrid_search,
    bm25_retriever,
    vector_retriever,
    legal_chunks,
):
    same_chunk = legal_chunks[0]

    bm25_retriever.search.return_value = [
        dict(same_chunk, score=0.90)
    ]

    vector_retriever.search.return_value = [
        dict(same_chunk, score=0.98)
    ]

    results = await execute_search(hybrid_search)

    assert sum(
        item["id"] == same_chunk["id"]
        for item in results
    ) == 1


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_results_belong_to_requested_tenant(
    hybrid_search,
):
    results = await execute_search(
        hybrid_search,
        tenant_id=TENANT_ID,
    )

    assert all(
        item["tenant_id"] == TENANT_ID
        for item in results
    )


async def test_cross_tenant_chunks_are_excluded(
    hybrid_search,
    vector_retriever,
    legal_chunks,
):
    foreign_chunk = dict(
        legal_chunks[0],
        id="foreign_chunk",
        tenant_id=OTHER_TENANT_ID,
        document_id="private_contract",
    )

    vector_retriever.search.return_value = [
        foreign_chunk,
        legal_chunks[1],
    ]

    results = await execute_search(
        hybrid_search,
        tenant_id=TENANT_ID,
    )

    assert all(
        item["tenant_id"] != OTHER_TENANT_ID
        for item in results
    )

    assert "foreign_chunk" not in {
        item["id"] for item in results
    }


async def test_tenant_context_is_required(hybrid_search):
    with pytest.raises((ValueError, TypeError)):
        await execute_search(
            hybrid_search,
            tenant_id="",
        )


# ============================================================
# METADATA FILTERING
# ============================================================

async def test_document_id_filter(
    hybrid_search,
):
    results = await execute_search(
        hybrid_search,
        filters={"document_id": "contract_001"},
    )

    assert all(
        item["document_id"] == "contract_001"
        for item in results
    )


async def test_clause_type_filter(
    hybrid_search,
):
    results = await execute_search(
        hybrid_search,
        filters={"clause_type": "termination"},
    )

    assert all(
        item["metadata"].get("clause_type")
        == "termination"
        for item in results
    )


async def test_nonmatching_filter_returns_empty(
    hybrid_search,
):
    results = await execute_search(
        hybrid_search,
        filters={"document_id": "nonexistent_contract"},
    )

    assert results == []


# ============================================================
# RERANKING
# ============================================================

async def test_reranker_called(
    hybrid_search,
    reranker,
):
    await execute_search(hybrid_search)

    reranker.rerank.assert_awaited()


async def test_results_are_sorted_by_score(hybrid_search):
    results = await execute_search(hybrid_search)

    scores = [item["score"] for item in results]

    assert scores == sorted(scores, reverse=True)


# ============================================================
# CITATION METADATA
# ============================================================

async def test_results_preserve_source(hybrid_search):
    results = await execute_search(hybrid_search)

    assert all(
        item.get("source")
        for item in results
    )


async def test_results_preserve_page_numbers(
    hybrid_search,
):
    results = await execute_search(hybrid_search)

    assert all(
        isinstance(item["page"], int)
        and item["page"] >= 1
        for item in results
    )


async def test_results_preserve_document_ids(
    hybrid_search,
):
    results = await execute_search(hybrid_search)

    assert all(
        item.get("document_id")
        for item in results
    )


# ============================================================
# EMPTY AND ERROR CASES
# ============================================================

async def test_empty_retrieval_results(
    hybrid_search,
    bm25_retriever,
    vector_retriever,
):
    bm25_retriever.search.return_value = []
    vector_retriever.search.return_value = []

    results = await execute_search(hybrid_search)

    assert results == []


async def test_empty_query_is_rejected(hybrid_search):
    with pytest.raises((ValueError, TypeError)):
        await execute_search(
            hybrid_search,
            query="",
        )


async def test_whitespace_query_is_rejected(
    hybrid_search,
):
    with pytest.raises((ValueError, TypeError)):
        await execute_search(
            hybrid_search,
            query="   ",
        )


async def test_invalid_top_k_is_rejected(
    hybrid_search,
):
    with pytest.raises((ValueError, TypeError)):
        await execute_search(
            hybrid_search,
            top_k=0,
        )


async def test_negative_top_k_is_rejected(
    hybrid_search,
):
    with pytest.raises((ValueError, TypeError)):
        await execute_search(
            hybrid_search,
            top_k=-5,
        )


async def test_bm25_failure_is_handled(
    hybrid_search,
    bm25_retriever,
):
    bm25_retriever.search.side_effect = RuntimeError(
        "BM25 retrieval unavailable"
    )

    try:
        results = await execute_search(hybrid_search)
        assert isinstance(results, list)
    except RuntimeError as exc:
        assert "BM25" in str(exc)


async def test_vector_failure_is_handled(
    hybrid_search,
    vector_retriever,
):
    vector_retriever.search.side_effect = RuntimeError(
        "Vector database unavailable"
    )

    try:
        results = await execute_search(hybrid_search)
        assert isinstance(results, list)
    except RuntimeError as exc:
        assert "Vector" in str(exc)


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_hybrid_search(hybrid_search):
    queries = [
        "termination clause",
        "payment obligations",
        "liability limitations",
        "renewal conditions",
    ]

    results = await asyncio.gather(
        *[
            execute_search(
                hybrid_search,
                query=query,
                tenant_id=TENANT_ID,
            )
            for query in queries
        ]
    )

    assert len(results) == len(queries)
    assert all(isinstance(item, list) for item in results)


# ============================================================
# DATA SAFETY
# ============================================================

async def test_original_retrieval_data_not_mutated(
    hybrid_search,
    bm25_results,
    vector_results,
):
    import copy

    original_bm25 = copy.deepcopy(bm25_results)
    original_vector = copy.deepcopy(vector_results)

    await execute_search(hybrid_search)

    assert bm25_results == original_bm25
    assert vector_results == original_vector


async def test_result_scores_are_finite(hybrid_search):
    results = await execute_search(hybrid_search)

    for item in results:
        assert math.isfinite(float(item["score"]))


async def test_no_duplicate_result_identifiers(
    hybrid_search,
):
    results = await execute_search(hybrid_search)

    keys = [
        (item["tenant_id"], item["document_id"], item["id"])
        for item in results
    ]

    assert len(keys) == len(set(keys))
