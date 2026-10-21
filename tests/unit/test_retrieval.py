
"""
tests/unit/test_retrieval.py

Lexora AI — Retrieval Pipeline Unit Tests

Expected module:
    backend.app.rag.retrieval

Expected class:
    RetrievalService

Expected constructor:
    RetrievalService(
        vector_retriever=...,
        bm25_retriever=...,
        permission_service=...,
        reranker=...,
        audit_service=...,
    )

Expected method:
    async retrieve(
        query: str,
        tenant_id: str,
        user_id: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]

Expected retriever interface:
    async search(
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict,
    ) -> list[dict]

Expected permission interface:
    async get_allowed_document_ids(
        user_id: str,
        tenant_id: str,
    ) -> list[str]

Expected reranker interface:
    async rerank(
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]

Expected result:
    {
        "id": str,
        "document_id": str,
        "tenant_id": str,
        "text": str,
        "score": float,
        "metadata": dict,
    }

Security requirements:
    - Enforce tenant isolation.
    - Enforce document-level permissions.
    - Never return unauthorized chunks.
    - Never trust caller-supplied tenant filters.
    - Preserve citation metadata.
    - Fail closed on permission errors.
    - Handle retrieval backend failures safely.
    - Avoid cross-request data leakage.

This is a reference contract test suite.
Adapt interfaces to your implementation.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import math
from copy import deepcopy
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

QUERY = "What is the termination notice period?"

TOP_K = 5


# ============================================================
# IMPORT
# ============================================================

def load_retrieval_service_class():
    candidates = (
        "backend.app.rag.retrieval",
        "backend.app.rag.retrieval_service",
        "backend.app.services.retrieval_service",
        "app.rag.retrieval",
        "app.rag.retrieval_service",
        "app.services.retrieval_service",
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
            "RetrievalService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "RetrievalService could not be imported. "
        "Update the import path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# DOCUMENT FACTORY
# ============================================================

def make_document(
    document_id,
    text,
    tenant_id=TENANT_A,
    score=0.5,
    page=1,
    department="legal",
    chunk_index=0,
):
    return {
        "id": f"{document_id}_chunk_{chunk_index}",
        "document_id": document_id,
        "tenant_id": tenant_id,
        "text": text,
        "score": score,
        "metadata": {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "source": f"{document_id}.pdf",
            "page": page,
            "department": department,
            "chunk_index": chunk_index,
        },
    }


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def legal_documents():
    return [
        make_document(
            "contract_termination",
            (
                "Either party may terminate the "
                "agreement with thirty days notice."
            ),
            score=0.95,
            page=4,
        ),
        make_document(
            "contract_payment",
            (
                "Invoices must be paid within "
                "thirty days."
            ),
            score=0.75,
            page=2,
        ),
        make_document(
            "contract_confidentiality",
            (
                "Confidential information must "
                "remain protected."
            ),
            score=0.70,
            page=7,
        ),
        make_document(
            "contract_renewal",
            (
                "The contract renews automatically "
                "every twelve months."
            ),
            score=0.65,
            page=8,
        ),
        make_document(
            "contract_liability",
            (
                "The liability limitation shall "
                "apply to all parties."
            ),
            score=0.60,
            page=9,
        ),
    ]


@pytest.fixture
def mixed_tenant_documents(legal_documents):
    return legal_documents + [
        make_document(
            "tenant_b_private",
            (
                "TENANT_B_SECRET: confidential "
                "termination and pricing terms."
            ),
            tenant_id=TENANT_B,
            score=0.99,
        )
    ]


@pytest.fixture
def vector_retriever(mixed_tenant_documents):
    retriever = MagicMock()

    async def search(
        query,
        tenant_id,
        top_k,
        filters,
    ):
        # Deliberately return mixed-tenant data.
        # The service must enforce isolation.
        return deepcopy(mixed_tenant_documents[:top_k])

    retriever.search = AsyncMock(
        side_effect=search
    )

    return retriever


@pytest.fixture
def bm25_retriever(mixed_tenant_documents):
    retriever = MagicMock()

    async def search(
        query,
        tenant_id,
        top_k,
        filters,
    ):
        return deepcopy(
            list(reversed(mixed_tenant_documents))[:top_k]
        )

    retriever.search = AsyncMock(
        side_effect=search
    )

    return retriever


@pytest.fixture
def permission_service():
    service = MagicMock()

    permissions = {
        (USER_A, TENANT_A): {
            "contract_termination",
            "contract_payment",
            "contract_confidentiality",
            "contract_renewal",
            "contract_liability",
        },
        (USER_B, TENANT_B): {
            "tenant_b_private",
        },
    }

    async def get_allowed_document_ids(
        user_id,
        tenant_id,
    ):
        return list(
            permissions.get(
                (user_id, tenant_id),
                set(),
            )
        )

    service.get_allowed_document_ids = AsyncMock(
        side_effect=get_allowed_document_ids
    )

    return service


@pytest.fixture
def reranker():
    service = MagicMock()

    async def rerank(query, documents, top_k):
        ranked = sorted(
            documents,
            key=lambda item: item["score"],
            reverse=True,
        )

        return deepcopy(ranked[:top_k])

    service.rerank = AsyncMock(
        side_effect=rerank
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
def retrieval_service(
    vector_retriever,
    bm25_retriever,
    permission_service,
    reranker,
    audit_service,
):
    service_class = load_retrieval_service_class()

    return service_class(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        permission_service=permission_service,
        reranker=reranker,
        audit_service=audit_service,
    )


# ============================================================
# HELPERS
# ============================================================

async def retrieve(
    service,
    query=QUERY,
    tenant_id=TENANT_A,
    user_id=USER_A,
    top_k=TOP_K,
    filters=None,
):
    result = service.retrieve(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        top_k=top_k,
        filters=filters,
    )

    if inspect.isawaitable(result):
        return await result

    return result


def result_ids(results):
    return {
        item["document_id"]
        for item in results
    }


def assert_valid_result(item):
    assert isinstance(item, dict)

    required = {
        "id",
        "document_id",
        "tenant_id",
        "text",
        "score",
        "metadata",
    }

    assert required.issubset(item.keys())

    assert isinstance(item["text"], str)
    assert isinstance(item["metadata"], dict)

    assert isinstance(item["score"], (int, float))
    assert math.isfinite(item["score"])


def assert_tenant_isolation(results, tenant_id):
    for item in results:
        assert item["tenant_id"] == tenant_id

        assert (
            item["metadata"]["tenant_id"]
            == tenant_id
        )


# ============================================================
# INITIALIZATION
# ============================================================

async def test_retrieval_service_initialization(
    retrieval_service,
):
    assert retrieval_service is not None
    assert callable(retrieval_service.retrieve)


# ============================================================
# BASIC RETRIEVAL
# ============================================================

async def test_retrieve_returns_list(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    assert isinstance(results, list)


async def test_retrieve_returns_documents(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    assert len(results) > 0


async def test_retrieval_result_schema(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    for item in results:
        assert_valid_result(item)


async def test_retrieval_contains_relevant_document(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    assert "contract_termination" in result_ids(
        results
    )


async def test_retrieval_scores_are_finite(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    for item in results:
        assert math.isfinite(item["score"])


# ============================================================
# HYBRID SEARCH
# ============================================================

async def test_vector_retriever_called(
    retrieval_service,
    vector_retriever,
):
    await retrieve(
        retrieval_service
    )

    vector_retriever.search.assert_awaited()


async def test_bm25_retriever_called(
    retrieval_service,
    bm25_retriever,
):
    await retrieve(
        retrieval_service
    )

    bm25_retriever.search.assert_awaited()


async def test_both_retrievers_receive_query(
    retrieval_service,
    vector_retriever,
    bm25_retriever,
):
    await retrieve(
        retrieval_service
    )

    vector_calls = repr(
        vector_retriever.search.await_args_list
    )

    bm25_calls = repr(
        bm25_retriever.search.await_args_list
    )

    assert QUERY in vector_calls
    assert QUERY in bm25_calls


async def test_hybrid_results_deduplicated(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    chunk_ids = [
        item["id"]
        for item in results
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


async def test_reranker_called(
    retrieval_service,
    reranker,
):
    await retrieve(
        retrieval_service
    )

    reranker.rerank.assert_awaited()


# ============================================================
# TOP-K
# ============================================================

@pytest.mark.parametrize(
    "top_k",
    [1, 2, 3, 4, 5],
)
async def test_top_k_limits_results(
    retrieval_service,
    top_k,
):
    results = await retrieve(
        retrieval_service,
        top_k=top_k,
    )

    assert len(results) <= top_k


async def test_top_k_zero_rejected(
    retrieval_service,
):
    with pytest.raises(ValueError):
        await retrieve(
            retrieval_service,
            top_k=0,
        )


async def test_negative_top_k_rejected(
    retrieval_service,
):
    with pytest.raises(ValueError):
        await retrieve(
            retrieval_service,
            top_k=-5,
        )


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_tenant_a_only_receives_own_documents(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        tenant_id=TENANT_A,
        user_id=USER_A,
    )

    assert_tenant_isolation(
        results,
        TENANT_A,
    )


async def test_cross_tenant_document_not_returned(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        tenant_id=TENANT_A,
        user_id=USER_A,
    )

    assert "tenant_b_private" not in result_ids(
        results
    )

    assert "TENANT_B_SECRET" not in repr(results)


async def test_tenant_b_only_receives_own_documents(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        tenant_id=TENANT_B,
        user_id=USER_B,
    )

    assert_tenant_isolation(
        results,
        TENANT_B,
    )


async def test_unknown_tenant_returns_no_documents(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        tenant_id="unknown_tenant",
        user_id="unknown_user",
    )

    assert results == []


# ============================================================
# DOCUMENT PERMISSIONS
# ============================================================

async def test_permission_service_called(
    retrieval_service,
    permission_service,
):
    await retrieve(
        retrieval_service
    )

    permission_service.get_allowed_document_ids.assert_awaited()


async def test_user_without_permissions_gets_no_results(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        user_id="unauthorized_user",
    )

    assert results == []


async def test_restricted_document_filtered(
    retrieval_service,
    permission_service,
):
    permission_service.get_allowed_document_ids = (
        AsyncMock(
            return_value=[
                "contract_termination",
            ]
        )
    )

    results = await retrieve(
        retrieval_service
    )

    assert result_ids(results) <= {
        "contract_termination"
    }


async def test_permission_backend_failure_fails_closed(
    retrieval_service,
    permission_service,
):
    permission_service.get_allowed_document_ids = (
        AsyncMock(
            side_effect=RuntimeError(
                "Permission database unavailable"
            )
        )
    )

    try:
        results = await retrieve(
            retrieval_service
        )
    except (RuntimeError, PermissionError):
        return

    assert results == []


# ============================================================
# METADATA FILTERS
# ============================================================

async def test_department_filter(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        filters={
            "department": "legal",
        },
    )

    assert all(
        item["metadata"]["department"] == "legal"
        for item in results
    )


async def test_tenant_filter_cannot_be_overridden(
    retrieval_service,
):
    try:
        results = await retrieve(
            retrieval_service,
            filters={
                "tenant_id": TENANT_B,
            },
        )
    except (ValueError, PermissionError):
        return

    assert_tenant_isolation(
        results,
        TENANT_A,
    )


async def test_document_id_filter(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service,
        filters={
            "document_id": "contract_termination",
        },
    )

    assert result_ids(results) <= {
        "contract_termination"
    }


# ============================================================
# CITATIONS
# ============================================================

async def test_citation_metadata_preserved(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    for item in results:
        assert item["metadata"]["source"]
        assert item["metadata"]["page"] >= 1


async def test_document_id_preserved(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    for item in results:
        assert (
            item["document_id"]
            == item["metadata"]["document_id"]
        )


async def test_chunk_id_preserved(
    retrieval_service,
):
    results = await retrieve(
        retrieval_service
    )

    for item in results:
        assert item["id"]


# ============================================================
# INVALID INPUT
# ============================================================

@pytest.mark.parametrize(
    "invalid_query",
    ["", "   ", None],
)
async def test_invalid_query_rejected(
    retrieval_service,
    invalid_query,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await retrieve(
            retrieval_service,
            query=invalid_query,
        )


@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_tenant_rejected(
    retrieval_service,
    invalid_tenant,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await retrieve(
            retrieval_service,
            tenant_id=invalid_tenant,
        )


@pytest.mark.parametrize(
    "invalid_user",
    ["", "   ", None],
)
async def test_invalid_user_rejected(
    retrieval_service,
    invalid_user,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await retrieve(
            retrieval_service,
            user_id=invalid_user,
        )


# ============================================================
# BACKEND FAILURES
# ============================================================

async def test_vector_backend_failure(
    retrieval_service,
    vector_retriever,
):
    vector_retriever.search = AsyncMock(
        side_effect=RuntimeError(
            "Vector database unavailable"
        )
    )

    try:
        results = await retrieve(
            retrieval_service
        )
    except RuntimeError:
        return

    assert isinstance(results, list)
    assert_tenant_isolation(results, TENANT_A)


async def test_bm25_backend_failure(
    retrieval_service,
    bm25_retriever,
):
    bm25_retriever.search = AsyncMock(
        side_effect=RuntimeError(
            "BM25 backend unavailable"
        )
    )

    try:
        results = await retrieve(
            retrieval_service
        )
    except RuntimeError:
        return

    assert isinstance(results, list)
    assert_tenant_isolation(results, TENANT_A)


async def test_reranker_failure(
    retrieval_service,
    reranker,
):
    reranker.rerank = AsyncMock(
        side_effect=RuntimeError(
            "Reranker unavailable"
        )
    )

    try:
        results = await retrieve(
            retrieval_service
        )
    except RuntimeError:
        return

    assert isinstance(results, list)
    assert_tenant_isolation(results, TENANT_A)


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_retrieval_is_audited(
    retrieval_service,
    audit_service,
):
    await retrieve(
        retrieval_service
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant_context(
    retrieval_service,
    audit_service,
):
    await retrieve(
        retrieval_service
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


async def test_audit_does_not_expose_private_content(
    retrieval_service,
    audit_service,
):
    await retrieve(
        retrieval_service
    )

    assert "TENANT_B_SECRET" not in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_retrieval(
    retrieval_service,
):
    queries = [
        "termination notice",
        "payment obligations",
        "confidentiality requirements",
        "contract renewal",
    ]

    results = await asyncio.gather(
        *[
            retrieve(
                retrieval_service,
                query=query,
            )
            for query in queries
        ]
    )

    assert len(results) == len(queries)

    for result in results:
        assert isinstance(result, list)
        assert_tenant_isolation(result, TENANT_A)


async def test_concurrent_tenant_retrieval(
    retrieval_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        retrieve(
            retrieval_service,
            tenant_id=TENANT_A,
            user_id=USER_A,
        ),
        retrieve(
            retrieval_service,
            tenant_id=TENANT_B,
            user_id=USER_B,
        ),
    )

    assert_tenant_isolation(
        tenant_a,
        TENANT_A,
    )

    assert_tenant_isolation(
        tenant_b,
        TENANT_B,
    )


# ============================================================
# END
# ============================================================
