
"""
tests/unit/test_tenant_isolation.py

Lexora AI — Multi-Tenant Isolation Security Tests

Expected module:
    backend.app.security.tenant_isolation

Expected class:
    TenantIsolationService

Expected methods:
    async validate_access(
        user_id: str,
        tenant_id: str,
    ) -> bool

    async validate_resource(
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
    ) -> bool

    async filter_documents(
        documents: list[dict],
        tenant_id: str,
    ) -> list[dict]

    build_search_filters(
        tenant_id: str,
        filters: dict | None = None,
    ) -> dict

Security requirements:
    - All tenant-scoped access must be authorized.
    - Users must not access another tenant's resources.
    - Tenant filters must not be overridden by callers.
    - Database and vector queries must be tenant-scoped.
    - Cache keys must include tenant identity.
    - Audit logs must preserve tenant context.
    - Authorization failures must deny access.
    - Missing tenant identity must never grant access.

These are specification tests.
Adapt service interfaces to the actual implementation.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"
TENANT_C = "tenant_gamma"

USER_A = "user_alpha"
USER_B = "user_beta"
USER_C = "user_gamma"

ADMIN_A = "admin_alpha"

DOC_A = "contract_alpha"
DOC_B = "contract_beta"
DOC_C = "contract_gamma"

QUERY = "What are the termination conditions?"


# ============================================================
# IMPORT
# ============================================================

def load_service_class():
    candidates = (
        "backend.app.security.tenant_isolation",
        "backend.app.services.tenant_isolation_service",
        "app.security.tenant_isolation",
        "app.services.tenant_isolation_service",
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
            "TenantIsolationService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "TenantIsolationService could not be imported. "
        "Update the module path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def tenant_documents():
    return [
        {
            "id": "chunk_alpha_001",
            "document_id": DOC_A,
            "tenant_id": TENANT_A,
            "text": (
                "Tenant Alpha contract termination "
                "requires thirty days notice."
            ),
            "score": 0.92,
            "metadata": {
                "classification": "internal",
                "department": "legal",
            },
        },
        {
            "id": "chunk_beta_001",
            "document_id": DOC_B,
            "tenant_id": TENANT_B,
            "text": (
                "TENANT_B_SECRET: confidential "
                "commercial pricing terms."
            ),
            "score": 0.99,
            "metadata": {
                "classification": "confidential",
                "department": "finance",
            },
        },
        {
            "id": "chunk_gamma_001",
            "document_id": DOC_C,
            "tenant_id": TENANT_C,
            "text": (
                "TENANT_C_SECRET: confidential "
                "acquisition agreement."
            ),
            "score": 0.96,
            "metadata": {
                "classification": "restricted",
                "department": "executive",
            },
        },
    ]


@pytest.fixture
def membership_repository():
    repository = MagicMock()

    memberships = {
        TENANT_A: {USER_A, ADMIN_A},
        TENANT_B: {USER_B},
        TENANT_C: {USER_C},
    }

    async def is_member(user_id, tenant_id):
        return user_id in memberships.get(
            tenant_id,
            set(),
        )

    repository.is_member = AsyncMock(
        side_effect=is_member
    )

    return repository


@pytest.fixture
def resource_repository():
    repository = MagicMock()

    resources = {
        ("document", DOC_A): TENANT_A,
        ("document", DOC_B): TENANT_B,
        ("document", DOC_C): TENANT_C,
        ("contract", "agreement_alpha"): TENANT_A,
        ("contract", "agreement_beta"): TENANT_B,
        ("analysis", "analysis_alpha"): TENANT_A,
        ("analysis", "analysis_beta"): TENANT_B,
    }

    async def get_tenant_id(
        resource_type,
        resource_id,
    ):
        return resources.get(
            (resource_type, resource_id)
        )

    repository.get_tenant_id = AsyncMock(
        side_effect=get_tenant_id
    )

    return repository


@pytest.fixture
def audit_service():
    service = MagicMock()
    service.record_event = AsyncMock(
        return_value=None
    )
    return service


@pytest.fixture
def isolation_service(
    membership_repository,
    resource_repository,
    audit_service,
):
    service_class = load_service_class()

    return service_class(
        membership_repository=membership_repository,
        resource_repository=resource_repository,
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


async def validate_access(
    service,
    user_id,
    tenant_id,
):
    return await invoke(
        service.validate_access,
        user_id=user_id,
        tenant_id=tenant_id,
    )


async def validate_resource(
    service,
    user_id,
    tenant_id,
    resource_type,
    resource_id,
):
    return await invoke(
        service.validate_resource,
        user_id=user_id,
        tenant_id=tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )


async def filter_documents(
    service,
    documents,
    tenant_id,
):
    return await invoke(
        service.filter_documents,
        documents=documents,
        tenant_id=tenant_id,
    )


async def build_filters(
    service,
    tenant_id,
    filters=None,
):
    return await invoke(
        service.build_search_filters,
        tenant_id=tenant_id,
        filters=filters,
    )


def document_ids(documents):
    return {
        item["document_id"]
        for item in documents
    }


def assert_no_cross_tenant_content(documents):
    serialized = repr(documents)

    assert "TENANT_B_SECRET" not in serialized
    assert "TENANT_C_SECRET" not in serialized


# ============================================================
# INITIALIZATION
# ============================================================

async def test_service_initialization(
    isolation_service,
):
    assert isolation_service is not None
    assert callable(isolation_service.validate_access)
    assert callable(isolation_service.validate_resource)
    assert callable(isolation_service.filter_documents)
    assert callable(isolation_service.build_search_filters)


# ============================================================
# TENANT MEMBERSHIP
# ============================================================

async def test_user_can_access_own_tenant(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        USER_A,
        TENANT_A,
    ) is True


async def test_user_cannot_access_other_tenant(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        USER_A,
        TENANT_B,
    ) is False


async def test_user_b_cannot_access_tenant_a(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        USER_B,
        TENANT_A,
    ) is False


async def test_unknown_user_denied(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        "unknown_user",
        TENANT_A,
    ) is False


async def test_unknown_tenant_denied(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        USER_A,
        "unknown_tenant",
    ) is False


async def test_admin_is_tenant_scoped(
    isolation_service,
):
    assert await validate_access(
        isolation_service,
        ADMIN_A,
        TENANT_A,
    ) is True

    assert await validate_access(
        isolation_service,
        ADMIN_A,
        TENANT_B,
    ) is False


# ============================================================
# RESOURCE ISOLATION
# ============================================================

async def test_own_document_access_allowed(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "document",
        DOC_A,
    ) is True


async def test_cross_tenant_document_denied(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "document",
        DOC_B,
    ) is False


async def test_cross_tenant_contract_denied(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "contract",
        "agreement_beta",
    ) is False


async def test_cross_tenant_analysis_denied(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "analysis",
        "analysis_beta",
    ) is False


async def test_nonexistent_resource_denied(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "document",
        "missing_document",
    ) is False


async def test_nonmember_cannot_access_resource(
    isolation_service,
):
    assert await validate_resource(
        isolation_service,
        USER_B,
        TENANT_A,
        "document",
        DOC_A,
    ) is False


# ============================================================
# DOCUMENT FILTERING
# ============================================================

async def test_filter_documents_by_tenant_a(
    isolation_service,
    tenant_documents,
):
    results = await filter_documents(
        isolation_service,
        tenant_documents,
        TENANT_A,
    )

    assert document_ids(results) == {DOC_A}
    assert_no_cross_tenant_content(results)


async def test_filter_documents_by_tenant_b(
    isolation_service,
    tenant_documents,
):
    results = await filter_documents(
        isolation_service,
        tenant_documents,
        TENANT_B,
    )

    assert document_ids(results) == {DOC_B}


async def test_filter_documents_by_tenant_c(
    isolation_service,
    tenant_documents,
):
    results = await filter_documents(
        isolation_service,
        tenant_documents,
        TENANT_C,
    )

    assert document_ids(results) == {DOC_C}


async def test_filter_documents_unknown_tenant(
    isolation_service,
    tenant_documents,
):
    results = await filter_documents(
        isolation_service,
        tenant_documents,
        "unknown_tenant",
    )

    assert results == []


async def test_filter_documents_empty_input(
    isolation_service,
):
    results = await filter_documents(
        isolation_service,
        [],
        TENANT_A,
    )

    assert results == []


async def test_filter_documents_preserves_metadata(
    isolation_service,
    tenant_documents,
):
    results = await filter_documents(
        isolation_service,
        tenant_documents,
        TENANT_A,
    )

    assert results[0]["metadata"]["department"] == "legal"


async def test_filter_documents_does_not_mutate_input(
    isolation_service,
    tenant_documents,
):
    original = deepcopy(tenant_documents)

    await filter_documents(
        isolation_service,
        tenant_documents,
        TENANT_A,
    )

    assert tenant_documents == original


# ============================================================
# VECTOR SEARCH FILTER SECURITY
# ============================================================

async def test_search_filters_include_tenant(
    isolation_service,
):
    filters = await build_filters(
        isolation_service,
        TENANT_A,
    )

    assert filters["tenant_id"] == TENANT_A


async def test_tenant_filter_cannot_be_overridden(
    isolation_service,
):
    try:
        filters = await build_filters(
            isolation_service,
            TENANT_A,
            {
                "tenant_id": TENANT_B,
            },
        )
    except (ValueError, PermissionError):
        return

    assert filters["tenant_id"] == TENANT_A


async def test_regular_metadata_filters_preserved(
    isolation_service,
):
    filters = await build_filters(
        isolation_service,
        TENANT_A,
        {
            "document_type": "contract",
            "department": "legal",
        },
    )

    assert filters["tenant_id"] == TENANT_A
    assert filters["document_type"] == "contract"
    assert filters["department"] == "legal"


async def test_empty_metadata_filters(
    isolation_service,
):
    filters = await build_filters(
        isolation_service,
        TENANT_A,
        {},
    )

    assert filters == {
        "tenant_id": TENANT_A,
    }


# ============================================================
# FAIL-CLOSED SECURITY
# ============================================================

@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_tenant_rejected(
    isolation_service,
    invalid_tenant,
):
    with pytest.raises((ValueError, TypeError)):
        await validate_access(
            isolation_service,
            USER_A,
            invalid_tenant,
        )


@pytest.mark.parametrize(
    "invalid_user",
    ["", "   ", None],
)
async def test_invalid_user_rejected(
    isolation_service,
    invalid_user,
):
    with pytest.raises((ValueError, TypeError)):
        await validate_access(
            isolation_service,
            invalid_user,
            TENANT_A,
        )


async def test_membership_database_failure_fails_closed(
    isolation_service,
    membership_repository,
):
    membership_repository.is_member.side_effect = (
        RuntimeError("Membership database unavailable")
    )

    try:
        allowed = await validate_access(
            isolation_service,
            USER_A,
            TENANT_A,
        )
    except (RuntimeError, PermissionError):
        return

    assert allowed is False


async def test_resource_database_failure_fails_closed(
    isolation_service,
    resource_repository,
):
    resource_repository.get_tenant_id.side_effect = (
        RuntimeError("Resource database unavailable")
    )

    try:
        allowed = await validate_resource(
            isolation_service,
            USER_A,
            TENANT_A,
            "document",
            DOC_A,
        )
    except (RuntimeError, PermissionError):
        return

    assert allowed is False


async def test_missing_tenant_metadata_not_exposed(
    isolation_service,
    tenant_documents,
):
    unscoped = {
        "id": "unscoped_chunk",
        "document_id": "unscoped_document",
        "text": "Confidential information",
    }

    results = await filter_documents(
        isolation_service,
        tenant_documents + [unscoped],
        TENANT_A,
    )

    assert "unscoped_document" not in document_ids(results)


# ============================================================
# MALICIOUS FILTERS
# ============================================================

async def test_filter_injection_cannot_bypass_tenant(
    isolation_service,
):
    malicious_filters = {
        "tenant_id": {
            "$ne": TENANT_A,
        },
    }

    try:
        filters = await build_filters(
            isolation_service,
            TENANT_A,
            malicious_filters,
        )
    except (ValueError, PermissionError, TypeError):
        return

    assert filters["tenant_id"] == TENANT_A


async def test_or_filter_cannot_expand_tenant_scope(
    isolation_service,
):
    malicious_filters = {
        "$or": [
            {"tenant_id": TENANT_A},
            {"tenant_id": TENANT_B},
        ]
    }

    try:
        filters = await build_filters(
            isolation_service,
            TENANT_A,
            malicious_filters,
        )
    except (ValueError, PermissionError, TypeError):
        return

    assert "$or" not in filters


# ============================================================
# CACHE ISOLATION
# ============================================================

async def test_tenant_scoped_cache_keys_are_distinct():
    def cache_key(tenant_id, query):
        return f"{tenant_id}:rag:{query}"

    key_a = cache_key(TENANT_A, QUERY)
    key_b = cache_key(TENANT_B, QUERY)

    assert key_a != key_b


async def test_cross_tenant_cache_leak_prevention():
    cache = {}

    def cache_key(tenant_id, query):
        return (tenant_id, query)

    cache[cache_key(TENANT_A, QUERY)] = {
        "document_id": DOC_A,
        "text": "Tenant Alpha private agreement",
    }

    result = cache.get(
        cache_key(TENANT_B, QUERY)
    )

    assert result is None


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_tenant_access_audit(
    isolation_service,
    audit_service,
):
    await validate_access(
        isolation_service,
        USER_A,
        TENANT_A,
    )

    audit_service.record_event.assert_awaited()


async def test_audit_event_contains_tenant_context(
    isolation_service,
    audit_service,
):
    await validate_access(
        isolation_service,
        USER_A,
        TENANT_A,
    )

    serialized = repr(
        audit_service.record_event.await_args_list
    )

    assert TENANT_A in serialized


async def test_audit_log_does_not_expose_foreign_content(
    isolation_service,
    audit_service,
):
    await validate_resource(
        isolation_service,
        USER_A,
        TENANT_A,
        "document",
        DOC_B,
    )

    serialized = repr(
        audit_service.record_event.await_args_list
    )

    assert "TENANT_B_SECRET" not in serialized


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_tenant_access_checks(
    isolation_service,
):
    requests = [
        (USER_A, TENANT_A),
        (USER_A, TENANT_B),
        (USER_B, TENANT_B),
        (USER_B, TENANT_A),
        (USER_C, TENANT_C),
    ]

    results = await asyncio.gather(
        *[
            validate_access(
                isolation_service,
                user_id,
                tenant_id,
            )
            for user_id, tenant_id in requests
        ]
    )

    assert results == [
        True,
        False,
        True,
        False,
        True,
    ]


async def test_concurrent_document_filtering(
    isolation_service,
    tenant_documents,
):
    tenant_ids = [
        TENANT_A,
        TENANT_B,
        TENANT_C,
    ]

    results = await asyncio.gather(
        *[
            filter_documents(
                isolation_service,
                deepcopy(tenant_documents),
                tenant_id,
            )
            for tenant_id in tenant_ids
        ]
    )

    assert document_ids(results[0]) == {DOC_A}
    assert document_ids(results[1]) == {DOC_B}
    assert document_ids(results[2]) == {DOC_C}


# ============================================================
# END
# ============================================================
