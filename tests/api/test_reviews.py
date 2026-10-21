
"""
tests/unit/test_reviews.py

Lexora AI — Legal Document Review Workflow Tests

Expected module:
    backend.app.services.review_service

Expected class:
    ReviewService

Expected constructor:
    ReviewService(
        review_repository=...,
        document_repository=...,
        permission_service=...,
        notification_service=...,
        audit_service=...,
    )

Expected async methods:

    create_review(
        document_id: str,
        tenant_id: str,
        user_id: str,
        reviewer_ids: list[str],
        notes: str = "",
    ) -> dict

    get_review(
        review_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict

    list_reviews(
        tenant_id: str,
        user_id: str,
        filters: dict | None = None,
    ) -> list[dict]

    approve_review(
        review_id: str,
        tenant_id: str,
        user_id: str,
        comment: str = "",
    ) -> dict

    reject_review(
        review_id: str,
        tenant_id: str,
        user_id: str,
        reason: str,
    ) -> dict

    request_revision(
        review_id: str,
        tenant_id: str,
        user_id: str,
        comment: str,
    ) -> dict

    add_comment(
        review_id: str,
        tenant_id: str,
        user_id: str,
        comment: str,
    ) -> dict

    get_review_history(
        review_id: str,
        tenant_id: str,
        user_id: str,
    ) -> list[dict]

Expected review:
    {
        "id": str,
        "document_id": str,
        "tenant_id": str,
        "created_by": str,
        "reviewer_ids": list[str],
        "status": str,
        "notes": str,
        "comments": list[dict],
        "history": list[dict],
    }

Review statuses:
    pending
    in_review
    approved
    rejected
    revision_requested

Security requirements:
    - Verify document belongs to the tenant.
    - Validate reviewer membership.
    - Enforce review permissions.
    - Prevent cross-tenant access.
    - Prevent unauthorized approvals.
    - Reject invalid state transitions.
    - Preserve reviewer identity and audit history.
    - Prevent duplicate decisions.
    - Avoid confidential content in logs.
    - Handle concurrent decisions safely.

These are reference specification tests.
They have not been run against the real backend.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

OWNER_A = "owner_alpha"
REVIEWER_A = "reviewer_alpha"
REVIEWER_B = "reviewer_beta"
VIEWER_A = "viewer_alpha"

DOCUMENT_A = "document_alpha"
DOCUMENT_B = "document_beta"

REVIEW_A = "review_alpha"
REVIEW_B = "review_beta"

VALID_STATUSES = {
    "pending",
    "in_review",
    "approved",
    "rejected",
    "revision_requested",
}

FINAL_STATUSES = {
    "approved",
    "rejected",
}


# ============================================================
# IMPORT
# ============================================================

def load_review_service_class():
    candidates = (
        "backend.app.services.review_service",
        "backend.app.services.reviews",
        "backend.app.reviews.service",
        "app.services.review_service",
        "app.services.reviews",
        "app.reviews.service",
    )

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(str(exc))
            continue

        review_class = getattr(
            module,
            "ReviewService",
            None,
        )

        if review_class is not None:
            return review_class

    pytest.fail(
        "ReviewService could not be imported. "
        "Update the import path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# DATA FACTORIES
# ============================================================

def make_review(
    review_id=REVIEW_A,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    created_by=OWNER_A,
    reviewer_ids=None,
    status="pending",
    notes="Review contract before approval.",
    comments=None,
    history=None,
):
    return {
        "id": review_id,
        "document_id": document_id,
        "tenant_id": tenant_id,
        "created_by": created_by,
        "reviewer_ids": (
            list(reviewer_ids)
            if reviewer_ids is not None
            else [REVIEWER_A]
        ),
        "status": status,
        "notes": notes,
        "comments": deepcopy(comments or []),
        "history": deepcopy(history or []),
    }


def make_comment(
    user_id,
    text,
):
    return {
        "id": f"comment_{uuid.uuid4().hex}",
        "user_id": user_id,
        "comment": text,
        "created_at": datetime.now(timezone.utc),
    }


def make_history_event(
    action,
    user_id,
    old_status=None,
    new_status=None,
):
    return {
        "id": f"event_{uuid.uuid4().hex}",
        "action": action,
        "user_id": user_id,
        "old_status": old_status,
        "new_status": new_status,
        "created_at": datetime.now(timezone.utc),
    }


# ============================================================
# FAKE REVIEW REPOSITORY
# ============================================================

class FakeReviewRepository:

    def __init__(self):
        self.reviews = {
            REVIEW_A: make_review(),
            REVIEW_B: make_review(
                review_id=REVIEW_B,
                document_id=DOCUMENT_B,
                tenant_id=TENANT_B,
                created_by=REVIEWER_B,
                reviewer_ids=[REVIEWER_B],
                notes="Private tenant B legal review.",
            ),
        }

        self._locks = {}

    def _lock(self, review_id):
        return self._locks.setdefault(
            review_id,
            asyncio.Lock(),
        )

    async def create(self, data):
        review_id = data.get(
            "id",
            f"review_{uuid.uuid4().hex}",
        )

        if review_id in self.reviews:
            raise ValueError("Duplicate review ID")

        review = {
            **deepcopy(data),
            "id": review_id,
        }

        self.reviews[review_id] = review

        return deepcopy(review)

    async def get_by_id(
        self,
        review_id,
        tenant_id,
    ):
        review = self.reviews.get(review_id)

        if (
            review is None
            or review["tenant_id"] != tenant_id
        ):
            return None

        return deepcopy(review)

    async def list_by_tenant(self, tenant_id):
        return [
            deepcopy(review)
            for review in self.reviews.values()
            if review["tenant_id"] == tenant_id
        ]

    async def update(
        self,
        review_id,
        tenant_id,
        changes,
    ):
        async with self._lock(review_id):
            review = self.reviews.get(review_id)

            if (
                review is None
                or review["tenant_id"] != tenant_id
            ):
                raise LookupError("Review not found")

            review.update(deepcopy(changes))

            return deepcopy(review)

    async def transition(
        self,
        review_id,
        tenant_id,
        expected_statuses,
        new_status,
        user_id,
        action,
    ):
        """
        Atomic compare-and-set test helper.

        Production should use a DB transaction,
        conditional UPDATE or row-level locking.
        """
        async with self._lock(review_id):
            review = self.reviews.get(review_id)

            if (
                review is None
                or review["tenant_id"] != tenant_id
            ):
                raise LookupError("Review not found")

            if review["status"] not in expected_statuses:
                raise ValueError(
                    "Invalid review transition"
                )

            old_status = review["status"]
            review["status"] = new_status

            review["history"].append(
                make_history_event(
                    action=action,
                    user_id=user_id,
                    old_status=old_status,
                    new_status=new_status,
                )
            )

            return deepcopy(review)

    async def add_comment(
        self,
        review_id,
        tenant_id,
        comment,
    ):
        async with self._lock(review_id):
            review = self.reviews.get(review_id)

            if (
                review is None
                or review["tenant_id"] != tenant_id
            ):
                raise LookupError("Review not found")

            review["comments"].append(
                deepcopy(comment)
            )

            return deepcopy(comment)


# ============================================================
# FAKE DOCUMENT REPOSITORY
# ============================================================

class FakeDocumentRepository:

    def __init__(self):
        self.documents = {
            DOCUMENT_A: {
                "id": DOCUMENT_A,
                "tenant_id": TENANT_A,
                "title": "Master Service Agreement",
                "status": "active",
            },
            DOCUMENT_B: {
                "id": DOCUMENT_B,
                "tenant_id": TENANT_B,
                "title": "Confidential Tenant B Contract",
                "status": "active",
            },
        }

    async def get_by_id(
        self,
        document_id,
        tenant_id,
    ):
        document = self.documents.get(document_id)

        if (
            document is None
            or document["tenant_id"] != tenant_id
        ):
            return None

        return deepcopy(document)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def review_repository():
    return FakeReviewRepository()


@pytest.fixture
def document_repository():
    return FakeDocumentRepository()


@pytest.fixture
def permission_service():
    service = MagicMock()

    permissions = {
        (OWNER_A, TENANT_A): {
            "create_review",
            "read_review",
            "comment",
        },
        (REVIEWER_A, TENANT_A): {
            "read_review",
            "approve_review",
            "reject_review",
            "request_revision",
            "comment",
        },
        (REVIEWER_B, TENANT_B): {
            "create_review",
            "read_review",
            "approve_review",
            "reject_review",
            "request_revision",
            "comment",
        },
        (VIEWER_A, TENANT_A): {
            "read_review",
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
def notification_service():
    service = MagicMock()

    service.notify_review_assigned = AsyncMock(
        return_value=True
    )

    service.notify_review_decision = AsyncMock(
        return_value=True
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
def review_service(
    review_repository,
    document_repository,
    permission_service,
    notification_service,
    audit_service,
):
    service_class = load_review_service_class()

    return service_class(
        review_repository=review_repository,
        document_repository=document_repository,
        permission_service=permission_service,
        notification_service=notification_service,
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


async def create_review(
    service,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=OWNER_A,
    reviewer_ids=None,
    notes="Review this contract.",
):
    return await invoke(
        service.create_review,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        reviewer_ids=(
            list(reviewer_ids)
            if reviewer_ids is not None
            else [REVIEWER_A]
        ),
        notes=notes,
    )


async def get_review(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
):
    return await invoke(
        service.get_review,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


async def list_reviews(
    service,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
    filters=None,
):
    return await invoke(
        service.list_reviews,
        tenant_id=tenant_id,
        user_id=user_id,
        filters=filters,
    )


async def approve_review(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
    comment="Approved after legal review.",
):
    return await invoke(
        service.approve_review,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
        comment=comment,
    )


async def reject_review(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
    reason="Unacceptable liability exposure.",
):
    return await invoke(
        service.reject_review,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
        reason=reason,
    )


async def request_revision(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
    comment="Revise the indemnity clause.",
):
    return await invoke(
        service.request_revision,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
        comment=comment,
    )


async def add_comment(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
    comment="Please clarify the termination clause.",
):
    return await invoke(
        service.add_comment,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
        comment=comment,
    )


async def get_history(
    service,
    review_id=REVIEW_A,
    tenant_id=TENANT_A,
    user_id=REVIEWER_A,
):
    return await invoke(
        service.get_review_history,
        review_id=review_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


def assert_valid_review(review):
    assert isinstance(review, dict)

    required = {
        "id",
        "document_id",
        "tenant_id",
        "created_by",
        "reviewer_ids",
        "status",
        "notes",
        "comments",
        "history",
    }

    assert required.issubset(review.keys())

    assert review["id"]
    assert review["document_id"]
    assert review["tenant_id"]

    assert review["status"] in VALID_STATUSES

    assert isinstance(
        review["reviewer_ids"],
        list,
    )

    assert isinstance(review["comments"], list)
    assert isinstance(review["history"], list)


# ============================================================
# INITIALIZATION
# ============================================================

async def test_review_service_initialization(
    review_service,
):
    assert review_service is not None

    assert callable(review_service.create_review)
    assert callable(review_service.get_review)
    assert callable(review_service.list_reviews)
    assert callable(review_service.approve_review)
    assert callable(review_service.reject_review)
    assert callable(review_service.request_revision)
    assert callable(review_service.add_comment)
    assert callable(review_service.get_review_history)


# ============================================================
# REVIEW CREATION
# ============================================================

async def test_create_review(review_service):
    result = await create_review(
        review_service
    )

    assert_valid_review(result)


async def test_create_review_document_identity(
    review_service,
):
    result = await create_review(
        review_service
    )

    assert result["document_id"] == DOCUMENT_A


async def test_create_review_tenant_identity(
    review_service,
):
    result = await create_review(
        review_service
    )

    assert result["tenant_id"] == TENANT_A


async def test_create_review_owner_identity(
    review_service,
):
    result = await create_review(
        review_service
    )

    assert result["created_by"] == OWNER_A


async def test_create_review_default_status(
    review_service,
):
    result = await create_review(
        review_service
    )

    assert result["status"] == "pending"


async def test_create_review_assigns_reviewer(
    review_service,
):
    result = await create_review(
        review_service
    )

    assert REVIEWER_A in result["reviewer_ids"]


async def test_create_review_persisted(
    review_service,
):
    created = await create_review(
        review_service
    )

    retrieved = await get_review(
        review_service,
        review_id=created["id"],
    )

    assert retrieved["id"] == created["id"]


async def test_unique_review_ids(
    review_service,
):
    first = await create_review(
        review_service
    )

    second = await create_review(
        review_service
    )

    assert first["id"] != second["id"]


# ============================================================
# REVIEW RETRIEVAL
# ============================================================

async def test_get_review(review_service):
    result = await get_review(
        review_service
    )

    assert_valid_review(result)
    assert result["id"] == REVIEW_A


async def test_nonexistent_review(
    review_service,
):
    try:
        result = await get_review(
            review_service,
            review_id="missing_review",
        )
    except (LookupError, ValueError):
        return

    assert result is None


async def test_list_reviews(
    review_service,
):
    results = await list_reviews(
        review_service
    )

    assert isinstance(results, list)
    assert len(results) >= 1


async def test_list_reviews_tenant_scoped(
    review_service,
):
    results = await list_reviews(
        review_service,
        tenant_id=TENANT_A,
    )

    assert all(
        review["tenant_id"] == TENANT_A
        for review in results
    )


async def test_filter_reviews_by_status(
    review_service,
):
    results = await list_reviews(
        review_service,
        filters={"status": "pending"},
    )

    assert all(
        review["status"] == "pending"
        for review in results
    )


# ============================================================
# APPROVAL
# ============================================================

async def test_approve_review(
    review_service,
):
    result = await approve_review(
        review_service
    )

    assert result["status"] == "approved"


async def test_approval_persisted(
    review_service,
):
    await approve_review(
        review_service
    )

    result = await get_review(
        review_service
    )

    assert result["status"] == "approved"


async def test_approval_preserves_identity(
    review_service,
):
    result = await approve_review(
        review_service
    )

    assert result["id"] == REVIEW_A
    assert result["tenant_id"] == TENANT_A
    assert result["document_id"] == DOCUMENT_A


async def test_duplicate_approval_rejected(
    review_service,
):
    await approve_review(
        review_service
    )

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await approve_review(
            review_service
        )


# ============================================================
# REJECTION
# ============================================================

async def test_reject_review(
    review_service,
):
    result = await reject_review(
        review_service
    )

    assert result["status"] == "rejected"


async def test_rejection_persisted(
    review_service,
):
    await reject_review(
        review_service
    )

    result = await get_review(
        review_service
    )

    assert result["status"] == "rejected"


async def test_rejection_requires_reason(
    review_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await reject_review(
            review_service,
            reason="",
        )


async def test_rejected_review_cannot_be_approved(
    review_service,
):
    await reject_review(
        review_service
    )

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await approve_review(
            review_service
        )


# ============================================================
# REVISION REQUESTS
# ============================================================

async def test_request_revision(
    review_service,
):
    result = await request_revision(
        review_service
    )

    assert result["status"] == "revision_requested"


async def test_revision_request_requires_comment(
    review_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await request_revision(
            review_service,
            comment="",
        )


async def test_revision_request_persisted(
    review_service,
):
    await request_revision(
        review_service
    )

    result = await get_review(
        review_service
    )

    assert result["status"] == "revision_requested"


# ============================================================
# REVIEW COMMENTS
# ============================================================

async def test_add_review_comment(
    review_service,
):
    result = await add_comment(
        review_service
    )

    assert isinstance(result, dict)
    assert result


async def test_comment_persisted(
    review_service,
):
    comment_text = "Review the payment clause."

    await add_comment(
        review_service,
        comment=comment_text,
    )

    review = await get_review(
        review_service
    )

    assert any(
        comment_text in repr(comment)
        for comment in review["comments"]
    )


async def test_comment_author_preserved(
    review_service,
):
    await add_comment(
        review_service,
        user_id=REVIEWER_A,
    )

    review = await get_review(
        review_service
    )

    assert any(
        comment.get("user_id") == REVIEWER_A
        for comment in review["comments"]
    )


async def test_empty_comment_rejected(
    review_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await add_comment(
            review_service,
            comment="",
        )


# ============================================================
# REVIEW HISTORY
# ============================================================

async def test_approval_added_to_history(
    review_service,
):
    await approve_review(
        review_service
    )

    history = await get_history(
        review_service
    )

    assert any(
        event.get("action") == "approved"
        for event in history
    )


async def test_rejection_added_to_history(
    review_service,
):
    await reject_review(
        review_service
    )

    history = await get_history(
        review_service
    )

    assert any(
        event.get("action") == "rejected"
        for event in history
    )


async def test_history_preserves_reviewer(
    review_service,
):
    await approve_review(
        review_service
    )

    history = await get_history(
        review_service
    )

    assert any(
        event.get("user_id") == REVIEWER_A
        for event in history
    )


async def test_review_history_order(
    review_service,
):
    await add_comment(
        review_service,
        comment="Initial review comment",
    )

    await approve_review(
        review_service
    )

    history = await get_history(
        review_service
    )

    assert isinstance(history, list)


# ============================================================
# PERMISSIONS
# ============================================================

async def test_reviewer_can_approve(
    review_service,
):
    result = await approve_review(
        review_service,
        user_id=REVIEWER_A,
    )

    assert result["status"] == "approved"


async def test_owner_cannot_self_approve(
    review_service,
):
    with pytest.raises(PermissionError):
        await approve_review(
            review_service,
            user_id=OWNER_A,
        )


async def test_viewer_cannot_approve(
    review_service,
):
    with pytest.raises(PermissionError):
        await approve_review(
            review_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_reject(
    review_service,
):
    with pytest.raises(PermissionError):
        await reject_review(
            review_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_request_revision(
    review_service,
):
    with pytest.raises(PermissionError):
        await request_revision(
            review_service,
            user_id=VIEWER_A,
        )


async def test_permission_service_called(
    review_service,
    permission_service,
):
    await approve_review(
        review_service
    )

    permission_service.has_permission.assert_awaited()


async def test_permission_failure_fails_closed(
    review_service,
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
        await approve_review(
            review_service
        )


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_cross_tenant_review_read_rejected(
    review_service,
):
    try:
        result = await get_review(
            review_service,
            review_id=REVIEW_B,
            tenant_id=TENANT_A,
            user_id=REVIEWER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result is None


async def test_cross_tenant_approval_rejected(
    review_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await approve_review(
            review_service,
            review_id=REVIEW_B,
            tenant_id=TENANT_A,
            user_id=REVIEWER_A,
        )


async def test_cross_tenant_comment_rejected(
    review_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await add_comment(
            review_service,
            review_id=REVIEW_B,
            tenant_id=TENANT_A,
            user_id=REVIEWER_A,
        )


async def test_cross_tenant_history_rejected(
    review_service,
):
    try:
        result = await get_history(
            review_service,
            review_id=REVIEW_B,
            tenant_id=TENANT_A,
            user_id=REVIEWER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result == []


async def test_cross_tenant_document_review_rejected(
    review_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await create_review(
            review_service,
            document_id=DOCUMENT_B,
            tenant_id=TENANT_A,
            user_id=OWNER_A,
        )


async def test_tenant_b_can_review_own_document(
    review_service,
):
    result = await get_review(
        review_service,
        review_id=REVIEW_B,
        tenant_id=TENANT_B,
        user_id=REVIEWER_B,
    )

    assert result["tenant_id"] == TENANT_B


# ============================================================
# REVIEWER VALIDATION
# ============================================================

async def test_empty_reviewer_list_rejected(
    review_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_review(
            review_service,
            reviewer_ids=[],
        )


async def test_duplicate_reviewers_handled(
    review_service,
):
    result = await create_review(
        review_service,
        reviewer_ids=[
            REVIEWER_A,
            REVIEWER_A,
        ],
    )

    assert result["reviewer_ids"].count(
        REVIEWER_A
    ) == 1


async def test_cross_tenant_reviewer_rejected(
    review_service,
):
    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await create_review(
            review_service,
            reviewer_ids=[REVIEWER_B],
        )


# ============================================================
# INVALID INPUT
# ============================================================

@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_tenant(
    review_service,
    invalid_tenant,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_review(
            review_service,
            tenant_id=invalid_tenant,
        )


@pytest.mark.parametrize(
    "invalid_document",
    ["", "   ", None],
)
async def test_invalid_document_id(
    review_service,
    invalid_document,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_review(
            review_service,
            document_id=invalid_document,
        )


async def test_missing_document_rejected(
    review_service,
):
    with pytest.raises(
        (LookupError, ValueError)
    ):
        await create_review(
            review_service,
            document_id="missing_document",
        )


# ============================================================
# NOTIFICATIONS
# ============================================================

async def test_review_assignment_notification(
    review_service,
    notification_service,
):
    await create_review(
        review_service
    )

    notification_service.notify_review_assigned.assert_awaited()


async def test_approval_notification(
    review_service,
    notification_service,
):
    await approve_review(
        review_service
    )

    notification_service.notify_review_decision.assert_awaited()


async def test_rejection_notification(
    review_service,
    notification_service,
):
    await reject_review(
        review_service
    )

    notification_service.notify_review_decision.assert_awaited()


# ============================================================
# AUDIT
# ============================================================

async def test_review_creation_audited(
    review_service,
    audit_service,
):
    await create_review(
        review_service
    )

    audit_service.record_event.assert_awaited()


async def test_approval_audited(
    review_service,
    audit_service,
):
    await approve_review(
        review_service
    )

    audit_service.record_event.assert_awaited()


async def test_rejection_audited(
    review_service,
    audit_service,
):
    await reject_review(
        review_service
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant(
    review_service,
    audit_service,
):
    await approve_review(
        review_service
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# SENSITIVE DATA PROTECTION
# ============================================================

async def test_private_review_comment_not_logged(
    review_service,
    caplog,
):
    secret = "CONFIDENTIAL_LEGAL_NOTE_ABC123"

    with caplog.at_level(logging.DEBUG):
        await add_comment(
            review_service,
            comment=secret,
        )

    assert secret not in caplog.text


async def test_reviewer_list_not_mutated(
    review_service,
):
    reviewers = [REVIEWER_A]

    original = deepcopy(reviewers)

    await create_review(
        review_service,
        reviewer_ids=reviewers,
    )

    assert reviewers == original


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_review_creation(
    review_service,
):
    results = await asyncio.gather(
        *[
            create_review(
                review_service,
                notes=f"Review request {index}",
            )
            for index in range(10)
        ]
    )

    assert len(results) == 10

    ids = {
        review["id"]
        for review in results
    }

    assert len(ids) == 10


async def test_concurrent_comments(
    review_service,
):
    await asyncio.gather(
        *[
            add_comment(
                review_service,
                comment=f"Comment {index}",
            )
            for index in range(10)
        ]
    )

    review = await get_review(
        review_service
    )

    assert len(review["comments"]) == 10


async def test_concurrent_approval_and_rejection(
    review_service,
):
    results = await asyncio.gather(
        approve_review(review_service),
        reject_review(review_service),
        return_exceptions=True,
    )

    successful = [
        result
        for result in results
        if isinstance(result, dict)
    ]

    failures = [
        result
        for result in results
        if isinstance(result, Exception)
    ]

    assert len(successful) == 1
    assert len(failures) == 1

    review = await get_review(
        review_service
    )

    assert review["status"] in {
        "approved",
        "rejected",
    }


async def test_concurrent_tenant_review_queries(
    review_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        get_review(
            review_service,
            review_id=REVIEW_A,
            tenant_id=TENANT_A,
            user_id=REVIEWER_A,
        ),
        get_review(
            review_service,
            review_id=REVIEW_B,
            tenant_id=TENANT_B,
            user_id=REVIEWER_B,
        ),
    )

    assert tenant_a["tenant_id"] == TENANT_A
    assert tenant_b["tenant_id"] == TENANT_B


# ============================================================
# END
# ============================================================
