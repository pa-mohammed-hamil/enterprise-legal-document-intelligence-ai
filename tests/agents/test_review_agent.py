
"""
tests/agents/test_review_agent.py

Enterprise Legal Document Intelligence AI
=========================================

Review Agent test suite.

Run:
    pytest tests/agents/test_review_agent.py -v

Dependencies:
    pytest
    pytest-asyncio

Reference interface:

    ReviewAgent(
        review_repository=...,
        audit_repository=...,
        llm=...,
    )

    await agent.create_review(
        document_id,
        tenant_id,
        requested_by,
        reviewer_id,
        priority,
    )

    await agent.analyze_review(
        review_id,
        tenant_id,
        actor_id,
    )

    await agent.assign_reviewer(
        review_id,
        tenant_id,
        actor_id,
        reviewer_id,
    )

    await agent.submit_decision(
        review_id,
        tenant_id,
        actor_id,
        decision,
        comments,
    )

    await agent.get_review(
        review_id,
        tenant_id,
        actor_id,
    )

Expected return values are dictionaries.

This suite defines expected behavior. It does not
claim compatibility with an uninspected implementation.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest


# ============================================================
# IMPORT
# ============================================================

try:
    from backend.app.agents.review_agent import ReviewAgent
except ImportError:
    try:
        from app.agents.review_agent import ReviewAgent
    except ImportError:
        ReviewAgent = None


# ============================================================
# TEST CONSTANTS
# ============================================================

TENANT_A = "TENANT-A"
TENANT_B = "TENANT-B"

ADMIN = "USER-ADMIN"
REVIEWER = "USER-REVIEWER"
REVIEWER_2 = "USER-REVIEWER-2"
REQUESTER = "USER-REQUESTER"
OUTSIDER = "USER-OUTSIDER"

DOCUMENT_ID = "DOC-2026-00128"

VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
    "critical",
}

VALID_STATUSES = {
    "pending",
    "in_review",
    "approved",
    "rejected",
    "changes_requested",
}

FINAL_STATUSES = {
    "approved",
    "rejected",
}


# ============================================================
# FAKE REVIEW REPOSITORY
# ============================================================

class FakeReviewRepository:
    """
    In-memory repository with tenant-aware lookups.

    This mock enforces tenant filtering itself.

    Passing tenant tests with this fake repository does
    not prove that the production agent or database
    independently enforces tenant isolation.
    """

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.sequence = 0
        self.lock = asyncio.Lock()

    async def create(
        self,
        *,
        document_id: str,
        tenant_id: str,
        requested_by: str,
        reviewer_id: str,
        priority: str,
    ) -> dict[str, Any]:

        async with self.lock:
            self.sequence += 1
            review_id = f"REV-{self.sequence:05d}"

            review = {
                "id": review_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "requested_by": requested_by,
                "reviewer_id": reviewer_id,
                "priority": priority,
                "status": "pending",
                "ai_recommendation": None,
                "decision": None,
                "comments": "",
                "version": 1,
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            self.records[review_id] = deepcopy(review)

            self.calls.append({
                "method": "create",
                "tenant_id": tenant_id,
                "review_id": review_id,
            })

            return deepcopy(review)

    async def get(
        self,
        review_id: str,
        tenant_id: str,
    ) -> dict[str, Any] | None:

        self.calls.append({
            "method": "get",
            "tenant_id": tenant_id,
            "review_id": review_id,
        })

        review = self.records.get(review_id)

        if review is None:
            return None

        if review["tenant_id"] != tenant_id:
            return None

        return deepcopy(review)

    async def update(
        self,
        review_id: str,
        tenant_id: str,
        changes: dict[str, Any],
        expected_version: int | None = None,
    ) -> dict[str, Any]:

        async with self.lock:
            review = self.records.get(review_id)

            if review is None:
                raise ValueError("Review not found")

            if review["tenant_id"] != tenant_id:
                raise PermissionError("Tenant mismatch")

            if (
                expected_version is not None
                and review["version"] != expected_version
            ):
                raise ValueError("Version conflict")

            review.update(deepcopy(changes))
            review["version"] += 1

            self.calls.append({
                "method": "update",
                "tenant_id": tenant_id,
                "review_id": review_id,
                "changes": deepcopy(changes),
            })

            return deepcopy(review)


# ============================================================
# FAKE AUDIT REPOSITORY
# ============================================================

class FakeAuditRepository:

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        review_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        event = {
            "tenant_id": tenant_id,
            "actor_id": actor_id,
            "action": action,
            "review_id": review_id,
            "metadata": deepcopy(metadata or {}),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self.events.append(event)

        return deepcopy(event)


# ============================================================
# FAKE LEGAL REVIEW LLM
# ============================================================

class FakeReviewLLM:
    """
    Produces deterministic recommendations.

    The output is advisory only.
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def analyze(
        self,
        document_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        self.calls.append({
            "document_id": document_id,
            "kwargs": kwargs,
        })

        return {
            "summary": (
                "Illustrative review of contract clauses "
                "and identified legal issues."
            ),
            "recommendation": "manual_review",
            "risk_level": "high",
            "findings": [
                {
                    "category": "indemnification",
                    "severity": "high",
                    "description": (
                        "Broad indemnification language "
                        "requires legal assessment."
                    ),
                    "document_id": document_id,
                },
                {
                    "category": "termination",
                    "severity": "medium",
                    "description": (
                        "Termination notice terms "
                        "require confirmation."
                    ),
                    "document_id": document_id,
                },
            ],
            "confidence": 0.82,
            "requires_human_approval": True,
        }


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def review_repository():
    return FakeReviewRepository()


@pytest.fixture
def audit_repository():
    return FakeAuditRepository()


@pytest.fixture
def llm():
    return FakeReviewLLM()


@pytest.fixture
def agent(
    review_repository,
    audit_repository,
    llm,
):
    if ReviewAgent is None:
        pytest.fail(
            "ReviewAgent import failed. "
            "Update the import path to match your project."
        )

    return ReviewAgent(
        review_repository=review_repository,
        audit_repository=audit_repository,
        llm=llm,
    )


# ============================================================
# ADAPTER HELPERS
# ============================================================

async def call_agent(
    agent: Any,
    method_name: str,
    **kwargs: Any,
) -> dict[str, Any]:

    method = getattr(agent, method_name)

    result = method(**kwargs)

    if inspect.isawaitable(result):
        result = await result

    assert isinstance(result, dict), (
        f"{method_name} must return a dictionary "
        "under the reference contract."
    )

    return result


async def create_review(
    agent: Any,
    *,
    document_id: str = DOCUMENT_ID,
    tenant_id: str = TENANT_A,
    requested_by: str = REQUESTER,
    reviewer_id: str = REVIEWER,
    priority: str = "high",
) -> dict[str, Any]:

    return await call_agent(
        agent,
        "create_review",
        document_id=document_id,
        tenant_id=tenant_id,
        requested_by=requested_by,
        reviewer_id=reviewer_id,
        priority=priority,
    )


async def analyze_review(
    agent: Any,
    review_id: str,
    tenant_id: str = TENANT_A,
    actor_id: str = REVIEWER,
) -> dict[str, Any]:

    return await call_agent(
        agent,
        "analyze_review",
        review_id=review_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
    )


async def assign_reviewer(
    agent: Any,
    review_id: str,
    reviewer_id: str,
    tenant_id: str = TENANT_A,
    actor_id: str = ADMIN,
) -> dict[str, Any]:

    return await call_agent(
        agent,
        "assign_reviewer",
        review_id=review_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        reviewer_id=reviewer_id,
    )


async def submit_decision(
    agent: Any,
    review_id: str,
    decision: str,
    comments: str,
    tenant_id: str = TENANT_A,
    actor_id: str = REVIEWER,
) -> dict[str, Any]:

    return await call_agent(
        agent,
        "submit_decision",
        review_id=review_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        decision=decision,
        comments=comments,
    )


# ============================================================
# REVIEW CREATION
# ============================================================

@pytest.mark.asyncio
async def test_create_review(agent):
    result = await create_review(agent)

    assert result["id"]
    assert result["document_id"] == DOCUMENT_ID
    assert result["tenant_id"] == TENANT_A
    assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_create_review_assigns_reviewer(agent):
    result = await create_review(
        agent,
        reviewer_id=REVIEWER_2,
    )

    assert result["reviewer_id"] == REVIEWER_2


@pytest.mark.asyncio
async def test_create_review_priority(agent):
    result = await create_review(
        agent,
        priority="critical",
    )

    assert result["priority"] == "critical"


@pytest.mark.asyncio
async def test_unique_review_identifiers(agent):
    first = await create_review(agent)
    second = await create_review(agent)

    assert first["id"] != second["id"]


@pytest.mark.asyncio
async def test_review_persisted_in_repository(
    agent,
    review_repository,
):
    created = await create_review(agent)

    stored = await review_repository.get(
        created["id"],
        TENANT_A,
    )

    assert stored is not None
    assert stored["id"] == created["id"]


@pytest.mark.asyncio
async def test_creation_audited(
    agent,
    audit_repository,
):
    created = await create_review(agent)

    assert any(
        event["review_id"] == created["id"]
        and event["actor_id"] == REQUESTER
        for event in audit_repository.events
    )


# ============================================================
# VALIDATION
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", ""),
        ("tenant_id", ""),
        ("requested_by", ""),
        ("reviewer_id", ""),
        ("priority", "invalid"),
    ],
)
async def test_invalid_creation_rejected(
    agent,
    field,
    value,
):
    kwargs = {field: value}

    with pytest.raises(
        (ValueError, TypeError, PermissionError)
    ):
        await create_review(agent, **kwargs)


@pytest.mark.asyncio
async def test_missing_review_rejected(agent):
    with pytest.raises(
        (ValueError, LookupError, PermissionError)
    ):
        await analyze_review(
            agent,
            "REV-DOES-NOT-EXIST",
        )


# ============================================================
# AI REVIEW ANALYSIS
# ============================================================

@pytest.mark.asyncio
async def test_ai_analysis_returns_summary(agent):
    review = await create_review(agent)

    result = await analyze_review(
        agent,
        review["id"],
    )

    assert isinstance(result["summary"], str)
    assert result["summary"].strip()


@pytest.mark.asyncio
async def test_ai_analysis_returns_findings(agent):
    review = await create_review(agent)

    result = await analyze_review(
        agent,
        review["id"],
    )

    assert isinstance(result["findings"], list)
    assert len(result["findings"]) > 0


@pytest.mark.asyncio
async def test_ai_analysis_calls_llm(agent, llm):
    review = await create_review(agent)

    await analyze_review(
        agent,
        review["id"],
    )

    assert llm.calls
    assert llm.calls[0]["document_id"] == DOCUMENT_ID


@pytest.mark.asyncio
async def test_ai_recommendation_requires_human_approval(agent):
    review = await create_review(agent)

    result = await analyze_review(
        agent,
        review["id"],
    )

    assert result["requires_human_approval"] is True
    assert result["recommendation"] == "manual_review"


@pytest.mark.asyncio
async def test_ai_analysis_does_not_approve_review(
    agent,
    review_repository,
):
    review = await create_review(agent)

    await analyze_review(
        agent,
        review["id"],
    )

    stored = await review_repository.get(
        review["id"],
        TENANT_A,
    )

    assert stored["status"] != "approved"
    assert stored["decision"] != "approved"


@pytest.mark.asyncio
async def test_ai_confidence_in_valid_range(agent):
    review = await create_review(agent)

    result = await analyze_review(
        agent,
        review["id"],
    )

    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_ai_findings_reference_correct_document(agent):
    review = await create_review(agent)

    result = await analyze_review(
        agent,
        review["id"],
    )

    assert all(
        finding["document_id"] == DOCUMENT_ID
        for finding in result["findings"]
    )


# ============================================================
# REVIEWER ASSIGNMENT
# ============================================================

@pytest.mark.asyncio
async def test_assign_reviewer(agent):
    review = await create_review(agent)

    updated = await assign_reviewer(
        agent,
        review["id"],
        REVIEWER_2,
    )

    assert updated["reviewer_id"] == REVIEWER_2


@pytest.mark.asyncio
async def test_assignment_persisted(
    agent,
    review_repository,
):
    review = await create_review(agent)

    await assign_reviewer(
        agent,
        review["id"],
        REVIEWER_2,
    )

    stored = await review_repository.get(
        review["id"],
        TENANT_A,
    )

    assert stored["reviewer_id"] == REVIEWER_2


@pytest.mark.asyncio
async def test_assignment_audited(
    agent,
    audit_repository,
):
    review = await create_review(agent)

    await assign_reviewer(
        agent,
        review["id"],
        REVIEWER_2,
    )

    assert any(
        event["review_id"] == review["id"]
        and event["actor_id"] == ADMIN
        and "assign" in event["action"].lower()
        for event in audit_repository.events
    )


# ============================================================
# HUMAN REVIEW DECISIONS
# ============================================================

@pytest.mark.asyncio
async def test_approve_review(agent):
    review = await create_review(agent)

    result = await submit_decision(
        agent,
        review["id"],
        "approved",
        "Legal review completed.",
    )

    assert result["status"] == "approved"


@pytest.mark.asyncio
async def test_reject_review(agent):
    review = await create_review(agent)

    result = await submit_decision(
        agent,
        review["id"],
        "rejected",
        "Unacceptable liability terms.",
    )

    assert result["status"] == "rejected"


@pytest.mark.asyncio
async def test_request_changes(agent):
    review = await create_review(agent)

    result = await submit_decision(
        agent,
        review["id"],
        "changes_requested",
        "Revise indemnification provisions.",
    )

    assert result["status"] == "changes_requested"


@pytest.mark.asyncio
async def test_decision_comments_preserved(agent):
    review = await create_review(agent)

    result = await submit_decision(
        agent,
        review["id"],
        "approved",
        "Reviewed all required clauses.",
    )

    assert (
        result["comments"]
        == "Reviewed all required clauses."
    )


@pytest.mark.asyncio
async def test_decision_persisted(
    agent,
    review_repository,
):
    review = await create_review(agent)

    await submit_decision(
        agent,
        review["id"],
        "approved",
        "Approved by legal reviewer.",
    )

    stored = await review_repository.get(
        review["id"],
        TENANT_A,
    )

    assert stored["status"] == "approved"


@pytest.mark.asyncio
async def test_decision_audited(
    agent,
    audit_repository,
):
    review = await create_review(agent)

    await submit_decision(
        agent,
        review["id"],
        "approved",
        "Approved after review.",
    )

    assert any(
        event["review_id"] == review["id"]
        and event["actor_id"] == REVIEWER
        and "approv" in event["action"].lower()
        for event in audit_repository.events
    )


# ============================================================
# DECISION VALIDATION
# ============================================================

@pytest.mark.asyncio
async def test_invalid_decision_rejected(agent):
    review = await create_review(agent)

    with pytest.raises((ValueError, TypeError)):
        await submit_decision(
            agent,
            review["id"],
            "auto_approve_everything",
            "Invalid decision.",
        )


@pytest.mark.asyncio
async def test_empty_decision_comment_rejected(agent):
    review = await create_review(agent)

    with pytest.raises((ValueError, TypeError)):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "",
        )


@pytest.mark.asyncio
async def test_double_approval_rejected(agent):
    review = await create_review(agent)

    await submit_decision(
        agent,
        review["id"],
        "approved",
        "First approval.",
    )

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "Second approval.",
        )


@pytest.mark.asyncio
async def test_rejected_review_cannot_be_approved_directly(agent):
    review = await create_review(agent)

    await submit_decision(
        agent,
        review["id"],
        "rejected",
        "Contract rejected.",
    )

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "Attempt to override rejection.",
        )


# ============================================================
# TENANT ISOLATION
# ============================================================

@pytest.mark.asyncio
async def test_cross_tenant_review_access_denied(agent):
    review = await create_review(
        agent,
        tenant_id=TENANT_A,
    )

    with pytest.raises(
        (ValueError, LookupError, PermissionError)
    ):
        await call_agent(
            agent,
            "get_review",
            review_id=review["id"],
            tenant_id=TENANT_B,
            actor_id=REVIEWER,
        )


@pytest.mark.asyncio
async def test_cross_tenant_approval_denied(agent):
    review = await create_review(
        agent,
        tenant_id=TENANT_A,
    )

    with pytest.raises(
        (ValueError, LookupError, PermissionError)
    ):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "Unauthorized cross-tenant approval.",
            tenant_id=TENANT_B,
        )


@pytest.mark.asyncio
async def test_cross_tenant_assignment_denied(agent):
    review = await create_review(
        agent,
        tenant_id=TENANT_A,
    )

    with pytest.raises(
        (ValueError, LookupError, PermissionError)
    ):
        await assign_reviewer(
            agent,
            review["id"],
            REVIEWER_2,
            tenant_id=TENANT_B,
        )


# ============================================================
# ACCESS CONTROL
# ============================================================

@pytest.mark.asyncio
async def test_unassigned_user_cannot_approve(agent):
    review = await create_review(agent)

    with pytest.raises(PermissionError):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "Unauthorized approval attempt.",
            actor_id=OUTSIDER,
        )


@pytest.mark.asyncio
async def test_requester_cannot_self_approve(agent):
    review = await create_review(agent)

    with pytest.raises(PermissionError):
        await submit_decision(
            agent,
            review["id"],
            "approved",
            "Self-approval attempt.",
            actor_id=REQUESTER,
        )


@pytest.mark.asyncio
async def test_unassigned_user_cannot_reassign(agent):
    review = await create_review(agent)

    with pytest.raises(PermissionError):
        await assign_reviewer(
            agent,
            review["id"],
            REVIEWER_2,
            actor_id=OUTSIDER,
        )


# ============================================================
# AUDIT INTEGRITY
# ============================================================

@pytest.mark.asyncio
async def test_audit_records_correct_tenant(
    agent,
    audit_repository,
):
    review = await create_review(
        agent,
        tenant_id=TENANT_A,
    )

    await submit_decision(
        agent,
        review["id"],
        "approved",
        "Reviewed.",
    )

    relevant = [
        event
        for event in audit_repository.events
        if event["review_id"] == review["id"]
    ]

    assert relevant
    assert all(
        event["tenant_id"] == TENANT_A
        for event in relevant
    )


@pytest.mark.asyncio
async def test_audit_events_have_timestamps(
    agent,
    audit_repository,
):
    await create_review(agent)

    assert audit_repository.events

    for event in audit_repository.events:
        assert event["timestamp"]
        assert datetime.fromisoformat(
            event["timestamp"]
        ).tzinfo is not None


# ============================================================
# CONCURRENCY
# ============================================================

@pytest.mark.asyncio
async def test_concurrent_review_creation_unique_ids(agent):
    results = await asyncio.gather(
        *(create_review(agent) for _ in range(10))
    )

    ids = [result["id"] for result in results]

    assert len(ids) == 10
    assert len(set(ids)) == 10


@pytest.mark.asyncio
async def test_concurrent_final_decisions_not_both_committed(
    agent,
    review_repository,
):
    review = await create_review(agent)

    async def approve():
        return await submit_decision(
            agent,
            review["id"],
            "approved",
            "Approval decision.",
        )

    async def reject():
        return await submit_decision(
            agent,
            review["id"],
            "rejected",
            "Rejection decision.",
        )

    results = await asyncio.gather(
        approve(),
        reject(),
        return_exceptions=True,
    )

    successful = [
        result
        for result in results
        if isinstance(result, dict)
    ]

    assert len(successful) == 1

    stored = await review_repository.get(
        review["id"],
        TENANT_A,
    )

    assert stored["status"] in FINAL_STATUSES
    assert stored["status"] == successful[0]["status"]


# ============================================================
# RESULT STRUCTURE
# ============================================================

@pytest.mark.asyncio
async def test_review_required_fields(agent):
    review = await create_review(agent)

    required = {
        "id",
        "document_id",
        "tenant_id",
        "reviewer_id",
        "requested_by",
        "priority",
        "status",
    }

    assert required.issubset(review.keys())


@pytest.mark.asyncio
async def test_review_json_serializable(agent):
    import json

    review = await create_review(agent)

    serialized = json.dumps(review)
    restored = json.loads(serialized)

    assert restored["id"] == review["id"]


@pytest.mark.asyncio
async def test_review_status_is_valid(agent):
    review = await create_review(agent)

    assert review["status"] in VALID_STATUSES


@pytest.mark.asyncio
async def test_review_priority_is_valid(agent):
    review = await create_review(agent)

    assert review["priority"] in VALID_PRIORITIES
