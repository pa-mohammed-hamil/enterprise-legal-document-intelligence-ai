
"""
tests/integration/test_agent_pipeline.py

Enterprise Legal Document Intelligence AI
=========================================

Multi-Agent Pipeline Integration Tests

Expected module:
    backend.app.pipelines.agent_pipeline

Expected class:
    AgentPipeline

Expected constructor:
    AgentPipeline(
        supervisor_agent=...,
        clause_agent=...,
        risk_agent=...,
        expiry_agent=...,
        legal_qa_agent=...,
        review_agent=...,
        verification_agent=...,
        audit_service=...,
    )

Expected method:
    await pipeline.run(
        document_id: str,
        tenant_id: str,
        user_id: str,
        task: str,
    ) -> dict

Expected output:
    {
        "document_id": str,
        "tenant_id": str,
        "status": "completed" | "partial" |
                  "pending_review" | "failed",
        "agent_results": dict,
        "citations": list,
        "requires_human_review": bool,
    }

IMPORTANT:
    These are contract-based integration tests.

    Mocked agents replace external LLMs, databases,
    and retrieval infrastructure, but the real
    AgentPipeline implementation is under test.

    Update the documented interface to match your
    production implementation if necessary.

Run:
    pytest tests/integration/test_agent_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from copy import deepcopy
from typing import Any
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

CONTRACT_A = "contract_001"

VALID_STATUSES = {
    "completed",
    "partial",
    "pending_review",
    "failed",
}


# ============================================================
# PRODUCTION IMPORT
# ============================================================

def load_agent_pipeline():
    """Import the real production pipeline."""

    candidates = [
        "backend.app.pipelines.agent_pipeline",
        "app.pipelines.agent_pipeline",
    ]

    errors = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)

            pipeline_class = getattr(
                module,
                "AgentPipeline",
            )

            return pipeline_class

        except (ImportError, AttributeError) as exc:
            errors.append(
                f"{module_name}: {exc}"
            )

    pytest.fail(
        "AgentPipeline could not be imported.\n"
        "Expected one of:\n"
        + "\n".join(candidates)
        + "\nErrors:\n"
        + "\n".join(errors)
    )


# ============================================================
# SAMPLE DATA
# ============================================================

@pytest.fixture
def sample_document():
    return {
        "document_id": DOCUMENT_A,
        "contract_id": CONTRACT_A,
        "tenant_id": TENANT_A,
        "filename": "master_services_agreement.pdf",
        "document_type": "contract",
        "text": (
            "MASTER SERVICES AGREEMENT\n"
            "The supplier shall maintain confidentiality.\n"
            "Either party may terminate the agreement "
            "with thirty days written notice.\n"
            "The agreement renews automatically unless "
            "notice is provided sixty days before expiry.\n"
            "Liability shall not exceed fees paid during "
            "the preceding twelve months, except for "
            "indemnification obligations."
        ),
        "metadata": {
            "pages": 8,
            "language": "en",
            "version": 1,
        },
    }


@pytest.fixture
def sample_clauses():
    return [
        {
            "clause_id": "clause_001",
            "type": "confidentiality",
            "text": (
                "The supplier shall maintain "
                "confidentiality."
            ),
            "document_id": DOCUMENT_A,
            "page": 2,
            "confidence": 0.98,
        },
        {
            "clause_id": "clause_002",
            "type": "termination",
            "text": (
                "Either party may terminate with "
                "thirty days written notice."
            ),
            "document_id": DOCUMENT_A,
            "page": 4,
            "confidence": 0.96,
        },
        {
            "clause_id": "clause_003",
            "type": "liability",
            "text": (
                "Liability is capped except for "
                "indemnification obligations."
            ),
            "document_id": DOCUMENT_A,
            "page": 6,
            "confidence": 0.94,
        },
    ]


@pytest.fixture
def sample_risks():
    return [
        {
            "risk_id": "risk_001",
            "category": "liability",
            "severity": "high",
            "description": (
                "Indemnification carve-out may "
                "create uncapped exposure."
            ),
            "document_id": DOCUMENT_A,
            "clause_id": "clause_003",
            "confidence": 0.93,
        },
        {
            "risk_id": "risk_002",
            "category": "renewal",
            "severity": "medium",
            "description": (
                "Automatic renewal requires "
                "advance notice."
            ),
            "document_id": DOCUMENT_A,
            "confidence": 0.91,
        },
    ]


@pytest.fixture
def sample_expiry():
    return {
        "document_id": DOCUMENT_A,
        "expiry_date": "2027-01-14",
        "notice_deadline": "2026-11-15",
        "auto_renewal": True,
        "risk_level": "medium",
        "requires_attention": True,
    }


@pytest.fixture
def sample_qa():
    return {
        "answer": (
            "Either party may terminate the agreement "
            "with thirty days written notice."
        ),
        "citations": [
            {
                "document_id": DOCUMENT_A,
                "page": 4,
                "chunk_id": "chunk_004",
            }
        ],
        "grounded": True,
        "confidence": 0.96,
    }


@pytest.fixture
def sample_review():
    return {
        "document_id": DOCUMENT_A,
        "status": "pending_review",
        "requires_human_review": True,
        "reason": (
            "High-severity liability risk "
            "requires legal approval."
        ),
        "assigned_role": "legal_reviewer",
    }


@pytest.fixture
def sample_verification():
    return {
        "verified": True,
        "grounded": True,
        "citation_accuracy": 1.0,
        "unsupported_claims": [],
        "requires_human_review": False,
    }


# ============================================================
# MOCK AGENTS
# ============================================================

@pytest.fixture
def mock_agents(
    sample_clauses,
    sample_risks,
    sample_expiry,
    sample_qa,
    sample_review,
    sample_verification,
):
    """
    Each agent exposes an asynchronous run() method.

    The production pipeline must invoke these
    dependencies rather than bypassing them.
    """

    supervisor = MagicMock()
    supervisor.run = AsyncMock(
        return_value={
            "selected_agents": [
                "clause",
                "risk",
                "expiry",
                "legal_qa",
                "review",
                "verification",
            ],
            "execution_mode": "sequential",
            "status": "ready",
        }
    )

    clause = MagicMock()
    clause.run = AsyncMock(
        return_value={
            "clauses": deepcopy(sample_clauses),
            "status": "completed",
        }
    )

    risk = MagicMock()
    risk.run = AsyncMock(
        return_value={
            "risks": deepcopy(sample_risks),
            "status": "completed",
        }
    )

    expiry = MagicMock()
    expiry.run = AsyncMock(
        return_value={
            "expiry": deepcopy(sample_expiry),
            "status": "completed",
        }
    )

    legal_qa = MagicMock()
    legal_qa.run = AsyncMock(
        return_value=deepcopy(sample_qa)
    )

    review = MagicMock()
    review.run = AsyncMock(
        return_value=deepcopy(sample_review)
    )

    verification = MagicMock()
    verification.run = AsyncMock(
        return_value=deepcopy(sample_verification)
    )

    audit = MagicMock()
    audit.record_event = AsyncMock(
        return_value=True
    )

    return {
        "supervisor_agent": supervisor,
        "clause_agent": clause,
        "risk_agent": risk,
        "expiry_agent": expiry,
        "legal_qa_agent": legal_qa,
        "review_agent": review,
        "verification_agent": verification,
        "audit_service": audit,
    }


# ============================================================
# PIPELINE FIXTURE
# ============================================================

@pytest.fixture
def pipeline(mock_agents):
    """Create the real pipeline with mocked dependencies."""

    pipeline_class = load_agent_pipeline()

    try:
        return pipeline_class(**mock_agents)
    except TypeError as exc:
        pytest.fail(
            "AgentPipeline constructor does not match "
            "the documented dependency contract: "
            f"{exc}"
        )


# ============================================================
# EXECUTION HELPER
# ============================================================

async def execute_pipeline(
    pipeline,
    *,
    document_id=DOCUMENT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
    task="analyze_contract",
):
    """Execute a sync or async pipeline method."""

    result = pipeline.run(
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        task=task,
    )

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# OUTPUT ASSERTIONS
# ============================================================

def assert_pipeline_result(result):
    assert isinstance(result, dict)

    assert result.get("status") in VALID_STATUSES

    assert isinstance(
        result.get("agent_results"),
        dict,
    )

    assert isinstance(
        result.get("citations"),
        list,
    )

    assert isinstance(
        result.get("requires_human_review"),
        bool,
    )


# ============================================================
# BASIC EXECUTION TESTS
# ============================================================

@pytest.mark.asyncio
async def test_agent_pipeline_initialization(pipeline):
    assert pipeline is not None

    assert hasattr(pipeline, "run")

    assert callable(pipeline.run)


@pytest.mark.asyncio
async def test_agent_pipeline_success(pipeline):
    result = await execute_pipeline(pipeline)

    assert_pipeline_result(result)

    assert result["document_id"] == DOCUMENT_A
    assert result["tenant_id"] == TENANT_A

    assert result["status"] in {
        "completed",
        "pending_review",
    }


@pytest.mark.asyncio
async def test_pipeline_returns_agent_results(pipeline):
    result = await execute_pipeline(pipeline)

    agent_results = result["agent_results"]

    assert isinstance(agent_results, dict)
    assert len(agent_results) > 0


# ============================================================
# SUPERVISOR ORCHESTRATION
# ============================================================

@pytest.mark.asyncio
async def test_supervisor_agent_is_invoked(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "supervisor_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_supervisor_selects_required_agents(
    pipeline,
    mock_agents,
):
    result = await execute_pipeline(pipeline)

    assert result["status"] in VALID_STATUSES

    mock_agents[
        "supervisor_agent"
    ].run.assert_awaited()


# ============================================================
# CLAUSE AGENT
# ============================================================

@pytest.mark.asyncio
async def test_clause_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "clause_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_clause_results_are_preserved(
    pipeline,
    sample_clauses,
):
    result = await execute_pipeline(pipeline)

    clause_result = result[
        "agent_results"
    ].get("clause")

    assert clause_result is not None

    assert clause_result["clauses"] == sample_clauses


# ============================================================
# RISK AGENT
# ============================================================

@pytest.mark.asyncio
async def test_risk_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "risk_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_risk_results_are_preserved(
    pipeline,
    sample_risks,
):
    result = await execute_pipeline(pipeline)

    risk_result = result[
        "agent_results"
    ].get("risk")

    assert risk_result is not None

    assert risk_result["risks"] == sample_risks


@pytest.mark.asyncio
async def test_high_risk_triggers_human_review(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    assert result["requires_human_review"] is True

    assert result["status"] in {
        "pending_review",
        "partial",
    }


# ============================================================
# EXPIRY AGENT
# ============================================================

@pytest.mark.asyncio
async def test_expiry_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "expiry_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_expiry_information_preserved(
    pipeline,
    sample_expiry,
):
    result = await execute_pipeline(pipeline)

    expiry_result = result[
        "agent_results"
    ].get("expiry")

    assert expiry_result is not None

    assert expiry_result["expiry"] == sample_expiry


# ============================================================
# LEGAL QA AGENT
# ============================================================

@pytest.mark.asyncio
async def test_legal_qa_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "legal_qa_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_legal_qa_citations_are_preserved(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    qa_result = result[
        "agent_results"
    ].get("legal_qa")

    assert qa_result is not None

    assert qa_result["grounded"] is True

    assert len(qa_result["citations"]) > 0


# ============================================================
# REVIEW AGENT
# ============================================================

@pytest.mark.asyncio
async def test_review_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "review_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_review_escalation(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    review_result = result[
        "agent_results"
    ].get("review")

    assert review_result is not None

    assert review_result["requires_human_review"] is True


# ============================================================
# VERIFICATION AGENT
# ============================================================

@pytest.mark.asyncio
async def test_verification_agent_execution(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "verification_agent"
    ].run.assert_awaited()


@pytest.mark.asyncio
async def test_verified_evidence(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    verification = result[
        "agent_results"
    ].get("verification")

    assert verification is not None

    assert verification["verified"] is True


@pytest.mark.asyncio
async def test_unverified_evidence_requires_review(
    pipeline,
    mock_agents,
):
    mock_agents[
        "verification_agent"
    ].run.return_value = {
        "verified": False,
        "grounded": False,
        "citation_accuracy": 0.0,
        "unsupported_claims": [
            "Unsupported contractual claim"
        ],
        "requires_human_review": True,
    }

    result = await execute_pipeline(pipeline)

    assert result["requires_human_review"] is True

    assert result["status"] != "completed"


# ============================================================
# TENANT ISOLATION
# ============================================================

@pytest.mark.asyncio
async def test_tenant_context_preserved(
    pipeline,
):
    result = await execute_pipeline(
        pipeline,
        tenant_id=TENANT_A,
    )

    assert result["tenant_id"] == TENANT_A


@pytest.mark.asyncio
async def test_different_tenants_have_distinct_context(
    pipeline,
):
    first = await execute_pipeline(
        pipeline,
        document_id=DOCUMENT_A,
        tenant_id=TENANT_A,
        user_id=USER_A,
    )

    second = await execute_pipeline(
        pipeline,
        document_id=DOCUMENT_B,
        tenant_id=TENANT_B,
        user_id=USER_B,
    )

    assert first["tenant_id"] == TENANT_A
    assert second["tenant_id"] == TENANT_B

    assert first["document_id"] == DOCUMENT_A
    assert second["document_id"] == DOCUMENT_B


@pytest.mark.asyncio
async def test_agent_calls_include_tenant_context(
    pipeline,
    mock_agents,
):
    await execute_pipeline(
        pipeline,
        tenant_id=TENANT_A,
    )

    agent = mock_agents["supervisor_agent"]

    assert agent.run.await_count >= 1

    call = agent.run.await_args

    serialized = str(call)

    assert TENANT_A in serialized


# ============================================================
# AUDIT LOGGING
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_creates_audit_event(
    pipeline,
    mock_agents,
):
    await execute_pipeline(pipeline)

    mock_agents[
        "audit_service"
    ].record_event.assert_awaited()


# ============================================================
# FAILURE HANDLING
# ============================================================

@pytest.mark.asyncio
async def test_clause_agent_failure(
    pipeline,
    mock_agents,
):
    mock_agents[
        "clause_agent"
    ].run.side_effect = RuntimeError(
        "Clause extraction failed"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        return

    assert result["status"] in {
        "partial",
        "failed",
        "pending_review",
    }


@pytest.mark.asyncio
async def test_risk_agent_failure(
    pipeline,
    mock_agents,
):
    mock_agents[
        "risk_agent"
    ].run.side_effect = RuntimeError(
        "Risk analysis unavailable"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        return

    assert result["status"] in {
        "partial",
        "failed",
        "pending_review",
    }


@pytest.mark.asyncio
async def test_supervisor_failure(
    pipeline,
    mock_agents,
):
    mock_agents[
        "supervisor_agent"
    ].run.side_effect = RuntimeError(
        "Supervisor unavailable"
    )

    try:
        result = await execute_pipeline(pipeline)
    except RuntimeError:
        return

    assert result["status"] == "failed"


# ============================================================
# INPUT VALIDATION
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document_id",
    [
        "",
        " ",
        None,
    ],
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

    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tenant_id",
    [
        "",
        " ",
        None,
    ],
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

    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_invalid_task(
    pipeline,
):
    try:
        result = await execute_pipeline(
            pipeline,
            task="unknown_task",
        )
    except (ValueError, TypeError):
        return

    assert result["status"] == "failed"


# ============================================================
# CONCURRENT EXECUTION
# ============================================================

@pytest.mark.asyncio
async def test_concurrent_pipeline_execution(
    pipeline,
):
    tasks = [
        execute_pipeline(
            pipeline,
            document_id=f"doc_{index}",
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
        for index in range(5)
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 5

    for index, result in enumerate(results):
        assert_pipeline_result(result)

        assert result["document_id"] == (
            f"doc_{index}"
        )


# ============================================================
# RESULT SERIALIZATION
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_output_json_serializable(
    pipeline,
):
    import json

    result = await execute_pipeline(pipeline)

    encoded = json.dumps(result)

    decoded = json.loads(encoded)

    assert decoded["document_id"] == DOCUMENT_A


# ============================================================
# NO UNSUPPORTED SUCCESS
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_does_not_claim_success_when_verification_fails(
    pipeline,
    mock_agents,
):
    mock_agents[
        "verification_agent"
    ].run.return_value = {
        "verified": False,
        "grounded": False,
        "citation_accuracy": 0.0,
        "unsupported_claims": [
            "Unverified liability interpretation"
        ],
        "requires_human_review": True,
    }

    result = await execute_pipeline(pipeline)

    assert result["status"] != "completed"

    assert result["requires_human_review"] is True


# ============================================================
# PIPELINE RESULT CONSISTENCY
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_result_schema(
    pipeline,
):
    result = await execute_pipeline(pipeline)

    required_fields = {
        "document_id",
        "tenant_id",
        "status",
        "agent_results",
        "citations",
        "requires_human_review",
    }

    assert required_fields.issubset(result.keys())


@pytest.mark.asyncio
async def test_pipeline_does_not_mutate_agent_outputs(
    pipeline,
    mock_agents,
    sample_clauses,
):
    original = deepcopy(sample_clauses)

    await execute_pipeline(pipeline)

    returned = mock_agents[
        "clause_agent"
    ].run.return_value

    assert returned["clauses"] == original
