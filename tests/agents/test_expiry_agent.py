
"""
tests/agents/test_expiry_agent.py

Enterprise Legal Document Intelligence AI
-----------------------------------------
Tests for contract expiry and renewal intelligence.

Run:
    pytest tests/agents/test_expiry_agent.py -v

Coverage:
    - Contract expiry detection
    - Renewal notice calculations
    - Overdue deadline detection
    - Automatic renewal handling
    - Risk prioritization
    - Invalid and missing dates
    - Tenant isolation
    - Alert recommendation generation
    - Batch analysis
    - Async agent execution

Integration contract:
    ExpiryAgent.analyze_contract(contract: dict) -> dict
    ExpiryAgent.analyze_batch(contracts: list[dict]) -> list[dict]

Expected result keys:
    contract_id
    tenant_id
    expiry_date
    days_until_expiry
    notice_deadline
    days_until_notice
    risk_level
    status
    requires_attention
    recommended_actions

If your existing agent uses different methods or schemas,
adapt the fixture below without weakening the assertions.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta
from typing import Any

import pytest


# ============================================================
# IMPORT
# ============================================================

try:
    from backend.app.agents.expiry_agent import ExpiryAgent
except ImportError:
    try:
        from app.agents.expiry_agent import ExpiryAgent
    except ImportError:
        ExpiryAgent = None


# ============================================================
# FIXED TEST CLOCK
# ============================================================

TODAY = date(2026, 10, 8)


def iso(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat()


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def agent():
    if ExpiryAgent is None:
        pytest.fail(
            "ExpiryAgent could not be imported. "
            "Update the import path to match your repository."
        )

    return ExpiryAgent()


@pytest.fixture
def contract_factory():
    def create(
        contract_id: str = "CON-001",
        tenant_id: str = "TENANT-A",
        expiry_days: int | None = 90,
        notice_days: int = 30,
        auto_renewal: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        contract = {
            "contract_id": contract_id,
            "tenant_id": tenant_id,
            "title": "Enterprise Services Agreement",
            "effective_date": iso(-180),
            "expiry_date": (
                iso(expiry_days)
                if expiry_days is not None
                else None
            ),
            "notice_period_days": notice_days,
            "auto_renewal": auto_renewal,
            "status": "active",
            "counterparty": "Example Corporation",
        }

        contract.update(overrides)
        return contract

    return create


# ============================================================
# AGENT ADAPTER
# ============================================================

async def analyze(
    agent: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """
    Call the agent using the documented reference interface.

    Supports synchronous or asynchronous implementations.
    """
    result = agent.analyze_contract(contract)

    if inspect.isawaitable(result):
        result = await result

    assert isinstance(result, dict), (
        "ExpiryAgent.analyze_contract must return a dictionary "
        "for this test contract."
    )

    return result


async def analyze_batch(
    agent: Any,
    contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = agent.analyze_batch(contracts)

    if inspect.isawaitable(result):
        result = await result

    assert isinstance(result, list)
    return result


# ============================================================
# BASIC EXPIRY TESTS
# ============================================================

@pytest.mark.asyncio
async def test_contract_expiry_date_preserved(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=90)

    result = await analyze(agent, contract)

    assert result["contract_id"] == "CON-001"
    assert result["expiry_date"] == iso(90)


@pytest.mark.asyncio
async def test_days_until_expiry(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=45)

    result = await analyze(agent, contract)

    assert result["days_until_expiry"] == 45


@pytest.mark.asyncio
async def test_contract_expiring_today(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=0)

    result = await analyze(agent, contract)

    assert result["days_until_expiry"] == 0
    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_contract_already_expired(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=-5)

    result = await analyze(agent, contract)

    assert result["days_until_expiry"] == -5
    assert result["status"] == "expired"
    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_future_contract_not_expired(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=180)

    result = await analyze(agent, contract)

    assert result["status"] != "expired"


# ============================================================
# NOTICE DEADLINE TESTS
# ============================================================

@pytest.mark.asyncio
async def test_notice_deadline_calculation(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=90,
        notice_days=30,
    )

    result = await analyze(agent, contract)

    assert result["notice_deadline"] == iso(60)


@pytest.mark.asyncio
async def test_notice_deadline_today(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=30,
        notice_days=30,
    )

    result = await analyze(agent, contract)

    assert result["days_until_notice"] == 0
    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_notice_deadline_overdue(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=20,
        notice_days=30,
    )

    result = await analyze(agent, contract)

    assert result["days_until_notice"] == -10
    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_zero_day_notice_period(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=15,
        notice_days=0,
    )

    result = await analyze(agent, contract)

    assert result["notice_deadline"] == iso(15)


@pytest.mark.asyncio
async def test_long_notice_period(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=60,
        notice_days=90,
    )

    result = await analyze(agent, contract)

    assert result["notice_deadline"] == iso(-30)
    assert result["days_until_notice"] == -30


# ============================================================
# RISK CLASSIFICATION
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expiry_days,expected_attention",
    [
        (0, True),
        (7, True),
        (15, True),
        (30, True),
        (180, False),
    ],
)
async def test_attention_thresholds(
    agent,
    contract_factory,
    expiry_days,
    expected_attention,
):
    contract = contract_factory(
        expiry_days=expiry_days,
        notice_days=0,
    )

    result = await analyze(agent, contract)

    assert result["requires_attention"] is expected_attention


@pytest.mark.asyncio
async def test_expired_contract_high_priority(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=-10)

    result = await analyze(agent, contract)

    assert result["risk_level"] in {"high", "critical"}


@pytest.mark.asyncio
async def test_distant_expiry_lower_priority(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=365,
        notice_days=30,
    )

    result = await analyze(agent, contract)

    assert result["risk_level"] in {"low", "none"}


# ============================================================
# AUTOMATIC RENEWAL
# ============================================================

@pytest.mark.asyncio
async def test_auto_renewal_detected(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=30,
        notice_days=30,
        auto_renewal=True,
    )

    result = await analyze(agent, contract)

    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_auto_renewal_notice_overdue(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_days=10,
        notice_days=30,
        auto_renewal=True,
    )

    result = await analyze(agent, contract)

    assert result["days_until_notice"] == -20
    assert result["requires_attention"] is True


# ============================================================
# VALIDATION AND EDGE CASES
# ============================================================

@pytest.mark.asyncio
async def test_missing_expiry_date(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=None)

    try:
        result = await analyze(agent, contract)
    except (ValueError, TypeError):
        return

    assert result["status"] in {
        "unknown",
        "missing_expiry",
        "requires_review",
    }


@pytest.mark.asyncio
async def test_invalid_expiry_date(
    agent,
    contract_factory,
):
    contract = contract_factory(
        expiry_date="not-a-date",
    )

    with pytest.raises((ValueError, TypeError)):
        await analyze(agent, contract)


@pytest.mark.asyncio
async def test_negative_notice_period_rejected(
    agent,
    contract_factory,
):
    contract = contract_factory(notice_days=-10)

    with pytest.raises((ValueError, TypeError)):
        await analyze(agent, contract)


@pytest.mark.asyncio
async def test_missing_contract_identifier_rejected(
    agent,
    contract_factory,
):
    contract = contract_factory(contract_id="")

    with pytest.raises((ValueError, TypeError)):
        await analyze(agent, contract)


# ============================================================
# TENANT ISOLATION
# ============================================================

@pytest.mark.asyncio
async def test_tenant_identifier_preserved(
    agent,
    contract_factory,
):
    contract = contract_factory(
        tenant_id="TENANT-B",
    )

    result = await analyze(agent, contract)

    assert result["tenant_id"] == "TENANT-B"


@pytest.mark.asyncio
async def test_different_tenants_not_conflated(
    agent,
    contract_factory,
):
    first = contract_factory(
        contract_id="CON-A",
        tenant_id="TENANT-A",
        expiry_days=15,
    )
    second = contract_factory(
        contract_id="CON-B",
        tenant_id="TENANT-B",
        expiry_days=90,
    )

    first_result = await analyze(agent, first)
    second_result = await analyze(agent, second)

    assert first_result["tenant_id"] == "TENANT-A"
    assert second_result["tenant_id"] == "TENANT-B"
    assert first_result["contract_id"] != second_result["contract_id"]


# ============================================================
# RECOMMENDATIONS
# ============================================================

@pytest.mark.asyncio
async def test_expiring_contract_has_recommendations(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=10)

    result = await analyze(agent, contract)

    assert isinstance(result["recommended_actions"], list)
    assert len(result["recommended_actions"]) > 0


@pytest.mark.asyncio
async def test_recommendations_are_text(
    agent,
    contract_factory,
):
    contract = contract_factory(expiry_days=7)

    result = await analyze(agent, contract)

    assert all(
        isinstance(action, str) and action.strip()
        for action in result["recommended_actions"]
    )


# ============================================================
# BATCH ANALYSIS
# ============================================================

@pytest.mark.asyncio
async def test_batch_analysis(
    agent,
    contract_factory,
):
    contracts = [
        contract_factory(
            contract_id="CON-001",
            expiry_days=10,
        ),
        contract_factory(
            contract_id="CON-002",
            expiry_days=60,
        ),
        contract_factory(
            contract_id="CON-003",
            expiry_days=180,
        ),
    ]

    results = await analyze_batch(agent, contracts)

    assert len(results) == 3
    assert {
        result["contract_id"]
        for result in results
    } == {"CON-001", "CON-002", "CON-003"}


@pytest.mark.asyncio
async def test_empty_batch(
    agent,
):
    results = await analyze_batch(agent, [])

    assert results == []


@pytest.mark.asyncio
async def test_batch_preserves_tenant_identifiers(
    agent,
    contract_factory,
):
    contracts = [
        contract_factory(
            contract_id="CON-A",
            tenant_id="TENANT-A",
        ),
        contract_factory(
            contract_id="CON-B",
            tenant_id="TENANT-B",
        ),
    ]

    results = await analyze_batch(agent, contracts)

    mapping = {
        result["contract_id"]: result["tenant_id"]
        for result in results
    }

    assert mapping["CON-A"] == "TENANT-A"
    assert mapping["CON-B"] == "TENANT-B"
