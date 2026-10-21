
"""
tests/unit/test_expiry.py

Lexora AI — Contract Expiry & Renewal Intelligence Tests

Expected module:
    backend.app.services.expiry_service

Expected class:
    ExpiryService

Expected constructor:
    ExpiryService(
        contract_repository=...,
        notification_service=...,
        audit_service=...,
    )

Expected methods:
    async calculate_status(contract, as_of=None) -> dict

    async get_expiring_contracts(
        tenant_id,
        as_of=None,
        days_ahead=30,
    ) -> list[dict]

    async calculate_notice_deadline(contract) -> date

    async process_expiry_alerts(
        tenant_id,
        as_of=None,
    ) -> list[dict]

Expected status result:
    {
        "contract_id": str,
        "tenant_id": str,
        "status": (
            "active" | "expiring_soon" |
            "expired" | "renewal_due"
        ),
        "days_remaining": int,
        "expiry_date": date,
        "notice_deadline": date | None,
    }

Business rules:
    - Expiry dates are inclusive through the end of the day.
    - A contract is expired when as_of > expiry_date.
    - An active contract expires soon within 30 days.
    - Renewal due takes precedence over expiring soon.
    - Notice deadlines use calendar days.
    - Missing/invalid expiry dates fail validation.
    - Tenant-scoped queries never expose other tenants.
    - Alerts must not be sent twice for the same event.
    - Processing failures must not silently mark alerts sent.

Reference specification tests.
Adapt to your actual service API and business policies.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

CONTRACT_ACTIVE = "contract_active"
CONTRACT_EXPIRING = "contract_expiring"
CONTRACT_EXPIRED = "contract_expired"
CONTRACT_RENEWAL = "contract_renewal"

TODAY = date(2026, 10, 8)

VALID_STATUSES = {
    "active",
    "expiring_soon",
    "expired",
    "renewal_due",
}


# ============================================================
# IMPORT
# ============================================================

def load_expiry_service_class():
    candidates = (
        "backend.app.services.expiry_service",
        "backend.app.services.expiry",
        "backend.app.legal.expiry_service",
        "app.services.expiry_service",
        "app.services.expiry",
        "app.legal.expiry_service",
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
            "ExpiryService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "ExpiryService could not be imported. "
        "Update the module path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# CONTRACT FACTORY
# ============================================================

def make_contract(
    contract_id="contract_001",
    tenant_id=TENANT_A,
    expiry_date=None,
    notice_period_days=0,
    auto_renew=False,
    status="active",
):
    return {
        "id": contract_id,
        "tenant_id": tenant_id,
        "title": "Master Service Agreement",
        "start_date": date(2026, 1, 1),
        "expiry_date": (
            expiry_date
            if expiry_date is not None
            else TODAY + timedelta(days=90)
        ),
        "notice_period_days": notice_period_days,
        "auto_renew": auto_renew,
        "status": status,
    }


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def contracts():
    return [
        make_contract(
            contract_id=CONTRACT_ACTIVE,
            expiry_date=TODAY + timedelta(days=120),
        ),
        make_contract(
            contract_id=CONTRACT_EXPIRING,
            expiry_date=TODAY + timedelta(days=15),
        ),
        make_contract(
            contract_id=CONTRACT_EXPIRED,
            expiry_date=TODAY - timedelta(days=5),
        ),
        make_contract(
            contract_id=CONTRACT_RENEWAL,
            expiry_date=TODAY + timedelta(days=60),
            notice_period_days=60,
            auto_renew=True,
        ),
        make_contract(
            contract_id="tenant_b_contract",
            tenant_id=TENANT_B,
            expiry_date=TODAY + timedelta(days=10),
        ),
    ]


@pytest.fixture
def contract_repository(contracts):
    repository = MagicMock()

    repository.contracts = deepcopy(contracts)

    async def list_by_tenant(tenant_id):
        return [
            deepcopy(contract)
            for contract in repository.contracts
            if contract["tenant_id"] == tenant_id
        ]

    async def get_by_id(contract_id, tenant_id):
        for contract in repository.contracts:
            if (
                contract["id"] == contract_id
                and contract["tenant_id"] == tenant_id
            ):
                return deepcopy(contract)

        return None

    repository.list_by_tenant = AsyncMock(
        side_effect=list_by_tenant
    )

    repository.get_by_id = AsyncMock(
        side_effect=get_by_id
    )

    return repository


@pytest.fixture
def notification_service():
    service = MagicMock()

    service.send_expiry_alert = AsyncMock(
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
def expiry_service(
    contract_repository,
    notification_service,
    audit_service,
):
    service_class = load_expiry_service_class()

    return service_class(
        contract_repository=contract_repository,
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


async def calculate_status(
    service,
    contract,
    as_of=TODAY,
):
    return await invoke(
        service.calculate_status,
        contract=contract,
        as_of=as_of,
    )


async def get_expiring(
    service,
    tenant_id=TENANT_A,
    as_of=TODAY,
    days_ahead=30,
):
    return await invoke(
        service.get_expiring_contracts,
        tenant_id=tenant_id,
        as_of=as_of,
        days_ahead=days_ahead,
    )


async def notice_deadline(service, contract):
    return await invoke(
        service.calculate_notice_deadline,
        contract=contract,
    )


async def process_alerts(
    service,
    tenant_id=TENANT_A,
    as_of=TODAY,
):
    return await invoke(
        service.process_expiry_alerts,
        tenant_id=tenant_id,
        as_of=as_of,
    )


def assert_valid_status(result):
    assert isinstance(result, dict)

    required = {
        "contract_id",
        "tenant_id",
        "status",
        "days_remaining",
        "expiry_date",
        "notice_deadline",
    }

    assert required.issubset(result.keys())

    assert result["status"] in VALID_STATUSES
    assert isinstance(result["days_remaining"], int)
    assert isinstance(result["expiry_date"], date)

    assert (
        result["notice_deadline"] is None
        or isinstance(result["notice_deadline"], date)
    )


def contract_ids(items):
    return {
        item.get("id", item.get("contract_id"))
        for item in items
    }


# ============================================================
# INITIALIZATION
# ============================================================

async def test_expiry_service_initialization(
    expiry_service,
):
    assert expiry_service is not None

    assert callable(expiry_service.calculate_status)
    assert callable(expiry_service.get_expiring_contracts)
    assert callable(expiry_service.calculate_notice_deadline)
    assert callable(expiry_service.process_expiry_alerts)


# ============================================================
# CONTRACT STATUS
# ============================================================

async def test_active_contract_status(expiry_service):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=90),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert_valid_status(result)
    assert result["status"] == "active"
    assert result["days_remaining"] == 90


async def test_expiring_soon_status(expiry_service):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=15),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "expiring_soon"
    assert result["days_remaining"] == 15


async def test_expired_contract_status(expiry_service):
    contract = make_contract(
        expiry_date=TODAY - timedelta(days=5),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "expired"
    assert result["days_remaining"] == -5


async def test_contract_expiring_today(expiry_service):
    contract = make_contract(
        expiry_date=TODAY,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["days_remaining"] == 0
    assert result["status"] == "expiring_soon"


async def test_contract_expired_yesterday(expiry_service):
    contract = make_contract(
        expiry_date=TODAY - timedelta(days=1),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "expired"


# ============================================================
# EXPIRY BOUNDARIES
# ============================================================

@pytest.mark.parametrize(
    "days,expected",
    [
        (-30, "expired"),
        (-1, "expired"),
        (0, "expiring_soon"),
        (1, "expiring_soon"),
        (7, "expiring_soon"),
        (15, "expiring_soon"),
        (29, "expiring_soon"),
        (30, "expiring_soon"),
        (31, "active"),
        (90, "active"),
    ],
)
async def test_expiry_status_boundaries(
    expiry_service,
    days,
    expected,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=days),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == expected


# ============================================================
# NOTICE DEADLINES
# ============================================================

async def test_notice_deadline_30_days(
    expiry_service,
):
    expiry = date(2026, 12, 31)

    contract = make_contract(
        expiry_date=expiry,
        notice_period_days=30,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == expiry - timedelta(days=30)


async def test_notice_deadline_60_days(
    expiry_service,
):
    expiry = date(2026, 12, 31)

    contract = make_contract(
        expiry_date=expiry,
        notice_period_days=60,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == expiry - timedelta(days=60)


async def test_notice_deadline_90_days(
    expiry_service,
):
    expiry = date(2027, 3, 31)

    contract = make_contract(
        expiry_date=expiry,
        notice_period_days=90,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == expiry - timedelta(days=90)


async def test_notice_deadline_zero_days(
    expiry_service,
):
    expiry = date(2026, 12, 31)

    contract = make_contract(
        expiry_date=expiry,
        notice_period_days=0,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == expiry


async def test_negative_notice_period_rejected(
    expiry_service,
):
    contract = make_contract(
        notice_period_days=-30,
    )

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await notice_deadline(
            expiry_service,
            contract,
        )


# ============================================================
# AUTO-RENEWAL
# ============================================================

async def test_auto_renewal_due(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=60),
        notice_period_days=60,
        auto_renew=True,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "renewal_due"


async def test_auto_renewal_notice_upcoming(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=90),
        notice_period_days=30,
        auto_renew=True,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "active"


async def test_auto_renewal_notice_passed(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=20),
        notice_period_days=30,
        auto_renew=True,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "renewal_due"


async def test_expired_auto_renewal_contract(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY - timedelta(days=2),
        notice_period_days=30,
        auto_renew=True,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "expired"


# ============================================================
# EXPIRING CONTRACT QUERIES
# ============================================================

async def test_get_expiring_contracts(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=30,
    )

    ids = contract_ids(results)

    assert CONTRACT_EXPIRING in ids
    assert CONTRACT_ACTIVE not in ids


async def test_expired_contracts_not_in_upcoming_window(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=30,
    )

    assert CONTRACT_EXPIRED not in contract_ids(results)


async def test_expiry_window_7_days(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=7,
    )

    assert CONTRACT_EXPIRING not in contract_ids(results)


async def test_expiry_window_90_days(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=90,
    )

    ids = contract_ids(results)

    assert CONTRACT_EXPIRING in ids
    assert CONTRACT_RENEWAL in ids


async def test_expiry_results_are_list(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
    )

    assert isinstance(results, list)


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_tenant_a_only_sees_own_contracts(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id=TENANT_A,
    )

    assert "tenant_b_contract" not in contract_ids(
        results
    )


async def test_tenant_b_only_sees_own_contracts(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id=TENANT_B,
    )

    assert contract_ids(results) == {
        "tenant_b_contract"
    }


async def test_unknown_tenant_returns_empty(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id="unknown_tenant",
    )

    assert results == []


async def test_tenant_identity_preserved(
    expiry_service,
):
    contract = make_contract(
        tenant_id=TENANT_B,
        expiry_date=TODAY + timedelta(days=10),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["tenant_id"] == TENANT_B


# ============================================================
# DATE VALIDATION
# ============================================================

async def test_missing_expiry_date_rejected(
    expiry_service,
):
    contract = make_contract()
    contract["expiry_date"] = None

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await calculate_status(
            expiry_service,
            contract,
        )


async def test_invalid_expiry_date_rejected(
    expiry_service,
):
    contract = make_contract()
    contract["expiry_date"] = "invalid-date"

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await calculate_status(
            expiry_service,
            contract,
        )


async def test_expiry_before_start_rejected(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2025, 12, 31),
    )

    with pytest.raises(ValueError):
        await calculate_status(
            expiry_service,
            contract,
        )


async def test_invalid_days_ahead_rejected(
    expiry_service,
):
    with pytest.raises(ValueError):
        await get_expiring(
            expiry_service,
            days_ahead=-1,
        )


# ============================================================
# CALENDAR DATE EDGE CASES
# ============================================================

async def test_leap_year_notice_deadline(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2028, 3, 1),
        notice_period_days=1,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == date(2028, 2, 29)


async def test_year_boundary_notice_deadline(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2027, 1, 15),
        notice_period_days=30,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == date(2026, 12, 16)


async def test_end_of_month_notice_deadline(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2026, 3, 31),
        notice_period_days=31,
    )

    result = await notice_deadline(
        expiry_service,
        contract,
    )

    assert result == date(2026, 2, 28)


# ============================================================
# ALERT PROCESSING
# ============================================================

async def test_process_expiry_alerts(
    expiry_service,
):
    results = await process_alerts(
        expiry_service,
    )

    assert isinstance(results, list)


async def test_expiry_alert_notification_called(
    expiry_service,
    notification_service,
):
    await process_alerts(
        expiry_service,
    )

    notification_service.send_expiry_alert.assert_awaited()


async def test_expiry_alert_tenant_scoped(
    expiry_service,
    notification_service,
):
    await process_alerts(
        expiry_service,
        tenant_id=TENANT_A,
    )

    calls = notification_service.send_expiry_alert.await_args_list

    assert calls

    for call in calls:
        assert "tenant_b_contract" not in repr(call)


async def test_notification_failure_propagates(
    expiry_service,
    notification_service,
):
    notification_service.send_expiry_alert.side_effect = (
        RuntimeError("Notification service unavailable")
    )

    with pytest.raises(RuntimeError):
        await process_alerts(
            expiry_service,
        )


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_alert_processing_is_audited(
    expiry_service,
    audit_service,
):
    await process_alerts(
        expiry_service,
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant_context(
    expiry_service,
    audit_service,
):
    await process_alerts(
        expiry_service,
        tenant_id=TENANT_A,
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# INPUT IMMUTABILITY
# ============================================================

async def test_calculate_status_does_not_mutate_contract(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=10),
    )

    original = deepcopy(contract)

    await calculate_status(
        expiry_service,
        contract,
    )

    assert contract == original


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_status_calculations(
    expiry_service,
):
    contracts = [
        make_contract(
            contract_id=f"contract_{index}",
            expiry_date=TODAY + timedelta(days=index),
        )
        for index in range(1, 21)
    ]

    results = await asyncio.gather(
        *[
            calculate_status(
                expiry_service,
                contract,
            )
            for contract in contracts
        ]
    )

    assert len(results) == 20

    for result in results:
        assert_valid_status(result)


async def test_concurrent_tenant_queries(
    expiry_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        get_expiring(
            expiry_service,
            tenant_id=TENANT_A,
        ),
        get_expiring(
            expiry_service,
            tenant_id=TENANT_B,
        ),
    )

    assert "tenant_b_contract" not in contract_ids(
        tenant_a
    )

    assert contract_ids(tenant_b) == {
        "tenant_b_contract"
    }


# ============================================================
# END
# ============================================================
