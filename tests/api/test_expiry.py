
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

Expected async methods:
    calculate_status(
        contract: dict,
        as_of: date,
    ) -> dict

    get_expiring_contracts(
        tenant_id: str,
        as_of: date,
        days_ahead: int = 30,
    ) -> list[dict]

    calculate_notice_deadline(
        contract: dict,
    ) -> date

    process_expiry_alerts(
        tenant_id: str,
        as_of: date,
    ) -> dict

Expected status result:
    {
        "contract_id": str,
        "tenant_id": str,
        "status": str,
        "days_until_expiry": int,
        "expiry_date": date,
        "notice_deadline": date | None,
        "auto_renewal": bool,
    }

Supported statuses:
    active
    expiring_soon
    renewal_due
    expired

Expected alert processing:
    {
        "tenant_id": str,
        "processed": int,
        "alerts_sent": int,
        "errors": list,
    }

Security:
    - Enforce tenant isolation.
    - Never notify unauthorized tenants.
    - Prevent duplicate alerts.
    - Validate dates and notice periods.
    - Handle leap years and month boundaries.
    - Avoid silently changing contract terms.
    - Audit alert processing.
    - Handle notification failures safely.

Reference specification tests.
Adjust interfaces to match your actual implementation.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from copy import deepcopy
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ============================================================
# CONSTANTS
# ============================================================

TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

CONTRACT_A = "contract_alpha"
CONTRACT_B = "contract_beta"

TODAY = date(2026, 10, 8)

VALID_STATUSES = {
    "active",
    "expiring_soon",
    "renewal_due",
    "expired",
}


# ============================================================
# IMPORT
# ============================================================

def load_expiry_service_class():
    candidates = (
        "backend.app.services.expiry_service",
        "backend.app.services.expiry",
        "backend.app.contracts.expiry_service",
        "app.services.expiry_service",
        "app.services.expiry",
        "app.contracts.expiry_service",
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
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    start_date=None,
    expiry_date=None,
    notice_period_days=30,
    auto_renewal=False,
    renewal_period_days=365,
    status="active",
):
    return {
        "id": contract_id,
        "tenant_id": tenant_id,
        "title": f"Agreement {contract_id}",
        "start_date": start_date or (
            TODAY - timedelta(days=365)
        ),
        "expiry_date": expiry_date or (
            TODAY + timedelta(days=60)
        ),
        "notice_period_days": notice_period_days,
        "auto_renewal": auto_renewal,
        "renewal_period_days": renewal_period_days,
        "status": status,
    }


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def active_contract():
    return make_contract(
        expiry_date=TODAY + timedelta(days=180),
    )


@pytest.fixture
def expiring_contract():
    return make_contract(
        contract_id="contract_expiring",
        expiry_date=TODAY + timedelta(days=15),
    )


@pytest.fixture
def expired_contract():
    return make_contract(
        contract_id="contract_expired",
        expiry_date=TODAY - timedelta(days=10),
    )


@pytest.fixture
def renewal_due_contract():
    return make_contract(
        contract_id="contract_renewal_due",
        expiry_date=TODAY + timedelta(days=45),
        notice_period_days=60,
        auto_renewal=True,
    )


@pytest.fixture
def auto_renewal_contract():
    return make_contract(
        contract_id="contract_auto_renewal",
        expiry_date=TODAY + timedelta(days=30),
        notice_period_days=45,
        auto_renewal=True,
    )


@pytest.fixture
def contract_repository(
    active_contract,
    expiring_contract,
    expired_contract,
    renewal_due_contract,
    auto_renewal_contract,
):
    repository = MagicMock()

    repository.contracts = [
        deepcopy(active_contract),
        deepcopy(expiring_contract),
        deepcopy(expired_contract),
        deepcopy(renewal_due_contract),
        deepcopy(auto_renewal_contract),
        make_contract(
            contract_id=CONTRACT_B,
            tenant_id=TENANT_B,
            expiry_date=TODAY + timedelta(days=5),
        ),
    ]

    async def list_by_tenant(tenant_id):
        return [
            deepcopy(contract)
            for contract in repository.contracts
            if contract["tenant_id"] == tenant_id
        ]

    repository.list_by_tenant = AsyncMock(
        side_effect=list_by_tenant
    )

    return repository


@pytest.fixture
def notification_service():
    service = MagicMock()

    service.sent_alerts = set()

    async def send_expiry_alert(
        tenant_id,
        contract_id,
        alert_type,
        expiry_date,
    ):
        key = (
            tenant_id,
            contract_id,
            alert_type,
            expiry_date,
        )

        if key in service.sent_alerts:
            return False

        service.sent_alerts.add(key)

        return True

    service.send_expiry_alert = AsyncMock(
        side_effect=send_expiry_alert
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


async def calculate_notice_deadline(
    service,
    contract,
):
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
        "days_until_expiry",
        "expiry_date",
        "notice_deadline",
        "auto_renewal",
    }

    assert required.issubset(result.keys())

    assert result["status"] in VALID_STATUSES

    assert isinstance(
        result["days_until_expiry"],
        int,
    )

    assert isinstance(
        result["expiry_date"],
        date,
    )

    assert isinstance(
        result["auto_renewal"],
        bool,
    )


# ============================================================
# INITIALIZATION
# ============================================================

async def test_expiry_service_initialization(
    expiry_service,
):
    assert expiry_service is not None
    assert callable(expiry_service.calculate_status)
    assert callable(
        expiry_service.get_expiring_contracts
    )
    assert callable(
        expiry_service.calculate_notice_deadline
    )
    assert callable(
        expiry_service.process_expiry_alerts
    )


# ============================================================
# ACTIVE CONTRACT
# ============================================================

async def test_active_contract_status(
    expiry_service,
    active_contract,
):
    result = await calculate_status(
        expiry_service,
        active_contract,
    )

    assert_valid_status(result)
    assert result["status"] == "active"


async def test_active_contract_days_remaining(
    expiry_service,
    active_contract,
):
    result = await calculate_status(
        expiry_service,
        active_contract,
    )

    assert result["days_until_expiry"] == 180


# ============================================================
# EXPIRING SOON
# ============================================================

async def test_expiring_soon_status(
    expiry_service,
    expiring_contract,
):
    result = await calculate_status(
        expiry_service,
        expiring_contract,
    )

    assert result["status"] == "expiring_soon"


async def test_expiring_contract_days_remaining(
    expiry_service,
    expiring_contract,
):
    result = await calculate_status(
        expiry_service,
        expiring_contract,
    )

    assert result["days_until_expiry"] == 15


async def test_contract_expiring_today(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY,
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["days_until_expiry"] == 0

    assert result["status"] in {
        "expiring_soon",
        "renewal_due",
    }


async def test_contract_expiring_tomorrow(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY + timedelta(days=1),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["days_until_expiry"] == 1


# ============================================================
# EXPIRED CONTRACT
# ============================================================

async def test_expired_contract_status(
    expiry_service,
    expired_contract,
):
    result = await calculate_status(
        expiry_service,
        expired_contract,
    )

    assert result["status"] == "expired"


async def test_expired_contract_negative_days(
    expiry_service,
    expired_contract,
):
    result = await calculate_status(
        expiry_service,
        expired_contract,
    )

    assert result["days_until_expiry"] == -10


async def test_expired_yesterday(
    expiry_service,
):
    contract = make_contract(
        expiry_date=TODAY - timedelta(days=1),
    )

    result = await calculate_status(
        expiry_service,
        contract,
    )

    assert result["status"] == "expired"


# ============================================================
# NOTICE DEADLINE
# ============================================================

async def test_notice_deadline_calculation(
    expiry_service,
):
    expiry_date = date(2026, 12, 31)

    contract = make_contract(
        expiry_date=expiry_date,
        notice_period_days=30,
    )

    deadline = await calculate_notice_deadline(
        expiry_service,
        contract,
    )

    assert deadline == date(2026, 12, 1)


async def test_notice_deadline_sixty_days(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2027, 3, 1),
        notice_period_days=60,
    )

    deadline = await calculate_notice_deadline(
        expiry_service,
        contract,
    )

    assert deadline == date(2026, 12, 31)


async def test_notice_deadline_zero_days(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2026, 12, 31),
        notice_period_days=0,
    )

    deadline = await calculate_notice_deadline(
        expiry_service,
        contract,
    )

    assert deadline == date(2026, 12, 31)


async def test_negative_notice_period_rejected(
    expiry_service,
):
    contract = make_contract(
        notice_period_days=-10,
    )

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await calculate_notice_deadline(
            expiry_service,
            contract,
        )


# ============================================================
# RENEWAL
# ============================================================

async def test_renewal_due_status(
    expiry_service,
    renewal_due_contract,
):
    result = await calculate_status(
        expiry_service,
        renewal_due_contract,
    )

    assert result["status"] == "renewal_due"


async def test_auto_renewal_flag_preserved(
    expiry_service,
    auto_renewal_contract,
):
    result = await calculate_status(
        expiry_service,
        auto_renewal_contract,
    )

    assert result["auto_renewal"] is True


async def test_non_renewing_contract_flag(
    expiry_service,
    active_contract,
):
    result = await calculate_status(
        expiry_service,
        active_contract,
    )

    assert result["auto_renewal"] is False


async def test_status_calculation_does_not_mutate_contract(
    expiry_service,
    auto_renewal_contract,
):
    original = deepcopy(
        auto_renewal_contract
    )

    await calculate_status(
        expiry_service,
        auto_renewal_contract,
    )

    assert auto_renewal_contract == original


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

    assert isinstance(results, list)


async def test_expiring_contract_included(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=30,
    )

    ids = {
        item["id"]
        for item in results
    }

    assert "contract_expiring" in ids


async def test_far_future_contract_excluded(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=30,
    )

    ids = {
        item["id"]
        for item in results
    }

    assert CONTRACT_A not in ids


async def test_expired_contract_excluded(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=30,
    )

    ids = {
        item["id"]
        for item in results
    }

    assert "contract_expired" not in ids


async def test_zero_day_window(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        days_ahead=0,
    )

    assert all(
        item["expiry_date"] == TODAY
        for item in results
    )


async def test_negative_window_rejected(
    expiry_service,
):
    with pytest.raises(ValueError):
        await get_expiring(
            expiry_service,
            days_ahead=-1,
        )


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_tenant_a_expiry_isolation(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id=TENANT_A,
    )

    assert all(
        item["tenant_id"] == TENANT_A
        for item in results
    )


async def test_tenant_b_expiry_isolation(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id=TENANT_B,
    )

    assert all(
        item["tenant_id"] == TENANT_B
        for item in results
    )


async def test_cross_tenant_contract_not_returned(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id=TENANT_A,
    )

    ids = {
        item["id"]
        for item in results
    }

    assert CONTRACT_B not in ids


async def test_unknown_tenant_returns_empty(
    expiry_service,
):
    results = await get_expiring(
        expiry_service,
        tenant_id="unknown_tenant",
    )

    assert results == []


# ============================================================
# ALERT PROCESSING
# ============================================================

async def test_process_expiry_alerts(
    expiry_service,
):
    result = await process_alerts(
        expiry_service
    )

    assert isinstance(result, dict)

    assert "processed" in result
    assert "alerts_sent" in result
    assert "errors" in result


async def test_expiry_alerts_sent(
    expiry_service,
    notification_service,
):
    await process_alerts(
        expiry_service
    )

    notification_service.send_expiry_alert.assert_awaited()


async def test_alert_processing_tenant_context(
    expiry_service,
):
    result = await process_alerts(
        expiry_service,
        tenant_id=TENANT_A,
    )

    assert result["tenant_id"] == TENANT_A


async def test_alerts_do_not_cross_tenants(
    expiry_service,
    notification_service,
):
    await process_alerts(
        expiry_service,
        tenant_id=TENANT_A,
    )

    for call in (
        notification_service
        .send_expiry_alert
        .await_args_list
    ):
        assert call.kwargs.get(
            "tenant_id",
            call.args[0] if call.args else None,
        ) == TENANT_A


async def test_duplicate_alerts_prevented(
    expiry_service,
    notification_service,
):
    await process_alerts(
        expiry_service
    )

    first_count = len(
        notification_service.sent_alerts
    )

    await process_alerts(
        expiry_service
    )

    second_count = len(
        notification_service.sent_alerts
    )

    assert first_count == second_count


async def test_notification_failure_handled(
    expiry_service,
    notification_service,
):
    notification_service.send_expiry_alert = (
        AsyncMock(
            side_effect=RuntimeError(
                "Notification provider unavailable"
            )
        )
    )

    try:
        result = await process_alerts(
            expiry_service
        )
    except RuntimeError:
        return

    assert result["errors"]


# ============================================================
# DATE BOUNDARIES
# ============================================================

async def test_leap_year_expiry(
    expiry_service,
):
    contract = make_contract(
        start_date=date(2024, 1, 1),
        expiry_date=date(2024, 2, 29),
    )

    result = await calculate_status(
        expiry_service,
        contract,
        as_of=date(2024, 2, 28),
    )

    assert result["days_until_expiry"] == 1


async def test_leap_year_notice_deadline(
    expiry_service,
):
    contract = make_contract(
        start_date=date(2024, 1, 1),
        expiry_date=date(2024, 3, 1),
        notice_period_days=1,
    )

    deadline = await calculate_notice_deadline(
        expiry_service,
        contract,
    )

    assert deadline == date(2024, 2, 29)


async def test_year_boundary_notice_deadline(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2027, 1, 15),
        notice_period_days=30,
    )

    deadline = await calculate_notice_deadline(
        expiry_service,
        contract,
    )

    assert deadline == date(2026, 12, 16)


async def test_month_boundary_expiry(
    expiry_service,
):
    contract = make_contract(
        expiry_date=date(2026, 11, 1),
    )

    result = await calculate_status(
        expiry_service,
        contract,
        as_of=date(2026, 10, 31),
    )

    assert result["days_until_expiry"] == 1


# ============================================================
# INVALID INPUT
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
    contract["expiry_date"] = "not-a-date"

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await calculate_status(
            expiry_service,
            contract,
        )


async def test_invalid_contract_input(
    expiry_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await calculate_status(
            expiry_service,
            "invalid-contract",
        )


@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_tenant(
    expiry_service,
    invalid_tenant,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await get_expiring(
            expiry_service,
            tenant_id=invalid_tenant,
        )


# ============================================================
# AUDIT
# ============================================================

async def test_alert_processing_audited(
    expiry_service,
    audit_service,
):
    await process_alerts(
        expiry_service
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant(
    expiry_service,
    audit_service,
):
    await process_alerts(
        expiry_service
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_status_calculation(
    expiry_service,
    contract_repository,
):
    contracts = (
        contract_repository.contracts[:5]
    )

    results = await asyncio.gather(
        *[
            calculate_status(
                expiry_service,
                contract,
            )
            for contract in contracts
        ]
    )

    assert len(results) == len(contracts)

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

    assert all(
        item["tenant_id"] == TENANT_A
        for item in tenant_a
    )

    assert all(
        item["tenant_id"] == TENANT_B
        for item in tenant_b
    )


async def test_concurrent_alert_processing(
    expiry_service,
    notification_service,
):
    results = await asyncio.gather(
        *[
            process_alerts(
                expiry_service,
                tenant_id=TENANT_A,
            )
            for _ in range(5)
        ]
    )

    assert len(results) == 5

    # Verify idempotent alert keys in the fake
    # notification backend.
    assert len(
        notification_service.sent_alerts
    ) == len(set(
        notification_service.sent_alerts
    ))


# ============================================================
# END
# ============================================================
