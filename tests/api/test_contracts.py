
"""
tests/unit/test_contracts.py

Lexora AI — Contract Management Unit Tests

Expected module:
    backend.app.services.contract_service

Expected class:
    ContractService

Expected constructor:
    ContractService(
        contract_repository=...,
        permission_service=...,
        audit_service=...,
    )

Expected async methods:
    create_contract(
        tenant_id: str,
        user_id: str,
        data: dict,
    ) -> dict

    get_contract(
        contract_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict

    list_contracts(
        tenant_id: str,
        user_id: str,
        filters: dict | None = None,
    ) -> list[dict]

    update_contract(
        contract_id: str,
        tenant_id: str,
        user_id: str,
        data: dict,
    ) -> dict

    change_status(
        contract_id: str,
        tenant_id: str,
        user_id: str,
        status: str,
    ) -> dict

    delete_contract(
        contract_id: str,
        tenant_id: str,
        user_id: str,
    ) -> bool

Expected contract fields:
    id, tenant_id, title, contract_type,
    counterparty, start_date, expiry_date,
    status, owner_id, metadata, deleted_at

Security:
    - Enforce tenant isolation on every operation.
    - Enforce permissions on every operation.
    - Never allow tenant_id or owner_id overrides.
    - Reject invalid contract dates and statuses.
    - Preserve contract identity during updates.
    - Soft-delete contracts without exposing them.
    - Audit sensitive contract modifications.
    - Prevent unauthorized lifecycle changes.
    - Avoid duplicate IDs during concurrent creation.

Reference tests only; not yet run against production.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
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
VIEWER_A = "viewer_alpha"

CONTRACT_A = "contract_alpha"
CONTRACT_B = "contract_beta"

VALID_STATUSES = {
    "draft",
    "active",
    "expired",
    "terminated",
    "archived",
}

START_DATE = date(2026, 1, 1)
EXPIRY_DATE = date(2027, 1, 1)


# ============================================================
# IMPORT
# ============================================================

def load_contract_service_class():
    candidates = (
        "backend.app.services.contract_service",
        "backend.app.services.contracts",
        "backend.app.contracts.service",
        "app.services.contract_service",
        "app.services.contracts",
        "app.contracts.service",
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
            "ContractService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "ContractService could not be imported. "
        "Update the module path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# DATA FACTORIES
# ============================================================

def make_contract(
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    owner_id=USER_A,
    title="Master Service Agreement",
    contract_type="msa",
    counterparty="Acme Technologies",
    start_date=START_DATE,
    expiry_date=EXPIRY_DATE,
    status="draft",
    metadata=None,
):
    return {
        "id": contract_id,
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "title": title,
        "contract_type": contract_type,
        "counterparty": counterparty,
        "start_date": start_date,
        "expiry_date": expiry_date,
        "status": status,
        "metadata": (
            deepcopy(metadata)
            if metadata is not None
            else {
                "department": "legal",
                "jurisdiction": "India",
                "currency": "INR",
            }
        ),
        "deleted_at": None,
    }


def make_create_data(
    title="New Service Agreement",
    contract_type="msa",
    counterparty="Example Corporation",
    start_date=START_DATE,
    expiry_date=EXPIRY_DATE,
    metadata=None,
):
    return {
        "title": title,
        "contract_type": contract_type,
        "counterparty": counterparty,
        "start_date": start_date,
        "expiry_date": expiry_date,
        "metadata": (
            deepcopy(metadata)
            if metadata is not None
            else {"department": "legal"}
        ),
    }


# ============================================================
# FAKE REPOSITORY
# ============================================================

class FakeContractRepository:

    def __init__(self):
        self.contracts = {
            CONTRACT_A: make_contract(),
            CONTRACT_B: make_contract(
                contract_id=CONTRACT_B,
                tenant_id=TENANT_B,
                owner_id=USER_B,
                title="Private Tenant B Contract",
            ),
        }

    async def create(self, contract_data):
        contract_id = contract_data.get(
            "id",
            f"contract_{uuid.uuid4().hex}",
        )

        if contract_id in self.contracts:
            raise ValueError("Duplicate contract ID")

        contract = {
            **deepcopy(contract_data),
            "id": contract_id,
            "deleted_at": None,
        }

        self.contracts[contract_id] = contract

        return deepcopy(contract)

    async def get_by_id(self, contract_id, tenant_id):
        contract = self.contracts.get(contract_id)

        if contract is None:
            return None

        if contract["tenant_id"] != tenant_id:
            return None

        if contract["deleted_at"] is not None:
            return None

        return deepcopy(contract)

    async def list_by_tenant(self, tenant_id):
        return [
            deepcopy(contract)
            for contract in self.contracts.values()
            if (
                contract["tenant_id"] == tenant_id
                and contract["deleted_at"] is None
            )
        ]

    async def update(
        self,
        contract_id,
        tenant_id,
        changes,
    ):
        contract = self.contracts.get(contract_id)

        if (
            contract is None
            or contract["tenant_id"] != tenant_id
            or contract["deleted_at"] is not None
        ):
            raise LookupError("Contract not found")

        contract.update(deepcopy(changes))

        return deepcopy(contract)

    async def soft_delete(
        self,
        contract_id,
        tenant_id,
    ):
        contract = self.contracts.get(contract_id)

        if (
            contract is None
            or contract["tenant_id"] != tenant_id
            or contract["deleted_at"] is not None
        ):
            return False

        contract["deleted_at"] = datetime.now(
            timezone.utc
        )

        return True


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def contract_repository():
    return FakeContractRepository()


@pytest.fixture
def permission_service():
    service = MagicMock()

    permissions = {
        (USER_A, TENANT_A): {
            "create",
            "read",
            "update",
            "delete",
            "change_status",
        },
        (USER_B, TENANT_B): {
            "create",
            "read",
            "update",
            "delete",
            "change_status",
        },
        (VIEWER_A, TENANT_A): {
            "read",
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
def audit_service():
    service = MagicMock()

    service.record_event = AsyncMock(
        return_value=None
    )

    return service


@pytest.fixture
def contract_service(
    contract_repository,
    permission_service,
    audit_service,
):
    service_class = load_contract_service_class()

    return service_class(
        contract_repository=contract_repository,
        permission_service=permission_service,
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


async def create_contract(
    service,
    data=None,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.create_contract,
        tenant_id=tenant_id,
        user_id=user_id,
        data=(
            deepcopy(data)
            if data is not None
            else make_create_data()
        ),
    )


async def get_contract(
    service,
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.get_contract,
        contract_id=contract_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


async def list_contracts(
    service,
    tenant_id=TENANT_A,
    user_id=USER_A,
    filters=None,
):
    return await invoke(
        service.list_contracts,
        tenant_id=tenant_id,
        user_id=user_id,
        filters=filters,
    )


async def update_contract(
    service,
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
    data=None,
):
    return await invoke(
        service.update_contract,
        contract_id=contract_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data=data or {"title": "Updated Agreement"},
    )


async def change_status(
    service,
    status,
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.change_status,
        contract_id=contract_id,
        tenant_id=tenant_id,
        user_id=user_id,
        status=status,
    )


async def delete_contract(
    service,
    contract_id=CONTRACT_A,
    tenant_id=TENANT_A,
    user_id=USER_A,
):
    return await invoke(
        service.delete_contract,
        contract_id=contract_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )


def assert_valid_contract(contract):
    assert isinstance(contract, dict)

    required = {
        "id",
        "tenant_id",
        "title",
        "contract_type",
        "counterparty",
        "start_date",
        "expiry_date",
        "status",
        "owner_id",
        "metadata",
        "deleted_at",
    }

    assert required.issubset(contract.keys())

    assert contract["id"]
    assert contract["tenant_id"]
    assert contract["title"]
    assert contract["status"] in VALID_STATUSES

    assert isinstance(contract["metadata"], dict)

    assert isinstance(contract["start_date"], date)
    assert isinstance(contract["expiry_date"], date)

    assert (
        contract["expiry_date"]
        >= contract["start_date"]
    )


# ============================================================
# INITIALIZATION
# ============================================================

async def test_contract_service_initialization(
    contract_service,
):
    assert contract_service is not None

    assert callable(contract_service.create_contract)
    assert callable(contract_service.get_contract)
    assert callable(contract_service.list_contracts)
    assert callable(contract_service.update_contract)
    assert callable(contract_service.change_status)
    assert callable(contract_service.delete_contract)


# ============================================================
# CONTRACT CREATION
# ============================================================

async def test_create_contract(contract_service):
    result = await create_contract(
        contract_service
    )

    assert_valid_contract(result)


async def test_create_contract_title(
    contract_service,
):
    result = await create_contract(
        contract_service,
        data=make_create_data(
            title="Enterprise NDA",
        ),
    )

    assert result["title"] == "Enterprise NDA"


async def test_create_contract_tenant_identity(
    contract_service,
):
    result = await create_contract(
        contract_service
    )

    assert result["tenant_id"] == TENANT_A


async def test_create_contract_owner_identity(
    contract_service,
):
    result = await create_contract(
        contract_service
    )

    assert result["owner_id"] == USER_A


async def test_create_contract_default_status(
    contract_service,
):
    result = await create_contract(
        contract_service
    )

    assert result["status"] == "draft"


async def test_create_contract_persisted(
    contract_service,
):
    created = await create_contract(
        contract_service
    )

    retrieved = await get_contract(
        contract_service,
        contract_id=created["id"],
    )

    assert retrieved["id"] == created["id"]


async def test_unique_contract_ids(
    contract_service,
):
    first = await create_contract(
        contract_service
    )

    second = await create_contract(
        contract_service
    )

    assert first["id"] != second["id"]


# ============================================================
# CONTRACT RETRIEVAL
# ============================================================

async def test_get_contract(contract_service):
    result = await get_contract(
        contract_service
    )

    assert_valid_contract(result)
    assert result["id"] == CONTRACT_A


async def test_get_nonexistent_contract(
    contract_service,
):
    try:
        result = await get_contract(
            contract_service,
            contract_id="missing_contract",
        )
    except (LookupError, ValueError):
        return

    assert result is None


async def test_list_contracts(
    contract_service,
):
    results = await list_contracts(
        contract_service
    )

    assert isinstance(results, list)
    assert len(results) >= 1


async def test_list_contracts_tenant_scoped(
    contract_service,
):
    results = await list_contracts(
        contract_service,
        tenant_id=TENANT_A,
    )

    assert all(
        item["tenant_id"] == TENANT_A
        for item in results
    )


# ============================================================
# CONTRACT UPDATES
# ============================================================

async def test_update_contract_title(
    contract_service,
):
    result = await update_contract(
        contract_service,
        data={
            "title": "Updated Master Agreement",
        },
    )

    assert result["title"] == (
        "Updated Master Agreement"
    )


async def test_update_contract_counterparty(
    contract_service,
):
    result = await update_contract(
        contract_service,
        data={
            "counterparty": "New Partner Ltd",
        },
    )

    assert result["counterparty"] == (
        "New Partner Ltd"
    )


async def test_update_contract_persisted(
    contract_service,
):
    await update_contract(
        contract_service,
        data={
            "title": "Persisted Update",
        },
    )

    result = await get_contract(
        contract_service
    )

    assert result["title"] == "Persisted Update"


async def test_update_preserves_contract_id(
    contract_service,
):
    result = await update_contract(
        contract_service,
    )

    assert result["id"] == CONTRACT_A


async def test_update_preserves_tenant(
    contract_service,
):
    result = await update_contract(
        contract_service,
    )

    assert result["tenant_id"] == TENANT_A


async def test_update_cannot_override_tenant(
    contract_service,
):
    try:
        result = await update_contract(
            contract_service,
            data={
                "tenant_id": TENANT_B,
            },
        )
    except (PermissionError, ValueError):
        return

    assert result["tenant_id"] == TENANT_A


async def test_update_cannot_override_owner(
    contract_service,
):
    try:
        result = await update_contract(
            contract_service,
            data={
                "owner_id": USER_B,
            },
        )
    except (PermissionError, ValueError):
        return

    assert result["owner_id"] == USER_A


# ============================================================
# CONTRACT LIFECYCLE
# ============================================================

async def test_draft_to_active(
    contract_service,
):
    result = await change_status(
        contract_service,
        status="active",
    )

    assert result["status"] == "active"


async def test_active_to_terminated(
    contract_service,
    contract_repository,
):
    contract_repository.contracts[
        CONTRACT_A
    ]["status"] = "active"

    result = await change_status(
        contract_service,
        status="terminated",
    )

    assert result["status"] == "terminated"


async def test_active_to_expired(
    contract_service,
    contract_repository,
):
    contract_repository.contracts[
        CONTRACT_A
    ]["status"] = "active"

    result = await change_status(
        contract_service,
        status="expired",
    )

    assert result["status"] == "expired"


async def test_invalid_status_rejected(
    contract_service,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await change_status(
            contract_service,
            status="unknown_status",
        )


async def test_terminated_to_active_rejected(
    contract_service,
    contract_repository,
):
    contract_repository.contracts[
        CONTRACT_A
    ]["status"] = "terminated"

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await change_status(
            contract_service,
            status="active",
        )


async def test_archived_to_active_rejected(
    contract_service,
    contract_repository,
):
    contract_repository.contracts[
        CONTRACT_A
    ]["status"] = "archived"

    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await change_status(
            contract_service,
            status="active",
        )


# ============================================================
# DATE VALIDATION
# ============================================================

async def test_contract_dates_preserved(
    contract_service,
):
    result = await create_contract(
        contract_service
    )

    assert result["start_date"] == START_DATE
    assert result["expiry_date"] == EXPIRY_DATE


async def test_expiry_before_start_rejected(
    contract_service,
):
    data = make_create_data(
        start_date=date(2027, 1, 1),
        expiry_date=date(2026, 1, 1),
    )

    with pytest.raises(ValueError):
        await create_contract(
            contract_service,
            data=data,
        )


async def test_same_start_and_expiry_date(
    contract_service,
):
    same_day = date(2026, 12, 31)

    result = await create_contract(
        contract_service,
        data=make_create_data(
            start_date=same_day,
            expiry_date=same_day,
        ),
    )

    assert result["start_date"] == same_day
    assert result["expiry_date"] == same_day


async def test_invalid_expiry_date_rejected(
    contract_service,
):
    data = make_create_data()
    data["expiry_date"] = "not-a-date"

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_contract(
            contract_service,
            data=data,
        )


# ============================================================
# METADATA
# ============================================================

async def test_contract_metadata_preserved(
    contract_service,
):
    metadata = {
        "department": "finance",
        "jurisdiction": "Germany",
        "currency": "EUR",
    }

    result = await create_contract(
        contract_service,
        data=make_create_data(
            metadata=metadata,
        ),
    )

    assert result["metadata"] == metadata


async def test_contract_metadata_not_mutated(
    contract_service,
):
    data = make_create_data()

    original = deepcopy(data)

    await create_contract(
        contract_service,
        data=data,
    )

    assert data == original


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_cross_tenant_contract_read_rejected(
    contract_service,
):
    try:
        result = await get_contract(
            contract_service,
            contract_id=CONTRACT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result is None


async def test_cross_tenant_update_rejected(
    contract_service,
):
    with pytest.raises(
        (PermissionError, LookupError, ValueError)
    ):
        await update_contract(
            contract_service,
            contract_id=CONTRACT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
            data={"title": "Unauthorized Update"},
        )


async def test_cross_tenant_delete_rejected(
    contract_service,
):
    try:
        result = await delete_contract(
            contract_service,
            contract_id=CONTRACT_B,
            tenant_id=TENANT_A,
            user_id=USER_A,
        )
    except (PermissionError, LookupError, ValueError):
        return

    assert result is False


async def test_tenant_b_can_read_own_contract(
    contract_service,
):
    result = await get_contract(
        contract_service,
        contract_id=CONTRACT_B,
        tenant_id=TENANT_B,
        user_id=USER_B,
    )

    assert result["tenant_id"] == TENANT_B


# ============================================================
# PERMISSIONS
# ============================================================

async def test_permission_checked_on_create(
    contract_service,
    permission_service,
):
    await create_contract(
        contract_service
    )

    permission_service.has_permission.assert_awaited()


async def test_viewer_can_read(
    contract_service,
):
    result = await get_contract(
        contract_service,
        user_id=VIEWER_A,
    )

    assert result["id"] == CONTRACT_A


async def test_viewer_cannot_create(
    contract_service,
):
    with pytest.raises(PermissionError):
        await create_contract(
            contract_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_update(
    contract_service,
):
    with pytest.raises(PermissionError):
        await update_contract(
            contract_service,
            user_id=VIEWER_A,
        )


async def test_viewer_cannot_delete(
    contract_service,
):
    with pytest.raises(PermissionError):
        await delete_contract(
            contract_service,
            user_id=VIEWER_A,
        )


async def test_permission_service_failure_fails_closed(
    contract_service,
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
        await create_contract(
            contract_service
        )


# ============================================================
# SOFT DELETION
# ============================================================

async def test_delete_contract(
    contract_service,
):
    result = await delete_contract(
        contract_service
    )

    assert result is True


async def test_deleted_contract_hidden(
    contract_service,
):
    await delete_contract(
        contract_service
    )

    try:
        result = await get_contract(
            contract_service
        )
    except (LookupError, ValueError):
        return

    assert result is None


async def test_deleted_contract_not_listed(
    contract_service,
):
    await delete_contract(
        contract_service
    )

    results = await list_contracts(
        contract_service
    )

    assert CONTRACT_A not in {
        item["id"]
        for item in results
    }


async def test_soft_delete_preserves_record(
    contract_service,
    contract_repository,
):
    await delete_contract(
        contract_service
    )

    assert CONTRACT_A in (
        contract_repository.contracts
    )

    assert (
        contract_repository.contracts[
            CONTRACT_A
        ]["deleted_at"]
        is not None
    )


async def test_repeated_delete_is_safe(
    contract_service,
):
    await delete_contract(
        contract_service
    )

    try:
        result = await delete_contract(
            contract_service
        )
    except (LookupError, ValueError):
        return

    assert result is False


# ============================================================
# INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_title",
    ["", "   ", None],
)
async def test_invalid_contract_title(
    contract_service,
    invalid_title,
):
    data = make_create_data()
    data["title"] = invalid_title

    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_contract(
            contract_service,
            data=data,
        )


async def test_missing_counterparty_rejected(
    contract_service,
):
    data = make_create_data()
    data["counterparty"] = ""

    with pytest.raises(ValueError):
        await create_contract(
            contract_service,
            data=data,
        )


@pytest.mark.parametrize(
    "invalid_tenant",
    ["", "   ", None],
)
async def test_invalid_tenant(
    contract_service,
    invalid_tenant,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await create_contract(
            contract_service,
            tenant_id=invalid_tenant,
        )


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_create_contract_audited(
    contract_service,
    audit_service,
):
    await create_contract(
        contract_service
    )

    audit_service.record_event.assert_awaited()


async def test_update_contract_audited(
    contract_service,
    audit_service,
):
    await update_contract(
        contract_service
    )

    audit_service.record_event.assert_awaited()


async def test_delete_contract_audited(
    contract_service,
    audit_service,
):
    await delete_contract(
        contract_service
    )

    audit_service.record_event.assert_awaited()


async def test_audit_contains_tenant_context(
    contract_service,
    audit_service,
):
    await create_contract(
        contract_service
    )

    assert TENANT_A in repr(
        audit_service.record_event.await_args_list
    )


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_contract_creation(
    contract_service,
):
    results = await asyncio.gather(
        *[
            create_contract(
                contract_service,
                data=make_create_data(
                    title=f"Agreement {index}",
                ),
            )
            for index in range(20)
        ]
    )

    assert len(results) == 20

    ids = {
        result["id"]
        for result in results
    }

    assert len(ids) == 20


async def test_concurrent_tenant_contract_reads(
    contract_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        get_contract(
            contract_service,
            contract_id=CONTRACT_A,
            tenant_id=TENANT_A,
            user_id=USER_A,
        ),
        get_contract(
            contract_service,
            contract_id=CONTRACT_B,
            tenant_id=TENANT_B,
            user_id=USER_B,
        ),
    )

    assert tenant_a["tenant_id"] == TENANT_A
    assert tenant_b["tenant_id"] == TENANT_B


async def test_concurrent_contract_listing(
    contract_service,
):
    results = await asyncio.gather(
        *[
            list_contracts(
                contract_service,
                tenant_id=TENANT_A,
            )
            for _ in range(10)
        ]
    )

    assert len(results) == 10

    for contracts in results:
        assert all(
            contract["tenant_id"] == TENANT_A
            for contract in contracts
        )


# ============================================================
# END
# ============================================================
