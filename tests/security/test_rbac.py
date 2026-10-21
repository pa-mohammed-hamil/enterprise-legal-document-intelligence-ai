
"""
tests/unit/test_rbac.py

Lexora AI — Role-Based Access Control (RBAC) Tests

Expected production module:
    backend.app.security.rbac

Expected class:
    RBACService

Expected constructor:
    RBACService(role_repository=..., audit_service=...)

Expected methods:
    async has_permission(
        user_id: str,
        tenant_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> bool

    async assign_role(
        user_id: str,
        tenant_id: str,
        role: str,
        actor_id: str,
    ) -> None

    async revoke_role(
        user_id: str,
        tenant_id: str,
        role: str,
        actor_id: str,
    ) -> None

    async get_user_roles(
        user_id: str,
        tenant_id: str,
    ) -> list[str]

Security requirements:
    - Deny access by default.
    - Enforce tenant isolation.
    - Enforce least-privilege permissions.
    - Prevent unauthorized role changes.
    - Protect privileged administrative actions.
    - Reject invalid identities and permissions.
    - Record successful role changes in audit logs.
    - Fail closed on authorization backend errors.

These are specification tests and may require
adapting the service interface to the repository.
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

ADMIN = "admin_001"
LEGAL_MANAGER = "legal_manager_001"
LEGAL_ANALYST = "legal_analyst_001"
VIEWER = "viewer_001"
GUEST = "guest_001"
SUPER_ADMIN = "super_admin_001"

DOCUMENT_ID = "contract_001"

ROLE_ADMIN = "admin"
ROLE_LEGAL_MANAGER = "legal_manager"
ROLE_LEGAL_ANALYST = "legal_analyst"
ROLE_VIEWER = "viewer"
ROLE_GUEST = "guest"
ROLE_SUPER_ADMIN = "super_admin"

READ_DOCUMENT = "document:read"
CREATE_DOCUMENT = "document:create"
UPDATE_DOCUMENT = "document:update"
DELETE_DOCUMENT = "document:delete"
APPROVE_CONTRACT = "contract:approve"
MANAGE_USERS = "user:manage"
MANAGE_ROLES = "role:manage"
VIEW_AUDIT = "audit:read"
MANAGE_SETTINGS = "settings:manage"

ALL_PERMISSIONS = {
    READ_DOCUMENT,
    CREATE_DOCUMENT,
    UPDATE_DOCUMENT,
    DELETE_DOCUMENT,
    APPROVE_CONTRACT,
    MANAGE_USERS,
    MANAGE_ROLES,
    VIEW_AUDIT,
    MANAGE_SETTINGS,
}

ROLE_PERMISSIONS = {
    ROLE_GUEST: set(),
    ROLE_VIEWER: {
        READ_DOCUMENT,
    },
    ROLE_LEGAL_ANALYST: {
        READ_DOCUMENT,
        CREATE_DOCUMENT,
        UPDATE_DOCUMENT,
    },
    ROLE_LEGAL_MANAGER: {
        READ_DOCUMENT,
        CREATE_DOCUMENT,
        UPDATE_DOCUMENT,
        APPROVE_CONTRACT,
    },
    ROLE_ADMIN: ALL_PERMISSIONS,
    ROLE_SUPER_ADMIN: ALL_PERMISSIONS,
}


# ============================================================
# IMPORT
# ============================================================

def load_rbac_class():
    candidates = (
        "backend.app.security.rbac",
        "backend.app.services.rbac_service",
        "app.security.rbac",
        "app.services.rbac_service",
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
            "RBACService",
            None,
        )

        if service_class is not None:
            return service_class

    pytest.fail(
        "RBACService could not be imported. "
        "Update the module path or class name.\n"
        + "\n".join(errors)
    )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def initial_roles():
    return {
        TENANT_A: {
            ADMIN: [ROLE_ADMIN],
            LEGAL_MANAGER: [ROLE_LEGAL_MANAGER],
            LEGAL_ANALYST: [ROLE_LEGAL_ANALYST],
            VIEWER: [ROLE_VIEWER],
            GUEST: [ROLE_GUEST],
            SUPER_ADMIN: [ROLE_SUPER_ADMIN],
        },
        TENANT_B: {
            ADMIN: [ROLE_VIEWER],
            LEGAL_MANAGER: [],
            LEGAL_ANALYST: [],
            VIEWER: [],
            GUEST: [],
            SUPER_ADMIN: [],
        },
    }


@pytest.fixture
def role_repository(initial_roles):
    repository = MagicMock()
    repository.roles = deepcopy(initial_roles)

    async def get_roles(user_id, tenant_id):
        return list(
            repository.roles
            .get(tenant_id, {})
            .get(user_id, [])
        )

    async def add_role(user_id, tenant_id, role):
        tenant_roles = repository.roles.setdefault(
            tenant_id, {}
        )

        user_roles = tenant_roles.setdefault(
            user_id, []
        )

        if role not in user_roles:
            user_roles.append(role)

    async def remove_role(user_id, tenant_id, role):
        roles = (
            repository.roles
            .get(tenant_id, {})
            .get(user_id, [])
        )

        if role in roles:
            roles.remove(role)

    repository.get_roles = AsyncMock(
        side_effect=get_roles
    )
    repository.add_role = AsyncMock(
        side_effect=add_role
    )
    repository.remove_role = AsyncMock(
        side_effect=remove_role
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
def rbac_service(
    role_repository,
    audit_service,
):
    service_class = load_rbac_class()

    return service_class(
        role_repository=role_repository,
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


async def check_permission(
    service,
    user_id,
    permission,
    tenant_id=TENANT_A,
    resource_id=None,
):
    return await invoke(
        service.has_permission,
        user_id=user_id,
        tenant_id=tenant_id,
        permission=permission,
        resource_id=resource_id,
    )


async def assign_role(
    service,
    user_id,
    role,
    actor_id=ADMIN,
    tenant_id=TENANT_A,
):
    return await invoke(
        service.assign_role,
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        actor_id=actor_id,
    )


async def revoke_role(
    service,
    user_id,
    role,
    actor_id=ADMIN,
    tenant_id=TENANT_A,
):
    return await invoke(
        service.revoke_role,
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        actor_id=actor_id,
    )


async def get_roles(
    service,
    user_id,
    tenant_id=TENANT_A,
):
    return await invoke(
        service.get_user_roles,
        user_id=user_id,
        tenant_id=tenant_id,
    )


# ============================================================
# INITIALIZATION
# ============================================================

async def test_rbac_initialization(rbac_service):
    assert rbac_service is not None
    assert callable(rbac_service.has_permission)
    assert callable(rbac_service.assign_role)
    assert callable(rbac_service.revoke_role)


# ============================================================
# ROLE PERMISSION MATRIX
# ============================================================

@pytest.mark.parametrize(
    "role_user,permission,expected",
    [
        (ADMIN, READ_DOCUMENT, True),
        (ADMIN, DELETE_DOCUMENT, True),
        (ADMIN, MANAGE_USERS, True),
        (ADMIN, MANAGE_ROLES, True),
        (ADMIN, VIEW_AUDIT, True),
        (LEGAL_MANAGER, READ_DOCUMENT, True),
        (LEGAL_MANAGER, APPROVE_CONTRACT, True),
        (LEGAL_MANAGER, DELETE_DOCUMENT, False),
        (LEGAL_MANAGER, MANAGE_ROLES, False),
        (LEGAL_ANALYST, READ_DOCUMENT, True),
        (LEGAL_ANALYST, CREATE_DOCUMENT, True),
        (LEGAL_ANALYST, UPDATE_DOCUMENT, True),
        (LEGAL_ANALYST, APPROVE_CONTRACT, False),
        (LEGAL_ANALYST, DELETE_DOCUMENT, False),
        (VIEWER, READ_DOCUMENT, True),
        (VIEWER, CREATE_DOCUMENT, False),
        (VIEWER, UPDATE_DOCUMENT, False),
        (VIEWER, DELETE_DOCUMENT, False),
        (VIEWER, MANAGE_USERS, False),
        (GUEST, READ_DOCUMENT, False),
        (GUEST, MANAGE_ROLES, False),
    ],
)
async def test_role_permission_matrix(
    rbac_service,
    role_user,
    permission,
    expected,
):
    allowed = await check_permission(
        rbac_service,
        user_id=role_user,
        permission=permission,
    )

    assert allowed is expected


# ============================================================
# ADMINISTRATIVE ACCESS
# ============================================================

async def test_admin_can_manage_users(rbac_service):
    assert await check_permission(
        rbac_service,
        ADMIN,
        MANAGE_USERS,
    )


async def test_admin_can_manage_roles(rbac_service):
    assert await check_permission(
        rbac_service,
        ADMIN,
        MANAGE_ROLES,
    )


async def test_viewer_cannot_manage_users(rbac_service):
    assert not await check_permission(
        rbac_service,
        VIEWER,
        MANAGE_USERS,
    )


async def test_analyst_cannot_manage_roles(rbac_service):
    assert not await check_permission(
        rbac_service,
        LEGAL_ANALYST,
        MANAGE_ROLES,
    )


async def test_guest_has_no_privileged_permissions(
    rbac_service,
):
    for permission in ALL_PERMISSIONS:
        assert not await check_permission(
            rbac_service,
            GUEST,
            permission,
        )


# ============================================================
# DEFAULT DENY
# ============================================================

async def test_unknown_user_denied(rbac_service):
    assert not await check_permission(
        rbac_service,
        "unknown_user",
        READ_DOCUMENT,
    )


async def test_unknown_permission_denied(rbac_service):
    assert not await check_permission(
        rbac_service,
        ADMIN,
        "system:unrestricted_access",
    )


async def test_unknown_role_does_not_grant_access(
    rbac_service,
    role_repository,
):
    role_repository.roles[TENANT_A][GUEST] = [
        "unknown_role"
    ]

    assert not await check_permission(
        rbac_service,
        GUEST,
        READ_DOCUMENT,
    )


# ============================================================
# TENANT ISOLATION
# ============================================================

async def test_admin_role_is_tenant_scoped(
    rbac_service,
):
    assert await check_permission(
        rbac_service,
        ADMIN,
        MANAGE_USERS,
        tenant_id=TENANT_A,
    )

    assert not await check_permission(
        rbac_service,
        ADMIN,
        MANAGE_USERS,
        tenant_id=TENANT_B,
    )


async def test_tenant_b_viewer_permissions(
    rbac_service,
):
    assert await check_permission(
        rbac_service,
        ADMIN,
        READ_DOCUMENT,
        tenant_id=TENANT_B,
    )

    assert not await check_permission(
        rbac_service,
        ADMIN,
        DELETE_DOCUMENT,
        tenant_id=TENANT_B,
    )


async def test_unknown_tenant_denied(rbac_service):
    assert not await check_permission(
        rbac_service,
        ADMIN,
        READ_DOCUMENT,
        tenant_id="unknown_tenant",
    )


async def test_cross_tenant_role_assignment_denied(
    rbac_service,
    role_repository,
):
    before = deepcopy(role_repository.roles)

    try:
        await assign_role(
            rbac_service,
            VIEWER,
            ROLE_ADMIN,
            actor_id=ADMIN,
            tenant_id=TENANT_B,
        )
    except PermissionError:
        pass

    assert role_repository.roles == before


# ============================================================
# ROLE ASSIGNMENT
# ============================================================

async def test_admin_can_assign_viewer_role(
    rbac_service,
):
    user_id = "new_employee"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    roles = await get_roles(
        rbac_service,
        user_id,
    )

    assert ROLE_VIEWER in roles


async def test_admin_can_assign_analyst_role(
    rbac_service,
):
    user_id = "new_analyst"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_LEGAL_ANALYST,
    )

    assert ROLE_LEGAL_ANALYST in await get_roles(
        rbac_service,
        user_id,
    )


async def test_viewer_cannot_assign_roles(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            "new_user",
            ROLE_ADMIN,
            actor_id=VIEWER,
        )


async def test_analyst_cannot_assign_roles(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            "new_user",
            ROLE_VIEWER,
            actor_id=LEGAL_ANALYST,
        )


async def test_guest_cannot_assign_roles(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            "new_user",
            ROLE_ADMIN,
            actor_id=GUEST,
        )


async def test_duplicate_role_assignment_is_idempotent(
    rbac_service,
):
    user_id = "employee_duplicate"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    roles = await get_roles(
        rbac_service,
        user_id,
    )

    assert roles.count(ROLE_VIEWER) == 1


# ============================================================
# ROLE REVOCATION
# ============================================================

async def test_admin_can_revoke_role(
    rbac_service,
):
    user_id = "employee_revoke"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    await revoke_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    assert ROLE_VIEWER not in await get_roles(
        rbac_service,
        user_id,
    )


async def test_revoked_role_loses_permissions(
    rbac_service,
):
    user_id = "employee_access"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    assert await check_permission(
        rbac_service,
        user_id,
        READ_DOCUMENT,
    )

    await revoke_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    assert not await check_permission(
        rbac_service,
        user_id,
        READ_DOCUMENT,
    )


async def test_viewer_cannot_revoke_roles(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await revoke_role(
            rbac_service,
            LEGAL_ANALYST,
            ROLE_LEGAL_ANALYST,
            actor_id=VIEWER,
        )


async def test_revoke_nonexistent_role_safe(
    rbac_service,
):
    await revoke_role(
        rbac_service,
        "unknown_employee",
        ROLE_VIEWER,
    )

    assert await get_roles(
        rbac_service,
        "unknown_employee",
    ) == []


# ============================================================
# PRIVILEGE ESCALATION
# ============================================================

async def test_viewer_cannot_promote_self(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            VIEWER,
            ROLE_ADMIN,
            actor_id=VIEWER,
        )


async def test_analyst_cannot_promote_self(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            LEGAL_ANALYST,
            ROLE_ADMIN,
            actor_id=LEGAL_ANALYST,
        )


async def test_unknown_actor_cannot_assign_admin(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            "new_employee",
            ROLE_ADMIN,
            actor_id="unknown_actor",
        )


async def test_invalid_role_rejected(
    rbac_service,
):
    with pytest.raises(
        (ValueError, PermissionError)
    ):
        await assign_role(
            rbac_service,
            "new_employee",
            "root_system_owner",
        )


# ============================================================
# SUPER ADMIN PROTECTION
# ============================================================

async def test_super_admin_has_privileged_permissions(
    rbac_service,
):
    assert await check_permission(
        rbac_service,
        SUPER_ADMIN,
        MANAGE_SETTINGS,
    )


async def test_viewer_cannot_assign_super_admin(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            VIEWER,
            ROLE_SUPER_ADMIN,
            actor_id=VIEWER,
        )


async def test_tenant_admin_cannot_assign_super_admin(
    rbac_service,
):
    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            VIEWER,
            ROLE_SUPER_ADMIN,
            actor_id=ADMIN,
        )


# ============================================================
# DOCUMENT PERMISSION CHECKS
# ============================================================

async def test_viewer_can_request_document_read(
    rbac_service,
):
    assert await check_permission(
        rbac_service,
        VIEWER,
        READ_DOCUMENT,
        resource_id=DOCUMENT_ID,
    )


async def test_viewer_cannot_delete_document(
    rbac_service,
):
    assert not await check_permission(
        rbac_service,
        VIEWER,
        DELETE_DOCUMENT,
        resource_id=DOCUMENT_ID,
    )


async def test_analyst_can_update_document(
    rbac_service,
):
    assert await check_permission(
        rbac_service,
        LEGAL_ANALYST,
        UPDATE_DOCUMENT,
        resource_id=DOCUMENT_ID,
    )


async def test_guest_cannot_read_document(
    rbac_service,
):
    assert not await check_permission(
        rbac_service,
        GUEST,
        READ_DOCUMENT,
        resource_id=DOCUMENT_ID,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_user_id",
    ["", "   ", None],
)
async def test_invalid_user_id_rejected(
    rbac_service,
    invalid_user_id,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await check_permission(
            rbac_service,
            invalid_user_id,
            READ_DOCUMENT,
        )


@pytest.mark.parametrize(
    "invalid_tenant_id",
    ["", "   ", None],
)
async def test_invalid_tenant_id_rejected(
    rbac_service,
    invalid_tenant_id,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await check_permission(
            rbac_service,
            ADMIN,
            READ_DOCUMENT,
            tenant_id=invalid_tenant_id,
        )


@pytest.mark.parametrize(
    "invalid_permission",
    ["", "   ", None],
)
async def test_invalid_permission_rejected(
    rbac_service,
    invalid_permission,
):
    with pytest.raises(
        (ValueError, TypeError)
    ):
        await check_permission(
            rbac_service,
            ADMIN,
            invalid_permission,
        )


# ============================================================
# BACKEND FAILURE — FAIL CLOSED
# ============================================================

async def test_repository_failure_denies_access(
    rbac_service,
    role_repository,
):
    role_repository.get_roles.side_effect = RuntimeError(
        "Authorization database unavailable"
    )

    try:
        allowed = await check_permission(
            rbac_service,
            ADMIN,
            MANAGE_USERS,
        )
    except (RuntimeError, PermissionError):
        return

    assert allowed is False


async def test_role_assignment_failure_not_silent(
    rbac_service,
    role_repository,
):
    role_repository.add_role.side_effect = RuntimeError(
        "Database write failed"
    )

    with pytest.raises(RuntimeError):
        await assign_role(
            rbac_service,
            "new_employee",
            ROLE_VIEWER,
        )


# ============================================================
# AUDIT LOGGING
# ============================================================

async def test_role_assignment_is_audited(
    rbac_service,
    audit_service,
):
    await assign_role(
        rbac_service,
        "audited_employee",
        ROLE_VIEWER,
    )

    audit_service.record_event.assert_awaited()


async def test_role_revocation_is_audited(
    rbac_service,
    audit_service,
):
    user_id = "audited_revoke_employee"

    await assign_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    audit_service.record_event.reset_mock()

    await revoke_role(
        rbac_service,
        user_id,
        ROLE_VIEWER,
    )

    audit_service.record_event.assert_awaited()


async def test_unauthorized_assignment_does_not_mutate_roles(
    rbac_service,
    role_repository,
):
    before = deepcopy(role_repository.roles)

    with pytest.raises(PermissionError):
        await assign_role(
            rbac_service,
            VIEWER,
            ROLE_ADMIN,
            actor_id=VIEWER,
        )

    assert role_repository.roles == before


# ============================================================
# CONCURRENCY
# ============================================================

async def test_concurrent_permission_checks(
    rbac_service,
):
    checks = [
        (ADMIN, MANAGE_USERS),
        (LEGAL_MANAGER, APPROVE_CONTRACT),
        (LEGAL_ANALYST, UPDATE_DOCUMENT),
        (VIEWER, READ_DOCUMENT),
        (GUEST, DELETE_DOCUMENT),
    ]

    results = await asyncio.gather(
        *[
            check_permission(
                rbac_service,
                user_id,
                permission,
            )
            for user_id, permission in checks
        ]
    )

    assert results == [
        True,
        True,
        True,
        True,
        False,
    ]


async def test_concurrent_tenant_isolation(
    rbac_service,
):
    tenant_a, tenant_b = await asyncio.gather(
        check_permission(
            rbac_service,
            ADMIN,
            MANAGE_USERS,
            tenant_id=TENANT_A,
        ),
        check_permission(
            rbac_service,
            ADMIN,
            MANAGE_USERS,
            tenant_id=TENANT_B,
        ),
    )

    assert tenant_a is True
    assert tenant_b is False


# ============================================================
# END
# ============================================================
