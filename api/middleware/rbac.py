"""
Role-Based Access Control (RBAC) middleware/dependencies.

Responsibilities:
- Resolve the authenticated user from request state.
- Enforce role-based permissions.
- Support organization/tenant isolation.
- Support hierarchical roles.
- Provide reusable FastAPI dependencies.
- Prevent privilege escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Sequence

from fastapi import Depends, HTTPException, Request, status


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    LEGAL_ADMIN = "legal_admin"
    LEGAL_REVIEWER = "legal_reviewer"
    LEGAL_ANALYST = "legal_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


# Higher value = more privileges.
ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 10,
    Role.USER: 20,
    Role.LEGAL_ANALYST: 30,
    Role.LEGAL_REVIEWER: 40,
    Role.COMPLIANCE_OFFICER: 50,
    Role.MANAGER: 60,
    Role.LEGAL_ADMIN: 70,
    Role.ORG_ADMIN: 80,
    Role.SUPER_ADMIN: 100,
}


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class Permission(str, Enum):
    # Users
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # Organizations
    ORGANIZATION_READ = "organization:read"
    ORGANIZATION_UPDATE = "organization:update"

    # Documents
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_SHARE = "document:share"

    # Contracts
    CONTRACT_READ = "contract:read"
    CONTRACT_CREATE = "contract:create"
    CONTRACT_UPDATE = "contract:update"
    CONTRACT_DELETE = "contract:delete"

    # Legal intelligence
    CLAUSE_READ = "clause:read"
    CLAUSE_ANALYZE = "clause:analyze"
    LEGAL_QA = "legal_qa"
    CONTRACT_COMPARE = "contract:compare"

    # Risk
    RISK_READ = "risk:read"
    RISK_ANALYZE = "risk:analyze"
    RISK_UPDATE = "risk:update"

    # Lifecycle
    EXPIRY_READ = "expiry:read"
    RENEWAL_READ = "renewal:read"
    RENEWAL_MANAGE = "renewal:manage"

    # Reviews
    REVIEW_READ = "review:read"
    REVIEW_CREATE = "review:create"
    REVIEW_APPROVE = "review:approve"

    # Analytics
    ANALYTICS_READ = "analytics:read"

    # Administration
    AUDIT_READ = "audit:read"
    MODEL_READ = "model:read"
    MODEL_MANAGE = "model:manage"


# ---------------------------------------------------------------------------
# Role permissions
# ---------------------------------------------------------------------------


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.DOCUMENT_READ,
        Permission.CONTRACT_READ,
        Permission.CLAUSE_READ,
        Permission.RISK_READ,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
    },
    Role.USER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.CONTRACT_READ,
        Permission.CONTRACT_CREATE,
        Permission.CONTRACT_UPDATE,
        Permission.CLAUSE_READ,
        Permission.CLAUSE_ANALYZE,
        Permission.LEGAL_QA,
        Permission.CONTRACT_COMPARE,
        Permission.RISK_READ,
        Permission.RISK_ANALYZE,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
        Permission.REVIEW_READ,
        Permission.REVIEW_CREATE,
    },
    Role.LEGAL_ANALYST: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_UPDATE,
        Permission.CONTRACT_READ,
        Permission.CONTRACT_CREATE,
        Permission.CONTRACT_UPDATE,
        Permission.CLAUSE_READ,
        Permission.CLAUSE_ANALYZE,
        Permission.LEGAL_QA,
        Permission.CONTRACT_COMPARE,
        Permission.RISK_READ,
        Permission.RISK_ANALYZE,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
        Permission.REVIEW_READ,
        Permission.REVIEW_CREATE,
        Permission.ANALYTICS_READ,
    },
    Role.LEGAL_REVIEWER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_UPDATE,
        Permission.DOCUMENT_SHARE,
        Permission.CONTRACT_READ,
        Permission.CONTRACT_CREATE,
        Permission.CONTRACT_UPDATE,
        Permission.CLAUSE_READ,
        Permission.CLAUSE_ANALYZE,
        Permission.LEGAL_QA,
        Permission.CONTRACT_COMPARE,
        Permission.RISK_READ,
        Permission.RISK_ANALYZE,
        Permission.RISK_UPDATE,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
        Permission.RENEWAL_MANAGE,
        Permission.REVIEW_READ,
        Permission.REVIEW_CREATE,
        Permission.REVIEW_APPROVE,
        Permission.ANALYTICS_READ,
    },
    Role.COMPLIANCE_OFFICER: {
        Permission.DOCUMENT_READ,
        Permission.CONTRACT_READ,
        Permission.CLAUSE_READ,
        Permission.LEGAL_QA,
        Permission.RISK_READ,
        Permission.RISK_ANALYZE,
        Permission.RISK_UPDATE,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
        Permission.RENEWAL_MANAGE,
        Permission.REVIEW_READ,
        Permission.ANALYTICS_READ,
        Permission.AUDIT_READ,
    },
    Role.MANAGER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_UPDATE,
        Permission.DOCUMENT_SHARE,
        Permission.CONTRACT_READ,
        Permission.CONTRACT_CREATE,
        Permission.CONTRACT_UPDATE,
        Permission.CLAUSE_READ,
        Permission.CLAUSE_ANALYZE,
        Permission.LEGAL_QA,
        Permission.CONTRACT_COMPARE,
        Permission.RISK_READ,
        Permission.RISK_ANALYZE,
        Permission.RISK_UPDATE,
        Permission.EXPIRY_READ,
        Permission.RENEWAL_READ,
        Permission.RENEWAL_MANAGE,
        Permission.REVIEW_READ,
        Permission.REVIEW_CREATE,
        Permission.REVIEW_APPROVE,
        Permission.ANALYTICS_READ,
    },
    Role.LEGAL_ADMIN: set(Permission),
    Role.ORG_ADMIN: set(Permission),
    Role.SUPER_ADMIN: set(Permission),
}


# ---------------------------------------------------------------------------
# User context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthenticatedUser:
    """
    Minimal authorization context.

    The authentication middleware should populate request.state.user
    with an equivalent object.
    """

    id: str
    organization_id: str
    role: Role
    is_active: bool = True
    permissions: frozenset[Permission] = frozenset()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def forbidden(detail: str = "Insufficient permissions.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def unauthorized(detail: str = "Authentication required.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# User resolution
# ---------------------------------------------------------------------------


def get_current_user(request: Request) -> AuthenticatedUser:
    """
    Resolve the authenticated user.

    Authentication should happen before RBAC. This function intentionally
    does not decode JWTs itself; authentication belongs in auth middleware.
    """

    user: Optional[AuthenticatedUser] = getattr(
        request.state,
        "user",
        None,
    )

    if user is None:
        raise unauthorized()

    if not user.is_active:
        raise forbidden("User account is inactive.")

    return user


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


def normalize_role(role: Role | str) -> Role:
    if isinstance(role, Role):
        return role

    try:
        return Role(role)
    except ValueError as exc:
        raise forbidden("Invalid user role.") from exc


def has_role(
    user: AuthenticatedUser,
    required_role: Role | str,
) -> bool:
    user_role = normalize_role(user.role)
    required = normalize_role(required_role)

    return ROLE_HIERARCHY[user_role] >= ROLE_HIERARCHY[required]


def role_permissions(role: Role | str) -> set[Permission]:
    normalized = normalize_role(role)

    # Administrative roles intentionally inherit all permissions.
    if normalized in {
        Role.SUPER_ADMIN,
        Role.ORG_ADMIN,
        Role.LEGAL_ADMIN,
    }:
        return set(Permission)

    return ROLE_PERMISSIONS.get(normalized, set()).copy()


def effective_permissions(
    user: AuthenticatedUser,
) -> set[Permission]:
    permissions = role_permissions(user.role)

    # Explicit permissions can only grant permissions already allowed
    # by the role unless your identity system deliberately supports
    # custom permission grants.
    permissions.update(user.permissions)

    return permissions


def has_permission(
    user: AuthenticatedUser,
    permission: Permission | str,
) -> bool:
    try:
        required = (
            permission
            if isinstance(permission, Permission)
            else Permission(permission)
        )
    except ValueError:
        return False

    return required in effective_permissions(user)


def require_permissions(
    *permissions: Permission | str,
    require_all: bool = True,
) -> Callable[..., AuthenticatedUser]:
    """
    FastAPI dependency factory.

    Example:

        @router.get(
            "/documents",
            dependencies=[Depends(
                require_permissions(Permission.DOCUMENT_READ)
            )],
        )
    """

    required = tuple(
        permission
        if isinstance(permission, Permission)
        else Permission(permission)
        for permission in permissions
    )

    def dependency(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not required:
            return user

        allowed = [has_permission(user, permission) for permission in required]

        authorized = all(allowed) if require_all else any(allowed)

        if not authorized:
            permission_names = ", ".join(permission.value for permission in required)

            raise forbidden(
                f"Missing required permission(s): {permission_names}"
            )

        return user

    return dependency


def require_role(
    *roles: Role | str,
) -> Callable[..., AuthenticatedUser]:
    """
    Require one of the supplied roles or a higher role in the hierarchy.
    """

    required_roles = tuple(normalize_role(role) for role in roles)

    def dependency(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not required_roles:
            return user

        if any(has_role(user, role) for role in required_roles):
            return user

        allowed = ", ".join(role.value for role in required_roles)

        raise forbidden(
            f"Required role: {allowed}"
        )

    return dependency


# ---------------------------------------------------------------------------
# Organization / tenant isolation
# ---------------------------------------------------------------------------


def require_same_organization(
    organization_id: str,
    user: AuthenticatedUser,
) -> None:
    """
    Prevent cross-tenant access.

    SUPER_ADMIN is allowed to operate across organizations.
    """

    if user.role == Role.SUPER_ADMIN:
        return

    if user.organization_id != organization_id:
        raise forbidden(
            "Access to resources belonging to another organization is denied."
        )


def enforce_tenant(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Store the resolved tenant in request state.

    Downstream repositories/services should use this tenant context when
    querying organization-scoped resources.
    """

    request.state.organization_id = user.organization_id
    return user


# ---------------------------------------------------------------------------
# Route-level authorization decorator
# ---------------------------------------------------------------------------


def authorize(
    permissions: Sequence[Permission | str] | None = None,
    roles: Sequence[Role | str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for service-layer or custom route authorization.

    Prefer FastAPI Depends() for normal route authorization.
    """

    required_permissions = tuple(
        permission
        if isinstance(permission, Permission)
        else Permission(permission)
        for permission in (permissions or ())
    )

    required_roles = tuple(
        normalize_role(role)
        for role in (roles or ())
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            user = kwargs.get("current_user")

            if user is None:
                raise unauthorized()

            if not isinstance(user, AuthenticatedUser):
                raise unauthorized()

            if required_roles and not any(
                has_role(user, role)
                for role in required_roles
            ):
                raise forbidden("Insufficient role.")

            if required_permissions and not all(
                has_permission(user, permission)
                for permission in required_permissions
            ):
                raise forbidden("Insufficient permissions.")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Convenience dependencies
# ---------------------------------------------------------------------------


require_document_read = require_permissions(
    Permission.DOCUMENT_READ
)

require_document_write = require_permissions(
    Permission.DOCUMENT_UPDATE
)

require_contract_read = require_permissions(
    Permission.CONTRACT_READ
)

require_contract_write = require_permissions(
    Permission.CONTRACT_UPDATE
)

require_legal_qa = require_permissions(
    Permission.LEGAL_QA
)

require_risk_analysis = require_permissions(
    Permission.RISK_ANALYZE
)

require_review_approval = require_permissions(
    Permission.REVIEW_APPROVE
)

require_admin = require_role(
    Role.ORG_ADMIN,
    Role.LEGAL_ADMIN,
)


__all__ = [
    "Role",
    "Permission",
    "AuthenticatedUser",
    "ROLE_HIERARCHY",
    "ROLE_PERMISSIONS",
    "get_current_user",
    "has_role",
    "has_permission",
    "effective_permissions",
    "require_role",
    "require_permissions",
    "require_same_organization",
    "enforce_tenant",
    "authorize",
    "require_document_read",
    "require_document_write",
    "require_contract_read",
    "require_contract_write",
    "require_legal_qa",
    "require_risk_analysis",
    "require_review_approval",
    "require_admin",
]