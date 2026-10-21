"""
Tenant / organization isolation middleware.

Responsibilities:
- Resolve the authenticated user's organization/tenant.
- Establish tenant context for the current request.
- Prevent unauthorized tenant switching.
- Validate optional tenant headers.
- Expose tenant context through request.state.
- Provide reusable FastAPI dependencies.

Important:
    Tenant isolation must also be enforced at the repository/database
    layer. Middleware alone is not sufficient protection against
    cross-tenant data access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TENANT_HEADER = "X-Tenant-ID"

# Tenant IDs should be opaque identifiers rather than user-controlled names.
TENANT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


# ---------------------------------------------------------------------------
# Tenant context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantContext:
    """
    Immutable tenant context attached to a request.
    """

    tenant_id: str
    user_id: str
    is_super_admin: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def tenant_forbidden(
    detail: str = "Tenant access denied.",
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def tenant_required(
    detail: str = "Tenant context is required.",
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_tenant_id(tenant_id: str) -> str:
    """
    Validate tenant identifiers before putting them into request state,
    logs, database queries, or downstream service calls.
    """

    if not tenant_id:
        raise tenant_required(
            "Tenant ID cannot be empty."
        )

    if len(tenant_id) > 128:
        raise tenant_forbidden(
            "Invalid tenant identifier."
        )

    if not TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise tenant_forbidden(
            "Invalid tenant identifier."
        )

    return tenant_id


# ---------------------------------------------------------------------------
# Authentication context
# ---------------------------------------------------------------------------


def get_authenticated_user(request: Request):
    """
    Retrieve the authentication context created by auth middleware.

    The authentication layer is responsible for validating the JWT/session.
    """

    user = getattr(request.state, "user", None)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not getattr(user, "is_active", True):
        raise tenant_forbidden(
            "User account is inactive."
        )

    return user


# ---------------------------------------------------------------------------
# Tenant resolution
# ---------------------------------------------------------------------------


def resolve_user_tenant(user) -> str:
    """
    Resolve the tenant from the authenticated identity.

    The tenant must come from trusted authentication/identity data,
    not from an arbitrary request parameter.
    """

    tenant_id = getattr(user, "organization_id", None)

    if not tenant_id:
        raise tenant_required(
            "Authenticated user has no organization."
        )

    return validate_tenant_id(str(tenant_id))


def resolve_requested_tenant(
    request: Request,
) -> Optional[str]:
    """
    Read an optional tenant header.

    This header is useful for super-admin / cross-organization operations,
    but must never override a normal user's authenticated tenant.
    """

    tenant_id = request.headers.get(TENANT_HEADER)

    if tenant_id is None:
        return None

    return validate_tenant_id(tenant_id)


# ---------------------------------------------------------------------------
# Tenant access control
# ---------------------------------------------------------------------------


def is_super_admin(user) -> bool:
    """
    Determine whether the authenticated identity has global tenant access.
    """

    role = getattr(user, "role", None)

    if hasattr(role, "value"):
        role = role.value

    return role == "super_admin"


def ensure_tenant_access(
    user,
    authenticated_tenant: str,
    requested_tenant: Optional[str],
) -> str:
    """
    Resolve the final tenant for this request.

    Normal users:
        authenticated tenant == effective tenant

    Super admins:
        may explicitly select another tenant.
    """

    if requested_tenant is None:
        return authenticated_tenant

    if requested_tenant == authenticated_tenant:
        return authenticated_tenant

    if not is_super_admin(user):
        raise tenant_forbidden(
            "Cross-tenant access is not permitted."
        )

    return requested_tenant


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TenantMiddleware:
    """
    Lightweight ASGI-compatible middleware.

    The middleware establishes tenant context before the request reaches
    application routes.

    It intentionally does not perform authentication itself.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)

        try:
            user = get_authenticated_user(request)

            authenticated_tenant = resolve_user_tenant(user)
            requested_tenant = resolve_requested_tenant(request)

            effective_tenant = ensure_tenant_access(
                user,
                authenticated_tenant,
                requested_tenant,
            )

            tenant_context = TenantContext(
                tenant_id=effective_tenant,
                user_id=str(user.id),
                is_super_admin=is_super_admin(user),
            )

            request.state.tenant = tenant_context
            request.state.tenant_id = effective_tenant

        except HTTPException:
            # Authentication/tenant errors are converted through the
            # application's global exception handling.
            raise

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_tenant_context(
    request: Request,
) -> TenantContext:
    """
    Retrieve the established tenant context.

    Use this dependency in routes/services that require tenant awareness.
    """

    context = getattr(
        request.state,
        "tenant",
        None,
    )

    if context is None:
        raise tenant_required()

    return context


def get_tenant_id(
    tenant: TenantContext = Depends(get_tenant_context),
) -> str:
    """
    Return the effective tenant ID.
    """

    return tenant.tenant_id


# ---------------------------------------------------------------------------
# Explicit tenant ownership check
# ---------------------------------------------------------------------------


def require_same_tenant(
    resource_tenant_id: str,
    tenant: TenantContext,
) -> None:
    """
    Verify that a resource belongs to the current tenant.

    Repositories/services should call this before operating on
    tenant-owned resources.
    """

    resource_tenant = validate_tenant_id(
        str(resource_tenant_id)
    )

    if tenant.is_super_admin:
        return

    if resource_tenant != tenant.tenant_id:
        raise tenant_forbidden(
            "Resource does not belong to the current tenant."
        )


# ---------------------------------------------------------------------------
# Repository helper
# ---------------------------------------------------------------------------


def tenant_filter(
    tenant: TenantContext,
) -> dict[str, str]:
    """
    Standard filter for tenant-scoped database queries.

    Example:

        filters = tenant_filter(tenant)

        query = select(Document).where(
            Document.organization_id == filters["organization_id"]
        )
    """

    return {
        "organization_id": tenant.tenant_id,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "TENANT_HEADER",
    "TenantContext",
    "TenantMiddleware",
    "get_tenant_context",
    "get_tenant_id",
    "validate_tenant_id",
    "require_same_tenant",
    "tenant_filter",
]