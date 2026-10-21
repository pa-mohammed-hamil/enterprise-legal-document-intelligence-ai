# backend/db/models/role.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(Base):
    """
    Tenant-aware RBAC role.

    Examples:
        - super_admin
        - organization_admin
        - legal_admin
        - legal_reviewer
        - contract_manager
        - risk_manager
        - compliance_officer
        - document_manager
        - analyst
        - standard_user
        - read_only

    Security model:

        Tenant
          -> Organization
              -> Role
                  -> Permissions
                      -> User

    `tenant_id` remains the hard security boundary.

    Roles define what a principal may attempt. Resource-level ACL
    checks must still be applied to protected documents/contracts.
    """

    __tablename__ = "roles"

    # ========================================================
    # IDENTITY / TENANCY
    # ========================================================

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # ROLE IDENTITY
    # ========================================================

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    code: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    display_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    role_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="custom",
        index=True,
    )

    # ========================================================
    # ROLE LEVEL
    # ========================================================

    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
    )

    # ========================================================
    # SYSTEM ROLE
    # ========================================================

    is_system: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    is_super_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ========================================================
    # STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    # ========================================================
    # PERMISSIONS
    # ========================================================

    permissions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    denied_permissions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # RESOURCE SCOPES
    # ========================================================

    resource_scopes: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_document_types: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_contract_types: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_departments: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_business_units: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # ROLE INHERITANCE
    # ========================================================

    parent_role_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "roles.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    inherit_permissions: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ========================================================
    # DOCUMENT PERMISSIONS
    # ========================================================

    can_view_documents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_upload_documents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_update_documents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_delete_documents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_download_documents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ========================================================
    # CONTRACT PERMISSIONS
    # ========================================================

    can_view_contracts: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_create_contracts: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_update_contracts: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_delete_contracts: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # CLAUSE / OBLIGATION PERMISSIONS
    # ========================================================

    can_analyze_clauses: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_obligations: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # RISK PERMISSIONS
    # ========================================================

    can_view_risks: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_manage_risks: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_resolve_risks: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_dismiss_risks: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # RENEWAL / EXPIRY
    # ========================================================

    can_manage_renewals: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_expiry: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # REVIEW
    # ========================================================

    can_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_approve: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_reject: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_assign_reviews: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # SEARCH / RAG / CHAT
    # ========================================================

    can_search: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_use_legal_chat: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_use_rag: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ========================================================
    # AI PERMISSIONS
    # ========================================================

    can_use_ai: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    can_run_agents: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_run_models: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_models: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_view_model_metrics: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # ANALYTICS
    # ========================================================

    can_view_analytics: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_export_data: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # AUDIT
    # ========================================================

    can_view_audit_logs: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_export_audit_logs: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # SECURITY
    # ========================================================

    can_view_security_events: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_security: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_view_pii: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_override_guardrails: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # ADMINISTRATION
    # ========================================================

    can_manage_users: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_roles: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_organization: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_settings: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_manage_integrations: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ========================================================
    # ASSIGNMENT LIMITS
    # ========================================================

    max_users: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    user_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # SESSION / AUTH POLICIES
    # ========================================================

    require_mfa: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    session_timeout_minutes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # CUSTOM POLICY
    # ========================================================

    policy: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    conditions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # SOFT DELETE
    # ========================================================

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )

    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    organization = relationship(
        "Organization",
        foreign_keys=[organization_id],
        lazy="selectin",
    )

    parent_role = relationship(
        "Role",
        remote_side=[id],
        foreign_keys=[parent_role_id],
        back_populates="child_roles",
    )

    child_roles = relationship(
        "Role",
        back_populates="parent_role",
        foreign_keys=[parent_role_id],
        lazy="selectin",
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "organization_id",
            "code",
            name="uq_roles_tenant_org_code",
        ),
        Index(
            "ix_roles_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_roles_tenant_org_active",
            "tenant_id",
            "organization_id",
            "is_active",
            "is_deleted",
        ),
        Index(
            "ix_roles_tenant_type",
            "tenant_id",
            "role_type",
        ),
        Index(
            "ix_roles_tenant_system",
            "tenant_id",
            "is_system",
        ),
        Index(
            "ix_roles_tenant_default",
            "tenant_id",
            "organization_id",
            "is_default",
        ),
        Index(
            "ix_roles_parent",
            "tenant_id",
            "parent_role_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def enabled(self) -> bool:
        return bool(
            self.is_active
            and not self.is_deleted
            and (self.status or "").lower()
            == "active"
        )

    @property
    def assignable(self) -> bool:
        if not self.enabled:
            return False

        if self.max_users is None:
            return True

        return self.user_count < self.max_users

    @property
    def full_name(self) -> str:
        return (
            self.display_name
            or self.name
        )

    # ========================================================
    # PERMISSION ENGINE
    # ========================================================

    def has_permission(
        self,
        permission: str,
        *,
        include_parent: bool = True,
    ) -> bool:
        """
        Explicit deny wins.

        "*" grants all permissions except explicitly denied ones.
        """

        permission = (
            permission
            .strip()
            .lower()
        )

        denied = {
            str(item).strip().lower()
            for item in (
                self.denied_permissions or []
            )
        }

        if (
            permission in denied
            or "*" in denied
        ):
            return False

        if self.is_super_admin:
            return True

        allowed = {
            str(item).strip().lower()
            for item in (
                self.permissions or []
            )
        }

        if (
            permission in allowed
            or "*" in allowed
        ):
            return True

        if (
            include_parent
            and self.inherit_permissions
            and self.parent_role is not None
        ):
            return self.parent_role.has_permission(
                permission,
                include_parent=True,
            )

        return False

    def grant_permission(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission
            .strip()
            .lower()
        )

        if not permission:
            raise ValueError(
                "Permission cannot be empty."
            )

        current = {
            str(item).strip().lower()
            for item in (
                self.permissions or []
            )
        }

        current.add(permission)

        self.permissions = sorted(current)

        denied = {
            str(item).strip().lower()
            for item in (
                self.denied_permissions or []
            )
        }

        denied.discard(permission)

        self.denied_permissions = sorted(
            denied
        )

    def revoke_permission(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission
            .strip()
            .lower()
        )

        current = {
            str(item).strip().lower()
            for item in (
                self.permissions or []
            )
        }

        current.discard(permission)

        self.permissions = sorted(current)

    def deny_permission(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission
            .strip()
            .lower()
        )

        if not permission:
            raise ValueError(
                "Permission cannot be empty."
            )

        denied = {
            str(item).strip().lower()
            for item in (
                self.denied_permissions or []
            )
        }

        denied.add(permission)

        self.denied_permissions = sorted(
            denied
        )

        allowed = {
            str(item).strip().lower()
            for item in (
                self.permissions or []
            )
        }

        allowed.discard(permission)

        self.permissions = sorted(allowed)

    def remove_denial(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission
            .strip()
            .lower()
        )

        denied = {
            str(item).strip().lower()
            for item in (
                self.denied_permissions or []
            )
        }

        denied.discard(permission)

        self.denied_permissions = sorted(
            denied
        )

    # ========================================================
    # RESOURCE SCOPE
    # ========================================================

    def allows_resource_scope(
        self,
        scope: str,
    ) -> bool:
        if self.is_super_admin:
            return True

        scopes = {
            str(item).strip().lower()
            for item in (
                self.resource_scopes or []
            )
        }

        normalized = (
            scope.strip().lower()
        )

        return (
            "*" in scopes
            or normalized in scopes
        )

    # ========================================================
    # USER ASSIGNMENT
    # ========================================================

    def can_assign_users(
        self,
        count: int = 1,
    ) -> bool:
        if count < 0:
            return False

        if not self.enabled:
            return False

        if self.max_users is None:
            return True

        return (
            self.user_count + count
            <= self.max_users
        )

    def record_assignment(self) -> None:
        if not self.can_assign_users(1):
            raise ValueError(
                "Role assignment limit reached."
            )

        self.user_count += 1

    def record_unassignment(self) -> None:
        self.user_count = max(
            0,
            self.user_count - 1,
        )

    # ========================================================
    # STATUS
    # ========================================================

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted role cannot be activated."
            )

        self.status = "active"
        self.is_active = True

    def deactivate(self) -> None:
        if self.is_system:
            raise ValueError(
                "System role cannot be deactivated "
                "through the standard workflow."
            )

        self.status = "inactive"
        self.is_active = False

    # ========================================================
    # SOFT DELETE
    # ========================================================

    def soft_delete(
        self,
        *,
        deleted_by: Optional[
            uuid.UUID
        ] = None,
    ) -> None:
        if self.is_system:
            raise ValueError(
                "System roles cannot be deleted."
            )

        if self.user_count > 0:
            raise ValueError(
                "Role still has assigned users."
            )

        self.is_deleted = True
        self.is_active = False
        self.status = "deleted"

        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

    def restore(self) -> None:
        if self.is_system:
            raise ValueError(
                "System role does not require restore."
            )

        self.is_deleted = False
        self.is_active = True
        self.status = "active"

        self.deleted_at = None
        self.deleted_by = None

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<Role("
            f"id={self.id}, "
            f"code={self.code!r}, "
            f"name={self.name!r}, "
            f"system={self.is_system}, "
            f"active={self.is_active}"
            ")>"
        )