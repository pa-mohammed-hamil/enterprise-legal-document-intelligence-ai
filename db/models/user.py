# backend/db/models/user.py

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


class User(Base):
    """
    Enterprise tenant-aware application user.

    Responsibilities:
        - authentication identity
        - organization membership
        - RBAC role assignment
        - account status
        - MFA / SSO state
        - security controls
        - login tracking
        - session/security metadata
        - notification preferences
        - legal-review capabilities
        - AI/RAG preferences
        - soft deletion

    Security boundary:

        tenant_id
            -> organization_id
                -> role_id
                    -> permissions
                        -> resource ACL

    Password hashes, MFA secrets and recovery data must never be
    returned by API response schemas.
    """

    __tablename__ = "users"

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
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    role_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "roles.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # USER IDENTITY
    # ========================================================

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
    )

    normalized_email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
    )

    username: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    first_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    last_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    display_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
    )

    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # EMPLOYMENT / BUSINESS CONTEXT
    # ========================================================

    job_title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    department: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    business_unit: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    employee_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    manager_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    password_hash: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    password_reset_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    authentication_provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="local",
        index=True,
    )

    provider_user_id: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    # ========================================================
    # EMAIL / PHONE VERIFICATION
    # ========================================================

    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    phone_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    phone_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # MFA
    # ========================================================

    mfa_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    mfa_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # Store encrypted secret material only.
    mfa_secret_encrypted: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    mfa_enabled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    recovery_codes_hashes: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # SSO
    # ========================================================

    sso_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    sso_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    sso_subject: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    # ========================================================
    # ACCOUNT STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    is_locked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    locked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    lock_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    disabled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    disabled_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # FAILED LOGIN / BRUTE FORCE PROTECTION
    # ========================================================

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    last_failed_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    lockout_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # LOGIN TRACKING
    # ========================================================

    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    last_login_ip: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    last_login_user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    login_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # SESSION SECURITY
    # ========================================================

    session_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    security_stamp: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default=lambda: uuid.uuid4().hex,
        index=True,
    )

    token_not_before: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # EXTRA PERMISSIONS
    # ========================================================

    # Role permissions remain the normal authorization source.
    # These fields allow controlled per-user overrides.

    additional_permissions: Mapped[list[Any]] = mapped_column(
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
    # RESOURCE ACCESS
    # ========================================================

    access_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="standard",
        index=True,
    )

    allowed_document_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    allowed_contract_ids: Mapped[list[Any]] = mapped_column(
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
    # LEGAL REVIEW PROFILE
    # ========================================================

    legal_reviewer: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    reviewer_specialties: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    reviewer_jurisdictions: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    max_active_reviews: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    active_review_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # LOCALE / UI
    # ========================================================

    timezone_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="UTC",
    )

    locale: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
    )

    preferred_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
    )

    theme: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="system",
    )

    # ========================================================
    # NOTIFICATIONS
    # ========================================================

    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    email_notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    push_notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    notification_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # AI / RAG PREFERENCES
    # ========================================================

    ai_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    preferred_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    ai_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    rag_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # CONSENT / POLICY
    # ========================================================

    terms_accepted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    privacy_policy_accepted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    privacy_policy_accepted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # METADATA
    # ========================================================

    tags: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

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

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    organization = relationship(
        "Organization",
        foreign_keys=[organization_id],
        lazy="selectin",
    )

    role = relationship(
        "Role",
        foreign_keys=[role_id],
        lazy="selectin",
    )

    manager = relationship(
        "User",
        remote_side=[id],
        foreign_keys=[manager_user_id],
        back_populates="direct_reports",
    )

    direct_reports = relationship(
        "User",
        back_populates="manager",
        foreign_keys=[manager_user_id],
        lazy="selectin",
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "normalized_email",
            name="uq_users_tenant_email",
        ),
        UniqueConstraint(
            "tenant_id",
            "username",
            name="uq_users_tenant_username",
        ),
        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_users_tenant_external_id",
        ),
        Index(
            "ix_users_tenant_org",
            "tenant_id",
            "organization_id",
        ),
        Index(
            "ix_users_tenant_role",
            "tenant_id",
            "role_id",
        ),
        Index(
            "ix_users_tenant_status",
            "tenant_id",
            "status",
            "is_active",
        ),
        Index(
            "ix_users_tenant_department",
            "tenant_id",
            "department",
        ),
        Index(
            "ix_users_tenant_reviewer",
            "tenant_id",
            "legal_reviewer",
            "is_active",
        ),
        Index(
            "ix_users_auth_provider",
            "tenant_id",
            "authentication_provider",
            "provider_user_id",
        ),
        Index(
            "ix_users_sso_subject",
            "tenant_id",
            "sso_provider",
            "sso_subject",
        ),
        Index(
            "ix_users_lockout",
            "tenant_id",
            "is_locked",
            "lockout_until",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def full_name(self) -> str:
        if self.display_name:
            return self.display_name

        value = " ".join(
            item
            for item in (
                self.first_name,
                self.last_name,
            )
            if item
        ).strip()

        return value or self.email

    @property
    def enabled(self) -> bool:
        return bool(
            self.is_active
            and not self.is_deleted
            and not self.is_locked
            and (self.status or "").lower()
            in {
                "active",
                "verified",
            }
        )

    @property
    def locked_out(self) -> bool:
        if self.is_locked:
            return True

        if self.lockout_until is None:
            return False

        return self.lockout_until > utcnow()

    @property
    def can_accept_review(self) -> bool:
        if not self.enabled:
            return False

        if not self.legal_reviewer:
            return False

        if self.max_active_reviews is None:
            return True

        return (
            self.active_review_count
            < self.max_active_reviews
        )

    # ========================================================
    # PERMISSION ENGINE
    # ========================================================

    def has_permission(
        self,
        permission: str,
    ) -> bool:
        """
        Permission precedence:

            user explicit deny
                >
            user explicit allow
                >
            role permission
                >
            deny

        Resource ACL checks must still be performed separately.
        """

        permission = (
            permission.strip().lower()
        )

        if not permission:
            return False

        if not self.enabled:
            return False

        denied = {
            str(value).strip().lower()
            for value in (
                self.denied_permissions or []
            )
        }

        if (
            "*" in denied
            or permission in denied
        ):
            return False

        allowed = {
            str(value).strip().lower()
            for value in (
                self.additional_permissions
                or []
            )
        }

        if (
            "*" in allowed
            or permission in allowed
        ):
            return True

        if self.role is not None:
            return self.role.has_permission(
                permission
            )

        return False

    def grant_permission(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission.strip().lower()
        )

        if not permission:
            raise ValueError(
                "Permission cannot be empty."
            )

        allowed = {
            str(value).strip().lower()
            for value in (
                self.additional_permissions
                or []
            )
        }

        allowed.add(permission)

        self.additional_permissions = sorted(
            allowed
        )

        denied = {
            str(value).strip().lower()
            for value in (
                self.denied_permissions or []
            )
        }

        denied.discard(permission)

        self.denied_permissions = sorted(
            denied
        )

    def deny_permission(
        self,
        permission: str,
    ) -> None:
        permission = (
            permission.strip().lower()
        )

        if not permission:
            raise ValueError(
                "Permission cannot be empty."
            )

        denied = {
            str(value).strip().lower()
            for value in (
                self.denied_permissions or []
            )
        }

        denied.add(permission)

        self.denied_permissions = sorted(
            denied
        )

        allowed = {
            str(value).strip().lower()
            for value in (
                self.additional_permissions
                or []
            )
        }

        allowed.discard(permission)

        self.additional_permissions = sorted(
            allowed
        )

    # ========================================================
    # AUTHENTICATION EVENTS
    # ========================================================

    def record_successful_login(
        self,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.last_login_at = utcnow()
        self.last_login_ip = ip_address
        self.last_login_user_agent = (
            user_agent
        )

        self.login_count += 1

        self.failed_login_attempts = 0
        self.last_failed_login_at = None
        self.lockout_until = None

    def record_failed_login(
        self,
        *,
        max_attempts: int = 5,
    ) -> bool:
        """
        Returns True when the configured failed-login threshold
        has been reached.

        The service layer should decide lockout duration.
        """

        self.failed_login_attempts += 1
        self.last_failed_login_at = utcnow()

        return (
            self.failed_login_attempts
            >= max_attempts
        )

    # ========================================================
    # LOCK / UNLOCK
    # ========================================================

    def lock(
        self,
        *,
        reason: Optional[str] = None,
        until: Optional[datetime] = None,
    ) -> None:
        self.is_locked = True
        self.locked_at = utcnow()
        self.lock_reason = reason
        self.lockout_until = until

        self.invalidate_sessions()

    def unlock(self) -> None:
        self.is_locked = False
        self.locked_at = None
        self.lock_reason = None
        self.lockout_until = None
        self.failed_login_attempts = 0

    # ========================================================
    # SESSION INVALIDATION
    # ========================================================

    def invalidate_sessions(self) -> None:
        """
        Increment session_version so previously issued sessions/JWTs
        can be rejected when the auth layer validates the version.
        """

        self.session_version += 1
        self.security_stamp = (
            uuid.uuid4().hex
        )
        self.token_not_before = utcnow()

    # ========================================================
    # PASSWORD
    # ========================================================

    def set_password_hash(
        self,
        password_hash: str,
    ) -> None:
        if not password_hash:
            raise ValueError(
                "Password hash cannot be empty."
            )

        self.password_hash = password_hash
        self.password_changed_at = utcnow()
        self.password_reset_required = False

        self.invalidate_sessions()

    # ========================================================
    # EMAIL VERIFICATION
    # ========================================================

    def verify_email(self) -> None:
        self.email_verified = True
        self.email_verified_at = utcnow()

        if self.status == "pending":
            self.status = "active"

    # ========================================================
    # MFA
    # ========================================================

    def enable_mfa(
        self,
        *,
        method: str,
        encrypted_secret: Optional[str] = None,
    ) -> None:
        self.mfa_enabled = True
        self.mfa_method = method
        self.mfa_secret_encrypted = (
            encrypted_secret
        )
        self.mfa_enabled_at = utcnow()

        self.invalidate_sessions()

    def disable_mfa(self) -> None:
        self.mfa_enabled = False
        self.mfa_method = None
        self.mfa_secret_encrypted = None
        self.recovery_codes_hashes = []

        self.invalidate_sessions()

    # ========================================================
    # REVIEW CAPACITY
    # ========================================================

    def assign_review(self) -> None:
        if not self.can_accept_review:
            raise ValueError(
                "User cannot accept another review."
            )

        self.active_review_count += 1

    def complete_review(self) -> None:
        self.active_review_count = max(
            0,
            self.active_review_count - 1,
        )

    # ========================================================
    # ACCOUNT STATUS
    # ========================================================

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted user cannot be activated."
            )

        self.status = "active"
        self.is_active = True
        self.disabled_at = None
        self.disabled_by = None

    def disable(
        self,
        *,
        disabled_by: Optional[
            uuid.UUID
        ] = None,
    ) -> None:
        self.status = "disabled"
        self.is_active = False
        self.disabled_at = utcnow()
        self.disabled_by = disabled_by

        self.invalidate_sessions()

    # ========================================================
    # ROLE
    # ========================================================

    def assign_role(
        self,
        role_id: uuid.UUID,
    ) -> None:
        self.role_id = role_id

        # Existing sessions should not retain old permissions.
        self.invalidate_sessions()

    def remove_role(self) -> None:
        self.role_id = None
        self.invalidate_sessions()

    # ========================================================
    # ORGANIZATION
    # ========================================================

    def move_to_organization(
        self,
        organization_id: Optional[
            uuid.UUID
        ],
    ) -> None:
        self.organization_id = (
            organization_id
        )

        # Organization change can change resource visibility.
        self.invalidate_sessions()

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
        self.is_deleted = True
        self.is_active = False
        self.status = "deleted"

        self.deleted_at = utcnow()
        self.deleted_by = deleted_by

        self.invalidate_sessions()

    def restore(self) -> None:
        self.is_deleted = False
        self.is_active = True
        self.status = "active"

        self.deleted_at = None
        self.deleted_by = None

        self.invalidate_sessions()

    # ========================================================
    # API-SAFE REPRESENTATION
    # ========================================================

    def public_identity(self) -> dict[str, Any]:
        """
        Deliberately excludes:
            password_hash
            MFA secret
            recovery codes
            security stamp
            authentication internals
        """

        return {
            "id": str(self.id),
            "tenant_id": str(
                self.tenant_id
            ),
            "organization_id": (
                str(self.organization_id)
                if self.organization_id
                else None
            ),
            "role_id": (
                str(self.role_id)
                if self.role_id
                else None
            ),
            "email": self.email,
            "username": self.username,
            "display_name": self.full_name,
            "job_title": self.job_title,
            "department": self.department,
            "status": self.status,
            "email_verified":
                self.email_verified,
            "mfa_enabled":
                self.mfa_enabled,
            "legal_reviewer":
                self.legal_reviewer,
            "created_at":
                self.created_at.isoformat()
                if self.created_at
                else None,
        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<User("
            f"id={self.id}, "
            f"email={self.email!r}, "
            f"status={self.status!r}, "
            f"organization_id="
            f"{self.organization_id}, "
            f"role_id={self.role_id}"
            ")>"
        )