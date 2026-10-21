"""
SQLAlchemy declarative base and model registration.

Responsibilities
----------------
- Provide the application's single SQLAlchemy DeclarativeBase.
- Expose shared model metadata for Alembic/migrations.
- Provide reusable model mixins.
- Provide UUID/timestamp/tenant-aware base helpers.
- Register all ORM models through explicit imports.
- Avoid creating database engines or sessions in this module.

Architecture
------------

    backend.app.db.base
            |
            +---- Base
            |
            +---- metadata
            |
            +---- shared mixins
            |
            +---- ORM model registration
                         |
                         +---- Alembic
                         +---- repositories
                         +---- services
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    MetaData,
    String,
    event,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
)


# ============================================================================
# Naming convention
# ============================================================================

# Stable constraint names are important for:
# - Alembic autogeneration
# - PostgreSQL migrations
# - deterministic CI/CD deployments
# - clean rollback operations

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": (
        "fk_%(table_name)s_%(column_0_name)s_"
        "%(referred_table_name)s"
    ),
    "pk": "pk_%(table_name)s",
}


# ============================================================================
# Metadata
# ============================================================================


metadata = MetaData(
    naming_convention=NAMING_CONVENTION,
)


# ============================================================================
# Declarative base
# ============================================================================


class Base(DeclarativeBase):
    """
    Root declarative base for all application ORM models.

    All models should inherit from this class directly or through one of
    the project-specific base mixins.
    """

    metadata = metadata

    # ------------------------------------------------------------------
    # Common table naming
    # ------------------------------------------------------------------

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Generate a predictable snake_case table name.

        Explicit __tablename__ declarations on individual models take
        precedence because SQLAlchemy uses those declarations directly.
        """

        name = cls.__name__

        # Convert:
        # DocumentVersion -> document_version
        # AuditLog -> audit_log
        result: list[str] = []

        for index, character in enumerate(name):
            if (
                character.isupper()
                and index > 0
            ):
                result.append("_")

            result.append(
                character.lower()
            )

        return "".join(result)

    def __repr__(self) -> str:
        """Safe ORM representation."""

        identity = getattr(
            self,
            "id",
            None,
        )

        return (
            f"<{self.__class__.__name__}"
            f"(id={identity!r})>"
        )


# ============================================================================
# UUID primary-key mixin
# ============================================================================


class UUIDPrimaryKeyMixin:
    """
    Reusable UUID primary-key definition.

    PostgreSQL uses native UUID while SQLAlchemy can still represent the
    value consistently throughout the application.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


# ============================================================================
# Timestamp mixin
# ============================================================================


class TimestampMixin:
    """
    Standard UTC creation/update timestamps.

    All timestamps are timezone-aware and should be treated as UTC.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(
            timezone.utc
        ),
        onupdate=lambda: datetime.now(
            timezone.utc
        ),
    )


# ============================================================================
# Soft-delete mixin
# ============================================================================


class SoftDeleteMixin:
    """
    Optional soft-delete support.

    Models using this mixin should ensure repositories consistently exclude
    deleted records from normal queries.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    deleted_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ============================================================================
# Tenant mixin
# ============================================================================


class TenantMixin:
    """
    Tenant ownership for multi-tenant resources.

    Tenant isolation is enforced at the security/repository layer;
    this mixin only provides the persisted ownership field.
    """

    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )


# ============================================================================
# Organization mixin
# ============================================================================


class OrganizationMixin:
    """
    Organization ownership/reference.

    This can coexist with TenantMixin when the application distinguishes
    deployment/security tenancy from business organization ownership.
    """

    organization_id: Mapped[
        Optional[uuid.UUID]
    ] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )


# ============================================================================
# Audit ownership mixin
# ============================================================================


class AuditOwnershipMixin:
    """
    Optional actor ownership fields for auditable resources.
    """

    created_by_id: Mapped[
        Optional[uuid.UUID]
    ] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    updated_by_id: Mapped[
        Optional[uuid.UUID]
    ] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )


# ============================================================================
# Standard model base combinations
# ============================================================================


class ModelBase(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Standard application model base.

    Suitable for non-tenant resources.
    """

    __abstract__ = True


class TenantModelBase(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    TenantMixin,
    OrganizationMixin,
    Base,
):
    """
    Standard tenant-scoped model base.

    Suitable for documents, contracts, clauses, risks, reviews, etc.
    """

    __abstract__ = True


class AuditedTenantModelBase(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    SoftDeleteMixin,
    TenantMixin,
    OrganizationMixin,
    AuditOwnershipMixin,
    Base,
):
    """
    Tenant-scoped model base with soft-delete and actor tracking.
    """

    __abstract__ = True


# ============================================================================
# UTC timestamp safety
# ============================================================================


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.

    Centralizing this helper makes timestamp generation easy to test.
    """

    return datetime.now(
        timezone.utc
    )


# ============================================================================
# SQLAlchemy lifecycle helpers
# ============================================================================


@event.listens_for(
    Base,
    "before_update",
    propagate=True,
)
def _touch_updated_at(
    mapper: Any,
    connection: Any,
    target: Any,
) -> None:
    """
    Keep updated_at current for models that expose the field.

    SQLAlchemy's onupdate normally handles this, but this listener also
    protects models where updates occur through less conventional ORM paths.
    """

    if hasattr(
        target,
        "updated_at",
    ):
        target.updated_at = utc_now()


# ============================================================================
# Model registration
# ============================================================================


def import_models() -> None:
    """
    Import all ORM models so SQLAlchemy metadata contains every table.

    This function is intentionally explicit.

    It is primarily used by:
        - Alembic env.py
        - application startup validation
        - metadata inspection
        - tests

    The imports are local to avoid circular-import problems during package
    initialization.
    """

    # Core identity / tenancy
    from backend.app.models.user import User  # noqa: F401
    from backend.app.models.organization import Organization  # noqa: F401
    from backend.app.models.role import Role  # noqa: F401

    # Documents
    from backend.app.models.document import Document  # noqa: F401
    from backend.app.models.document_version import (  # noqa: F401
        DocumentVersion,
    )
    from backend.app.models.document_chunk import (  # noqa: F401
        DocumentChunk,
    )

    # Legal domain
    from backend.app.models.contract import Contract  # noqa: F401
    from backend.app.models.clause import Clause  # noqa: F401
    from backend.app.models.obligation import Obligation  # noqa: F401
    from backend.app.models.risk import Risk  # noqa: F401

    # Workflow
    from backend.app.models.renewal import Renewal  # noqa: F401
    from backend.app.models.review import Review  # noqa: F401
    from backend.app.models.notification import Notification  # noqa: F401

    # Security / observability
    from backend.app.models.audit_log import AuditLog  # noqa: F401
    from backend.app.models.model_run import ModelRun  # noqa: F401


# ============================================================================
# Metadata helpers
# ============================================================================


def get_metadata() -> MetaData:
    """Return application SQLAlchemy metadata."""

    return Base.metadata


def get_registered_table_names() -> list[str]:
    """
    Return registered table names in deterministic order.
    """

    return sorted(
        Base.metadata.tables.keys()
    )


def is_model_registered(
    table_name: str,
) -> bool:
    """Check whether a table is registered in metadata."""

    return table_name in Base.metadata.tables


def get_model_table(
    table_name: str,
) -> Any:
    """
    Return a SQLAlchemy Table object.

    Raises KeyError when the table is not registered.
    """

    return Base.metadata.tables[
        table_name
    ]


# ============================================================================
# Metadata validation
# ============================================================================


EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "user",
        "organization",
        "role",
        "document",
        "document_version",
        "document_chunk",
        "contract",
        "clause",
        "obligation",
        "risk",
        "renewal",
        "review",
        "notification",
        "audit_log",
        "model_run",
    }
)


def validate_model_registration(
    *,
    require_all: bool = False,
) -> list[str]:
    """
    Validate that expected ORM models are registered.

    Returns missing table names.

    `require_all=False` is useful during partial unit-test imports.
    """

    registered = set(
        Base.metadata.tables.keys()
    )

    missing = sorted(
        EXPECTED_TABLES - registered
    )

    if (
        require_all
        and missing
    ):
        raise RuntimeError(
            "Required ORM models are not registered: "
            + ", ".join(missing)
        )

    return missing


# ============================================================================
# Alembic helper
# ============================================================================


def prepare_metadata_for_migrations() -> MetaData:
    """
    Ensure all models are imported before Alembic accesses metadata.

    Usage in Alembic env.py:

        from backend.app.db.base import (
            prepare_metadata_for_migrations,
        )

        target_metadata = (
            prepare_metadata_for_migrations()
        )
    """

    import_models()

    return Base.metadata


# ============================================================================
# Public exports
# ============================================================================


__all__ = [
    "AuditedTenantModelBase",
    "Base",
    "EXPECTED_TABLES",
    "ModelBase",
    "NAMING_CONVENTION",
    "OrganizationMixin",
    "SoftDeleteMixin",
    "TenantMixin",
    "TenantModelBase",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "get_metadata",
    "get_model_table",
    "get_registered_table_names",
    "import_models",
    "is_model_registered",
    "prepare_metadata_for_migrations",
    "utc_now",
    "validate_model_registration",
]