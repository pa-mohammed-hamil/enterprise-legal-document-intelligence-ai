"""
Database engine configuration and lifecycle management.

Responsibilities
----------------
- Create and configure the SQLAlchemy engine.
- Configure PostgreSQL connection pooling.
- Support SQLite for lightweight local/test environments.
- Expose database health checks.
- Provide engine lifecycle/shutdown helpers.
- Keep session/transaction management in session.py.
- Keep ORM metadata/model definitions in base.py.

Architecture
------------

    config.py
        |
        v
    database.py
        |
        +---- SQLAlchemy Engine
        |
        +---- Connection Pool
        |
        +---- Health Check
        |
        +---- Shutdown
        |
        v
    session.py
        |
        v
    repositories/services
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from sqlalchemy import (
    Engine,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import (
    DBAPIError,
    OperationalError,
    SQLAlchemyError,
)

from backend.app.core.logging import get_logger


logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class DatabaseError(Exception):
    """Base database error."""


class DatabaseConfigurationError(
    DatabaseError
):
    """Raised when database configuration is invalid."""


class DatabaseConnectionError(
    DatabaseError
):
    """Raised when the database cannot be reached."""


class DatabaseHealthError(
    DatabaseError
):
    """Raised when the database health check fails."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database engine configuration.

    Values can be supplied directly or loaded from environment variables.
    """

    url: str

    pool_size: int = 10
    max_overflow: int = 20

    pool_timeout: int = 30
    pool_recycle: int = 1800

    pool_pre_ping: bool = True

    echo: bool = False
    echo_pool: bool = False

    connect_timeout: int = 10

    application_name: str = (
        "enterprise-legal-document-intelligence-ai"
    )

    ssl_mode: Optional[str] = None

    @classmethod
    def from_environment(
        cls,
    ) -> "DatabaseConfig":
        """
        Build configuration from environment variables.

        Supported variables
        --------------------
        DATABASE_URL
        DB_POOL_SIZE
        DB_MAX_OVERFLOW
        DB_POOL_TIMEOUT
        DB_POOL_RECYCLE
        DB_POOL_PRE_PING
        DB_ECHO
        DB_ECHO_POOL
        DB_CONNECT_TIMEOUT
        DB_APPLICATION_NAME
        DB_SSL_MODE
        """

        url = (
            os.getenv(
                "DATABASE_URL"
            )
            or os.getenv(
                "DB_URL"
            )
        )

        if not url:
            raise DatabaseConfigurationError(
                "DATABASE_URL is not configured."
            )

        return cls(
            url=url,
            pool_size=_env_int(
                "DB_POOL_SIZE",
                10,
            ),
            max_overflow=_env_int(
                "DB_MAX_OVERFLOW",
                20,
            ),
            pool_timeout=_env_int(
                "DB_POOL_TIMEOUT",
                30,
            ),
            pool_recycle=_env_int(
                "DB_POOL_RECYCLE",
                1800,
            ),
            pool_pre_ping=_env_bool(
                "DB_POOL_PRE_PING",
                True,
            ),
            echo=_env_bool(
                "DB_ECHO",
                False,
            ),
            echo_pool=_env_bool(
                "DB_ECHO_POOL",
                False,
            ),
            connect_timeout=_env_int(
                "DB_CONNECT_TIMEOUT",
                10,
            ),
            application_name=os.getenv(
                "DB_APPLICATION_NAME",
                "enterprise-legal-document-intelligence-ai",
            ),
            ssl_mode=os.getenv(
                "DB_SSL_MODE"
            ),
        )

    def validate(self) -> None:
        """Validate configuration before engine creation."""

        if not self.url.strip():
            raise DatabaseConfigurationError(
                "Database URL cannot be empty."
            )

        if self.pool_size < 1:
            raise DatabaseConfigurationError(
                "DB pool size must be >= 1."
            )

        if self.max_overflow < 0:
            raise DatabaseConfigurationError(
                "DB max overflow cannot be negative."
            )

        if self.pool_timeout < 1:
            raise DatabaseConfigurationError(
                "DB pool timeout must be >= 1."
            )

        if self.pool_recycle < -1:
            raise DatabaseConfigurationError(
                "DB pool recycle must be >= -1."
            )

        if self.connect_timeout < 1:
            raise DatabaseConfigurationError(
                "DB connect timeout must be >= 1."
            )


# ============================================================================
# Environment helpers
# ============================================================================


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


def _env_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise DatabaseConfigurationError(
            f"{name} must be an integer."
        ) from exc


# ============================================================================
# URL helpers
# ============================================================================


def _normalize_database_url(
    database_url: str,
) -> str:
    """
    Normalize common PostgreSQL URL schemes.

    Some deployment platforms expose:

        postgres://...

    while SQLAlchemy expects:

        postgresql://...

    or:

        postgresql+psycopg://...
    """

    url = database_url.strip()

    if url.startswith(
        "postgres://"
    ):
        url = (
            "postgresql://"
            + url[len("postgres://") :]
        )

    return url


def _is_sqlite_url(
    database_url: str,
) -> bool:
    return database_url.startswith(
        "sqlite://"
    )


def _is_postgresql_url(
    database_url: str,
) -> bool:
    return database_url.startswith(
        "postgresql://"
    ) or database_url.startswith(
        "postgresql+"
    )


def _safe_database_url(
    database_url: str,
) -> str:
    """
    Return a log-safe URL with credentials hidden.
    """

    try:
        parsed = make_url(
            database_url
        )

        return str(
            parsed.render_as_string(
                hide_password=True
            )
        )

    except Exception:
        return "<invalid-database-url>"


# ============================================================================
# Engine creation
# ============================================================================


def create_database_engine(
    config: Optional[
        DatabaseConfig
    ] = None,
) -> Engine:
    """
    Create the SQLAlchemy database engine.

    Production:
        PostgreSQL + psycopg

    Tests/local:
        SQLite is supported.

    Pool configuration is disabled for SQLite because SQLite handles
    connections differently from PostgreSQL.
    """

    config = (
        config
        or DatabaseConfig.from_environment()
    )

    config.validate()

    database_url = _normalize_database_url(
        config.url
    )

    connect_args: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    if _is_sqlite_url(
        database_url
    ):
        connect_args[
            "check_same_thread"
        ] = False

        engine = create_engine(
            database_url,
            echo=config.echo,
            echo_pool=config.echo_pool,
            connect_args=connect_args,
        )

        logger.info(
            "database.engine_created",
            extra={
                "database": "sqlite",
                "url": _safe_database_url(
                    database_url
                ),
            },
        )

        return engine

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    if _is_postgresql_url(
        database_url
    ):
        connect_args[
            "connect_timeout"
        ] = config.connect_timeout

        if config.application_name:
            connect_args[
                "application_name"
            ] = config.application_name

        if config.ssl_mode:
            connect_args[
                "sslmode"
            ] = config.ssl_mode

        engine = create_engine(
            database_url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=config.pool_pre_ping,
            echo=config.echo,
            echo_pool=config.echo_pool,
            connect_args=connect_args,
        )

        logger.info(
            "database.engine_created",
            extra={
                "database": "postgresql",
                "url": _safe_database_url(
                    database_url
                ),
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
            },
        )

        return engine

    # ------------------------------------------------------------------
    # Generic SQLAlchemy-compatible database
    # ------------------------------------------------------------------

    engine = create_engine(
        database_url,
        pool_pre_ping=config.pool_pre_ping,
        echo=config.echo,
        echo_pool=config.echo_pool,
    )

    logger.info(
        "database.engine_created",
        extra={
            "database": "other",
            "url": _safe_database_url(
                database_url
            ),
        },
    )

    return engine


# ============================================================================
# Engine events
# ============================================================================


def _register_engine_events(
    engine: Engine,
) -> None:
    """
    Register database connection events.

    PostgreSQL-specific settings can be placed here without affecting
    SQLite/local development.
    """

    if not _is_postgresql_url(
        str(engine.url)
    ):
        return

    @event.listens_for(
        engine,
        "connect",
    )
    def _on_connect(
        dbapi_connection: Any,
        connection_record: Any,
    ) -> None:
        """
        Configure PostgreSQL connection behavior.

        The application stores timestamps in UTC.
        """

        try:
            cursor = dbapi_connection.cursor()

            cursor.execute(
                "SET TIME ZONE 'UTC'"
            )

            cursor.close()

        except Exception:
            logger.debug(
                "database.connection_configuration_failed",
                exc_info=True,
            )


# ============================================================================
# Global engine
# ============================================================================


_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """
    Return the process-wide database engine.

    The engine is lazily initialized to avoid creating connections during
    module import.
    """

    global _engine

    if _engine is None:
        _engine = create_database_engine()

        _register_engine_events(
            _engine
        )

    return _engine


def set_engine(
    engine: Engine,
) -> None:
    """
    Replace the global engine.

    Primarily useful for tests and controlled application initialization.
    """

    global _engine

    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            logger.warning(
                "database.previous_engine_dispose_failed",
                exc_info=True,
            )

    _engine = engine

    _register_engine_events(
        engine
    )


# ============================================================================
# Connection health
# ============================================================================


def check_database_connection(
    engine: Optional[Engine] = None,
) -> bool:
    """
    Execute a minimal database connectivity check.

    Returns:
        True when the database responds successfully.

    Raises:
        DatabaseConnectionError when the connection fails.
    """

    engine = engine or get_engine()

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True

    except (
        OperationalError,
        DBAPIError,
        SQLAlchemyError,
    ) as exc:
        logger.error(
            "database.connection_check_failed",
            extra={
                "error_type": type(
                    exc
                ).__name__,
            },
        )

        raise DatabaseConnectionError(
            "Unable to connect to database."
        ) from exc


def database_health(
    engine: Optional[Engine] = None,
) -> dict[str, Any]:
    """
    Return a health-check payload suitable for /health.
    """

    engine = engine or get_engine()

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        pool = engine.pool

        result: dict[str, Any] = {
            "status": "healthy",
            "database": engine.url.get_backend_name(),
            "driver": engine.url.get_driver_name(),
        }

        # QueuePool exposes these methods. SQLite's pool may not.
        for attribute in (
            "size",
            "checkedout",
            "overflow",
            "checkedin",
        ):
            method = getattr(
                pool,
                attribute,
                None,
            )

            if callable(method):
                try:
                    result[
                        attribute
                    ] = method()
                except Exception:
                    pass

        return result

    except Exception as exc:
        logger.error(
            "database.health_check_failed",
            extra={
                "error_type": type(
                    exc
                ).__name__,
            },
        )

        return {
            "status": "unhealthy",
            "database": engine.url.get_backend_name(),
            "driver": engine.url.get_driver_name(),
            "error": type(
                exc
            ).__name__,
        }


# ============================================================================
# Pool information
# ============================================================================


def get_pool_status(
    engine: Optional[Engine] = None,
) -> dict[str, Any]:
    """Return connection-pool statistics."""

    engine = engine or get_engine()

    pool = engine.pool

    result: dict[str, Any] = {
        "pool_class": type(
            pool
        ).__name__,
    }

    for attribute in (
        "size",
        "checkedout",
        "overflow",
        "checkedin",
    ):
        method = getattr(
            pool,
            attribute,
            None,
        )

        if callable(method):
            try:
                result[
                    attribute
                ] = method()
            except Exception:
                result[
                    attribute
                ] = None

    return result


# ============================================================================
# Database lifecycle
# ============================================================================


def dispose_engine() -> None:
    """
    Dispose all pooled database connections.

    Call during application shutdown.
    """

    global _engine

    if _engine is None:
        return

    try:
        _engine.dispose()

        logger.info(
            "database.engine_disposed"
        )

    finally:
        _engine = None


def initialize_database(
    *,
    check_connection: bool = True,
) -> Engine:
    """
    Initialize the database subsystem.

    This does NOT create database tables.

    Schema creation belongs to Alembic migrations.
    """

    engine = get_engine()

    if check_connection:
        check_database_connection(
            engine
        )

    logger.info(
        "database.initialized"
    )

    return engine


# ============================================================================
# Testing helpers
# ============================================================================


def create_test_engine(
    database_url: str = (
        "sqlite+pysqlite:///:memory:"
    ),
) -> Engine:
    """
    Create an isolated test engine.

    This function intentionally does not modify the global engine.
    """

    config = DatabaseConfig(
        url=database_url,
        pool_pre_ping=False,
    )

    return create_database_engine(
        config
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "DatabaseConfig",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseHealthError",
    "check_database_connection",
    "create_database_engine",
    "create_test_engine",
    "database_health",
    "dispose_engine",
    "get_engine",
    "get_pool_status",
    "initialize_database",
    "set_engine",
]