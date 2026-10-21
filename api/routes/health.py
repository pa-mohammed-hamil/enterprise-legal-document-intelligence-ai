# backend/api/routes/health.py

from __future__ import annotations

import asyncio
import os
import platform
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ============================================================
# ENUMS
# ============================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DependencyType(str, Enum):
    DATABASE = "database"
    CACHE = "cache"
    VECTOR_DATABASE = "vector_database"
    OBJECT_STORAGE = "object_storage"
    MODEL_SERVER = "model_server"
    RAG = "rag"
    WORKER = "worker"
    EXTERNAL = "external"


# ============================================================
# RESPONSE MODELS
# ============================================================

class DependencyHealth(BaseModel):
    name: str

    type: DependencyType

    status: HealthStatus

    latency_ms: float = 0.0

    message: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class LivenessResponse(BaseModel):
    status: HealthStatus

    service: str

    version: str

    timestamp: datetime

    uptime_seconds: float


class ReadinessResponse(BaseModel):
    status: HealthStatus

    ready: bool

    timestamp: datetime

    dependencies: list[DependencyHealth]


class DetailedHealthResponse(BaseModel):
    status: HealthStatus

    service: str

    version: str

    environment: str

    timestamp: datetime

    uptime_seconds: float

    python_version: str

    dependencies: list[DependencyHealth]

    healthy_dependencies: int

    degraded_dependencies: int

    unhealthy_dependencies: int


# ============================================================
# APPLICATION STATE
# ============================================================

SERVICE_NAME = os.getenv(
    "SERVICE_NAME",
    "enterprise-legal-document-intelligence-ai",
)

SERVICE_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
)

_STARTED_AT = time.monotonic()


# ============================================================
# DEPENDENCY PROVIDERS
#
# Replace these with providers from backend.dependencies.py.
#
# Returning None is intentional:
# an unconfigured optional dependency is reported as UNKNOWN,
# rather than crashing the health endpoint.
# ============================================================

async def get_database() -> Any:
    return None


async def get_redis_client() -> Any:
    return None


async def get_vector_client() -> Any:
    return None


async def get_object_storage() -> Any:
    return None


async def get_model_server() -> Any:
    return None


async def get_rag_service() -> Any:
    return None


async def get_worker_client() -> Any:
    return None


# ============================================================
# UTILITIES
# ============================================================

def _utcnow() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _uptime() -> float:
    return max(
        0.0,
        time.monotonic() - _STARTED_AT,
    )


async def _maybe_await(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Awaitable,
    ):
        return await value

    return value


async def _timed_check(
    *,
    name: str,
    dependency_type: DependencyType,
    check: Callable[
        [],
        Awaitable[Any],
    ],
    timeout: float = 3.0,
) -> DependencyHealth:

    started = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            check(),
            timeout=timeout,
        )

        latency = (
            time.perf_counter()
            - started
        ) * 1000

        if isinstance(
            result,
            DependencyHealth,
        ):
            result.latency_ms = round(
                latency,
                2,
            )
            return result

        if result is False:
            return DependencyHealth(
                name=name,
                type=dependency_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(
                    latency,
                    2,
                ),
                message="Health check failed.",
            )

        return DependencyHealth(
            name=name,
            type=dependency_type,
            status=HealthStatus.HEALTHY,
            latency_ms=round(
                latency,
                2,
            ),
        )

    except asyncio.TimeoutError:
        latency = (
            time.perf_counter()
            - started
        ) * 1000

        return DependencyHealth(
            name=name,
            type=dependency_type,
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(
                latency,
                2,
            ),
            message=(
                f"Health check timed out "
                f"after {timeout:.1f}s."
            ),
        )

    except Exception as exc:
        latency = (
            time.perf_counter()
            - started
        ) * 1000

        # Do not expose credentials, connection URLs,
        # SQL queries, tokens, or stack traces here.
        return DependencyHealth(
            name=name,
            type=dependency_type,
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(
                latency,
                2,
            ),
            message=type(exc).__name__,
        )


# ============================================================
# DATABASE CHECK
# ============================================================

async def _check_database(
    database: Any,
) -> DependencyHealth:

    if database is None:
        return DependencyHealth(
            name="postgresql",
            type=DependencyType.DATABASE,
            status=HealthStatus.UNKNOWN,
            message="Database dependency not configured.",
        )

    async def check() -> Any:

        # AsyncSession / AsyncConnection
        if hasattr(
            database,
            "execute",
        ):
            try:
                from sqlalchemy import text

                await database.execute(
                    text("SELECT 1")
                )

            except ImportError:
                await database.execute(
                    "SELECT 1"
                )

            return True

        if hasattr(
            database,
            "ping",
        ):
            return await _maybe_await(
                database.ping()
            )

        if hasattr(
            database,
            "health_check",
        ):
            return await _maybe_await(
                database.health_check()
            )

        raise RuntimeError(
            "Unsupported database health interface."
        )

    return await _timed_check(
        name="postgresql",
        dependency_type=DependencyType.DATABASE,
        check=check,
    )


# ============================================================
# REDIS CHECK
# ============================================================

async def _check_redis(
    redis: Any,
) -> DependencyHealth:

    if redis is None:
        return DependencyHealth(
            name="redis",
            type=DependencyType.CACHE,
            status=HealthStatus.UNKNOWN,
            message="Redis dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            redis,
            "ping",
        ):
            result = await _maybe_await(
                redis.ping()
            )

            return bool(
                result
            )

        if hasattr(
            redis,
            "health_check",
        ):
            return await _maybe_await(
                redis.health_check()
            )

        raise RuntimeError(
            "Unsupported Redis health interface."
        )

    return await _timed_check(
        name="redis",
        dependency_type=DependencyType.CACHE,
        check=check,
    )


# ============================================================
# VECTOR DATABASE CHECK
# ============================================================

async def _check_vector_db(
    client: Any,
) -> DependencyHealth:

    if client is None:
        return DependencyHealth(
            name="vector-database",
            type=DependencyType.VECTOR_DATABASE,
            status=HealthStatus.UNKNOWN,
            message="Vector DB dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            client,
            "health_check",
        ):
            return await _maybe_await(
                client.health_check()
            )

        # Qdrant client
        if hasattr(
            client,
            "get_collections",
        ):
            await _maybe_await(
                client.get_collections()
            )

            return True

        if hasattr(
            client,
            "ping",
        ):
            return await _maybe_await(
                client.ping()
            )

        raise RuntimeError(
            "Unsupported vector database interface."
        )

    return await _timed_check(
        name="vector-database",
        dependency_type=DependencyType.VECTOR_DATABASE,
        check=check,
    )


# ============================================================
# OBJECT STORAGE CHECK
# ============================================================

async def _check_storage(
    storage: Any,
) -> DependencyHealth:

    if storage is None:
        return DependencyHealth(
            name="object-storage",
            type=DependencyType.OBJECT_STORAGE,
            status=HealthStatus.UNKNOWN,
            message="Object storage dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            storage,
            "health_check",
        ):
            return await _maybe_await(
                storage.health_check()
            )

        if hasattr(
            storage,
            "ping",
        ):
            return await _maybe_await(
                storage.ping()
            )

        # S3-style wrapper
        if hasattr(
            storage,
            "head_bucket",
        ):
            return await _maybe_await(
                storage.head_bucket()
            )

        raise RuntimeError(
            "Unsupported storage health interface."
        )

    return await _timed_check(
        name="object-storage",
        dependency_type=DependencyType.OBJECT_STORAGE,
        check=check,
    )


# ============================================================
# MODEL SERVER CHECK
# ============================================================

async def _check_model_server(
    model_server: Any,
) -> DependencyHealth:

    if model_server is None:
        return DependencyHealth(
            name="model-server",
            type=DependencyType.MODEL_SERVER,
            status=HealthStatus.UNKNOWN,
            message="Model server dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            model_server,
            "health_check",
        ):
            result = await _maybe_await(
                model_server.health_check()
            )

        elif hasattr(
            model_server,
            "health",
        ):
            result = await _maybe_await(
                model_server.health()
            )

        elif hasattr(
            model_server,
            "is_ready",
        ):
            result = await _maybe_await(
                model_server.is_ready()
            )

        else:
            raise RuntimeError(
                "Unsupported model server interface."
            )

        if isinstance(
            result,
            dict,
        ):
            state = str(
                result.get(
                    "status",
                    "healthy",
                )
            ).lower()

            if state in {
                "healthy",
                "ready",
                "ok",
            }:
                return True

            return False

        return bool(
            result
        )

    return await _timed_check(
        name="model-server",
        dependency_type=DependencyType.MODEL_SERVER,
        check=check,
        timeout=5.0,
    )


# ============================================================
# RAG CHECK
# ============================================================

async def _check_rag(
    rag_service: Any,
) -> DependencyHealth:

    if rag_service is None:
        return DependencyHealth(
            name="rag",
            type=DependencyType.RAG,
            status=HealthStatus.UNKNOWN,
            message="RAG dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            rag_service,
            "health_check",
        ):
            return await _maybe_await(
                rag_service.health_check()
            )

        if hasattr(
            rag_service,
            "is_ready",
        ):
            return await _maybe_await(
                rag_service.is_ready()
            )

        # Avoid performing an actual LLM/RAG query
        # in health probes because that adds cost,
        # latency, and external failure coupling.
        return True

    return await _timed_check(
        name="rag",
        dependency_type=DependencyType.RAG,
        check=check,
        timeout=5.0,
    )


# ============================================================
# WORKER CHECK
# ============================================================

async def _check_worker(
    worker: Any,
) -> DependencyHealth:

    if worker is None:
        return DependencyHealth(
            name="background-worker",
            type=DependencyType.WORKER,
            status=HealthStatus.UNKNOWN,
            message="Worker dependency not configured.",
        )

    async def check() -> Any:

        if hasattr(
            worker,
            "health_check",
        ):
            return await _maybe_await(
                worker.health_check()
            )

        if hasattr(
            worker,
            "ping",
        ):
            result = await _maybe_await(
                worker.ping()
            )

            return bool(
                result
            )

        if hasattr(
            worker,
            "inspect",
        ):
            inspector = worker.inspect()

            ping = await _maybe_await(
                inspector.ping()
            )

            return bool(
                ping
            )

        raise RuntimeError(
            "Unsupported worker health interface."
        )

    return await _timed_check(
        name="background-worker",
        dependency_type=DependencyType.WORKER,
        check=check,
        timeout=5.0,
    )


# ============================================================
# RUN DEPENDENCY CHECKS
# ============================================================

async def _dependency_health(
    *,
    database: Any,
    redis: Any,
    vector_client: Any,
    storage: Any,
    model_server: Any,
    rag_service: Any,
    worker: Any,
) -> list[DependencyHealth]:

    results = await asyncio.gather(
        _check_database(
            database
        ),
        _check_redis(
            redis
        ),
        _check_vector_db(
            vector_client
        ),
        _check_storage(
            storage
        ),
        _check_model_server(
            model_server
        ),
        _check_rag(
            rag_service
        ),
        _check_worker(
            worker
        ),
    )

    return list(
        results
    )


def _overall_status(
    dependencies: list[
        DependencyHealth
    ],
) -> HealthStatus:

    statuses = {
        dependency.status
        for dependency
        in dependencies
    }

    if HealthStatus.UNHEALTHY in statuses:
        return HealthStatus.UNHEALTHY

    if (
        HealthStatus.DEGRADED
        in statuses
        or HealthStatus.UNKNOWN
        in statuses
    ):
        return HealthStatus.DEGRADED

    return HealthStatus.HEALTHY


# ============================================================
# ROOT HEALTH
# ============================================================

@router.get(
    "",
    response_model=LivenessResponse,
    summary="Application health",
)
async def health() -> LivenessResponse:

    return LivenessResponse(
        status=HealthStatus.HEALTHY,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=_utcnow(),
        uptime_seconds=round(
            _uptime(),
            3,
        ),
    )


# ============================================================
# LIVENESS
#
# Must remain lightweight.
# Do NOT check PostgreSQL/Redis/Qdrant here.
# Kubernetes should restart the process only when the
# application itself is unhealthy.
# ============================================================

@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Kubernetes liveness probe",
)
async def liveness() -> LivenessResponse:

    return LivenessResponse(
        status=HealthStatus.HEALTHY,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=_utcnow(),
        uptime_seconds=round(
            _uptime(),
            3,
        ),
    )


# ============================================================
# READINESS
#
# Determines whether this instance should receive traffic.
# ============================================================

@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Kubernetes readiness probe",
)
async def readiness(
    response: Response,

    database: Any = Depends(
        get_database
    ),

    redis: Any = Depends(
        get_redis_client
    ),

    vector_client: Any = Depends(
        get_vector_client
    ),

    storage: Any = Depends(
        get_object_storage
    ),

    model_server: Any = Depends(
        get_model_server
    ),

    rag_service: Any = Depends(
        get_rag_service
    ),

    worker: Any = Depends(
        get_worker_client
    ),
) -> ReadinessResponse:

    dependencies = (
        await _dependency_health(
            database=database,
            redis=redis,
            vector_client=vector_client,
            storage=storage,
            model_server=model_server,
            rag_service=rag_service,
            worker=worker,
        )
    )

    overall = _overall_status(
        dependencies
    )

    # UNKNOWN usually means an optional integration is
    # not configured. It degrades observability but does
    # not necessarily mean the API cannot serve traffic.
    unhealthy = any(
        item.status
        == HealthStatus.UNHEALTHY
        for item
        in dependencies
    )

    ready = not unhealthy

    if not ready:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return ReadinessResponse(
        status=overall,
        ready=ready,
        timestamp=_utcnow(),
        dependencies=dependencies,
    )


# ============================================================
# DETAILED HEALTH
#
# In production this endpoint should normally be protected
# by admin/internal-network authorization.
# ============================================================

@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed system health",
)
async def detailed_health(
    response: Response,

    database: Any = Depends(
        get_database
    ),

    redis: Any = Depends(
        get_redis_client
    ),

    vector_client: Any = Depends(
        get_vector_client
    ),

    storage: Any = Depends(
        get_object_storage
    ),

    model_server: Any = Depends(
        get_model_server
    ),

    rag_service: Any = Depends(
        get_rag_service
    ),

    worker: Any = Depends(
        get_worker_client
    ),
) -> DetailedHealthResponse:

    dependencies = (
        await _dependency_health(
            database=database,
            redis=redis,
            vector_client=vector_client,
            storage=storage,
            model_server=model_server,
            rag_service=rag_service,
            worker=worker,
        )
    )

    overall = _overall_status(
        dependencies
    )

    if overall == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    healthy_count = sum(
        1
        for item in dependencies
        if item.status
        == HealthStatus.HEALTHY
    )

    degraded_count = sum(
        1
        for item in dependencies
        if item.status
        in {
            HealthStatus.DEGRADED,
            HealthStatus.UNKNOWN,
        }
    )

    unhealthy_count = sum(
        1
        for item in dependencies
        if item.status
        == HealthStatus.UNHEALTHY
    )

    return DetailedHealthResponse(
        status=overall,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        environment=ENVIRONMENT,
        timestamp=_utcnow(),
        uptime_seconds=round(
            _uptime(),
            3,
        ),
        python_version=(
            platform.python_version()
        ),
        dependencies=dependencies,
        healthy_dependencies=(
            healthy_count
        ),
        degraded_dependencies=(
            degraded_count
        ),
        unhealthy_dependencies=(
            unhealthy_count
        ),
    )


# ============================================================
# INDIVIDUAL DEPENDENCY ENDPOINTS
# ============================================================

@router.get(
    "/database",
    response_model=DependencyHealth,
    summary="Database health",
)
async def database_health(
    response: Response,

    database: Any = Depends(
        get_database
    ),
) -> DependencyHealth:

    result = await _check_database(
        database
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/redis",
    response_model=DependencyHealth,
    summary="Redis health",
)
async def redis_health(
    response: Response,

    redis: Any = Depends(
        get_redis_client
    ),
) -> DependencyHealth:

    result = await _check_redis(
        redis
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/vector-db",
    response_model=DependencyHealth,
    summary="Vector database health",
)
async def vector_database_health(
    response: Response,

    vector_client: Any = Depends(
        get_vector_client
    ),
) -> DependencyHealth:

    result = await _check_vector_db(
        vector_client
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/storage",
    response_model=DependencyHealth,
    summary="Object storage health",
)
async def storage_health(
    response: Response,

    storage: Any = Depends(
        get_object_storage
    ),
) -> DependencyHealth:

    result = await _check_storage(
        storage
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/models",
    response_model=DependencyHealth,
    summary="Model serving health",
)
async def model_health(
    response: Response,

    model_server: Any = Depends(
        get_model_server
    ),
) -> DependencyHealth:

    result = await _check_model_server(
        model_server
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/rag",
    response_model=DependencyHealth,
    summary="RAG subsystem health",
)
async def rag_health(
    response: Response,

    rag_service: Any = Depends(
        get_rag_service
    ),
) -> DependencyHealth:

    result = await _check_rag(
        rag_service
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result


@router.get(
    "/workers",
    response_model=DependencyHealth,
    summary="Background worker health",
)
async def worker_health(
    response: Response,

    worker: Any = Depends(
        get_worker_client
    ),
) -> DependencyHealth:

    result = await _check_worker(
        worker
    )

    if result.status == HealthStatus.UNHEALTHY:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return result