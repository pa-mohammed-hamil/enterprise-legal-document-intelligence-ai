# backend/api/routes/models.py

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/models",
    tags=["Models"],
)


# ============================================================
# ENUMS
# ============================================================

class ModelStatus(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class ModelType(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    DOCUMENT_CLASSIFIER = "document_classifier"
    CLAUSE_CLASSIFIER = "clause_classifier"
    RISK_CLASSIFIER = "risk_classifier"
    LEGAL_NER = "legal_ner"
    OCR = "ocr"
    OTHER = "other"


class ModelProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies.py
# ============================================================

async def get_model_registry() -> Any:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model registry dependency is not configured.",
    )


async def get_model_router() -> Any:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model router dependency is not configured.",
    )


async def get_model_server() -> Any:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model server dependency is not configured.",
    )


async def get_model_metrics() -> Any:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model metrics dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ModelRegisterRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=255,
    )

    version: str = Field(
        min_length=1,
        max_length=100,
    )

    model_type: ModelType

    provider: ModelProvider = (
        ModelProvider.HUGGINGFACE
    )

    model_id: str = Field(
        min_length=1,
        max_length=1000,
    )

    revision: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    device: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    enabled: bool = True

    is_default: bool = False

    configuration: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ModelUpdateRequest(BaseModel):
    enabled: Optional[bool] = None

    is_default: Optional[bool] = None

    device: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    configuration: Optional[
        dict[str, Any]
    ] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


class ModelLoadRequest(BaseModel):
    force_reload: bool = False

    warmup: bool = True


class ModelInferenceRequest(BaseModel):
    inputs: Any

    task: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )

    timeout_seconds: float = Field(
        default=60.0,
        ge=0.1,
        le=600.0,
    )


class ModelRouteRequest(BaseModel):
    task: str = Field(
        min_length=1,
        max_length=100,
    )

    preferred_model: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    require_local: bool = False

    require_gpu: bool = False

    max_latency_ms: Optional[float] = Field(
        default=None,
        gt=0,
    )

    max_cost: Optional[float] = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ModelResponse(BaseModel):
    id: str

    name: str
    version: str

    model_type: str
    provider: str

    model_id: str

    revision: Optional[str] = None
    device: Optional[str] = None

    status: str

    enabled: bool = True
    is_default: bool = False

    loaded: bool = False

    configuration: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    loaded_at: Optional[datetime] = None


class ModelListResponse(BaseModel):
    items: list[ModelResponse]

    total: int


class ModelHealthResponse(BaseModel):
    model_id: str

    status: str

    loaded: bool = False

    latency_ms: Optional[float] = None

    device: Optional[str] = None

    memory_usage_mb: Optional[float] = None

    error: Optional[str] = None

    checked_at: datetime


class ModelMetricsResponse(BaseModel):
    model_id: str

    requests: int = 0
    successes: int = 0
    failures: int = 0

    average_latency_ms: float = 0.0

    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None

    input_tokens: int = 0
    output_tokens: int = 0

    estimated_cost: float = 0.0

    model_loads: int = 0
    fallbacks: int = 0


class ModelInferenceResponse(BaseModel):
    model_id: str

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    output: Any

    latency_ms: Optional[float] = None

    fallback_used: bool = False

    fallback_model: Optional[str] = None

    request_id: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ModelRouteResponse(BaseModel):
    task: str

    selected_model: str

    model_type: Optional[str] = None

    reason: Optional[str] = None

    fallback_models: list[str] = Field(
        default_factory=list,
    )


class ModelSummaryResponse(BaseModel):
    total_models: int = 0

    ready: int = 0
    loading: int = 0
    degraded: int = 0
    failed: int = 0
    disabled: int = 0

    loaded: int = 0

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_provider: dict[str, int] = Field(
        default_factory=dict,
    )

    generated_at: datetime


class MessageResponse(BaseModel):
    message: str


# ============================================================
# HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def _enum_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


def _optional_str(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    return str(value)


def _user_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return str(value)


def _tenant_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "tenant_id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    return str(value)


def _roles(
    user: Any,
) -> set[str]:

    values = (
        _attr(
            user,
            "roles",
            [],
        )
        or []
    )

    if isinstance(
        values,
        str,
    ):
        values = [
            values
        ]

    role = _attr(
        user,
        "role",
    )

    if role:
        values = [
            *values,
            role,
        ]

    return {
        str(
            _enum_value(value)
        ).lower()
        for value in values
    }


def _require_model_admin(
    user: Any,
) -> None:

    allowed = {
        "admin",
        "super_admin",
        "ml_admin",
        "model_admin",
        "ai_admin",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Model administration "
                "permission is required."
            ),
        )


def _request_id(
    request: Request,
) -> Optional[str]:

    value = (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )

    return _optional_str(
        value
    )


async def _call(
    target: Any,
    operation: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        target,
        operation,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Model operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(
        **kwargs
    )


def _serialize_model(
    model: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                model,
                "id",
                _attr(
                    model,
                    "model_id",
                    "",
                ),
            )
        ),

        "name": str(
            _attr(
                model,
                "name",
                "",
            )
        ),

        "version": str(
            _attr(
                model,
                "version",
                "",
            )
        ),

        "model_type": str(
            _enum_value(
                _attr(
                    model,
                    "model_type",
                    "other",
                )
            )
        ),

        "provider": str(
            _enum_value(
                _attr(
                    model,
                    "provider",
                    "custom",
                )
            )
        ),

        "model_id": str(
            _attr(
                model,
                "model_id",
                _attr(
                    model,
                    "id",
                    "",
                ),
            )
        ),

        "revision": _attr(
            model,
            "revision",
        ),

        "device": _attr(
            model,
            "device",
        ),

        "status": str(
            _enum_value(
                _attr(
                    model,
                    "status",
                    "registered",
                )
            )
        ),

        "enabled": bool(
            _attr(
                model,
                "enabled",
                True,
            )
        ),

        "is_default": bool(
            _attr(
                model,
                "is_default",
                False,
            )
        ),

        "loaded": bool(
            _attr(
                model,
                "loaded",
                False,
            )
        ),

        "configuration": (
            _attr(
                model,
                "configuration",
            )
            or _attr(
                model,
                "config",
            )
            or {}
        ),

        "metadata": (
            _attr(
                model,
                "model_metadata",
            )
            or _attr(
                model,
                "metadata_json",
            )
            or {}
        ),

        "created_at": _attr(
            model,
            "created_at",
        ),

        "updated_at": _attr(
            model,
            "updated_at",
        ),

        "loaded_at": _attr(
            model,
            "loaded_at",
        ),
    }


# ============================================================
# LIST MODELS
# ============================================================

@router.get(
    "",
    response_model=ModelListResponse,
    summary="List registered AI models",
)
async def list_models(
    model_type: Optional[
        ModelType
    ] = None,

    provider: Optional[
        ModelProvider
    ] = None,

    model_status: Optional[
        ModelStatus
    ] = Query(
        default=None,
        alias="status",
    ),

    enabled: Optional[
        bool
    ] = None,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    models = await _call(
        registry,
        "list_models",

        model_type=(
            model_type.value
            if model_type
            else None
        ),

        provider=(
            provider.value
            if provider
            else None
        ),

        status=(
            model_status.value
            if model_status
            else None
        ),

        enabled=enabled,
    )

    items = (
        _attr(
            models,
            "items",
            models,
        )
        or []
    )

    return {
        "items": [
            _serialize_model(
                model
            )
            for model
            in items
        ],

        "total": int(
            _attr(
                models,
                "total",
                len(items),
            )
        ),
    }


# ============================================================
# SUMMARY
# Keep static routes above /{model_id}
# ============================================================

@router.get(
    "/summary",
    response_model=ModelSummaryResponse,
    summary="Get model registry summary",
)
async def model_summary(
    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> Any:

    result = await _call(
        registry,
        "summary",
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            **result,
            "generated_at":
                result.get(
                    "generated_at"
                )
                or datetime.now(
                    timezone.utc
                ),
        }

    return {
        "total_models":
            _attr(
                result,
                "total_models",
                0,
            ),

        "ready":
            _attr(
                result,
                "ready",
                0,
            ),

        "loading":
            _attr(
                result,
                "loading",
                0,
            ),

        "degraded":
            _attr(
                result,
                "degraded",
                0,
            ),

        "failed":
            _attr(
                result,
                "failed",
                0,
            ),

        "disabled":
            _attr(
                result,
                "disabled",
                0,
            ),

        "loaded":
            _attr(
                result,
                "loaded",
                0,
            ),

        "by_type":
            _attr(
                result,
                "by_type",
                {},
            ),

        "by_provider":
            _attr(
                result,
                "by_provider",
                {},
            ),

        "generated_at":
            datetime.now(
                timezone.utc
            ),
    }


# ============================================================
# ROUTER TEST / MODEL SELECTION
# ============================================================

@router.post(
    "/route",
    response_model=ModelRouteResponse,
    summary="Route task to appropriate model",
)
async def route_model(
    payload: ModelRouteRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    model_router: Any = Depends(
        get_model_router
    ),
) -> Any:

    result = await _call(
        model_router,
        "route",

        task=payload.task,

        preferred_model=(
            payload.preferred_model
        ),

        require_local=(
            payload.require_local
        ),

        require_gpu=(
            payload.require_gpu
        ),

        max_latency_ms=(
            payload.max_latency_ms
        ),

        max_cost=(
            payload.max_cost
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        metadata=payload.metadata,
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {
        "task":
            payload.task,

        "selected_model":
            str(
                _attr(
                    result,
                    "selected_model",
                    _attr(
                        result,
                        "model_id",
                        "",
                    ),
                )
            ),

        "model_type":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "model_type",
                    )
                )
            ),

        "reason":
            _attr(
                result,
                "reason",
            ),

        "fallback_models":
            list(
                _attr(
                    result,
                    "fallback_models",
                    [],
                )
                or []
            ),
    }


# ============================================================
# REGISTER MODEL
# ============================================================

@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register AI model",
)
async def register_model(
    payload: ModelRegisterRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    _require_model_admin(
        current_user
    )

    result = await _call(
        registry,
        "register",

        name=payload.name,

        version=payload.version,

        model_type=(
            payload.model_type.value
        ),

        provider=(
            payload.provider.value
        ),

        model_id=payload.model_id,

        revision=payload.revision,

        device=payload.device,

        enabled=payload.enabled,

        is_default=(
            payload.is_default
        ),

        configuration=(
            payload.configuration
        ),

        metadata=(
            payload.metadata
        ),

        registered_by=_user_id(
            current_user
        ),
    )

    return _serialize_model(
        result
    )


# ============================================================
# GET MODEL
# ============================================================

@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get registered model",
)
async def get_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    result = await _call(
        registry,
        "get",

        model_id=model_id,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model was not found.",
        )

    return _serialize_model(
        result
    )


# ============================================================
# UPDATE MODEL
# ============================================================

@router.patch(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Update model configuration",
)
async def update_model(
    model_id: str,
    payload: ModelUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    _require_model_admin(
        current_user
    )

    values = payload.model_dump(
        exclude_unset=True
    )

    result = await _call(
        registry,
        "update",

        model_id=model_id,

        updated_by=_user_id(
            current_user
        ),

        **values,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model was not found.",
        )

    return _serialize_model(
        result
    )


# ============================================================
# LOAD MODEL
# ============================================================

@router.post(
    "/{model_id}/load",
    response_model=MessageResponse,
    summary="Load model into serving memory",
)
async def load_model(
    model_id: str,
    payload: ModelLoadRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    server: Any = Depends(
        get_model_server
    ),
) -> dict[str, str]:

    _require_model_admin(
        current_user
    )

    await _call(
        server,
        "load_model",

        model_id=model_id,

        force_reload=(
            payload.force_reload
        ),

        warmup=payload.warmup,
    )

    return {
        "message":
            f"Model '{model_id}' loaded."
    }


# ============================================================
# UNLOAD MODEL
# ============================================================

@router.post(
    "/{model_id}/unload",
    response_model=MessageResponse,
    summary="Unload model",
)
async def unload_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    server: Any = Depends(
        get_model_server
    ),
) -> dict[str, str]:

    _require_model_admin(
        current_user
    )

    await _call(
        server,
        "unload_model",

        model_id=model_id,
    )

    return {
        "message":
            f"Model '{model_id}' unloaded."
    }


# ============================================================
# MODEL HEALTH
# ============================================================

@router.get(
    "/{model_id}/health",
    response_model=ModelHealthResponse,
    summary="Get model health",
)
async def model_health(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    server: Any = Depends(
        get_model_server
    ),
) -> Any:

    result = await _call(
        server,
        "model_health",

        model_id=model_id,
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            **result,
            "model_id":
                result.get(
                    "model_id",
                    model_id,
                ),
            "checked_at":
                result.get(
                    "checked_at"
                )
                or datetime.now(
                    timezone.utc
                ),
        }

    return {
        "model_id":
            model_id,

        "status":
            str(
                _enum_value(
                    _attr(
                        result,
                        "status",
                        "unknown",
                    )
                )
            ),

        "loaded":
            bool(
                _attr(
                    result,
                    "loaded",
                    False,
                )
            ),

        "latency_ms":
            _attr(
                result,
                "latency_ms",
            ),

        "device":
            _attr(
                result,
                "device",
            ),

        "memory_usage_mb":
            _attr(
                result,
                "memory_usage_mb",
            ),

        "error":
            _attr(
                result,
                "error",
            ),

        "checked_at":
            datetime.now(
                timezone.utc
            ),
    }


# ============================================================
# MODEL METRICS
# ============================================================

@router.get(
    "/{model_id}/metrics",
    response_model=ModelMetricsResponse,
    summary="Get model metrics",
)
async def model_metrics(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    metrics: Any = Depends(
        get_model_metrics
    ),
) -> Any:

    result = await _call(
        metrics,
        "get_model_metrics",

        model_id=model_id,
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            "model_id":
                model_id,
            **result,
        }

    return {
        "model_id":
            model_id,

        "requests":
            _attr(
                result,
                "requests",
                0,
            ),

        "successes":
            _attr(
                result,
                "successes",
                0,
            ),

        "failures":
            _attr(
                result,
                "failures",
                0,
            ),

        "average_latency_ms":
            _attr(
                result,
                "average_latency_ms",
                0.0,
            ),

        "p50_latency_ms":
            _attr(
                result,
                "p50_latency_ms",
            ),

        "p95_latency_ms":
            _attr(
                result,
                "p95_latency_ms",
            ),

        "p99_latency_ms":
            _attr(
                result,
                "p99_latency_ms",
            ),

        "input_tokens":
            _attr(
                result,
                "input_tokens",
                0,
            ),

        "output_tokens":
            _attr(
                result,
                "output_tokens",
                0,
            ),

        "estimated_cost":
            _attr(
                result,
                "estimated_cost",
                0.0,
            ),

        "model_loads":
            _attr(
                result,
                "model_loads",
                0,
            ),

        "fallbacks":
            _attr(
                result,
                "fallbacks",
                0,
            ),
    }


# ============================================================
# DIRECT INFERENCE
#
# Keep this protected. In most application flows, inference
# should go through services/agents/RAG rather than exposing
# arbitrary model execution to ordinary users.
# ============================================================

@router.post(
    "/{model_id}/infer",
    response_model=ModelInferenceResponse,
    summary="Run direct model inference",
)
async def model_inference(
    model_id: str,
    payload: ModelInferenceRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    server: Any = Depends(
        get_model_server
    ),
) -> Any:

    _require_model_admin(
        current_user
    )

    result = await _call(
        server,
        "infer",

        model_id=model_id,

        inputs=payload.inputs,

        task=payload.task,

        parameters=(
            payload.parameters
        ),

        timeout_seconds=(
            payload.timeout_seconds
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return {
            "model_id":
                result.get(
                    "model_id",
                    model_id,
                ),

            "model_name":
                result.get(
                    "model_name"
                ),

            "model_version":
                result.get(
                    "model_version"
                ),

            "output":
                result.get(
                    "output"
                ),

            "latency_ms":
                result.get(
                    "latency_ms"
                ),

            "fallback_used":
                bool(
                    result.get(
                        "fallback_used",
                        False,
                    )
                ),

            "fallback_model":
                result.get(
                    "fallback_model"
                ),

            "request_id":
                result.get(
                    "request_id"
                )
                or _request_id(
                    request
                ),

            "metadata":
                result.get(
                    "metadata",
                    {},
                ),
        }

    return {
        "model_id":
            model_id,

        "model_name":
            _attr(
                result,
                "model_name",
            ),

        "model_version":
            _attr(
                result,
                "model_version",
            ),

        "output":
            _attr(
                result,
                "output",
                result,
            ),

        "latency_ms":
            _attr(
                result,
                "latency_ms",
            ),

        "fallback_used":
            bool(
                _attr(
                    result,
                    "fallback_used",
                    False,
                )
            ),

        "fallback_model":
            _attr(
                result,
                "fallback_model",
            ),

        "request_id":
            (
                _attr(
                    result,
                    "request_id",
                )
                or _request_id(
                    request
                )
            ),

        "metadata":
            _attr(
                result,
                "metadata",
                {},
            ),
    }


# ============================================================
# SET DEFAULT
# ============================================================

@router.post(
    "/{model_id}/default",
    response_model=ModelResponse,
    summary="Set default model",
)
async def set_default_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    _require_model_admin(
        current_user
    )

    result = await _call(
        registry,
        "set_default",

        model_id=model_id,

        updated_by=_user_id(
            current_user
        ),
    )

    return _serialize_model(
        result
    )


# ============================================================
# ENABLE
# ============================================================

@router.post(
    "/{model_id}/enable",
    response_model=ModelResponse,
    summary="Enable model",
)
async def enable_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    _require_model_admin(
        current_user
    )

    result = await _call(
        registry,
        "enable",

        model_id=model_id,

        updated_by=_user_id(
            current_user
        ),
    )

    return _serialize_model(
        result
    )


# ============================================================
# DISABLE
# ============================================================

@router.post(
    "/{model_id}/disable",
    response_model=ModelResponse,
    summary="Disable model",
)
async def disable_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, Any]:

    _require_model_admin(
        current_user
    )

    result = await _call(
        registry,
        "disable",

        model_id=model_id,

        updated_by=_user_id(
            current_user
        ),
    )

    return _serialize_model(
        result
    )


# ============================================================
# DELETE REGISTRY ENTRY
# ============================================================

@router.delete(
    "/{model_id}",
    response_model=MessageResponse,
    summary="Remove model from registry",
)
async def delete_model(
    model_id: str,

    current_user: Any = Depends(
        get_current_user
    ),

    registry: Any = Depends(
        get_model_registry
    ),
) -> dict[str, str]:

    _require_model_admin(
        current_user
    )

    deleted = await _call(
        registry,
        "delete",

        model_id=model_id,

        deleted_by=_user_id(
            current_user
        ),
    )

    if deleted is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model was not found.",
        )

    return {
        "message":
            f"Model '{model_id}' removed from registry."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Model management subsystem health",
)
async def models_service_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "model-management",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }