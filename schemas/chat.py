"""
backend/app/api/v1/chat.py

Legal RAG chat API.

Responsibilities
----------------
- Accept legal-document questions
- Use authenticated user / tenant context
- Validate chat requests
- Invoke LegalRAGPipeline
- Return answers, citations, sources, validation and uncertainty
- Maintain conversation identifiers
- Support document-scoped queries
- Expose health/status endpoint

Security
--------
- Never trust user_id / tenant_id supplied by the request body.
- Identity comes from authenticated request.state.
- Tenant and document authorization must be enforced downstream.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Legal AI Chat"],
)


# ============================================================================
# Schemas
# ============================================================================


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        examples=["user"],
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
    )

    @field_validator("role")
    @classmethod
    def validate_role(
        cls,
        value: str,
    ) -> str:

        value = value.strip().lower()

        allowed = {
            "user",
            "assistant",
        }

        if value not in allowed:
            raise ValueError(
                "role must be 'user' or 'assistant'"
            )

        return value

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "message content cannot be empty"
            )

        return value


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
    )

    conversation_id: str | None = None

    document_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    include_sources: bool = True
    include_citations: bool = True
    include_validation: bool = True
    include_uncertainty: bool = True

    @field_validator("message")
    @classmethod
    def clean_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "message cannot be empty"
            )

        return value

    @field_validator("document_ids")
    @classmethod
    def clean_document_ids(
        cls,
        values: list[str],
    ) -> list[str]:

        output: list[str] = []
        seen: set[str] = set()

        for value in values:

            value = str(value).strip()

            if (
                not value
                or value in seen
            ):
                continue

            seen.add(value)
            output.append(value)

        return output


class CitationResponse(BaseModel):
    citation_id: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    chunk_id: str | None = None

    page: int | str | None = None

    clause: str | None = None
    section: str | None = None

    quote: str | None = None

    relevance_score: float | None = None

    source_uri: str | None = None


class SourceResponse(BaseModel):
    id: str | None = None

    document_id: str | None = None
    document_name: str | None = None

    page: int | str | None = None

    clause: str | None = None
    section: str | None = None

    text: str | None = None

    score: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ChatResponse(BaseModel):
    conversation_id: str

    answer: str

    query: str

    rewritten_query: str | None = None

    query_type: str | None = None
    route: str | None = None

    citations: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    sources: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    validation: dict[str, Any] | None = None

    uncertainty: dict[str, Any] | None = None

    retrieved_count: int = 0
    reranked_count: int = 0

    latency_ms: float = 0.0

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Dependency placeholder
# ============================================================================


def get_rag_pipeline(
    request: Request,
) -> Any:
    """
    Retrieve the initialized LegalRAGPipeline.

    Recommended initialization in app/main.py:

        app.state.rag_pipeline = LegalRAGPipeline(...)

    This avoids rebuilding embedding models, rerankers and LLM clients for
    every HTTP request.
    """

    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )

    if pipeline is None:
        logger.error(
            "RAG pipeline is not initialized."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "RAG_PIPELINE_UNAVAILABLE",
                "message": (
                    "Legal AI service is temporarily unavailable."
                ),
            },
        )

    return pipeline


# ============================================================================
# Identity
# ============================================================================


def _identity(
    request: Request,
) -> tuple[str, str | None, str | None]:

    user_id = getattr(
        request.state,
        "user_id",
        None,
    )

    tenant_id = getattr(
        request.state,
        "tenant_id",
        None,
    )

    organization_id = getattr(
        request.state,
        "organization_id",
        None,
    )

    if not user_id:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTHENTICATION_REQUIRED",
                "message": (
                    "Authentication is required."
                ),
            },
        )

    return (
        str(user_id),
        str(tenant_id)
        if tenant_id
        else None,
        str(organization_id)
        if organization_id
        else None,
    )


# ============================================================================
# Main chat endpoint
# ============================================================================


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    payload: ChatRequest,
    request: Request,
    pipeline: Any = Depends(
        get_rag_pipeline
    ),
) -> ChatResponse:

    user_id, tenant_id, organization_id = (
        _identity(request)
    )

    conversation_id = (
        payload.conversation_id
        or str(uuid.uuid4())
    )

    # ------------------------------------------------------------------------
    # Build permission-aware retrieval filters
    # ------------------------------------------------------------------------

    filters = dict(
        payload.filters
    )

    if payload.document_ids:

        filters["document_ids"] = (
            payload.document_ids
        )

    # Never accept identity/tenant authorization filters from the client.
    for forbidden in (
        "user_id",
        "tenant_id",
        "organization_id",
        "org_id",
        "permissions",
        "roles",
    ):
        filters.pop(
            forbidden,
            None,
        )

    # ------------------------------------------------------------------------
    # Conversation metadata
    # ------------------------------------------------------------------------

    metadata: dict[str, Any] = {
        "conversation_id": conversation_id,
        "history": [
            message.model_dump()
            for message
            in payload.history
        ],
    }

    if payload.top_k is not None:
        metadata["requested_top_k"] = (
            payload.top_k
        )

    # ------------------------------------------------------------------------
    # Run legal RAG
    # ------------------------------------------------------------------------

    try:

        result = await _run_pipeline(
            pipeline=pipeline,
            query=payload.message,
            user_id=user_id,
            tenant_id=tenant_id,
            organization_id=organization_id,
            filters=filters,
            metadata=metadata,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Legal chat request failed "
            "conversation_id=%s user_id=%s",
            conversation_id,
            user_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "CHAT_GENERATION_FAILED",
                "message": (
                    "The legal AI assistant could not "
                    "process this request."
                ),
            },
        ) from exc

    # ------------------------------------------------------------------------
    # Normalize pipeline result
    # ------------------------------------------------------------------------

    answer = str(
        _get(
            result,
            "answer",
            default="",
        )
        or ""
    ).strip()

    if not answer:

        logger.warning(
            "Pipeline returned empty answer "
            "conversation_id=%s",
            conversation_id,
        )

        answer = (
            "The available legal documents do not provide "
            "enough information to answer this question."
        )

    citations = _serialize_collection(
        _get(
            result,
            "citations",
            default=[],
        )
    )

    sources = _serialize_collection(
        _get(
            result,
            "sources",
            default=[],
        )
    )

    validation = _serialize_optional(
        _get(
            result,
            "validation",
            default=None,
        )
    )

    uncertainty = _serialize_optional(
        _get(
            result,
            "uncertainty",
            default=None,
        )
    )

    if not payload.include_citations:
        citations = []

    if not payload.include_sources:
        sources = []

    if not payload.include_validation:
        validation = None

    if not payload.include_uncertainty:
        uncertainty = None

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        query=str(
            _get(
                result,
                "query",
                default=payload.message,
            )
        ),
        rewritten_query=_optional_string(
            _get(
                result,
                "rewritten_query",
                default=None,
            )
        ),
        query_type=_enum_string(
            _get(
                result,
                "query_type",
                default=None,
            )
        ),
        route=_enum_string(
            _get(
                result,
                "route",
                default=None,
            )
        ),
        citations=citations,
        sources=sources,
        validation=validation,
        uncertainty=uncertainty,
        retrieved_count=_safe_int(
            _get(
                result,
                "retrieved_count",
                default=0,
            )
        ),
        reranked_count=_safe_int(
            _get(
                result,
                "reranked_count",
                default=0,
            )
        ),
        latency_ms=_safe_float(
            _get(
                result,
                "latency_ms",
                default=0.0,
            )
        ),
        metadata={
            "tenant_id": tenant_id,
            "organization_id": organization_id,
        },
    )


# ============================================================================
# Pipeline execution
# ============================================================================


async def _run_pipeline(
    *,
    pipeline: Any,
    query: str,
    user_id: str,
    tenant_id: str | None,
    organization_id: str | None,
    filters: dict[str, Any],
    metadata: dict[str, Any],
) -> Any:

    kwargs = {
        "query": query,
        "filters": filters,
        "user_id": user_id,
        "org_id": organization_id,
        "metadata": metadata,
    }

    # Tenant context can be consumed by permission-aware retrieval.
    if tenant_id:
        kwargs["tenant_id"] = tenant_id

    # Prefer native async pipeline.
    arun = getattr(
        pipeline,
        "arun",
        None,
    )

    if callable(arun):

        try:
            return await arun(
                **kwargs
            )

        except TypeError as exc:
            # Compatibility with the previously defined LegalRAGPipeline
            # signature if tenant_id/metadata are not yet exposed.
            if (
                "tenant_id" not in str(exc)
                and "metadata" not in str(exc)
            ):
                raise

            compatible_kwargs = {
                "query": query,
                "filters": filters,
                "user_id": user_id,
                "org_id": organization_id,
            }

            return await arun(
                **compatible_kwargs
            )

    raise RuntimeError(
        "RAG pipeline does not provide an async arun() method."
    )


# ============================================================================
# Health endpoint
# ============================================================================


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
)
async def chat_status(
    request: Request,
) -> dict[str, Any]:

    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )

    return {
        "service": "legal-ai-chat",
        "status": (
            "ready"
            if pipeline is not None
            else "unavailable"
        ),
        "rag_pipeline": (
            pipeline is not None
        ),
    }


# ============================================================================
# Serialization
# ============================================================================


def _serialize_collection(
    value: Any,
) -> list[dict[str, Any]]:

    if value is None:
        return []

    # CitationResult support.
    nested_citations = getattr(
        value,
        "citations",
        None,
    )

    if nested_citations is not None:
        value = nested_citations

    if isinstance(
        value,
        Mapping,
    ):
        return [
            dict(value)
        ]

    if not isinstance(
        value,
        (list, tuple),
    ):
        return []

    output: list[dict[str, Any]] = []

    for item in value:

        serialized = _serialize_optional(
            item
        )

        if serialized is not None:
            output.append(serialized)

    return output


def _serialize_optional(
    value: Any,
) -> dict[str, Any] | None:

    if value is None:
        return None

    if isinstance(
        value,
        Mapping,
    ):
        return dict(value)

    to_dict = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(to_dict):

        result = to_dict()

        if isinstance(
            result,
            Mapping,
        ):
            return dict(result)

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):

        result = model_dump()

        if isinstance(
            result,
            Mapping,
        ):
            return dict(result)

    if hasattr(
        value,
        "__dict__",
    ):
        return dict(
            value.__dict__
        )

    return {
        "value": str(value)
    }


# ============================================================================
# Helpers
# ============================================================================


def _get(
    obj: Any,
    name: str,
    *,
    default: Any = None,
) -> Any:

    if isinstance(
        obj,
        Mapping,
    ):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def _optional_string(
    value: Any,
) -> str | None:

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _enum_string(
    value: Any,
) -> str | None:

    if value is None:
        return None

    value = getattr(
        value,
        "value",
        value,
    )

    return str(value)


def _safe_int(
    value: Any,
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0


def _safe_float(
    value: Any,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0