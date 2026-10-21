# backend/api/routes/chat.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator


router = APIRouter(
    prefix="/chat",
    tags=["Legal Chat"],
)


# ============================================================
# DEPENDENCIES
# ============================================================

async def get_chat_service() -> Any:
    """
    TEMPORARY MOCK: Returns demo responses until real service is implemented.
    
    TODO: Replace with real ChatService from backend.dependencies
    
    Recommended service dependencies:
        ChatService(
            rag_pipeline=...,
            legal_qa_agent=...,
            document_repository=...,
            audit_service=...,
            security_service=...,
        )
    """
    class MockChatService:
        async def chat(self, **kwargs):
            """Main chat method that the router calls"""
            import uuid
            from datetime import datetime, timezone
            
            message = kwargs.get("message", "")
            message_id = str(uuid.uuid4())
            conversation_id = kwargs.get("conversation_id") or str(uuid.uuid4())
            document_ids = kwargs.get("document_ids", [])
            mode = kwargs.get("mode", "legal_qa")
            model = kwargs.get("model")
            
            # Generate mock response
            user_message = message.lower()
            
            if "risk" in user_message or "liability" in user_message:
                response_text = """Based on the selected documents, I've identified the following key risks:

1. **Indemnification Scope**: The agreement contains broad indemnification clauses that may expose your organization to significant liability.

2. **Limitation of Liability**: The liability cap is set at contract value, which may be insufficient for certain breach scenarios.

3. **Termination Rights**: Notice period for termination is 90 days, which limits flexibility in case of vendor underperformance.

⚠️ **Recommendation**: Have legal counsel review the indemnification and liability provisions before signing."""
            
            elif "obligation" in user_message:
                response_text = """Key contractual obligations identified:

**Your Organization's Obligations:**
• Monthly payment of $50,000 within 30 days of invoice
• Provide necessary access and cooperation
• Maintain confidentiality of vendor proprietary information
• 45-day notice for termination

**Vendor's Obligations:**
• Deliver services per Statement of Work
• Maintain professional liability insurance ($2M minimum)
• Respond to support requests within 4 business hours
• Quarterly business reviews and reporting

All obligations are subject to the terms outlined in Sections 3-7 of the Master Services Agreement."""
            
            elif "termination" in user_message or "renewal" in user_message:
                response_text = """**Termination and Renewal Terms:**

**Initial Term**: 24 months from effective date (January 1, 2025 - December 31, 2026)

**Renewal**: Automatically renews for successive 12-month periods unless either party provides written notice 90 days before term end.

**Termination for Convenience**: Either party may terminate with 90 days' written notice.

**Termination for Cause**: Immediate termination upon 30 days' written notice if material breach is not cured.

**Notice Deadline**: For current term ending Dec 31, 2026, notice must be provided by **October 2, 2026**."""
            
            else:
                response_text = f"""Thank you for your legal question. 

⚠️ **Demo Mode**: This is a mock response from the development backend. The full RAG pipeline with document retrieval, LLM analysis, and citation extraction is under development.

**Your Question**: {message}

**Selected Documents**: {len(document_ids)} document(s)
**Analysis Mode**: {mode}
**Model Requested**: {model or 'auto'}

In production, this would:
1. Retrieve relevant document chunks using hybrid search
2. Rerank results for relevance
3. Generate AI analysis using {model or 'GPT-4o'}
4. Provide verified citations with page numbers
5. Include confidence scores and source validation

**Next Steps**: Implement the ChatService class to connect RAG retrieval + LLM generation."""
            
            # Create mock result object
            class MockResult:
                pass
            
            result = MockResult()
            result.message_id = message_id
            result.conversation_id = conversation_id
            result.answer = response_text  # Changed from response to answer
            result.citations = []
            result.retrieval = None
            result.usage = None
            result.model = model or "gpt-4o-demo"
            result.latency_ms = 580
            result.timestamp = datetime.now(timezone.utc)
            result.metadata = {"mode": mode, "demo_mode": True}
            
            return result
    
    return MockChatService()


async def get_current_user() -> Any:
    """
    Replace with the real JWT authentication dependency.

    Expected:
        id
        tenant_id
        organization_id
        role / roles
        is_admin
    """
    # TEMPORARY: Mock user for development
    # TODO: Replace with real JWT authentication
    return {
        "id": "dev-user-001",
        "tenant_id": "dev-tenant-001",
        "organization_id": "dev-org-001",
        "email": "dev@example.com",
        "role": "admin",
        "is_admin": True,
    }


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ChatMessageRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=50_000,
    )

    conversation_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    document_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    contract_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    clause_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    model: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    mode: str = Field(
        default="legal_qa",
        max_length=100,
    )

    use_rag: bool = True
    use_agents: bool = True

    include_citations: bool = True

    max_sources: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("message")
    @classmethod
    def normalize_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Chat message cannot be empty."
            )

        return value


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    document_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    contract_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    archived: Optional[bool] = None


class FeedbackRequest(BaseModel):
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
    )

    helpful: Optional[bool] = None

    feedback: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    citation_issue: bool = False
    hallucination_issue: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class CitationResponse(BaseModel):
    citation_id: str

    document_id: Optional[str] = None
    contract_id: Optional[str] = None
    clause_id: Optional[str] = None
    chunk_id: Optional[str] = None

    document_name: Optional[str] = None

    page_number: Optional[int] = None

    section: Optional[str] = None
    clause_number: Optional[str] = None

    quote: Optional[str] = None

    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RetrievalResponse(BaseModel):
    query: Optional[str] = None

    strategy: Optional[str] = None

    retrieved_count: int = 0
    reranked_count: int = 0

    retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None


class UsageResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    estimated_cost: Optional[float] = None


class ChatResponse(BaseModel):
    message_id: str
    conversation_id: str

    answer: str

    citations: list[CitationResponse] = Field(
        default_factory=list,
    )

    retrieval: Optional[
        RetrievalResponse
    ] = None

    model: Optional[str] = None
    model_version: Optional[str] = None

    agent: Optional[str] = None

    grounded: Optional[bool] = None
    groundedness_score: Optional[float] = None

    confidence: Optional[float] = None
    uncertainty: Optional[float] = None

    validation_passed: Optional[bool] = None

    usage: Optional[UsageResponse] = None

    latency_ms: Optional[float] = None

    created_at: datetime


class ConversationResponse(BaseModel):
    id: str

    title: Optional[str] = None

    user_id: str
    tenant_id: str

    organization_id: Optional[str] = None

    message_count: int = 0

    archived: bool = False

    created_at: datetime
    updated_at: Optional[datetime] = None


class ConversationPageResponse(BaseModel):
    items: list[ConversationResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class MessageResponse(BaseModel):
    id: str

    conversation_id: str

    role: str
    content: str

    citations: list[CitationResponse] = Field(
        default_factory=list,
    )

    created_at: datetime


class MessagePageResponse(BaseModel):
    items: list[MessageResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class SimpleMessageResponse(BaseModel):
    message: str


# ============================================================
# SECURITY CONTEXT
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(
        obj,
        name,
        default,
    )


def _security_context(
    user: Any,
) -> dict[str, Optional[str]]:

    tenant_id = _attr(
        user,
        "tenant_id",
    )

    user_id = _attr(
        user,
        "id",
    )

    organization_id = _attr(
        user,
        "organization_id",
    )

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return {
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),

        "organization_id": (
            str(organization_id)
            if organization_id
            else None
        ),
    }


# ============================================================
# REQUEST HELPERS
# ============================================================

def _request_id(
    request: Request,
) -> str:

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

    return str(
        value or uuid4()
    )


def _client_ip(
    request: Request,
) -> Optional[str]:

    if request.client is None:
        return None

    return request.client.host


# ============================================================
# SERIALIZATION
# ============================================================

def _enum_value(
    value: Any,
) -> Any:

    return getattr(
        value,
        "value",
        value,
    )


def _citation(
    citation: Any,
    index: int,
) -> dict[str, Any]:

    metadata = (
        _attr(
            citation,
            "metadata",
        )
        or _attr(
            citation,
            "metadata_json",
        )
        or {}
    )

    return {
        "citation_id": str(
            _attr(
                citation,
                "citation_id",
                f"source-{index + 1}",
            )
        ),

        "document_id": _optional_str(
            _attr(
                citation,
                "document_id",
            )
        ),

        "contract_id": _optional_str(
            _attr(
                citation,
                "contract_id",
            )
        ),

        "clause_id": _optional_str(
            _attr(
                citation,
                "clause_id",
            )
        ),

        "chunk_id": _optional_str(
            _attr(
                citation,
                "chunk_id",
            )
        ),

        "document_name": _attr(
            citation,
            "document_name",
        ),

        "page_number": _attr(
            citation,
            "page_number",
        ),

        "section": _attr(
            citation,
            "section",
        ),

        "clause_number": _attr(
            citation,
            "clause_number",
        ),

        "quote": _attr(
            citation,
            "quote",
        ),

        "retrieval_score": _attr(
            citation,
            "retrieval_score",
        ),

        "reranker_score": _attr(
            citation,
            "reranker_score",
        ),

        "metadata": metadata,
    }


def _serialize_chat_result(
    result: Any,
) -> dict[str, Any]:

    citations = (
        _attr(
            result,
            "citations",
            [],
        )
        or []
    )

    retrieval = _attr(
        result,
        "retrieval",
    )

    usage = _attr(
        result,
        "usage",
    )

    created_at = _attr(
        result,
        "created_at",
    )

    if created_at is None:
        created_at = datetime.now(
            timezone.utc
        )

    return {
        "message_id": str(
            _attr(
                result,
                "message_id",
                uuid4(),
            )
        ),

        "conversation_id": str(
            _attr(
                result,
                "conversation_id",
                uuid4(),
            )
        ),

        "answer": str(
            _attr(
                result,
                "answer",
                "",
            )
        ),

        "citations": [
            _citation(
                citation,
                index,
            )
            for index, citation
            in enumerate(citations)
        ],

        "retrieval": (
            {
                "query": _attr(
                    retrieval,
                    "query",
                ),

                "strategy": _enum_value(
                    _attr(
                        retrieval,
                        "strategy",
                    )
                ),

                "retrieved_count": int(
                    _attr(
                        retrieval,
                        "retrieved_count",
                        0,
                    )
                    or 0
                ),

                "reranked_count": int(
                    _attr(
                        retrieval,
                        "reranked_count",
                        0,
                    )
                    or 0
                ),

                "retrieval_ms": _attr(
                    retrieval,
                    "retrieval_ms",
                ),

                "reranking_ms": _attr(
                    retrieval,
                    "reranking_ms",
                ),
            }
            if retrieval
            else None
        ),

        "model": _optional_str(
            _attr(
                result,
                "model",
            )
        ),

        "model_version": _optional_str(
            _attr(
                result,
                "model_version",
            )
        ),

        "agent": _optional_str(
            _attr(
                result,
                "agent",
            )
        ),

        "grounded": _attr(
            result,
            "grounded",
        ),

        "groundedness_score": _attr(
            result,
            "groundedness_score",
        ),

        "confidence": _attr(
            result,
            "confidence",
        ),

        "uncertainty": _attr(
            result,
            "uncertainty",
        ),

        "validation_passed": _attr(
            result,
            "validation_passed",
        ),

        "usage": (
            {
                "prompt_tokens": int(
                    _attr(
                        usage,
                        "prompt_tokens",
                        0,
                    )
                    or 0
                ),

                "completion_tokens": int(
                    _attr(
                        usage,
                        "completion_tokens",
                        0,
                    )
                    or 0
                ),

                "total_tokens": int(
                    _attr(
                        usage,
                        "total_tokens",
                        0,
                    )
                    or 0
                ),

                "estimated_cost": _attr(
                    usage,
                    "estimated_cost",
                ),
            }
            if usage
            else None
        ),

        "latency_ms": _attr(
            result,
            "latency_ms",
        ),

        "created_at": created_at,
    }


def _serialize_conversation(
    item: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                item,
                "id",
                "",
            )
        ),

        "title": _attr(
            item,
            "title",
        ),

        "user_id": str(
            _attr(
                item,
                "user_id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                item,
                "tenant_id",
                "",
            )
        ),

        "organization_id": _optional_str(
            _attr(
                item,
                "organization_id",
            )
        ),

        "message_count": int(
            _attr(
                item,
                "message_count",
                0,
            )
            or 0
        ),

        "archived": bool(
            _attr(
                item,
                "archived",
                False,
            )
        ),

        "created_at": (
            _attr(
                item,
                "created_at",
            )
            or datetime.now(
                timezone.utc
            )
        ),

        "updated_at": _attr(
            item,
            "updated_at",
        ),
    }


def _serialize_message(
    item: Any,
) -> dict[str, Any]:

    citations = (
        _attr(
            item,
            "citations",
            [],
        )
        or []
    )

    return {
        "id": str(
            _attr(
                item,
                "id",
                "",
            )
        ),

        "conversation_id": str(
            _attr(
                item,
                "conversation_id",
                "",
            )
        ),

        "role": str(
            _enum_value(
                _attr(
                    item,
                    "role",
                    "assistant",
                )
            )
        ),

        "content": str(
            _attr(
                item,
                "content",
                "",
            )
        ),

        "citations": [
            _citation(
                citation,
                index,
            )
            for index, citation
            in enumerate(citations)
        ],

        "created_at": (
            _attr(
                item,
                "created_at",
            )
            or datetime.now(
                timezone.utc
            )
        ),
    }


def _optional_str(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    return str(value)


# ============================================================
# SERVICE INVOCATION
# ============================================================

async def _invoke(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Chat operation "
                f"'{method_name}' is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# CHAT
# ============================================================

@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask legal AI",
)
async def chat(
    payload: ChatMessageRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    try:
        result = await _invoke(
            service,
            "chat",

            message=payload.message,

            conversation_id=(
                payload.conversation_id
            ),

            document_ids=(
                payload.document_ids
            ),

            contract_ids=(
                payload.contract_ids
            ),

            clause_ids=(
                payload.clause_ids
            ),

            tenant_id=context[
                "tenant_id"
            ],

            organization_id=context[
                "organization_id"
            ],

            user_id=context[
                "user_id"
            ],

            model=payload.model,

            mode=payload.mode,

            use_rag=payload.use_rag,

            use_agents=(
                payload.use_agents
            ),

            include_citations=(
                payload.include_citations
            ),

            max_sources=(
                payload.max_sources
            ),

            temperature=(
                payload.temperature
            ),

            metadata=(
                payload.metadata
            ),

            request_id=(
                _request_id(request)
            ),

            ip_address=(
                _client_ip(request)
            ),
        )

    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Access to one or more requested "
                "legal resources was denied."
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return _serialize_chat_result(
        result
    )


# ============================================================
# STREAM CHAT
# ============================================================

@router.post(
    "/stream",
    summary="Stream legal AI answer",
)
async def stream_chat(
    payload: ChatMessageRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> StreamingResponse:

    context = _security_context(
        current_user
    )

    async def event_stream() -> AsyncIterator[str]:

        try:
            stream_method = getattr(
                service,
                "stream_chat",
                None,
            )

            if stream_method is None:
                raise RuntimeError(
                    "Streaming chat is not configured."
                )

            stream = stream_method(
                message=payload.message,

                conversation_id=(
                    payload.conversation_id
                ),

                document_ids=(
                    payload.document_ids
                ),

                contract_ids=(
                    payload.contract_ids
                ),

                clause_ids=(
                    payload.clause_ids
                ),

                tenant_id=context[
                    "tenant_id"
                ],

                organization_id=context[
                    "organization_id"
                ],

                user_id=context[
                    "user_id"
                ],

                model=payload.model,

                mode=payload.mode,

                use_rag=payload.use_rag,

                use_agents=(
                    payload.use_agents
                ),

                include_citations=(
                    payload.include_citations
                ),

                max_sources=(
                    payload.max_sources
                ),

                temperature=(
                    payload.temperature
                ),

                metadata=(
                    payload.metadata
                ),

                request_id=(
                    _request_id(request)
                ),
            )

            if hasattr(
                stream,
                "__await__",
            ):
                stream = await stream

            async for event in stream:

                if await request.is_disconnected():
                    break

                if isinstance(
                    event,
                    str,
                ):
                    data = {
                        "type": "token",
                        "content": event,
                    }

                elif isinstance(
                    event,
                    dict,
                ):
                    data = event

                else:
                    data = {
                        "type": "event",
                        "data": str(event),
                    }

                yield (
                    "data: "
                    + json.dumps(
                        data,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n\n"
                )

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done"
                    }
                )
                + "\n\n"
            )

        except Exception:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": (
                            "Unable to complete "
                            "the chat request."
                        ),
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# CREATE CONVERSATION
# ============================================================

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create chat conversation",
)
async def create_conversation(
    payload: ConversationCreateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    conversation = await _invoke(
        service,
        "create_conversation",

        tenant_id=context[
            "tenant_id"
        ],

        organization_id=context[
            "organization_id"
        ],

        user_id=context[
            "user_id"
        ],

        title=payload.title,

        document_ids=(
            payload.document_ids
        ),

        contract_ids=(
            payload.contract_ids
        ),

        metadata=payload.metadata,
    )

    return _serialize_conversation(
        conversation
    )


# ============================================================
# LIST CONVERSATIONS
# ============================================================

@router.get(
    "/conversations",
    response_model=ConversationPageResponse,
    summary="List my chat conversations",
)
async def list_conversations(
    archived: Optional[bool] = Query(
        default=None
    ),

    search: Optional[str] = Query(
        default=None,
        max_length=500,
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=250,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    page = await _invoke(
        service,
        "list_conversations",

        tenant_id=context[
            "tenant_id"
        ],

        organization_id=context[
            "organization_id"
        ],

        user_id=context[
            "user_id"
        ],

        archived=archived,
        search=search,

        limit=limit,
        offset=offset,
    )

    return _conversation_page(
        page
    )


# ============================================================
# GET CONVERSATION
# ============================================================

@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get conversation",
)
async def get_conversation(
    conversation_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    conversation = await _invoke(
        service,
        "get_conversation",

        conversation_id=str(
            conversation_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        user_id=context[
            "user_id"
        ],
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation was not found.",
        )

    return _serialize_conversation(
        conversation
    )


# ============================================================
# UPDATE CONVERSATION
# ============================================================

@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Update conversation",
)
async def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    conversation = await _invoke(
        service,
        "update_conversation",

        conversation_id=str(
            conversation_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        user_id=context[
            "user_id"
        ],

        title=payload.title,

        archived=payload.archived,
    )

    return _serialize_conversation(
        conversation
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================

@router.delete(
    "/conversations/{conversation_id}",
    response_model=SimpleMessageResponse,
    summary="Delete conversation",
)
async def delete_conversation(
    conversation_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, str]:

    context = _security_context(
        current_user
    )

    await _invoke(
        service,
        "delete_conversation",

        conversation_id=str(
            conversation_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        user_id=context[
            "user_id"
        ],
    )

    return {
        "message":
            "Conversation deleted successfully."
    }


# ============================================================
# CONVERSATION MESSAGES
# ============================================================

@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessagePageResponse,
    summary="Get conversation messages",
)
async def get_conversation_messages(
    conversation_id: UUID,

    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    page = await _invoke(
        service,
        "list_messages",

        conversation_id=str(
            conversation_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        user_id=context[
            "user_id"
        ],

        limit=limit,
        offset=offset,
    )

    items = list(
        _attr(
            page,
            "items",
            [],
        )
        or []
    )

    total = int(
        _attr(
            page,
            "total",
            len(items),
        )
    )

    return {
        "items": [
            _serialize_message(item)
            for item in items
        ],

        "total": total,

        "limit": int(
            _attr(
                page,
                "limit",
                limit,
            )
        ),

        "offset": int(
            _attr(
                page,
                "offset",
                offset,
            )
        ),

        "has_more": bool(
            _attr(
                page,
                "has_more",
                (
                    offset
                    + len(items)
                    < total
                ),
            )
        ),
    }


# ============================================================
# REGENERATE
# ============================================================

@router.post(
    "/messages/{message_id}/regenerate",
    response_model=ChatResponse,
    summary="Regenerate assistant answer",
)
async def regenerate_message(
    message_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, Any]:

    context = _security_context(
        current_user
    )

    result = await _invoke(
        service,
        "regenerate",

        message_id=str(
            message_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        organization_id=context[
            "organization_id"
        ],

        user_id=context[
            "user_id"
        ],

        request_id=(
            _request_id(request)
        ),
    )

    return _serialize_chat_result(
        result
    )


# ============================================================
# FEEDBACK
# ============================================================

@router.post(
    "/messages/{message_id}/feedback",
    response_model=SimpleMessageResponse,
    summary="Submit answer feedback",
)
async def submit_feedback(
    message_id: UUID,
    payload: FeedbackRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_chat_service
    ),
) -> dict[str, str]:

    context = _security_context(
        current_user
    )

    await _invoke(
        service,
        "submit_feedback",

        message_id=str(
            message_id
        ),

        tenant_id=context[
            "tenant_id"
        ],

        user_id=context[
            "user_id"
        ],

        rating=payload.rating,

        helpful=payload.helpful,

        feedback=payload.feedback,

        citation_issue=(
            payload.citation_issue
        ),

        hallucination_issue=(
            payload.hallucination_issue
        ),

        metadata=payload.metadata,
    )

    return {
        "message":
            "Feedback submitted successfully."
    }


# ============================================================
# PAGE HELPER
# ============================================================

def _conversation_page(
    page: Any,
) -> dict[str, Any]:

    items = list(
        _attr(
            page,
            "items",
            [],
        )
        or []
    )

    total = int(
        _attr(
            page,
            "total",
            len(items),
        )
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
    )

    return {
        "items": [
            _serialize_conversation(
                item
            )
            for item in items
        ],

        "total": total,
        "limit": limit,
        "offset": offset,

        "has_more": bool(
            _attr(
                page,
                "has_more",
                (
                    offset
                    + len(items)
                    < total
                ),
            )
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    summary="Legal chat subsystem health",
)
async def chat_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "legal-chat",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }