# backend/api/routes/search.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field, field_validator


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


# ============================================================
# DEPENDENCIES
# Replace with providers from backend/dependencies.py
# ============================================================

async def get_search_service() -> Any:
    """
    SearchService should orchestrate:

        QueryClassifier
        QueryRewriter
        QueryRouter
        MultiQuery
        PermissionAwareSearch
        BM25
        VectorSearch
        HybridSearch
        MetadataFilter
        Reranker
        AuditService
        EvaluationService
        Observability / tracing

    IMPORTANT:
    Authorization filtering must happen before protected text
    reaches reranking, generation, or any LLM.
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Search service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# TYPES
# ============================================================

SearchMode = Literal[
    "auto",
    "hybrid",
    "vector",
    "bm25",
    "semantic",
    "keyword",
]

SearchResourceType = Literal[
    "all",
    "document",
    "contract",
    "clause",
    "table",
    "chunk",
]


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class SearchFiltersRequest(BaseModel):
    organization_id: Optional[UUID] = None

    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=100,
    )

    contract_ids: list[UUID] = Field(
        default_factory=list,
        max_length=100,
    )

    clause_ids: list[UUID] = Field(
        default_factory=list,
        max_length=200,
    )

    document_version_ids: list[UUID] = Field(
        default_factory=list,
        max_length=100,
    )

    document_types: list[str] = Field(
        default_factory=list,
        max_length=50,
    )

    clause_types: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    risk_levels: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    languages: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    parent_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=200,
    )

    page_from: Optional[int] = Field(
        default=None,
        ge=1,
    )

    page_to: Optional[int] = Field(
        default=None,
        ge=1,
    )

    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None

    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None

    expiry_from: Optional[datetime] = None
    expiry_to: Optional[datetime] = None

    governing_law: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    jurisdiction: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    mode: SearchMode = "auto"

    resource_type: SearchResourceType = "all"

    filters: SearchFiltersRequest = Field(
        default_factory=SearchFiltersRequest,
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    candidate_k: int = Field(
        default=100,
        ge=1,
        le=500,
    )

    rerank_top_k: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
    )

    use_query_classifier: bool = True
    use_query_rewrite: bool = True
    use_multi_query: bool = True
    use_reranker: bool = True

    include_parent_context: bool = True
    include_neighbors: bool = False

    include_content: bool = True
    include_metadata: bool = True

    include_scores: bool = True
    include_explanation: bool = False

    include_query_plan: bool = False

    minimum_score: Optional[float] = None

    model: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Search query cannot be empty."
            )

        return value


class SearchSuggestRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=100,
    )


class SimilarSearchRequest(BaseModel):
    chunk_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    clause_id: Optional[UUID] = None

    document_id: Optional[UUID] = None

    text: Optional[str] = Field(
        default=None,
        max_length=20_000,
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    minimum_score: Optional[float] = None


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class SearchCitationResponse(BaseModel):
    document_id: Optional[str] = None
    document_name: Optional[str] = None

    document_version_id: Optional[str] = None

    contract_id: Optional[str] = None

    clause_id: Optional[str] = None
    clause_number: Optional[str] = None

    chunk_id: Optional[str] = None

    page_number: Optional[int] = None

    section: Optional[str] = None
    heading: Optional[str] = None

    quote: Optional[str] = None


class SearchScoresResponse(BaseModel):
    final_score: Optional[float] = None

    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None

    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None

    normalized_score: Optional[float] = None


class SearchResultItem(BaseModel):
    id: str

    resource_type: str

    document_id: Optional[str] = None
    document_version_id: Optional[str] = None

    contract_id: Optional[str] = None
    clause_id: Optional[str] = None

    chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    title: Optional[str] = None

    content: Optional[str] = None

    page_number: Optional[int] = None

    section: Optional[str] = None
    heading: Optional[str] = None

    document_type: Optional[str] = None
    clause_type: Optional[str] = None

    risk_level: Optional[str] = None

    language: Optional[str] = None

    scores: Optional[
        SearchScoresResponse
    ] = None

    citation: Optional[
        SearchCitationResponse
    ] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class QueryClassificationResponse(BaseModel):
    query_type: Optional[str] = None
    scope: Optional[str] = None
    complexity: Optional[str] = None

    confidence: Optional[float] = None

    requires_rag: bool = True
    requires_structured_search: bool = False
    requires_comparison: bool = False
    requires_table_search: bool = False

    signals: list[str] = Field(
        default_factory=list,
    )


class QueryRewriteResponse(BaseModel):
    original_query: str

    rewritten_query: Optional[str] = None

    lexical_query: Optional[str] = None
    semantic_query: Optional[str] = None

    expanded_queries: list[str] = Field(
        default_factory=list,
    )


class SearchPlanResponse(BaseModel):
    mode: str

    use_bm25: bool = False
    use_vector: bool = False
    use_hybrid: bool = False
    use_reranker: bool = False

    use_multi_query: bool = False

    permission_filtering: bool = True

    candidate_k: int = 0
    top_k: int = 0


class SearchTimingResponse(BaseModel):
    classification_ms: Optional[float] = None
    rewrite_ms: Optional[float] = None

    permission_ms: Optional[float] = None

    bm25_ms: Optional[float] = None
    vector_ms: Optional[float] = None

    fusion_ms: Optional[float] = None
    reranker_ms: Optional[float] = None

    total_ms: Optional[float] = None


class SearchResponse(BaseModel):
    query: str

    request_id: Optional[str] = None

    mode: str

    total: int

    results: list[SearchResultItem]

    classification: Optional[
        QueryClassificationResponse
    ] = None

    rewrite: Optional[
        QueryRewriteResponse
    ] = None

    plan: Optional[
        SearchPlanResponse
    ] = None

    timing: Optional[
        SearchTimingResponse
    ] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SearchSuggestionResponse(BaseModel):
    suggestions: list[str]

    total: int


class SearchHealthResponse(BaseModel):
    status: str
    service: str

    bm25: Optional[str] = None
    vector: Optional[str] = None
    reranker: Optional[str] = None

    timestamp: datetime


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


def _float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


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


def _organization_id(
    user: Any,
) -> Optional[str]:

    return _optional_str(
        _attr(
            user,
            "organization_id",
        )
    )


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


def _roles(
    user: Any,
) -> list[str]:

    values = (
        _attr(
            user,
            "roles",
            [],
        )
        or []
    )

    if isinstance(values, str):
        values = [values]

    role = _attr(
        user,
        "role",
    )

    if role:
        values = [
            *values,
            role,
        ]

    return list(
        {
            str(
                _enum_value(value)
            ).lower()
            for value in values
        }
    )


def _request_id(
    request: Request,
) -> Optional[str]:

    return _optional_str(
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )


def _serialize_scores(
    item: Any,
) -> dict[str, Any]:

    scores = _attr(
        item,
        "scores",
        None,
    )

    source = (
        scores
        if scores is not None
        else item
    )

    return {
        "final_score":
            _float(
                _attr(
                    source,
                    "final_score",
                    _attr(
                        item,
                        "score",
                    ),
                )
            ),

        "vector_score":
            _float(
                _attr(
                    source,
                    "vector_score",
                )
            ),

        "bm25_score":
            _float(
                _attr(
                    source,
                    "bm25_score",
                )
            ),

        "hybrid_score":
            _float(
                _attr(
                    source,
                    "hybrid_score",
                )
            ),

        "reranker_score":
            _float(
                _attr(
                    source,
                    "reranker_score",
                )
            ),

        "normalized_score":
            _float(
                _attr(
                    source,
                    "normalized_score",
                )
            ),
    }


def _serialize_citation(
    item: Any,
) -> dict[str, Any]:

    citation = (
        _attr(
            item,
            "citation",
        )
        or item
    )

    return {
        "document_id":
            _optional_str(
                _attr(
                    citation,
                    "document_id",
                    _attr(
                        item,
                        "document_id",
                    ),
                )
            ),

        "document_name":
            _optional_str(
                _attr(
                    citation,
                    "document_name",
                )
            ),

        "document_version_id":
            _optional_str(
                _attr(
                    citation,
                    "document_version_id",
                    _attr(
                        item,
                        "document_version_id",
                    ),
                )
            ),

        "contract_id":
            _optional_str(
                _attr(
                    citation,
                    "contract_id",
                    _attr(
                        item,
                        "contract_id",
                    ),
                )
            ),

        "clause_id":
            _optional_str(
                _attr(
                    citation,
                    "clause_id",
                    _attr(
                        item,
                        "clause_id",
                    ),
                )
            ),

        "clause_number":
            _optional_str(
                _attr(
                    citation,
                    "clause_number",
                )
            ),

        "chunk_id":
            _optional_str(
                _attr(
                    citation,
                    "chunk_id",
                    _attr(
                        item,
                        "chunk_id",
                    ),
                )
            ),

        "page_number":
            _attr(
                citation,
                "page_number",
                _attr(
                    item,
                    "page_number",
                ),
            ),

        "section":
            _optional_str(
                _attr(
                    citation,
                    "section",
                    _attr(
                        item,
                        "section",
                    ),
                )
            ),

        "heading":
            _optional_str(
                _attr(
                    citation,
                    "heading",
                    _attr(
                        item,
                        "heading",
                    ),
                )
            ),

        "quote":
            _optional_str(
                _attr(
                    citation,
                    "quote",
                )
            ),
    }


def _result_metadata(
    item: Any,
) -> dict[str, Any]:

    return (
        _attr(
            item,
            "metadata",
        )
        or _attr(
            item,
            "metadata_json",
        )
        or {}
    )


def _serialize_result(
    item: Any,
    *,
    include_content: bool = True,
    include_metadata: bool = True,
    include_scores: bool = True,
) -> dict[str, Any]:

    chunk_id = _optional_str(
        _attr(
            item,
            "chunk_id",
        )
    )

    clause_id = _optional_str(
        _attr(
            item,
            "clause_id",
        )
    )

    document_id = _optional_str(
        _attr(
            item,
            "document_id",
        )
    )

    item_id = (
        _optional_str(
            _attr(
                item,
                "id",
            )
        )
        or chunk_id
        or clause_id
        or document_id
        or ""
    )

    return {
        "id": item_id,

        "resource_type": str(
            _enum_value(
                _attr(
                    item,
                    "resource_type",
                    _attr(
                        item,
                        "type",
                        "chunk",
                    ),
                )
            )
        ),

        "document_id":
            document_id,

        "document_version_id":
            _optional_str(
                _attr(
                    item,
                    "document_version_id",
                )
            ),

        "contract_id":
            _optional_str(
                _attr(
                    item,
                    "contract_id",
                )
            ),

        "clause_id":
            clause_id,

        "chunk_id":
            chunk_id,

        "parent_chunk_id":
            _optional_str(
                _attr(
                    item,
                    "parent_chunk_id",
                )
            ),

        "title":
            _optional_str(
                _attr(
                    item,
                    "title",
                )
            ),

        "content":
            (
                _optional_str(
                    _attr(
                        item,
                        "content",
                        _attr(
                            item,
                            "text",
                        ),
                    )
                )
                if include_content
                else None
            ),

        "page_number":
            _attr(
                item,
                "page_number",
            ),

        "section":
            _optional_str(
                _attr(
                    item,
                    "section",
                )
            ),

        "heading":
            _optional_str(
                _attr(
                    item,
                    "heading",
                )
            ),

        "document_type":
            _optional_str(
                _enum_value(
                    _attr(
                        item,
                        "document_type",
                    )
                )
            ),

        "clause_type":
            _optional_str(
                _enum_value(
                    _attr(
                        item,
                        "clause_type",
                    )
                )
            ),

        "risk_level":
            _optional_str(
                _enum_value(
                    _attr(
                        item,
                        "risk_level",
                    )
                )
            ),

        "language":
            _optional_str(
                _attr(
                    item,
                    "language",
                )
            ),

        "scores":
            (
                _serialize_scores(
                    item
                )
                if include_scores
                else None
            ),

        "citation":
            _serialize_citation(
                item
            ),

        "metadata":
            (
                _result_metadata(
                    item
                )
                if include_metadata
                else {}
            ),
    }


def _serialize_classification(
    classification: Any,
) -> Optional[dict[str, Any]]:

    if classification is None:
        return None

    signals = (
        _attr(
            classification,
            "signals",
            [],
        )
        or []
    )

    return {
        "query_type":
            _optional_str(
                _enum_value(
                    _attr(
                        classification,
                        "query_type",
                    )
                )
            ),

        "scope":
            _optional_str(
                _enum_value(
                    _attr(
                        classification,
                        "scope",
                    )
                )
            ),

        "complexity":
            _optional_str(
                _enum_value(
                    _attr(
                        classification,
                        "complexity",
                    )
                )
            ),

        "confidence":
            _float(
                _attr(
                    classification,
                    "confidence",
                )
            ),

        "requires_rag":
            bool(
                _attr(
                    classification,
                    "requires_rag",
                    True,
                )
            ),

        "requires_structured_search":
            bool(
                _attr(
                    classification,
                    "requires_structured_search",
                    False,
                )
            ),

        "requires_comparison":
            bool(
                _attr(
                    classification,
                    "requires_comparison",
                    False,
                )
            ),

        "requires_table_search":
            bool(
                _attr(
                    classification,
                    "requires_table_search",
                    False,
                )
            ),

        "signals": [
            str(
                _enum_value(signal)
            )
            for signal in signals
        ],
    }


def _serialize_rewrite(
    rewrite: Any,
    original_query: str,
) -> Optional[dict[str, Any]]:

    if rewrite is None:
        return None

    return {
        "original_query":
            _optional_str(
                _attr(
                    rewrite,
                    "original_query",
                    original_query,
                )
            )
            or original_query,

        "rewritten_query":
            _optional_str(
                _attr(
                    rewrite,
                    "rewritten_query",
                )
            ),

        "lexical_query":
            _optional_str(
                _attr(
                    rewrite,
                    "lexical_query",
                )
            ),

        "semantic_query":
            _optional_str(
                _attr(
                    rewrite,
                    "semantic_query",
                )
            ),

        "expanded_queries": [
            str(query)
            for query in (
                _attr(
                    rewrite,
                    "expanded_queries",
                    [],
                )
                or []
            )
        ],
    }


def _serialize_plan(
    plan: Any,
) -> Optional[dict[str, Any]]:

    if plan is None:
        return None

    return {
        "mode":
            str(
                _enum_value(
                    _attr(
                        plan,
                        "mode",
                        "auto",
                    )
                )
            ),

        "use_bm25":
            bool(
                _attr(
                    plan,
                    "use_bm25",
                    False,
                )
            ),

        "use_vector":
            bool(
                _attr(
                    plan,
                    "use_vector",
                    False,
                )
            ),

        "use_hybrid":
            bool(
                _attr(
                    plan,
                    "use_hybrid",
                    False,
                )
            ),

        "use_reranker":
            bool(
                _attr(
                    plan,
                    "use_reranker",
                    False,
                )
            ),

        "use_multi_query":
            bool(
                _attr(
                    plan,
                    "use_multi_query",
                    False,
                )
            ),

        "permission_filtering":
            bool(
                _attr(
                    plan,
                    "permission_filtering",
                    True,
                )
            ),

        "candidate_k":
            int(
                _attr(
                    plan,
                    "candidate_k",
                    0,
                )
                or 0
            ),

        "top_k":
            int(
                _attr(
                    plan,
                    "top_k",
                    0,
                )
                or 0
            ),
    }


def _serialize_timing(
    timing: Any,
) -> Optional[dict[str, Any]]:

    if timing is None:
        return None

    return {
        "classification_ms":
            _float(
                _attr(
                    timing,
                    "classification_ms",
                )
            ),

        "rewrite_ms":
            _float(
                _attr(
                    timing,
                    "rewrite_ms",
                )
            ),

        "permission_ms":
            _float(
                _attr(
                    timing,
                    "permission_ms",
                )
            ),

        "bm25_ms":
            _float(
                _attr(
                    timing,
                    "bm25_ms",
                )
            ),

        "vector_ms":
            _float(
                _attr(
                    timing,
                    "vector_ms",
                )
            ),

        "fusion_ms":
            _float(
                _attr(
                    timing,
                    "fusion_ms",
                )
            ),

        "reranker_ms":
            _float(
                _attr(
                    timing,
                    "reranker_ms",
                )
            ),

        "total_ms":
            _float(
                _attr(
                    timing,
                    "total_ms",
                )
            ),
    }


def _serialize_filters(
    filters: SearchFiltersRequest,
) -> dict[str, Any]:

    data = filters.model_dump(
        exclude_none=True,
    )

    for field in (
        "organization_id",
        "document_ids",
        "contract_ids",
        "clause_ids",
        "document_version_ids",
    ):
        if field not in data:
            continue

        value = data[field]

        if isinstance(value, list):
            data[field] = [
                str(item)
                for item in value
            ]

        elif value is not None:
            data[field] = str(value)

    return data


async def _service_call(
    service: Any,
    operation: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        operation,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Search operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# MAIN SEARCH
# ============================================================

@router.post(
    "",
    response_model=SearchResponse,
    summary="Search legal knowledge",
)
async def search(
    payload: SearchRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    tenant_id = _tenant_id(
        current_user
    )

    organization_id = _organization_id(
        current_user
    )

    # --------------------------------------------------------
    # SECURITY
    #
    # Client filters are REQUESTED scope only.
    # SearchService / PermissionAwareSearch must intersect them
    # with server-side authorization.
    #
    # Never treat document_ids sent by the browser as proof
    # that the user may read those documents.
    # --------------------------------------------------------

    requested_filters = _serialize_filters(
        payload.filters
    )

    # Do not allow a browser-provided organization to escape
    # the authenticated organization.
    requested_filters.pop(
        "organization_id",
        None,
    )

    result = await _service_call(
        service,
        "search",

        query=payload.query,

        tenant_id=tenant_id,

        organization_id=organization_id,

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        mode=payload.mode,

        resource_type=(
            payload.resource_type
        ),

        filters=requested_filters,

        top_k=payload.top_k,

        candidate_k=payload.candidate_k,

        rerank_top_k=(
            payload.rerank_top_k
        ),

        use_query_classifier=(
            payload.use_query_classifier
        ),

        use_query_rewrite=(
            payload.use_query_rewrite
        ),

        use_multi_query=(
            payload.use_multi_query
        ),

        use_reranker=(
            payload.use_reranker
        ),

        include_parent_context=(
            payload.include_parent_context
        ),

        include_neighbors=(
            payload.include_neighbors
        ),

        minimum_score=(
            payload.minimum_score
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),

        metadata=payload.metadata,
    )

    items = list(
        _attr(
            result,
            "results",
            _attr(
                result,
                "items",
                [],
            ),
        )
        or []
    )

    serialized = [
        _serialize_result(
            item,
            include_content=(
                payload.include_content
            ),
            include_metadata=(
                payload.include_metadata
            ),
            include_scores=(
                payload.include_scores
            ),
        )
        for item in items
    ]

    return {
        "query":
            payload.query,

        "request_id":
            _request_id(
                request
            ),

        "mode":
            str(
                _enum_value(
                    _attr(
                        result,
                        "mode",
                        payload.mode,
                    )
                )
            ),

        "total":
            int(
                _attr(
                    result,
                    "total",
                    len(serialized),
                )
                or 0
            ),

        "results":
            serialized,

        "classification":
            (
                _serialize_classification(
                    _attr(
                        result,
                        "classification",
                    )
                )
                if payload.include_explanation
                or payload.include_query_plan
                else None
            ),

        "rewrite":
            (
                _serialize_rewrite(
                    _attr(
                        result,
                        "rewrite",
                    ),
                    payload.query,
                )
                if payload.include_explanation
                or payload.include_query_plan
                else None
            ),

        "plan":
            (
                _serialize_plan(
                    _attr(
                        result,
                        "plan",
                    )
                )
                if payload.include_query_plan
                else None
            ),

        "timing":
            _serialize_timing(
                _attr(
                    result,
                    "timing",
                )
            ),

        "metadata":
            (
                _attr(
                    result,
                    "metadata",
                    {},
                )
                or {}
            ),
    }


# ============================================================
# SIMPLE GET SEARCH
#
# Useful for frontend search bars.
# ============================================================

@router.get(
    "",
    response_model=SearchResponse,
    summary="Quick legal search",
)
async def quick_search(
    request: Request,

    q: str = Query(
        min_length=1,
        max_length=2000,
    ),

    mode: SearchMode = Query(
        default="auto",
    ),

    resource_type: SearchResourceType = Query(
        default="all",
    ),

    top_k: int = Query(
        default=20,
        ge=1,
        le=100,
    ),

    document_id: Optional[UUID] = None,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    filters: dict[str, Any] = {}

    if document_id:
        filters["document_ids"] = [
            str(document_id)
        ]

    result = await _service_call(
        service,
        "search",

        query=q.strip(),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        mode=mode,

        resource_type=resource_type,

        filters=filters,

        top_k=top_k,

        candidate_k=max(
            top_k * 5,
            50,
        ),

        use_query_classifier=True,
        use_query_rewrite=True,
        use_multi_query=True,
        use_reranker=True,

        include_parent_context=True,
        include_neighbors=False,

        minimum_score=None,

        model=None,

        request_id=_request_id(
            request
        ),

        metadata={},
    )

    items = list(
        _attr(
            result,
            "results",
            _attr(
                result,
                "items",
                [],
            ),
        )
        or []
    )

    serialized = [
        _serialize_result(item)
        for item in items
    ]

    return {
        "query": q,

        "request_id":
            _request_id(
                request
            ),

        "mode":
            str(
                _enum_value(
                    _attr(
                        result,
                        "mode",
                        mode,
                    )
                )
            ),

        "total":
            int(
                _attr(
                    result,
                    "total",
                    len(serialized),
                )
                or 0
            ),

        "results":
            serialized,

        "classification": None,
        "rewrite": None,
        "plan": None,

        "timing":
            _serialize_timing(
                _attr(
                    result,
                    "timing",
                )
            ),

        "metadata":
            (
                _attr(
                    result,
                    "metadata",
                    {},
                )
                or {}
            ),
    }


# ============================================================
# DOCUMENT SEARCH
# ============================================================

@router.post(
    "/documents/{document_id}",
    response_model=SearchResponse,
    summary="Search inside document",
)
async def search_document(
    document_id: UUID,
    payload: SearchRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    filters = _serialize_filters(
        payload.filters
    )

    filters.pop(
        "organization_id",
        None,
    )

    filters["document_ids"] = [
        str(document_id)
    ]

    result = await _service_call(
        service,
        "search",

        query=payload.query,

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        mode=payload.mode,

        resource_type=(
            payload.resource_type
        ),

        filters=filters,

        top_k=payload.top_k,

        candidate_k=payload.candidate_k,

        rerank_top_k=(
            payload.rerank_top_k
        ),

        use_query_classifier=(
            payload.use_query_classifier
        ),

        use_query_rewrite=(
            payload.use_query_rewrite
        ),

        use_multi_query=(
            payload.use_multi_query
        ),

        use_reranker=(
            payload.use_reranker
        ),

        include_parent_context=(
            payload.include_parent_context
        ),

        include_neighbors=(
            payload.include_neighbors
        ),

        minimum_score=(
            payload.minimum_score
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),

        metadata=payload.metadata,
    )

    return _build_search_response(
        result=result,
        payload=payload,
        request=request,
    )


# ============================================================
# CONTRACT SEARCH
# ============================================================

@router.post(
    "/contracts/{contract_id}",
    response_model=SearchResponse,
    summary="Search inside contract",
)
async def search_contract(
    contract_id: UUID,
    payload: SearchRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    filters = _serialize_filters(
        payload.filters
    )

    filters.pop(
        "organization_id",
        None,
    )

    filters["contract_ids"] = [
        str(contract_id)
    ]

    result = await _service_call(
        service,
        "search",

        query=payload.query,

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        mode=payload.mode,

        resource_type=(
            payload.resource_type
        ),

        filters=filters,

        top_k=payload.top_k,

        candidate_k=payload.candidate_k,

        rerank_top_k=(
            payload.rerank_top_k
        ),

        use_query_classifier=(
            payload.use_query_classifier
        ),

        use_query_rewrite=(
            payload.use_query_rewrite
        ),

        use_multi_query=(
            payload.use_multi_query
        ),

        use_reranker=(
            payload.use_reranker
        ),

        include_parent_context=(
            payload.include_parent_context
        ),

        include_neighbors=(
            payload.include_neighbors
        ),

        minimum_score=(
            payload.minimum_score
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),

        metadata=payload.metadata,
    )

    return _build_search_response(
        result=result,
        payload=payload,
        request=request,
    )


# ============================================================
# CLAUSE SEARCH
# ============================================================

@router.post(
    "/clauses",
    response_model=SearchResponse,
    summary="Search clauses",
)
async def search_clauses(
    payload: SearchRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    filters = _serialize_filters(
        payload.filters
    )

    filters.pop(
        "organization_id",
        None,
    )

    result = await _service_call(
        service,
        "search",

        query=payload.query,

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        mode=payload.mode,

        resource_type="clause",

        filters=filters,

        top_k=payload.top_k,

        candidate_k=payload.candidate_k,

        rerank_top_k=(
            payload.rerank_top_k
        ),

        use_query_classifier=(
            payload.use_query_classifier
        ),

        use_query_rewrite=(
            payload.use_query_rewrite
        ),

        use_multi_query=(
            payload.use_multi_query
        ),

        use_reranker=(
            payload.use_reranker
        ),

        include_parent_context=(
            payload.include_parent_context
        ),

        include_neighbors=(
            payload.include_neighbors
        ),

        minimum_score=(
            payload.minimum_score
        ),

        model=payload.model,

        request_id=_request_id(
            request
        ),

        metadata=payload.metadata,
    )

    return _build_search_response(
        result=result,
        payload=payload,
        request=request,
    )


# ============================================================
# SIMILAR CONTENT
# ============================================================

@router.post(
    "/similar",
    response_model=SearchResponse,
    summary="Find similar legal content",
)
async def similar_search(
    payload: SimilarSearchRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    if not any(
        (
            payload.chunk_id,
            payload.clause_id,
            payload.document_id,
            payload.text,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Provide chunk_id, clause_id, "
                "document_id, or text."
            ),
        )

    result = await _service_call(
        service,
        "find_similar",

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        chunk_id=payload.chunk_id,

        clause_id=(
            str(payload.clause_id)
            if payload.clause_id
            else None
        ),

        document_id=(
            str(payload.document_id)
            if payload.document_id
            else None
        ),

        text=payload.text,

        top_k=payload.top_k,

        minimum_score=(
            payload.minimum_score
        ),

        request_id=_request_id(
            request
        ),
    )

    items = list(
        _attr(
            result,
            "results",
            _attr(
                result,
                "items",
                [],
            ),
        )
        or []
    )

    serialized = [
        _serialize_result(item)
        for item in items
    ]

    return {
        "query":
            payload.text
            or payload.chunk_id
            or _optional_str(
                payload.clause_id
            )
            or _optional_str(
                payload.document_id
            )
            or "",

        "request_id":
            _request_id(
                request
            ),

        "mode":
            "vector",

        "total":
            len(serialized),

        "results":
            serialized,

        "classification": None,
        "rewrite": None,
        "plan": None,

        "timing":
            _serialize_timing(
                _attr(
                    result,
                    "timing",
                )
            ),

        "metadata":
            (
                _attr(
                    result,
                    "metadata",
                    {},
                )
                or {}
            ),
    }


# ============================================================
# SEARCH SUGGESTIONS
# ============================================================

@router.post(
    "/suggest",
    response_model=SearchSuggestionResponse,
    summary="Get search suggestions",
)
async def search_suggestions(
    payload: SearchSuggestRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    result = await _service_call(
        service,
        "suggest",

        query=payload.query.strip(),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        roles=_roles(
            current_user
        ),

        document_ids=[
            str(value)
            for value
            in payload.document_ids
        ],

        limit=payload.limit,
    )

    suggestions = list(
        _attr(
            result,
            "suggestions",
            result
            if isinstance(
                result,
                (list, tuple),
            )
            else [],
        )
        or []
    )

    suggestions = [
        str(value)
        for value in suggestions
    ][: payload.limit]

    return {
        "suggestions":
            suggestions,

        "total":
            len(suggestions),
    }


# ============================================================
# RESPONSE BUILDER
# ============================================================

def _build_search_response(
    *,
    result: Any,
    payload: SearchRequest,
    request: Request,
) -> dict[str, Any]:

    items = list(
        _attr(
            result,
            "results",
            _attr(
                result,
                "items",
                [],
            ),
        )
        or []
    )

    serialized = [
        _serialize_result(
            item,
            include_content=(
                payload.include_content
            ),
            include_metadata=(
                payload.include_metadata
            ),
            include_scores=(
                payload.include_scores
            ),
        )
        for item in items
    ]

    return {
        "query":
            payload.query,

        "request_id":
            _request_id(
                request
            ),

        "mode":
            str(
                _enum_value(
                    _attr(
                        result,
                        "mode",
                        payload.mode,
                    )
                )
            ),

        "total":
            int(
                _attr(
                    result,
                    "total",
                    len(serialized),
                )
                or 0
            ),

        "results":
            serialized,

        "classification":
            (
                _serialize_classification(
                    _attr(
                        result,
                        "classification",
                    )
                )
                if payload.include_explanation
                or payload.include_query_plan
                else None
            ),

        "rewrite":
            (
                _serialize_rewrite(
                    _attr(
                        result,
                        "rewrite",
                    ),
                    payload.query,
                )
                if payload.include_explanation
                or payload.include_query_plan
                else None
            ),

        "plan":
            (
                _serialize_plan(
                    _attr(
                        result,
                        "plan",
                    )
                )
                if payload.include_query_plan
                else None
            ),

        "timing":
            _serialize_timing(
                _attr(
                    result,
                    "timing",
                )
            ),

        "metadata":
            (
                _attr(
                    result,
                    "metadata",
                    {},
                )
                or {}
            ),
    }


# ============================================================
# HEALTH
#
# Keep static routes before broad dynamic route patterns.
# ============================================================

@router.get(
    "/health/service",
    response_model=SearchHealthResponse,
    summary="Search subsystem health",
)
async def search_health(
    service: Any = Depends(
        get_search_service
    ),
) -> dict[str, Any]:

    health_method = getattr(
        service,
        "health_check",
        None,
    )

    if health_method is None:
        return {
            "status": "unknown",
            "service": "search",
            "bm25": None,
            "vector": None,
            "reranker": None,
            "timestamp": datetime.now(
                timezone.utc
            ),
        }

    result = await health_method()

    return {
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

        "service":
            "search",

        "bm25":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "bm25",
                    )
                )
            ),

        "vector":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "vector",
                    )
                )
            ),

        "reranker":
            _optional_str(
                _enum_value(
                    _attr(
                        result,
                        "reranker",
                    )
                )
            ),

        "timestamp":
            datetime.now(
                timezone.utc
            ),
    }