# backend/app/schemas/search.py

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# Enums
# ============================================================


class SearchMode(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    VECTOR = "vector"
    HYBRID = "hybrid"
    LEGAL = "legal"


class SearchEntityType(str, Enum):
    DOCUMENT = "document"
    CONTRACT = "contract"
    CLAUSE = "clause"
    OBLIGATION = "obligation"
    RISK = "risk"
    RENEWAL = "renewal"
    REVIEW = "review"


class SearchSortBy(str, Enum):
    RELEVANCE = "relevance"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    TITLE = "title"
    RISK_SCORE = "risk_score"
    EXPIRY_DATE = "expiry_date"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class SearchMatchType(str, Enum):
    EXACT = "exact"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# ============================================================
# Filters
# ============================================================


class SearchDateRange(BaseModel):
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "SearchDateRange":
        if self.start and self.end and self.start > self.end:
            raise ValueError(
                "start date cannot be after end date"
            )

        return self


class SearchFilters(BaseModel):
    organization_id: UUID | None = None

    entity_types: list[SearchEntityType] | None = None

    document_ids: list[UUID] | None = None
    contract_ids: list[UUID] | None = None
    clause_ids: list[UUID] | None = None

    document_types: list[str] | None = None
    contract_types: list[str] | None = None
    clause_types: list[str] | None = None

    statuses: list[str] | None = None

    risk_levels: list[str] | None = None

    tags: list[str] | None = None

    languages: list[str] | None = None

    jurisdictions: list[str] | None = None

    counterparties: list[str] | None = None

    created_range: SearchDateRange | None = None
    updated_range: SearchDateRange | None = None
    expiry_range: SearchDateRange | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Search request
# ============================================================


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    mode: SearchMode = SearchMode.HYBRID

    filters: SearchFilters = Field(
        default_factory=SearchFilters
    )

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    sort_by: SearchSortBy = SearchSortBy.RELEVANCE

    sort_order: SortOrder = SortOrder.DESC

    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    include_content: bool = True

    include_highlights: bool = True

    include_metadata: bool = True

    include_scores: bool = True


# ============================================================
# Search scoring
# ============================================================


class SearchScores(BaseModel):
    score: float = Field(
        default=0.0,
        ge=0.0,
    )

    normalized_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    bm25_score: float | None = None

    vector_score: float | None = None

    reranker_score: float | None = None


# ============================================================
# Highlight
# ============================================================


class SearchHighlight(BaseModel):
    field: str

    text: str

    start_offset: int | None = Field(
        default=None,
        ge=0,
    )

    end_offset: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# Citation / source
# ============================================================


class SearchCitation(BaseModel):
    document_id: UUID

    document_title: str | None = None

    contract_id: UUID | None = None

    clause_id: UUID | None = None

    chunk_id: UUID | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    clause_number: str | None = None

    source_text: str | None = None


# ============================================================
# Search result
# ============================================================


class SearchResult(BaseModel):
    id: UUID

    entity_type: SearchEntityType

    title: str | None = None

    content: str | None = None

    summary: str | None = None

    match_type: SearchMatchType

    scores: SearchScores = Field(
        default_factory=SearchScores
    )

    highlights: list[SearchHighlight] = Field(
        default_factory=list
    )

    citation: SearchCitation | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime | None = None

    updated_at: datetime | None = None


# ============================================================
# Search response
# ============================================================


class SearchResponse(BaseModel):
    query: str

    mode: SearchMode

    results: list[SearchResult] = Field(
        default_factory=list
    )

    total: int = Field(
        default=0,
        ge=0,
    )

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
    )

    pages: int = Field(
        default=0,
        ge=0,
    )

    processing_time_ms: float | None = None

    rewritten_query: str | None = None


# ============================================================
# Vector search
# ============================================================


class VectorSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    filters: SearchFilters = Field(
        default_factory=SearchFilters
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    include_vectors: bool = False


class VectorSearchResult(BaseModel):
    id: UUID

    document_id: UUID

    chunk_id: UUID | None = None

    contract_id: UUID | None = None

    clause_id: UUID | None = None

    content: str

    vector_score: float

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    citation: SearchCitation | None = None


class VectorSearchResponse(BaseModel):
    query: str

    results: list[VectorSearchResult] = Field(
        default_factory=list
    )

    total: int = 0

    processing_time_ms: float | None = None


# ============================================================
# Hybrid search
# ============================================================


class HybridSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    filters: SearchFilters = Field(
        default_factory=SearchFilters
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    bm25_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
    )

    vector_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    rerank: bool = True

    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "HybridSearchRequest":
        total = self.bm25_weight + self.vector_weight

        if abs(total - 1.0) > 0.001:
            raise ValueError(
                "bm25_weight + vector_weight must equal 1.0"
            )

        return self


class HybridSearchResult(SearchResult):
    bm25_rank: int | None = None
    vector_rank: int | None = None
    reranker_rank: int | None = None


class HybridSearchResponse(BaseModel):
    query: str

    results: list[HybridSearchResult] = Field(
        default_factory=list
    )

    total: int = 0

    processing_time_ms: float | None = None


# ============================================================
# Multi-query search
# ============================================================


class MultiQuerySearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    number_of_queries: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    filters: SearchFilters = Field(
        default_factory=SearchFilters
    )


class GeneratedSearchQuery(BaseModel):
    query: str

    reason: str | None = None


class MultiQuerySearchResponse(BaseModel):
    original_query: str

    generated_queries: list[GeneratedSearchQuery] = Field(
        default_factory=list
    )

    results: list[SearchResult] = Field(
        default_factory=list
    )

    total: int = 0

    processing_time_ms: float | None = None


# ============================================================
# Reranking
# ============================================================


class RerankDocument(BaseModel):
    id: UUID

    text: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RerankRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    documents: list[RerankDocument] = Field(
        ...,
        min_length=1,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )


class RerankResult(BaseModel):
    id: UUID

    score: float

    rank: int = Field(
        ge=1,
    )

    text: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RerankResponse(BaseModel):
    query: str

    results: list[RerankResult] = Field(
        default_factory=list
    )

    model_name: str | None = None

    processing_time_ms: float | None = None


# ============================================================
# Permission-aware search
# ============================================================


class PermissionAwareSearchRequest(SearchRequest):
    user_id: UUID | None = None

    enforce_permissions: bool = True


class SearchPermissionInfo(BaseModel):
    can_view: bool = True

    can_download: bool = False

    can_edit: bool = False

    permission_source: str | None = None


class PermissionAwareSearchResult(SearchResult):
    permissions: SearchPermissionInfo = Field(
        default_factory=SearchPermissionInfo
    )


class PermissionAwareSearchResponse(BaseModel):
    query: str

    results: list[PermissionAwareSearchResult] = Field(
        default_factory=list
    )

    total: int = 0

    filtered_by_permissions: int = 0

    processing_time_ms: float | None = None


# ============================================================
# Search suggestions
# ============================================================


class SearchSuggestionRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
    )


class SearchSuggestion(BaseModel):
    text: str

    entity_type: SearchEntityType | None = None

    score: float | None = None


class SearchSuggestionResponse(BaseModel):
    suggestions: list[SearchSuggestion] = Field(
        default_factory=list
    )


# ============================================================
# Facets
# ============================================================


class SearchFacetValue(BaseModel):
    value: str

    count: int = Field(
        default=0,
        ge=0,
    )


class SearchFacet(BaseModel):
    field: str

    values: list[SearchFacetValue] = Field(
        default_factory=list
    )


class SearchFacetsResponse(BaseModel):
    facets: list[SearchFacet] = Field(
        default_factory=list
    )


# ============================================================
# Search analytics
# ============================================================


class SearchAnalyticsEvent(BaseModel):
    query: str

    mode: SearchMode

    result_count: int = Field(
        default=0,
        ge=0,
    )

    clicked_result_id: UUID | None = None

    processing_time_ms: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# Search history
# ============================================================


class SearchHistoryItem(BaseModel):
    id: UUID

    query: str

    mode: SearchMode

    result_count: int = 0

    created_at: datetime


class SearchHistoryResponse(BaseModel):
    items: list[SearchHistoryItem] = Field(
        default_factory=list
    )

    total: int = Field(
        default=0,
        ge=0,
    )