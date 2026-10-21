"""
Legal RAG Query Router
======================

Routes normalized/rewritten legal queries to the appropriate retrieval
and processing strategy.

Typical flow:

User Query
    ↓
Query Classifier
    ↓
Query Rewriter
    ↓
Query Router
    ├── Hybrid Retrieval
    ├── Vector Retrieval
    ├── BM25 Retrieval
    ├── Metadata Search
    ├── Permission-aware Search
    ├── Multi-query Retrieval
    ├── Contract Comparison
    └── Direct/Clarification Route
    ↓
Reranker
    ↓
Legal LLM / Agents
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueryRoute(str, Enum):
    """Primary processing route for a legal query."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"
    METADATA = "metadata"
    MULTI_QUERY = "multi_query"

    LEGAL_QA = "legal_qa"
    DOCUMENT_SEARCH = "document_search"
    CLAUSE_SEARCH = "clause_search"
    CONTRACT_SEARCH = "contract_search"

    RISK_ANALYSIS = "risk_analysis"
    EXPIRY_ANALYSIS = "expiry_analysis"
    RENEWAL_ANALYSIS = "renewal_analysis"
    OBLIGATION_ANALYSIS = "obligation_analysis"
    COMPARISON = "comparison"
    REVIEW = "review"

    DIRECT = "direct"
    CLARIFICATION = "clarification"


class RetrievalMode(str, Enum):
    NONE = "none"
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    MULTI_QUERY = "multi_query"
    METADATA = "metadata"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QueryRouterRequest:
    query: str

    rewritten_query: str | None = None
    keyword_query: str | None = None
    semantic_query: str | None = None

    classification: str | None = None

    alternative_queries: Sequence[str] = field(default_factory=tuple)

    metadata: Mapping[str, Any] = field(default_factory=dict)

    user_id: str | None = None
    organization_id: str | None = None

    document_ids: Sequence[str] = field(default_factory=tuple)
    contract_ids: Sequence[str] = field(default_factory=tuple)

    top_k: int = 10


@dataclass(slots=True)
class QueryRoutingResult:
    route: QueryRoute
    retrieval_mode: RetrievalMode

    query: str
    retrieval_query: str

    keyword_query: str | None = None
    semantic_query: str | None = None

    alternative_queries: list[str] = field(default_factory=list)

    filters: dict[str, Any] = field(default_factory=dict)

    top_k: int = 10

    use_reranker: bool = True
    use_permissions: bool = True
    require_citations: bool = True

    agent: str | None = None

    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_retrieval(self) -> bool:
        return self.retrieval_mode is not RetrievalMode.NONE

    @property
    def is_agent_route(self) -> bool:
        return self.agent is not None


# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------


RISK_TERMS = (
    "risk",
    "risky",
    "exposure",
    "liability",
    "penalty",
    "compliance risk",
    "legal risk",
    "red flag",
    "unfavorable",
    "dangerous clause",
)

EXPIRY_TERMS = (
    "expire",
    "expires",
    "expiry",
    "expiration",
    "end date",
    "termination date",
    "valid until",
)

RENEWAL_TERMS = (
    "renew",
    "renewal",
    "auto-renew",
    "automatic renewal",
    "renewal date",
    "renewal notice",
    "notice period",
)

OBLIGATION_TERMS = (
    "obligation",
    "obligations",
    "responsibility",
    "responsibilities",
    "must",
    "shall",
    "required to",
    "duty",
    "duties",
    "commitment",
)

COMPARISON_TERMS = (
    "compare",
    "comparison",
    "difference",
    "differences",
    "versus",
    " vs ",
    "similarities",
    "which contract",
    "which agreement",
)

REVIEW_TERMS = (
    "review this",
    "review contract",
    "review agreement",
    "analyze contract",
    "analyse contract",
    "contract review",
    "legal review",
)

CLAUSE_TERMS = (
    "clause",
    "provision",
    "section",
    "termination clause",
    "indemnity",
    "indemnification",
    "confidentiality",
    "limitation of liability",
    "governing law",
    "intellectual property",
    "non-compete",
    "non compete",
    "force majeure",
    "warranty",
)

DOCUMENT_SEARCH_TERMS = (
    "find document",
    "find documents",
    "search document",
    "search documents",
    "show document",
    "show documents",
    "list documents",
)

CONTRACT_SEARCH_TERMS = (
    "find contract",
    "find contracts",
    "search contract",
    "search contracts",
    "show contract",
    "show contracts",
    "list contracts",
)

METADATA_TERMS = (
    "uploaded by",
    "created by",
    "uploaded on",
    "created on",
    "document type",
    "contract type",
    "file name",
    "filename",
    "author",
)

LEGAL_QA_TERMS = (
    "what",
    "which",
    "why",
    "how",
    "when",
    "where",
    "who",
    "explain",
    "summarize",
    "summarise",
    "tell me",
)


CLASSIFICATION_ROUTE_MAP: dict[str, QueryRoute] = {
    "legal_qa": QueryRoute.LEGAL_QA,
    "qa": QueryRoute.LEGAL_QA,

    "document_search": QueryRoute.DOCUMENT_SEARCH,
    "document": QueryRoute.DOCUMENT_SEARCH,

    "contract_search": QueryRoute.CONTRACT_SEARCH,
    "contract": QueryRoute.CONTRACT_SEARCH,

    "clause_search": QueryRoute.CLAUSE_SEARCH,
    "clause": QueryRoute.CLAUSE_SEARCH,

    "risk": QueryRoute.RISK_ANALYSIS,
    "risk_analysis": QueryRoute.RISK_ANALYSIS,

    "expiry": QueryRoute.EXPIRY_ANALYSIS,
    "expiry_analysis": QueryRoute.EXPIRY_ANALYSIS,

    "renewal": QueryRoute.RENEWAL_ANALYSIS,
    "renewal_analysis": QueryRoute.RENEWAL_ANALYSIS,

    "obligation": QueryRoute.OBLIGATION_ANALYSIS,
    "obligations": QueryRoute.OBLIGATION_ANALYSIS,
    "obligation_analysis": QueryRoute.OBLIGATION_ANALYSIS,

    "comparison": QueryRoute.COMPARISON,
    "contract_comparison": QueryRoute.COMPARISON,

    "review": QueryRoute.REVIEW,
    "contract_review": QueryRoute.REVIEW,

    "metadata": QueryRoute.METADATA,

    "keyword": QueryRoute.KEYWORD,
    "vector": QueryRoute.VECTOR,
    "hybrid": QueryRoute.HYBRID,
    "multi_query": QueryRoute.MULTI_QUERY,
}


AGENT_MAP: dict[QueryRoute, str] = {
    QueryRoute.LEGAL_QA: "legal_qa_agent",
    QueryRoute.CLAUSE_SEARCH: "clause_agent",
    QueryRoute.RISK_ANALYSIS: "risk_agent",
    QueryRoute.EXPIRY_ANALYSIS: "expiry_agent",
    QueryRoute.RENEWAL_ANALYSIS: "renewal_agent",
    QueryRoute.OBLIGATION_ANALYSIS: "obligation_agent",
    QueryRoute.COMPARISON: "comparison_agent",
    QueryRoute.REVIEW: "review_agent",
    QueryRoute.DOCUMENT_SEARCH: "document_agent",
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """
    Route legal queries to retrieval strategies and domain agents.

    The router is intentionally deterministic. Classification models may
    supply a label, but safe rule-based routing remains available as a
    fallback.
    """

    def __init__(
        self,
        *,
        default_top_k: int = 10,
        max_top_k: int = 100,
        multi_query_threshold: int = 2,
    ) -> None:

        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")

        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive")

        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.multi_query_threshold = multi_query_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        request: QueryRouterRequest,
    ) -> QueryRoutingResult:

        query = self._validate_query(request.query)

        retrieval_query = (
            request.rewritten_query
            or request.semantic_query
            or query
        ).strip()

        route, confidence, reasons = self._determine_route(
            query=retrieval_query,
            classification=request.classification,
            alternative_queries=request.alternative_queries,
        )

        retrieval_mode = self._select_retrieval_mode(
            route,
            request,
        )

        filters = self._build_filters(request)

        top_k = self._normalize_top_k(request.top_k)

        agent = AGENT_MAP.get(route)

        use_reranker = retrieval_mode in {
            RetrievalMode.HYBRID,
            RetrievalMode.MULTI_QUERY,
            RetrievalMode.VECTOR,
        }

        require_citations = route not in {
            QueryRoute.DIRECT,
            QueryRoute.CLARIFICATION,
        }

        return QueryRoutingResult(
            route=route,
            retrieval_mode=retrieval_mode,
            query=query,
            retrieval_query=retrieval_query,
            keyword_query=request.keyword_query,
            semantic_query=request.semantic_query,
            alternative_queries=list(request.alternative_queries),
            filters=filters,
            top_k=top_k,
            use_reranker=use_reranker,
            use_permissions=True,
            require_citations=require_citations,
            agent=agent,
            confidence=confidence,
            reasons=reasons,
            metadata=dict(request.metadata),
        )

    def route_query(
        self,
        query: str,
        *,
        classification: str | None = None,
        rewritten_query: str | None = None,
        keyword_query: str | None = None,
        semantic_query: str | None = None,
        alternative_queries: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        contract_ids: Sequence[str] | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        top_k: int | None = None,
    ) -> QueryRoutingResult:

        return self.route(
            QueryRouterRequest(
                query=query,
                classification=classification,
                rewritten_query=rewritten_query,
                keyword_query=keyword_query,
                semantic_query=semantic_query,
                alternative_queries=alternative_queries or (),
                metadata=metadata or {},
                document_ids=document_ids or (),
                contract_ids=contract_ids or (),
                user_id=user_id,
                organization_id=organization_id,
                top_k=top_k or self.default_top_k,
            )
        )

    # ------------------------------------------------------------------
    # Route selection
    # ------------------------------------------------------------------

    def _determine_route(
        self,
        *,
        query: str,
        classification: str | None,
        alternative_queries: Sequence[str],
    ) -> tuple[QueryRoute, float, list[str]]:

        reasons: list[str] = []

        # Prefer explicit classifier output when recognized.
        if classification:
            normalized_classification = (
                classification.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            mapped = CLASSIFICATION_ROUTE_MAP.get(
                normalized_classification
            )

            if mapped is not None:
                reasons.append(
                    f"classifier:{normalized_classification}"
                )

                return mapped, 0.98, reasons

        lowered = f" {query.casefold()} "

        # Comparison should be checked early because queries commonly
        # contain other clause/risk terms as well.
        if self._contains_any(lowered, COMPARISON_TERMS):
            reasons.append("comparison terminology detected")
            return QueryRoute.COMPARISON, 0.95, reasons

        if self._contains_any(lowered, REVIEW_TERMS):
            reasons.append("contract review intent detected")
            return QueryRoute.REVIEW, 0.94, reasons

        if self._contains_any(lowered, RENEWAL_TERMS):
            reasons.append("renewal intent detected")
            return QueryRoute.RENEWAL_ANALYSIS, 0.94, reasons

        if self._contains_any(lowered, EXPIRY_TERMS):
            reasons.append("expiry intent detected")
            return QueryRoute.EXPIRY_ANALYSIS, 0.94, reasons

        if self._contains_any(lowered, RISK_TERMS):
            reasons.append("legal risk terminology detected")
            return QueryRoute.RISK_ANALYSIS, 0.92, reasons

        if self._contains_any(lowered, OBLIGATION_TERMS):
            reasons.append("contract obligation terminology detected")
            return QueryRoute.OBLIGATION_ANALYSIS, 0.90, reasons

        if self._contains_any(lowered, CLAUSE_TERMS):
            reasons.append("clause-specific terminology detected")
            return QueryRoute.CLAUSE_SEARCH, 0.90, reasons

        if self._contains_any(lowered, DOCUMENT_SEARCH_TERMS):
            reasons.append("document search intent detected")
            return QueryRoute.DOCUMENT_SEARCH, 0.91, reasons

        if self._contains_any(lowered, CONTRACT_SEARCH_TERMS):
            reasons.append("contract search intent detected")
            return QueryRoute.CONTRACT_SEARCH, 0.91, reasons

        if self._contains_any(lowered, METADATA_TERMS):
            reasons.append("metadata search terminology detected")
            return QueryRoute.METADATA, 0.88, reasons

        if (
            len(alternative_queries)
            >= self.multi_query_threshold
            and self._is_complex_query(query)
        ):
            reasons.append(
                "complex query with multiple retrieval formulations"
            )
            return QueryRoute.MULTI_QUERY, 0.86, reasons

        if self._looks_like_question(query):
            reasons.append("general legal question detected")
            return QueryRoute.LEGAL_QA, 0.82, reasons

        reasons.append("default hybrid retrieval")

        return QueryRoute.HYBRID, 0.75, reasons

    # ------------------------------------------------------------------
    # Retrieval strategy
    # ------------------------------------------------------------------

    def _select_retrieval_mode(
        self,
        route: QueryRoute,
        request: QueryRouterRequest,
    ) -> RetrievalMode:

        if route in {
            QueryRoute.DIRECT,
            QueryRoute.CLARIFICATION,
        }:
            return RetrievalMode.NONE

        if route is QueryRoute.KEYWORD:
            return RetrievalMode.BM25

        if route is QueryRoute.VECTOR:
            return RetrievalMode.VECTOR

        if route is QueryRoute.METADATA:
            return RetrievalMode.METADATA

        if route is QueryRoute.MULTI_QUERY:
            return RetrievalMode.MULTI_QUERY

        if (
            len(request.alternative_queries)
            >= self.multi_query_threshold
            and route in {
                QueryRoute.LEGAL_QA,
                QueryRoute.CLAUSE_SEARCH,
                QueryRoute.RISK_ANALYSIS,
                QueryRoute.REVIEW,
                QueryRoute.COMPARISON,
            }
        ):
            return RetrievalMode.MULTI_QUERY

        return RetrievalMode.HYBRID

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        request: QueryRouterRequest,
    ) -> dict[str, Any]:

        filters: dict[str, Any] = {}

        if request.organization_id:
            filters["organization_id"] = request.organization_id

        if request.document_ids:
            filters["document_ids"] = self._unique(
                request.document_ids
            )

        if request.contract_ids:
            filters["contract_ids"] = self._unique(
                request.contract_ids
            )

        allowed_metadata_filters = {
            "document_type",
            "contract_type",
            "jurisdiction",
            "language",
            "status",
            "risk_level",
            "date_from",
            "date_to",
            "owner_id",
        }

        for key in allowed_metadata_filters:
            value = request.metadata.get(key)

            if value is not None:
                filters[key] = value

        return filters

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _contains_any(
        text: str,
        terms: Sequence[str],
    ) -> bool:

        return any(term.casefold() in text for term in terms)

    @staticmethod
    def _looks_like_question(query: str) -> bool:

        query = query.strip()

        if query.endswith("?"):
            return True

        lowered = query.casefold()

        return any(
            lowered.startswith(f"{term} ")
            for term in LEGAL_QA_TERMS
        )

    @staticmethod
    def _is_complex_query(query: str) -> bool:

        words = query.split()

        if len(words) >= 18:
            return True

        conjunctions = len(
            re.findall(
                r"\b(?:and|or|versus|vs|but|while)\b",
                query,
                flags=re.IGNORECASE,
            )
        )

        if conjunctions >= 2:
            return True

        if query.count("?") > 1:
            return True

        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if "\x00" in query:
            raise ValueError(
                "query cannot contain null characters"
            )

        return query

    def _normalize_top_k(self, top_k: int) -> int:

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            return self.default_top_k

        return min(top_k, self.max_top_k)

    @staticmethod
    def _unique(
        values: Sequence[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = str(value).strip()

            if not value:
                continue

            if value not in seen:
                seen.add(value)
                result.append(value)

        return result


# ---------------------------------------------------------------------------
# Default router
# ---------------------------------------------------------------------------


_default_router = QueryRouter()


def route_query(
    query: str,
    *,
    classification: str | None = None,
    rewritten_query: str | None = None,
    keyword_query: str | None = None,
    semantic_query: str | None = None,
    alternative_queries: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    document_ids: Sequence[str] | None = None,
    contract_ids: Sequence[str] | None = None,
    user_id: str | None = None,
    organization_id: str | None = None,
    top_k: int = 10,
) -> QueryRoutingResult:
    """
    Convenience function using the default router.
    """

    return _default_router.route_query(
        query,
        classification=classification,
        rewritten_query=rewritten_query,
        keyword_query=keyword_query,
        semantic_query=semantic_query,
        alternative_queries=alternative_queries,
        metadata=metadata,
        document_ids=document_ids,
        contract_ids=contract_ids,
        user_id=user_id,
        organization_id=organization_id,
        top_k=top_k,
    )


__all__ = [
    "QueryRoute",
    "RetrievalMode",
    "QueryRouterRequest",
    "QueryRoutingResult",
    "QueryRouter",
    "route_query",
]