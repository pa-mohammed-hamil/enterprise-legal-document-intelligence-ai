# backend/rag/retrieval/vector_search.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, Sequence
import math
import time
import uuid


class VectorSearchStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


class VectorSearchMode(str, Enum):
    DENSE = "dense"
    EXACT = "exact"


@dataclass(slots=True)
class VectorSearchConfig:
    collection_name: str = "legal_documents"

    top_k: int = 20
    score_threshold: Optional[float] = None

    search_mode: VectorSearchMode = VectorSearchMode.DENSE

    with_payload: bool = True
    with_vectors: bool = False

    tenant_filter_required: bool = True
    organization_filter_required: bool = False

    default_namespace: str = "legal"

    max_top_k: int = 100
    timeout_seconds: float = 10.0

    enable_deduplication: bool = True

    # Number of extra results to fetch before local filtering/dedup.
    candidate_multiplier: int = 2


@dataclass(slots=True)
class VectorSearchFilter:
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None

    document_ids: list[str] = field(default_factory=list)
    excluded_document_ids: list[str] = field(default_factory=list)

    document_types: list[str] = field(default_factory=list)
    clause_types: list[str] = field(default_factory=list)

    page_numbers: list[int] = field(default_factory=list)

    parent_chunk_ids: list[str] = field(default_factory=list)

    language: Optional[str] = None
    version: Optional[str] = None
    namespace: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorSearchHit:
    chunk_id: str

    score: float

    text: str

    document_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    title: Optional[str] = None
    section: Optional[str] = None
    clause_type: Optional[str] = None
    document_type: Optional[str] = None

    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None

    vector_id: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    vector: Optional[list[float]] = None

    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "document_id": self.document_id,
            "parent_chunk_id": self.parent_chunk_id,
            "page_number": self.page_number,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "title": self.title,
            "section": self.section,
            "clause_type": self.clause_type,
            "document_type": self.document_type,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "vector_id": self.vector_id,
            "metadata": self.metadata,
            "vector": self.vector,
            "rank": self.rank,
        }


@dataclass(slots=True)
class VectorSearchResult:
    status: VectorSearchStatus

    query: str

    hits: list[VectorSearchHit]

    total_hits: int

    collection_name: str

    top_k: int

    latency_ms: float

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    embedding_dimension: Optional[int] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None

    @property
    def documents(self) -> list[VectorSearchHit]:
        return self.hits

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "total_hits": self.total_hits,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "latency_ms": self.latency_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dimension": self.embedding_dimension,
            "metadata": self.metadata,
            "error": self.error,
        }


class VectorSearchError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "VECTOR_SEARCH_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class EmbeddingServiceProtocol(Protocol):
    def embed_query(
        self,
        text: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        ...


class VectorSearch:
    """
    Dense vector retrieval layer for Legal RAG.

    Pipeline:

        User Query
            ↓
        QueryClassifier
            ↓
        QueryRewriter
            ↓
        EmbeddingService
            ↓
        VectorSearch
            ↓
        Qdrant
            ↓
        MetadataFilter
            ↓
        PermissionAwareSearch
            ↓
        Reranker
            ↓
        Legal LLM + Citations

    Security note:
        tenant_id / organization_id filtering should happen inside
        the vector database query, not only after retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        *,
        vector_client: Any,
        config: Optional[VectorSearchConfig] = None,
    ) -> None:
        if embedding_service is None:
            raise ValueError("embedding_service is required.")

        if vector_client is None:
            raise ValueError("vector_client is required.")

        self.embedding_service = embedding_service
        self.vector_client = vector_client

        self.config = config or VectorSearchConfig()

        self._validate_config()

    # ==========================================================
    # Main text search
    # ==========================================================

    def search(
        self,
        query: str,
        *,
        filters: Optional[VectorSearchFilter] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> VectorSearchResult:

        started = time.perf_counter()

        query = self._prepare_query(query)

        if not query:
            return VectorSearchResult(
                status=VectorSearchStatus.EMPTY,
                query="",
                hits=[],
                total_hits=0,
                collection_name=self.config.collection_name,
                top_k=0,
                latency_ms=self._elapsed(started),
            )

        resolved_filter = filters or VectorSearchFilter()

        self._validate_security_filters(
            resolved_filter
        )

        resolved_top_k = self._resolve_top_k(top_k)

        threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )

        try:
            query_vector = self.embedding_service.embed_query(
                query,
                metadata={
                    "operation": "vector_search",
                    **(metadata or {}),
                },
            )

            self._validate_vector(query_vector)

            result = self.search_by_vector(
                query_vector,
                query=query,
                filters=resolved_filter,
                top_k=resolved_top_k,
                score_threshold=threshold,
                metadata=metadata,
                _started=started,
            )

            result.model_name = getattr(
                self.embedding_service,
                "model_name",
                None,
            )

            result.model_version = getattr(
                self.embedding_service,
                "model_version",
                None,
            )

            result.embedding_dimension = len(query_vector)

            return result

        except VectorSearchError:
            raise

        except Exception as exc:
            raise VectorSearchError(
                "Vector search failed.",
                code="VECTOR_SEARCH_FAILED",
                cause=exc,
            ) from exc

    # ==========================================================
    # Search using existing vector
    # ==========================================================

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        *,
        query: str = "",
        filters: Optional[VectorSearchFilter] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        _started: Optional[float] = None,
    ) -> VectorSearchResult:

        started = _started or time.perf_counter()

        vector = [
            float(value)
            for value in query_vector
        ]

        self._validate_vector(vector)

        resolved_filter = filters or VectorSearchFilter()

        self._validate_security_filters(
            resolved_filter
        )

        resolved_top_k = self._resolve_top_k(top_k)

        threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )

        backend_filter = self._build_filter(
            resolved_filter
        )

        backend_limit = min(
            self.config.max_top_k,
            max(
                resolved_top_k,
                resolved_top_k
                * self.config.candidate_multiplier,
            ),
        )

        try:
            raw_results = self._qdrant_search(
                vector=vector,
                query_filter=backend_filter,
                limit=backend_limit,
                score_threshold=threshold,
            )

            hits = [
                self._to_hit(item)
                for item in raw_results
            ]

            hits = self._apply_local_filters(
                hits,
                resolved_filter,
            )

            if threshold is not None:
                hits = [
                    hit
                    for hit in hits
                    if hit.score >= threshold
                ]

            hits.sort(
                key=lambda item: item.score,
                reverse=True,
            )

            if self.config.enable_deduplication:
                hits = self._deduplicate(hits)

            hits = hits[:resolved_top_k]

            for rank, hit in enumerate(
                hits,
                start=1,
            ):
                hit.rank = rank

            status = (
                VectorSearchStatus.SUCCESS
                if hits
                else VectorSearchStatus.EMPTY
            )

            return VectorSearchResult(
                status=status,
                query=query,
                hits=hits,
                total_hits=len(hits),
                collection_name=self.config.collection_name,
                top_k=resolved_top_k,
                latency_ms=self._elapsed(started),
                embedding_dimension=len(vector),
                metadata={
                    **(metadata or {}),
                    "candidate_count": len(raw_results),
                    "filter": self._filter_metadata(
                        resolved_filter
                    ),
                },
            )

        except VectorSearchError:
            raise

        except Exception as exc:
            raise VectorSearchError(
                "Vector database query failed.",
                code="VECTOR_DATABASE_QUERY_FAILED",
                cause=exc,
            ) from exc

    # ==========================================================
    # Qdrant adapter
    # ==========================================================

    def _qdrant_search(
        self,
        *,
        vector: Sequence[float],
        query_filter: Any,
        limit: int,
        score_threshold: Optional[float],
    ) -> list[Any]:
        """
        Supports both newer Qdrant `query_points()` and older
        `search()` clients.
        """

        client = self.vector_client

        # Newer qdrant-client API.
        if hasattr(client, "query_points"):
            kwargs: dict[str, Any] = {
                "collection_name": self.config.collection_name,
                "query": list(vector),
                "limit": limit,
                "with_payload": self.config.with_payload,
                "with_vectors": self.config.with_vectors,
            }

            if query_filter is not None:
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            response = client.query_points(**kwargs)

            if hasattr(response, "points"):
                return list(response.points)

            if isinstance(response, (list, tuple)):
                return list(response)

            return []

        # Older qdrant-client API.
        if hasattr(client, "search"):
            kwargs = {
                "collection_name": self.config.collection_name,
                "query_vector": list(vector),
                "limit": limit,
                "with_payload": self.config.with_payload,
                "with_vectors": self.config.with_vectors,
            }

            if query_filter is not None:
                kwargs["query_filter"] = query_filter

            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold

            response = client.search(**kwargs)

            return list(response or [])

        # Custom vector adapter.
        if hasattr(client, "vector_search"):
            response = client.vector_search(
                collection_name=self.config.collection_name,
                vector=list(vector),
                filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            return list(response or [])

        raise VectorSearchError(
            "Vector client does not expose query_points(), "
            "search(), or vector_search().",
            code="UNSUPPORTED_VECTOR_CLIENT",
        )

    # ==========================================================
    # Qdrant filter construction
    # ==========================================================

    def _build_filter(
        self,
        filters: VectorSearchFilter,
    ) -> Any:
        """
        Builds a Qdrant Filter when qdrant-client is installed.

        If Qdrant models are unavailable, a generic dictionary is
        returned for custom adapters.
        """

        must_values: list[
            tuple[str, Any]
        ] = []

        should_values: list[
            tuple[str, list[Any]]
        ] = []

        must_not_values: list[
            tuple[str, list[Any]]
        ] = []

        if filters.tenant_id:
            must_values.append(
                ("tenant_id", filters.tenant_id)
            )

        if filters.organization_id:
            must_values.append(
                (
                    "organization_id",
                    filters.organization_id,
                )
            )

        if filters.language:
            must_values.append(
                ("language", filters.language)
            )

        if filters.version:
            must_values.append(
                ("version", filters.version)
            )

        namespace = (
            filters.namespace
            or self.config.default_namespace
        )

        if namespace:
            must_values.append(
                ("namespace", namespace)
            )

        if filters.document_ids:
            should_values.append(
                (
                    "document_id",
                    filters.document_ids,
                )
            )

        if filters.document_types:
            should_values.append(
                (
                    "document_type",
                    filters.document_types,
                )
            )

        if filters.clause_types:
            should_values.append(
                (
                    "clause_type",
                    filters.clause_types,
                )
            )

        if filters.page_numbers:
            should_values.append(
                (
                    "page_number",
                    filters.page_numbers,
                )
            )

        if filters.parent_chunk_ids:
            should_values.append(
                (
                    "parent_chunk_id",
                    filters.parent_chunk_ids,
                )
            )

        if filters.excluded_document_ids:
            must_not_values.append(
                (
                    "document_id",
                    filters.excluded_document_ids,
                )
            )

        for key, value in filters.metadata.items():
            if value is None:
                continue

            if isinstance(
                value,
                (list, tuple, set),
            ):
                should_values.append(
                    (
                        f"metadata.{key}",
                        list(value),
                    )
                )
            else:
                must_values.append(
                    (
                        f"metadata.{key}",
                        value,
                    )
                )

        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchAny,
                MatchValue,
            )

            must = [
                FieldCondition(
                    key=key,
                    match=MatchValue(
                        value=value
                    ),
                )
                for key, value
                in must_values
            ]

            # Important:
            # each multi-value field is added to MUST with MatchAny.
            # This means:
            #
            # document_id IN [...]
            # AND document_type IN [...]
            #
            # rather than a global OR across unrelated fields.
            for key, values in should_values:
                must.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(
                            any=values
                        ),
                    )
                )

            must_not = []

            for key, values in must_not_values:
                must_not.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(
                            any=values
                        ),
                    )
                )

            if not must and not must_not:
                return None

            return Filter(
                must=must or None,
                must_not=must_not or None,
            )

        except ImportError:
            # Generic representation for tests/custom clients.
            return {
                "must": [
                    {
                        "key": key,
                        "value": value,
                    }
                    for key, value
                    in must_values
                ],
                "match_any": [
                    {
                        "key": key,
                        "values": values,
                    }
                    for key, values
                    in should_values
                ],
                "must_not": [
                    {
                        "key": key,
                        "values": values,
                    }
                    for key, values
                    in must_not_values
                ],
            }

    # ==========================================================
    # Backend result conversion
    # ==========================================================

    def _to_hit(
        self,
        item: Any,
    ) -> VectorSearchHit:

        if isinstance(item, dict):
            payload = item.get(
                "payload",
                {},
            ) or {}

            score = self._safe_float(
                item.get("score", 0.0)
            )

            vector_id = item.get("id")

            vector = item.get("vector")

        else:
            payload = getattr(
                item,
                "payload",
                {},
            ) or {}

            score = self._safe_float(
                getattr(
                    item,
                    "score",
                    0.0,
                )
            )

            vector_id = getattr(
                item,
                "id",
                None,
            )

            vector = getattr(
                item,
                "vector",
                None,
            )

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = (
            payload.get("chunk_id")
            or metadata.get("chunk_id")
            or vector_id
            or uuid.uuid4().hex
        )

        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("page_content")
            or metadata.get("text")
            or ""
        )

        return VectorSearchHit(
            chunk_id=str(chunk_id),
            score=score,
            text=str(text),
            document_id=self._optional_string(
                payload.get("document_id")
                or metadata.get("document_id")
            ),
            parent_chunk_id=self._optional_string(
                payload.get("parent_chunk_id")
                or metadata.get("parent_chunk_id")
            ),
            page_number=self._optional_int(
                payload.get("page_number")
                or metadata.get("page_number")
            ),
            page_start=self._optional_int(
                payload.get("page_start")
                or metadata.get("page_start")
            ),
            page_end=self._optional_int(
                payload.get("page_end")
                or metadata.get("page_end")
            ),
            title=self._optional_string(
                payload.get("title")
                or metadata.get("title")
            ),
            section=self._optional_string(
                payload.get("section")
                or payload.get("section_number")
                or metadata.get("section")
            ),
            clause_type=self._optional_string(
                payload.get("clause_type")
                or metadata.get("clause_type")
            ),
            document_type=self._optional_string(
                payload.get("document_type")
                or metadata.get("document_type")
            ),
            tenant_id=self._optional_string(
                payload.get("tenant_id")
                or metadata.get("tenant_id")
            ),
            organization_id=self._optional_string(
                payload.get("organization_id")
                or metadata.get("organization_id")
            ),
            vector_id=(
                str(vector_id)
                if vector_id is not None
                else None
            ),
            metadata=metadata,
            vector=self._vector_to_list(
                vector
            ),
        )

    # ==========================================================
    # Local defense-in-depth filtering
    # ==========================================================

    def _apply_local_filters(
        self,
        hits: Sequence[VectorSearchHit],
        filters: VectorSearchFilter,
    ) -> list[VectorSearchHit]:

        output: list[
            VectorSearchHit
        ] = []

        for hit in hits:
            if (
                filters.tenant_id
                and hit.tenant_id
                != filters.tenant_id
            ):
                continue

            if (
                filters.organization_id
                and hit.organization_id
                != filters.organization_id
            ):
                continue

            if (
                filters.document_ids
                and hit.document_id
                not in filters.document_ids
            ):
                continue

            if (
                filters.excluded_document_ids
                and hit.document_id
                in filters.excluded_document_ids
            ):
                continue

            if (
                filters.document_types
                and hit.document_type
                not in filters.document_types
            ):
                continue

            if (
                filters.clause_types
                and hit.clause_type
                not in filters.clause_types
            ):
                continue

            if (
                filters.parent_chunk_ids
                and hit.parent_chunk_id
                not in filters.parent_chunk_ids
            ):
                continue

            if filters.page_numbers:
                pages = {
                    page
                    for page in (
                        hit.page_number,
                        hit.page_start,
                        hit.page_end,
                    )
                    if page is not None
                }

                if not pages.intersection(
                    filters.page_numbers
                ):
                    continue

            output.append(hit)

        return output

    # ==========================================================
    # Deduplication
    # ==========================================================

    @staticmethod
    def _deduplicate(
        hits: Sequence[VectorSearchHit],
    ) -> list[VectorSearchHit]:

        output: list[
            VectorSearchHit
        ] = []

        seen: set[str] = set()

        for hit in hits:
            identity = (
                hit.chunk_id
                or hit.vector_id
                or (
                    f"{hit.document_id}:"
                    f"{hit.page_number}:"
                    f"{hit.text}"
                )
            )

            if identity in seen:
                continue

            seen.add(identity)

            output.append(hit)

        return output

    # ==========================================================
    # Security
    # ==========================================================

    def _validate_security_filters(
        self,
        filters: VectorSearchFilter,
    ) -> None:

        if (
            self.config.tenant_filter_required
            and not filters.tenant_id
        ):
            raise VectorSearchError(
                "tenant_id is required for vector search.",
                code="TENANT_FILTER_REQUIRED",
            )

        if (
            self.config.organization_filter_required
            and not filters.organization_id
        ):
            raise VectorSearchError(
                "organization_id is required for vector search.",
                code="ORGANIZATION_FILTER_REQUIRED",
            )

    # ==========================================================
    # Validation
    # ==========================================================

    def _resolve_top_k(
        self,
        top_k: Optional[int],
    ) -> int:

        value = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        if value <= 0:
            raise VectorSearchError(
                "top_k must be greater than zero.",
                code="INVALID_TOP_K",
            )

        return min(
            value,
            self.config.max_top_k,
        )

    @staticmethod
    def _validate_vector(
        vector: Sequence[float],
    ) -> None:

        if not vector:
            raise VectorSearchError(
                "Query embedding is empty.",
                code="EMPTY_QUERY_VECTOR",
            )

        for value in vector:
            try:
                number = float(value)
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise VectorSearchError(
                    "Query embedding contains a non-numeric value.",
                    code="INVALID_QUERY_VECTOR",
                    cause=exc,
                ) from exc

            if not math.isfinite(number):
                raise VectorSearchError(
                    "Query embedding contains NaN or infinity.",
                    code="INVALID_QUERY_VECTOR",
                )

    def _validate_config(
        self,
    ) -> None:

        if not self.config.collection_name:
            raise ValueError(
                "collection_name is required."
            )

        if self.config.top_k <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        if self.config.max_top_k <= 0:
            raise ValueError(
                "max_top_k must be > 0."
            )

        if (
            self.config.candidate_multiplier
            <= 0
        ):
            raise ValueError(
                "candidate_multiplier must be > 0."
            )

    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _prepare_query(
        query: Any,
    ) -> str:

        if query is None:
            return ""

        return str(query).strip()

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:

        try:
            result = float(value)

            if math.isfinite(result):
                return result

        except (
            TypeError,
            ValueError,
        ):
            pass

        return 0.0

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        value = str(value)

        return value if value else None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> Optional[int]:

        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _vector_to_list(
        vector: Any,
    ) -> Optional[list[float]]:

        if vector is None:
            return None

        if isinstance(vector, dict):
            # Named-vector payload. Leave it out of the generic
            # single-vector response.
            return None

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        try:
            return [
                float(value)
                for value in vector
            ]

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _filter_metadata(
        filters: VectorSearchFilter,
    ) -> dict[str, Any]:

        return {
            "tenant_id": filters.tenant_id,
            "organization_id":
                filters.organization_id,
            "document_ids":
                filters.document_ids,
            "excluded_document_ids":
                filters.excluded_document_ids,
            "document_types":
                filters.document_types,
            "clause_types":
                filters.clause_types,
            "page_numbers":
                filters.page_numbers,
            "parent_chunk_ids":
                filters.parent_chunk_ids,
            "language":
                filters.language,
            "version":
                filters.version,
            "namespace":
                filters.namespace,
        }

    @staticmethod
    def _elapsed(
        started: float,
    ) -> float:

        return round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            3,
        )


def vector_search(
    query: str,
    *,
    embedding_service: EmbeddingServiceProtocol,
    vector_client: Any,
    tenant_id: str,
    organization_id: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    top_k: int = 20,
    collection_name: str = "legal_documents",
) -> VectorSearchResult:
    """
    Convenience API for simple retrieval.
    """

    searcher = VectorSearch(
        embedding_service,
        vector_client=vector_client,
        config=VectorSearchConfig(
            collection_name=collection_name,
            top_k=top_k,
        ),
    )

    return searcher.search(
        query,
        filters=VectorSearchFilter(
            tenant_id=tenant_id,
            organization_id=organization_id,
            document_ids=list(
                document_ids or []
            ),
        ),
    )