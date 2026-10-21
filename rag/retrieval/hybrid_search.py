# backend/app/rag/retrieval/hybrid_search.py

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

from backend.rag.retrieval.bm25 import (
    BM25Retriever,
    BM25SearchResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================


class HybridSearchError(Exception):
    """Base hybrid-search error."""


class HybridSearchValidationError(HybridSearchError):
    """Invalid hybrid-search request/configuration."""


class HybridSearchBackendError(HybridSearchError):
    """A retrieval backend failed."""


# ============================================================
# Fusion mode
# ============================================================


class FusionMode(str, Enum):
    WEIGHTED = "weighted"
    RRF = "rrf"


# ============================================================
# Result
# ============================================================


@dataclass(slots=True)
class HybridSearchResult:
    id: str
    text: str
    score: float
    rank: int

    bm25_score: float = 0.0
    vector_score: float = 0.0

    normalized_bm25_score: float = 0.0
    normalized_vector_score: float = 0.0

    bm25_rank: int | None = None
    vector_rank: int | None = None

    organization_id: str | None = None
    document_id: str | None = None
    contract_id: str | None = None
    parent_id: str | None = None

    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None

    matched_terms: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "normalized_bm25_score":
                self.normalized_bm25_score,
            "normalized_vector_score":
                self.normalized_vector_score,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "organization_id": self.organization_id,
            "document_id": self.document_id,
            "contract_id": self.contract_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "matched_terms": self.matched_terms,
            "sources": self.sources,
            "metadata": self.metadata,
        }


# ============================================================
# Internal candidate
# ============================================================


@dataclass(slots=True)
class _Candidate:
    id: str

    text: str = ""

    organization_id: str | None = None
    document_id: str | None = None
    contract_id: str | None = None
    parent_id: str | None = None

    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    bm25_score: float = 0.0
    vector_score: float = 0.0

    normalized_bm25_score: float = 0.0
    normalized_vector_score: float = 0.0

    bm25_rank: int | None = None
    vector_rank: int | None = None

    matched_terms: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    final_score: float = 0.0


# ============================================================
# Hybrid Search
# ============================================================


class HybridSearch:
    """
    Enterprise hybrid lexical + semantic retrieval.

    Pipeline
    --------
        query
          │
          ├── BM25Retriever
          │       └── lexical candidates
          │
          └── VectorSearch
                  └── semantic candidates
                         │
                         ▼
                  score normalization
                         │
                         ▼
                  weighted fusion / RRF
                         │
                         ▼
                    deduplication
                         │
                         ▼
                  hybrid candidates
                         │
                         ▼
              permission-aware search
                         │
                         ▼
                    reranker
                         │
                         ▼
                Legal RAG generation

    Default weighted fusion:

        hybrid_score =
            bm25_weight * normalized_bm25
            +
            vector_weight * normalized_vector

    Default:
        BM25  = 0.40
        Vector = 0.60
    """

    def __init__(
        self,
        *,
        bm25: BM25Retriever,
        vector_search: Any,
        bm25_weight: float = 0.40,
        vector_weight: float = 0.60,
        fusion_mode: FusionMode | str = FusionMode.WEIGHTED,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        self.bm25 = bm25
        self.vector_search = vector_search

        self.bm25_weight = float(bm25_weight)
        self.vector_weight = float(vector_weight)

        self.fusion_mode = FusionMode(fusion_mode)

        self.rrf_k = max(1, int(rrf_k))

        self.candidate_multiplier = max(
            1,
            int(candidate_multiplier),
        )

        self._validate_weights(
            self.bm25_weight,
            self.vector_weight,
        )

    # ========================================================
    # Main search
    # ========================================================

    async def search(
        self,
        query: str,
        *,
        organization_id: UUID | str | None = None,
        top_k: int = 10,
        candidate_k: int | None = None,
        document_ids: Sequence[UUID | str] | None = None,
        contract_ids: Sequence[UUID | str] | None = None,
        filters: Mapping[str, Any] | None = None,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        fusion_mode: FusionMode | str | None = None,
        min_score: float = 0.0,
        vector: Sequence[float] | None = None,
    ) -> list[HybridSearchResult]:
        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            return []

        candidate_k = (
            candidate_k
            or top_k * self.candidate_multiplier
        )

        candidate_k = max(
            top_k,
            candidate_k,
        )

        lexical_weight = (
            self.bm25_weight
            if bm25_weight is None
            else float(bm25_weight)
        )

        semantic_weight = (
            self.vector_weight
            if vector_weight is None
            else float(vector_weight)
        )

        self._validate_weights(
            lexical_weight,
            semantic_weight,
        )

        mode = FusionMode(
            fusion_mode or self.fusion_mode
        )

        bm25_task = asyncio.create_task(
            self._run_bm25(
                query=query,
                organization_id=organization_id,
                top_k=candidate_k,
                document_ids=document_ids,
                contract_ids=contract_ids,
                filters=filters,
            )
        )

        vector_task = asyncio.create_task(
            self._run_vector_search(
                query=query,
                vector=vector,
                organization_id=organization_id,
                top_k=candidate_k,
                document_ids=document_ids,
                contract_ids=contract_ids,
                filters=filters,
            )
        )

        bm25_results, vector_results = (
            await asyncio.gather(
                bm25_task,
                vector_task,
            )
        )

        candidates = self._merge_candidates(
            bm25_results=bm25_results,
            vector_results=vector_results,
        )

        if not candidates:
            return []

        self._normalize_scores(candidates)

        if mode == FusionMode.WEIGHTED:
            self._weighted_fusion(
                candidates,
                bm25_weight=lexical_weight,
                vector_weight=semantic_weight,
            )

        elif mode == FusionMode.RRF:
            self._rrf_fusion(
                candidates,
                bm25_weight=lexical_weight,
                vector_weight=semantic_weight,
            )

        else:
            raise HybridSearchValidationError(
                f"Unsupported fusion mode: {mode}"
            )

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.final_score,
                candidate.normalized_vector_score,
                candidate.normalized_bm25_score,
            ),
            reverse=True,
        )

        results: list[HybridSearchResult] = []

        for candidate in ranked:
            if candidate.final_score < min_score:
                continue

            results.append(
                self._to_result(
                    candidate,
                    rank=len(results) + 1,
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # Search dictionaries
    # ========================================================

    async def search_dicts(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        results = await self.search(
            query,
            **kwargs,
        )

        return [
            result.to_dict()
            for result in results
        ]

    # ========================================================
    # Batch search
    # ========================================================

    async def batch_search(
        self,
        queries: Sequence[str],
        *,
        organization_id: UUID | str | None = None,
        top_k: int = 10,
        document_ids: Sequence[UUID | str] | None = None,
        contract_ids: Sequence[UUID | str] | None = None,
        filters: Mapping[str, Any] | None = None,
        fusion_mode: FusionMode | str | None = None,
    ) -> dict[str, list[HybridSearchResult]]:
        tasks = [
            self.search(
                query,
                organization_id=organization_id,
                top_k=top_k,
                document_ids=document_ids,
                contract_ids=contract_ids,
                filters=filters,
                fusion_mode=fusion_mode,
            )
            for query in queries
        ]

        results = await asyncio.gather(*tasks)

        return {
            query: result
            for query, result in zip(
                queries,
                results,
                strict=True,
            )
        }

    # ========================================================
    # BM25
    # ========================================================

    async def _run_bm25(
        self,
        *,
        query: str,
        organization_id: UUID | str | None,
        top_k: int,
        document_ids: Sequence[UUID | str] | None,
        contract_ids: Sequence[UUID | str] | None,
        filters: Mapping[str, Any] | None,
    ) -> list[Any]:
        try:
            method = getattr(
                self.bm25,
                "search",
                None,
            )

            if method is None:
                raise HybridSearchBackendError(
                    "BM25 backend exposes no search() method"
                )

            kwargs = {
                "top_k": top_k,
                "organization_id": organization_id,
                "document_ids": document_ids,
                "contract_ids": contract_ids,
                "filters": filters,
            }

            if inspect.iscoroutinefunction(method):
                result = await method(
                    query,
                    **kwargs,
                )
            else:
                # BM25 implementation is CPU/local synchronous.
                # For larger indexes this prevents blocking the
                # FastAPI event loop.
                result = await asyncio.to_thread(
                    method,
                    query,
                    **kwargs,
                )

            return list(result or [])

        except Exception as exc:
            logger.exception(
                "BM25 retrieval failed"
            )

            raise HybridSearchBackendError(
                "BM25 retrieval failed"
            ) from exc

    # ========================================================
    # Vector search
    # ========================================================

    async def _run_vector_search(
        self,
        *,
        query: str,
        vector: Sequence[float] | None,
        organization_id: UUID | str | None,
        top_k: int,
        document_ids: Sequence[UUID | str] | None,
        contract_ids: Sequence[UUID | str] | None,
        filters: Mapping[str, Any] | None,
    ) -> list[Any]:
        if self.vector_search is None:
            return []

        methods = (
            "search",
            "similarity_search",
            "query",
            "retrieve",
        )

        last_error: Exception | None = None

        for method_name in methods:
            method = getattr(
                self.vector_search,
                method_name,
                None,
            )

            if method is None:
                continue

            kwargs = {
                "organization_id": organization_id,
                "top_k": top_k,
                "document_ids": document_ids,
                "contract_ids": contract_ids,
                "filters": filters,
            }

            try:
                result = await self._call_vector_method(
                    method=method,
                    query=query,
                    vector=vector,
                    kwargs=kwargs,
                )

                return list(result or [])

            except TypeError as exc:
                last_error = exc
                continue

            except Exception as exc:
                logger.exception(
                    "Vector retrieval failed"
                )

                raise HybridSearchBackendError(
                    "Vector retrieval failed"
                ) from exc

        if last_error:
            raise HybridSearchBackendError(
                "Vector backend has no compatible "
                "search signature"
            ) from last_error

        raise HybridSearchBackendError(
            "Vector backend exposes no supported "
            "search method"
        )

    async def _call_vector_method(
        self,
        *,
        method: Any,
        query: str,
        vector: Sequence[float] | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Support common vector search APIs:

            search(query, ...)
            search(query_text=query, ...)
            search(vector=embedding, ...)
            similarity_search(query, ...)
        """

        attempts: list[
            tuple[
                tuple[Any, ...],
                dict[str, Any],
            ]
        ] = []

        if vector is not None:
            attempts.extend(
                [
                    (
                        (),
                        {
                            "vector": vector,
                            **kwargs,
                        },
                    ),
                    (
                        (),
                        {
                            "query_vector": vector,
                            **kwargs,
                        },
                    ),
                ]
            )

        attempts.extend(
            [
                (
                    (query,),
                    kwargs,
                ),
                (
                    (),
                    {
                        "query": query,
                        **kwargs,
                    },
                ),
                (
                    (),
                    {
                        "query_text": query,
                        **kwargs,
                    },
                ),
            ]
        )

        last_error: TypeError | None = None

        for args, call_kwargs in attempts:
            cleaned = {
                key: value
                for key, value in call_kwargs.items()
                if value is not None
            }

            try:
                value = method(
                    *args,
                    **cleaned,
                )

                if inspect.isawaitable(value):
                    return await value

                return value

            except TypeError as exc:
                last_error = exc

        if last_error:
            raise last_error

        return []

    # ========================================================
    # Candidate merge
    # ========================================================

    def _merge_candidates(
        self,
        *,
        bm25_results: Sequence[Any],
        vector_results: Sequence[Any],
    ) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}

        for index, result in enumerate(
            bm25_results,
            start=1,
        ):
            data = self._result_to_mapping(result)

            candidate_id = self._candidate_id(data)

            if not candidate_id:
                continue

            candidate = candidates.setdefault(
                candidate_id,
                self._candidate_from_data(
                    candidate_id,
                    data,
                ),
            )

            self._merge_metadata(
                candidate,
                data,
            )

            candidate.bm25_score = self._score_from_data(
                data
            )

            candidate.bm25_rank = self._rank_from_data(
                data,
                default=index,
            )

            candidate.sources.add("bm25")

            terms = data.get(
                "matched_terms",
                [],
            )

            if isinstance(
                terms,
                (list, tuple, set),
            ):
                candidate.matched_terms = list(
                    dict.fromkeys(
                        [
                            *candidate.matched_terms,
                            *[
                                str(term)
                                for term in terms
                            ],
                        ]
                    )
                )

        for index, result in enumerate(
            vector_results,
            start=1,
        ):
            data = self._result_to_mapping(result)

            candidate_id = self._candidate_id(data)

            if not candidate_id:
                continue

            candidate = candidates.setdefault(
                candidate_id,
                self._candidate_from_data(
                    candidate_id,
                    data,
                ),
            )

            self._merge_metadata(
                candidate,
                data,
            )

            candidate.vector_score = (
                self._vector_score_from_data(
                    data
                )
            )

            candidate.vector_rank = (
                self._rank_from_data(
                    data,
                    default=index,
                )
            )

            candidate.sources.add("vector")

        return candidates

    # ========================================================
    # Score normalization
    # ========================================================

    def _normalize_scores(
        self,
        candidates: Mapping[str, _Candidate],
    ) -> None:
        bm25_values = [
            candidate.bm25_score
            for candidate in candidates.values()
            if candidate.bm25_rank is not None
        ]

        vector_values = [
            candidate.vector_score
            for candidate in candidates.values()
            if candidate.vector_rank is not None
        ]

        normalized_bm25 = self._min_max_map(
            bm25_values
        )

        normalized_vector = self._min_max_map(
            vector_values
        )

        for candidate in candidates.values():
            if candidate.bm25_rank is not None:
                candidate.normalized_bm25_score = (
                    normalized_bm25.get(
                        candidate.bm25_score,
                        0.0,
                    )
                )

            if candidate.vector_rank is not None:
                candidate.normalized_vector_score = (
                    normalized_vector.get(
                        candidate.vector_score,
                        0.0,
                    )
                )

    @staticmethod
    def _min_max_map(
        values: Sequence[float],
    ) -> dict[float, float]:
        if not values:
            return {}

        minimum = min(values)
        maximum = max(values)

        if math.isclose(
            minimum,
            maximum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return {
                value: 1.0
                for value in values
            }

        difference = maximum - minimum

        return {
            value: (
                value - minimum
            )
            / difference
            for value in values
        }

    # ========================================================
    # Weighted fusion
    # ========================================================

    @staticmethod
    def _weighted_fusion(
        candidates: Mapping[str, _Candidate],
        *,
        bm25_weight: float,
        vector_weight: float,
    ) -> None:
        for candidate in candidates.values():
            candidate.final_score = (
                bm25_weight
                * candidate.normalized_bm25_score
                +
                vector_weight
                * candidate.normalized_vector_score
            )

            # Small consensus bonus when both lexical and
            # semantic retrieval independently found a chunk.
            if (
                candidate.bm25_rank is not None
                and candidate.vector_rank is not None
            ):
                candidate.final_score *= 1.05

            candidate.final_score = min(
                candidate.final_score,
                1.0,
            )

    # ========================================================
    # Reciprocal Rank Fusion
    # ========================================================

    def _rrf_fusion(
        self,
        candidates: Mapping[str, _Candidate],
        *,
        bm25_weight: float,
        vector_weight: float,
    ) -> None:
        max_possible = (
            bm25_weight
            / (self.rrf_k + 1)
            +
            vector_weight
            / (self.rrf_k + 1)
        )

        for candidate in candidates.values():
            score = 0.0

            if candidate.bm25_rank is not None:
                score += (
                    bm25_weight
                    / (
                        self.rrf_k
                        + candidate.bm25_rank
                    )
                )

            if candidate.vector_rank is not None:
                score += (
                    vector_weight
                    / (
                        self.rrf_k
                        + candidate.vector_rank
                    )
                )

            # Normalize RRF to roughly 0..1 for consistent
            # downstream min_score handling.
            candidate.final_score = (
                score / max_possible
                if max_possible
                else 0.0
            )

    # ========================================================
    # Generic result conversion
    # ========================================================

    @staticmethod
    def _result_to_mapping(
        result: Any,
    ) -> dict[str, Any]:
        if result is None:
            return {}

        if isinstance(result, Mapping):
            return dict(result)

        if hasattr(result, "to_dict"):
            value = result.to_dict()

            if isinstance(value, Mapping):
                return dict(value)

        if hasattr(result, "model_dump"):
            value = result.model_dump()

            if isinstance(value, Mapping):
                return dict(value)

        data: dict[str, Any] = {}

        for field_name in (
            "id",
            "chunk_id",
            "point_id",
            "text",
            "content",
            "chunk_text",
            "score",
            "similarity",
            "distance",
            "rank",
            "organization_id",
            "document_id",
            "contract_id",
            "parent_id",
            "parent_chunk_id",
            "title",
            "page_number",
            "chunk_index",
            "matched_terms",
            "metadata",
            "payload",
        ):
            if hasattr(result, field_name):
                data[field_name] = getattr(
                    result,
                    field_name,
                )

        return data

    @staticmethod
    def _candidate_id(
        data: Mapping[str, Any],
    ) -> str | None:
        value = (
            data.get("id")
            or data.get("chunk_id")
            or data.get("point_id")
        )

        if value is None:
            metadata = data.get("metadata")

            if isinstance(metadata, Mapping):
                value = (
                    metadata.get("chunk_id")
                    or metadata.get("id")
                )

        return (
            str(value)
            if value is not None
            else None
        )

    @classmethod
    def _candidate_from_data(
        cls,
        candidate_id: str,
        data: Mapping[str, Any],
    ) -> _Candidate:
        metadata = cls._extract_metadata(data)

        return _Candidate(
            id=candidate_id,
            text=str(
                data.get("text")
                or data.get("content")
                or data.get("chunk_text")
                or metadata.get("text")
                or ""
            ),
            organization_id=cls._string_or_none(
                data.get("organization_id")
                or metadata.get("organization_id")
            ),
            document_id=cls._string_or_none(
                data.get("document_id")
                or metadata.get("document_id")
            ),
            contract_id=cls._string_or_none(
                data.get("contract_id")
                or metadata.get("contract_id")
            ),
            parent_id=cls._string_or_none(
                data.get("parent_id")
                or data.get("parent_chunk_id")
                or metadata.get("parent_id")
                or metadata.get("parent_chunk_id")
            ),
            title=(
                data.get("title")
                or metadata.get("title")
            ),
            page_number=cls._int_or_none(
                data.get("page_number")
                or metadata.get("page_number")
            ),
            chunk_index=cls._int_or_none(
                data.get("chunk_index")
                if data.get("chunk_index") is not None
                else metadata.get("chunk_index")
            ),
            metadata=metadata,
        )

    @classmethod
    def _merge_metadata(
        cls,
        candidate: _Candidate,
        data: Mapping[str, Any],
    ) -> None:
        metadata = cls._extract_metadata(data)

        # Preserve fields from either retrieval backend.
        candidate.metadata = {
            **metadata,
            **candidate.metadata,
        }

        if not candidate.text:
            candidate.text = str(
                data.get("text")
                or data.get("content")
                or data.get("chunk_text")
                or metadata.get("text")
                or ""
            )

        if candidate.organization_id is None:
            candidate.organization_id = (
                cls._string_or_none(
                    data.get("organization_id")
                    or metadata.get("organization_id")
                )
            )

        if candidate.document_id is None:
            candidate.document_id = (
                cls._string_or_none(
                    data.get("document_id")
                    or metadata.get("document_id")
                )
            )

        if candidate.contract_id is None:
            candidate.contract_id = (
                cls._string_or_none(
                    data.get("contract_id")
                    or metadata.get("contract_id")
                )
            )

        if candidate.parent_id is None:
            candidate.parent_id = (
                cls._string_or_none(
                    data.get("parent_id")
                    or data.get("parent_chunk_id")
                    or metadata.get("parent_id")
                    or metadata.get("parent_chunk_id")
                )
            )

        if candidate.title is None:
            candidate.title = (
                data.get("title")
                or metadata.get("title")
            )

        if candidate.page_number is None:
            candidate.page_number = (
                cls._int_or_none(
                    data.get("page_number")
                    or metadata.get("page_number")
                )
            )

        if candidate.chunk_index is None:
            value = data.get("chunk_index")

            if value is None:
                value = metadata.get(
                    "chunk_index"
                )

            candidate.chunk_index = (
                cls._int_or_none(value)
            )

    @staticmethod
    def _extract_metadata(
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        raw_metadata = data.get("metadata")

        if isinstance(raw_metadata, Mapping):
            metadata.update(raw_metadata)

        payload = data.get("payload")

        if isinstance(payload, Mapping):
            metadata.update(payload)

        return metadata

    # ========================================================
    # Score extraction
    # ========================================================

    @staticmethod
    def _score_from_data(
        data: Mapping[str, Any],
    ) -> float:
        for key in (
            "score",
            "bm25_score",
            "relevance_score",
        ):
            value = data.get(key)

            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return 0.0

    @staticmethod
    def _vector_score_from_data(
        data: Mapping[str, Any],
    ) -> float:
        for key in (
            "score",
            "vector_score",
            "similarity",
            "similarity_score",
            "relevance_score",
        ):
            value = data.get(key)

            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        # Some vector stores return distance instead of
        # similarity. Convert positive distance into a bounded
        # similarity value.
        distance = data.get("distance")

        if distance is not None:
            try:
                distance = max(
                    0.0,
                    float(distance),
                )

                return 1.0 / (
                    1.0 + distance
                )

            except (TypeError, ValueError):
                pass

        return 0.0

    @staticmethod
    def _rank_from_data(
        data: Mapping[str, Any],
        *,
        default: int,
    ) -> int:
        value = data.get("rank")

        try:
            rank = int(value)

            if rank > 0:
                return rank

        except (TypeError, ValueError):
            pass

        return default

    # ========================================================
    # Result conversion
    # ========================================================

    @staticmethod
    def _to_result(
        candidate: _Candidate,
        *,
        rank: int,
    ) -> HybridSearchResult:
        return HybridSearchResult(
            id=candidate.id,
            text=candidate.text,
            score=round(
                candidate.final_score,
                6,
            ),
            rank=rank,
            bm25_score=round(
                candidate.bm25_score,
                6,
            ),
            vector_score=round(
                candidate.vector_score,
                6,
            ),
            normalized_bm25_score=round(
                candidate.normalized_bm25_score,
                6,
            ),
            normalized_vector_score=round(
                candidate.normalized_vector_score,
                6,
            ),
            bm25_rank=candidate.bm25_rank,
            vector_rank=candidate.vector_rank,
            organization_id=(
                candidate.organization_id
            ),
            document_id=candidate.document_id,
            contract_id=candidate.contract_id,
            parent_id=candidate.parent_id,
            title=candidate.title,
            page_number=candidate.page_number,
            chunk_index=candidate.chunk_index,
            matched_terms=list(
                candidate.matched_terms
            ),
            sources=sorted(
                candidate.sources
            ),
            metadata=dict(
                candidate.metadata
            ),
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_weights(
        bm25_weight: float,
        vector_weight: float,
    ) -> None:
        if not 0 <= bm25_weight <= 1:
            raise HybridSearchValidationError(
                "bm25_weight must be between 0 and 1"
            )

        if not 0 <= vector_weight <= 1:
            raise HybridSearchValidationError(
                "vector_weight must be between 0 and 1"
            )

        if not math.isclose(
            bm25_weight + vector_weight,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise HybridSearchValidationError(
                "bm25_weight + vector_weight "
                "must equal 1.0"
            )

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> str | None:
        return (
            str(value)
            if value is not None
            else None
        )

    @staticmethod
    def _int_or_none(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # ========================================================
    # Health
    # ========================================================

    async def health_check(
        self,
    ) -> dict[str, Any]:
        bm25_health: Any = None
        vector_health: Any = None

        try:
            method = getattr(
                self.bm25,
                "health_check",
                None,
            )

            if method:
                bm25_health = method()

                if inspect.isawaitable(
                    bm25_health
                ):
                    bm25_health = (
                        await bm25_health
                    )

        except Exception as exc:
            bm25_health = {
                "status": "unhealthy",
                "error": str(exc),
            }

        try:
            method = getattr(
                self.vector_search,
                "health_check",
                None,
            )

            if method:
                vector_health = method()

                if inspect.isawaitable(
                    vector_health
                ):
                    vector_health = (
                        await vector_health
                    )

        except Exception as exc:
            vector_health = {
                "status": "unhealthy",
                "error": str(exc),
            }

        return {
            "service": "hybrid_search",
            "status": "healthy",
            "fusion_mode":
                self.fusion_mode.value,
            "bm25_weight":
                self.bm25_weight,
            "vector_weight":
                self.vector_weight,
            "rrf_k": self.rrf_k,
            "candidate_multiplier":
                self.candidate_multiplier,
            "bm25": bm25_health,
            "vector": vector_health,
        }


# ============================================================
# Compatibility aliases
# ============================================================

HybridRetriever = HybridSearch
HybridSearchEngine = HybridSearch