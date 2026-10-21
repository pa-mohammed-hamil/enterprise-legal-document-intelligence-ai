"""
backend/app/rag/reranking/reranker.py

Provider-independent reranking layer for the Legal Document Intelligence RAG
pipeline.

Responsibilities
----------------
- Accept candidates from BM25/vector/hybrid retrieval.
- Normalize heterogeneous retrieval results.
- Rerank candidates with an injected reranker model.
- Combine semantic reranker score with retrieval score.
- Preserve document/chunk metadata and citations.
- Support top-k, thresholds, deduplication and graceful fallback.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RerankCandidate:
    """Normalized candidate supplied to the reranker."""

    text: str
    id: str | None = None
    retrieval_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_result(cls, result: Any) -> "RerankCandidate":
        if isinstance(result, cls):
            return result

        if isinstance(result, Mapping):
            text = (
                result.get("text")
                or result.get("content")
                or result.get("chunk_text")
                or result.get("page_content")
                or ""
            )

            metadata = dict(result.get("metadata") or {})

            candidate_id = (
                result.get("id")
                or result.get("chunk_id")
                or metadata.get("chunk_id")
                or metadata.get("id")
            )

            score = (
                result.get("score")
                if result.get("score") is not None
                else result.get("retrieval_score", 0.0)
            )

            source = (
                result.get("source")
                or metadata.get("source")
                or metadata.get("document_name")
            )

            return cls(
                id=str(candidate_id) if candidate_id is not None else None,
                text=str(text),
                retrieval_score=_safe_float(score),
                metadata=metadata,
                source=str(source) if source is not None else None,
            )

        # Object returned by another retrieval implementation.
        text = (
            getattr(result, "text", None)
            or getattr(result, "content", None)
            or getattr(result, "page_content", None)
            or ""
        )

        metadata = dict(getattr(result, "metadata", {}) or {})

        candidate_id = (
            getattr(result, "id", None)
            or getattr(result, "chunk_id", None)
            or metadata.get("chunk_id")
        )

        score = getattr(
            result,
            "score",
            getattr(result, "retrieval_score", 0.0),
        )

        return cls(
            id=str(candidate_id) if candidate_id is not None else None,
            text=str(text),
            retrieval_score=_safe_float(score),
            metadata=metadata,
            source=metadata.get("source"),
        )


@dataclass(slots=True)
class RerankedResult:
    """Final result returned by the reranking layer."""

    id: str | None
    text: str
    score: float
    reranker_score: float
    retrieval_score: float
    rank: int
    metadata: dict[str, Any]
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "reranker_score": self.reranker_score,
            "retrieval_score": self.retrieval_score,
            "rank": self.rank,
            "metadata": self.metadata,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

class RerankerModel(Protocol):
    """
    Interface implemented by hf_reranker.py or another reranking provider.
    """

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RerankerConfig:
    top_k: int = 10

    # Number of retrieved candidates allowed into cross-encoder inference.
    max_candidates: int = 50

    # Final score:
    # alpha * reranker_score + (1-alpha) * retrieval_score
    reranker_weight: float = 0.85

    min_score: float | None = None

    normalize_scores: bool = True
    deduplicate: bool = True

    # If model inference fails, retain retrieval ordering.
    fallback_on_error: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero.")

        if not 0.0 <= self.reranker_weight <= 1.0:
            raise ValueError("reranker_weight must be between 0 and 1.")


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    High-level reranking service.

    Typical flow:

        BM25 ─┐
              ├─ Hybrid Search → candidates → Reranker → LLM
        Vector┘

    The class is deliberately independent from Hugging Face so another
    cross-encoder/reranking provider can be injected without changing the
    RAG pipeline.
    """

    def __init__(
        self,
        model: RerankerModel,
        config: RerankerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or RerankerConfig()

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankedResult]:

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        if not candidates:
            return []

        started_at = time.perf_counter()

        normalized = [
            RerankCandidate.from_result(candidate)
            for candidate in candidates
        ]

        # Ignore malformed/empty chunks.
        normalized = [
            candidate
            for candidate in normalized
            if candidate.text.strip()
        ]

        if self.config.deduplicate:
            normalized = self._deduplicate(normalized)

        if not normalized:
            return []

        # Avoid running a costly cross encoder over hundreds of chunks.
        normalized.sort(
            key=lambda candidate: candidate.retrieval_score,
            reverse=True,
        )

        normalized = normalized[: self.config.max_candidates]

        documents = [candidate.text for candidate in normalized]

        try:
            raw_scores = list(self.model.score(query, documents))

            if len(raw_scores) != len(normalized):
                raise RuntimeError(
                    "Reranker returned an invalid number of scores: "
                    f"expected={len(normalized)}, received={len(raw_scores)}"
                )

            reranker_scores = [_safe_float(score) for score in raw_scores]

        except Exception:
            logger.exception("Reranker inference failed.")

            if not self.config.fallback_on_error:
                raise

            return self._retrieval_fallback(
                normalized,
                top_k=top_k,
                min_score=min_score,
            )

        retrieval_scores = [
            candidate.retrieval_score for candidate in normalized
        ]

        if self.config.normalize_scores:
            reranker_scores = _min_max_normalize(reranker_scores)
            retrieval_scores = _min_max_normalize(retrieval_scores)

        alpha = self.config.reranker_weight

        ranked: list[tuple[RerankCandidate, float, float, float]] = []

        for candidate, rerank_score, retrieval_score in zip(
            normalized,
            reranker_scores,
            retrieval_scores,
        ):
            final_score = (
                alpha * rerank_score
                + (1.0 - alpha) * retrieval_score
            )

            ranked.append(
                (
                    candidate,
                    final_score,
                    rerank_score,
                    retrieval_score,
                )
            )

        ranked.sort(key=lambda item: item[1], reverse=True)

        threshold = (
            self.config.min_score
            if min_score is None
            else min_score
        )

        limit = top_k or self.config.top_k

        output: list[RerankedResult] = []

        for candidate, final_score, rerank_score, retrieval_score in ranked:
            if threshold is not None and final_score < threshold:
                continue

            output.append(
                RerankedResult(
                    id=candidate.id,
                    text=candidate.text,
                    score=final_score,
                    reranker_score=rerank_score,
                    retrieval_score=retrieval_score,
                    rank=len(output) + 1,
                    metadata=candidate.metadata,
                    source=candidate.source,
                )
            )

            if len(output) >= limit:
                break

        logger.debug(
            "Reranked %d candidates into %d results in %.2f ms",
            len(normalized),
            len(output),
            (time.perf_counter() - started_at) * 1000,
        )

        return output

    async def arerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankedResult]:
        """
        Async wrapper.

        Transformer inference is normally blocking, so it is moved away from
        FastAPI's event loop.
        """

        return await asyncio.to_thread(
            self.rerank,
            query,
            candidates,
            top_k=top_k,
            min_score=min_score,
        )

    def rerank_as_dicts(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result in self.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        ]

    def _retrieval_fallback(
        self,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None,
        min_score: float | None,
    ) -> list[RerankedResult]:

        limit = top_k or self.config.top_k

        ordered = sorted(
            candidates,
            key=lambda item: item.retrieval_score,
            reverse=True,
        )

        results: list[RerankedResult] = []

        for candidate in ordered:
            score = candidate.retrieval_score

            if min_score is not None and score < min_score:
                continue

            results.append(
                RerankedResult(
                    id=candidate.id,
                    text=candidate.text,
                    score=score,
                    reranker_score=0.0,
                    retrieval_score=score,
                    rank=len(results) + 1,
                    metadata=candidate.metadata,
                    source=candidate.source,
                )
            )

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _deduplicate(
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankCandidate]:

        seen: set[str] = set()
        output: list[RerankCandidate] = []

        for candidate in candidates:
            # Prefer stable chunk ID. Otherwise use normalized text.
            key = candidate.id or " ".join(
                candidate.text.lower().split()
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(candidate)

        return output


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(number) or math.isinf(number):
        return 0.0

    return number


def _min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        # Equal scores contain no ranking information.
        return [1.0 for _ in values]

    scale = maximum - minimum

    return [
        (value - minimum) / scale
        for value in values
    ]