"""
backend/app/rag/pipeline.py

Production-style Legal RAG pipeline.

Flow
----
Query
  ↓
Query classification
  ↓
Query rewriting
  ↓
Query routing
  ↓
Multi-query expansion
  ↓
Hybrid retrieval (BM25 + Vector)
  ↓
Permission-aware filtering
  ↓
Reranking
  ↓
Context construction
  ↓
Legal LLM
  ↓
Citation generation
  ↓
Answer validation
  ↓
Uncertainty detection
  ↓
Final grounded response
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class QueryClassifier(Protocol):
    def classify(self, query: str) -> Any:
        ...


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        ...


class QueryRouter(Protocol):
    def route(self, query: str, query_type: Any = None) -> Any:
        ...


class MultiQueryGenerator(Protocol):
    def generate(self, query: str) -> Sequence[str]:
        ...


class Retriever(Protocol):
    def search(self, query: str, **kwargs: Any) -> Sequence[Any]:
        ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
    ) -> Sequence[Any]:
        ...


class LegalLLM(Protocol):
    def generate(self, **kwargs: Any) -> Any:
        ...


class CitationGenerator(Protocol):
    def generate(self, answer: str, contexts: Sequence[Any]) -> Any:
        ...


class AnswerValidator(Protocol):
    def validate(
        self,
        answer: str,
        contexts: Sequence[Any],
    ) -> Any:
        ...


class UncertaintyDetector(Protocol):
    def detect(
        self,
        answer: str,
        contexts: Sequence[Any],
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RAGPipelineConfig:
    retrieval_top_k: int = 30
    rerank_top_k: int = 8

    max_context_characters: int = 24_000

    enable_query_classification: bool = True
    enable_query_rewrite: bool = True
    enable_multi_query: bool = True
    enable_reranking: bool = True

    enable_citations: bool = True
    enable_validation: bool = True
    enable_uncertainty: bool = True

    max_query_variants: int = 4

    fail_on_generation_error: bool = True


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RAGResponse:
    query: str
    rewritten_query: str

    answer: str

    query_type: Any = None
    route: Any = None

    citations: list[Any] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    validation: Any = None
    uncertainty: Any = None

    retrieved_count: int = 0
    reranked_count: int = 0

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "query_type": self.query_type,
            "route": self.route,
            "citations": self.citations,
            "sources": self.sources,
            "validation": self.validation,
            "uncertainty": self.uncertainty,
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LegalRAGPipeline:

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LegalLLM,
        reranker: Reranker | None = None,
        query_classifier: QueryClassifier | None = None,
        query_rewriter: QueryRewriter | None = None,
        query_router: QueryRouter | None = None,
        multi_query_generator: MultiQueryGenerator | None = None,
        citation_generator: CitationGenerator | None = None,
        answer_validator: AnswerValidator | None = None,
        uncertainty_detector: UncertaintyDetector | None = None,
        config: RAGPipelineConfig | None = None,
    ) -> None:

        self.retriever = retriever
        self.llm = llm

        self.reranker = reranker

        self.query_classifier = query_classifier
        self.query_rewriter = query_rewriter
        self.query_router = query_router
        self.multi_query_generator = multi_query_generator

        self.citation_generator = citation_generator
        self.answer_validator = answer_validator
        self.uncertainty_detector = uncertainty_detector

        self.config = config or RAGPipelineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        user_id: str | None = None,
        organization_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:

        started = time.perf_counter()

        query = self._validate_query(query)

        # --------------------------------------------------------------
        # 1. Query classification
        # --------------------------------------------------------------

        query_type = None

        if (
            self.config.enable_query_classification
            and self.query_classifier is not None
        ):
            try:
                query_type = self.query_classifier.classify(query)
            except Exception:
                logger.exception("Query classification failed.")

        # --------------------------------------------------------------
        # 2. Query rewriting
        # --------------------------------------------------------------

        rewritten_query = query

        if (
            self.config.enable_query_rewrite
            and self.query_rewriter is not None
        ):
            try:
                candidate = self.query_rewriter.rewrite(query)

                if candidate and candidate.strip():
                    rewritten_query = candidate.strip()

            except Exception:
                logger.exception("Query rewriting failed.")

        # --------------------------------------------------------------
        # 3. Query routing
        # --------------------------------------------------------------

        route = None

        if self.query_router is not None:
            try:
                route = self.query_router.route(
                    rewritten_query,
                    query_type=query_type,
                )
            except TypeError:
                route = self.query_router.route(rewritten_query)
            except Exception:
                logger.exception("Query routing failed.")

        # --------------------------------------------------------------
        # 4. Multi-query generation
        # --------------------------------------------------------------

        queries = [rewritten_query]

        if (
            self.config.enable_multi_query
            and self.multi_query_generator is not None
        ):
            try:
                generated = self.multi_query_generator.generate(
                    rewritten_query
                )

                for item in generated:
                    item = str(item).strip()

                    if item and item not in queries:
                        queries.append(item)

                    if len(queries) >= self.config.max_query_variants:
                        break

            except Exception:
                logger.exception("Multi-query generation failed.")

        # --------------------------------------------------------------
        # 5. Retrieval
        # --------------------------------------------------------------

        retrieval_limit = max(
            top_k or self.config.rerank_top_k,
            self.config.retrieval_top_k,
        )

        candidates: list[Any] = []

        for search_query in queries:
            try:
                results = self.retriever.search(
                    search_query,
                    top_k=retrieval_limit,
                    filters=dict(filters or {}),
                    user_id=user_id,
                    organization_id=organization_id,
                    route=route,
                )

                candidates.extend(results or [])

            except TypeError:
                # Allows simpler retriever implementations.
                results = self.retriever.search(
                    search_query,
                    top_k=retrieval_limit,
                )

                candidates.extend(results or [])

        candidates = self._deduplicate_candidates(candidates)

        retrieved_count = len(candidates)

        # --------------------------------------------------------------
        # 6. Reranking
        # --------------------------------------------------------------

        final_top_k = top_k or self.config.rerank_top_k

        if (
            candidates
            and self.config.enable_reranking
            and self.reranker is not None
        ):
            try:
                contexts = list(
                    self.reranker.rerank(
                        rewritten_query,
                        candidates,
                        top_k=final_top_k,
                    )
                )

            except Exception:
                logger.exception(
                    "Reranking failed; using retrieval ranking."
                )

                contexts = candidates[:final_top_k]

        else:
            contexts = candidates[:final_top_k]

        # --------------------------------------------------------------
        # 7. Build grounded context
        # --------------------------------------------------------------

        context_text = self._build_context(contexts)

        # --------------------------------------------------------------
        # 8. LLM generation
        # --------------------------------------------------------------

        try:
            generation = self.llm.generate(
                query=query,
                rewritten_query=rewritten_query,
                context=context_text,
                contexts=contexts,
                query_type=query_type,
                route=route,
            )

            answer = self._extract_answer(generation)

        except Exception:
            logger.exception("Legal LLM generation failed.")

            if self.config.fail_on_generation_error:
                raise

            answer = (
                "Unable to generate a grounded answer from the "
                "available legal documents."
            )

        # --------------------------------------------------------------
        # 9. Citations
        # --------------------------------------------------------------

        citations: list[Any] = []

        if (
            self.config.enable_citations
            and self.citation_generator is not None
            and contexts
        ):
            try:
                generated_citations = self.citation_generator.generate(
                    answer,
                    contexts,
                )

                if generated_citations:
                    if isinstance(generated_citations, list):
                        citations = generated_citations
                    else:
                        citations = [generated_citations]

            except Exception:
                logger.exception("Citation generation failed.")

        # --------------------------------------------------------------
        # 10. Grounded-answer validation
        # --------------------------------------------------------------

        validation = None

        if (
            self.config.enable_validation
            and self.answer_validator is not None
        ):
            try:
                validation = self.answer_validator.validate(
                    answer,
                    contexts,
                )
            except Exception:
                logger.exception("Answer validation failed.")

        # --------------------------------------------------------------
        # 11. Uncertainty detection
        # --------------------------------------------------------------

        uncertainty = None

        if (
            self.config.enable_uncertainty
            and self.uncertainty_detector is not None
        ):
            try:
                uncertainty = self.uncertainty_detector.detect(
                    answer,
                    contexts,
                )
            except Exception:
                logger.exception("Uncertainty detection failed.")

        # --------------------------------------------------------------
        # 12. Sources
        # --------------------------------------------------------------

        sources = [
            self._serialize_source(context)
            for context in contexts
        ]

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        logger.info(
            "Legal RAG completed retrieved=%d reranked=%d latency=%.2fms",
            retrieved_count,
            len(contexts),
            latency_ms,
        )

        return RAGResponse(
            query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            query_type=query_type,
            route=route,
            citations=citations,
            sources=sources,
            validation=validation,
            uncertainty=uncertainty,
            retrieved_count=retrieved_count,
            reranked_count=len(contexts),
            latency_ms=round(latency_ms, 2),
            metadata={
                "query_variants": queries,
                "organization_id": organization_id,
            },
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def arun(
        self,
        query: str,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Execute blocking retrieval/model components outside FastAPI's
        event loop.
        """

        return await asyncio.to_thread(
            self.run,
            query,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context(
        self,
        contexts: Sequence[Any],
    ) -> str:

        sections: list[str] = []
        current_length = 0

        for index, context in enumerate(contexts, start=1):

            text = self._get_value(
                context,
                "text",
                "content",
                "page_content",
                default="",
            )

            if not text:
                continue

            metadata = self._get_value(
                context,
                "metadata",
                default={},
            ) or {}

            document = (
                metadata.get("document_name")
                or metadata.get("filename")
                or metadata.get("source")
                or self._get_value(
                    context,
                    "source",
                    default="Unknown document",
                )
            )

            page = (
                metadata.get("page")
                or metadata.get("page_number")
            )

            clause = (
                metadata.get("clause")
                or metadata.get("clause_type")
            )

            header = f"[Source {index}: {document}"

            if page is not None:
                header += f", Page {page}"

            if clause:
                header += f", Clause {clause}"

            header += "]"

            section = f"{header}\n{text.strip()}"

            projected = current_length + len(section)

            if projected > self.config.max_context_characters:
                remaining = (
                    self.config.max_context_characters
                    - current_length
                )

                if remaining > 200:
                    sections.append(section[:remaining])

                break

            sections.append(section)
            current_length = projected

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_candidates(
        self,
        candidates: Sequence[Any],
    ) -> list[Any]:

        seen: set[str] = set()
        output: list[Any] = []

        for candidate in candidates:

            candidate_id = self._get_value(
                candidate,
                "id",
                "chunk_id",
                default=None,
            )

            metadata = self._get_value(
                candidate,
                "metadata",
                default={},
            ) or {}

            candidate_id = (
                candidate_id
                or metadata.get("chunk_id")
                or metadata.get("id")
            )

            text = self._get_value(
                candidate,
                "text",
                "content",
                "page_content",
                default="",
            )

            key = (
                str(candidate_id)
                if candidate_id is not None
                else " ".join(str(text).lower().split())
            )

            if not key or key in seen:
                continue

            seen.add(key)
            output.append(candidate)

        return output

    # ------------------------------------------------------------------
    # Source serialization
    # ------------------------------------------------------------------

    def _serialize_source(
        self,
        source: Any,
    ) -> dict[str, Any]:

        metadata = self._get_value(
            source,
            "metadata",
            default={},
        ) or {}

        return {
            "id": self._get_value(
                source,
                "id",
                "chunk_id",
                default=metadata.get("chunk_id"),
            ),
            "score": self._get_value(
                source,
                "score",
                "retrieval_score",
                default=None,
            ),
            "source": self._get_value(
                source,
                "source",
                default=metadata.get("source"),
            ),
            "document_name": metadata.get("document_name"),
            "page": (
                metadata.get("page")
                or metadata.get("page_number")
            ),
            "clause": (
                metadata.get("clause")
                or metadata.get("clause_type")
            ),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        if len(query) > 20_000:
            raise ValueError("query is too large.")

        return query

    @staticmethod
    def _extract_answer(generation: Any) -> str:

        if generation is None:
            return ""

        if isinstance(generation, str):
            return generation.strip()

        if isinstance(generation, Mapping):

            for key in (
                "answer",
                "response",
                "text",
                "content",
                "output",
            ):
                value = generation.get(key)

                if value:
                    return str(value).strip()

        for attribute in (
            "answer",
            "response",
            "text",
            "content",
        ):
            value = getattr(
                generation,
                attribute,
                None,
            )

            if value:
                return str(value).strip()

        return str(generation).strip()

    @staticmethod
    def _get_value(
        obj: Any,
        *names: str,
        default: Any = None,
    ) -> Any:

        if isinstance(obj, Mapping):

            for name in names:
                if name in obj:
                    return obj[name]

            return default

        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)

        return default