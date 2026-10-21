# backend/app/rag/retrieval/multi_query.py

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================


class MultiQueryError(Exception):
    """Base multi-query retrieval error."""


class MultiQueryValidationError(MultiQueryError):
    """Invalid multi-query request."""


class MultiQueryGenerationError(MultiQueryError):
    """Query expansion/generation failed."""


class MultiQueryRetrievalError(MultiQueryError):
    """Underlying retrieval failed."""


# ============================================================
# Query types
# ============================================================


class QueryVariantType(str, Enum):
    ORIGINAL = "original"
    REWRITE = "rewrite"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    LEGAL = "legal"
    BROAD = "broad"
    SPECIFIC = "specific"
    DECOMPOSED = "decomposed"
    GENERATED = "generated"


# ============================================================
# Query variant
# ============================================================


@dataclass(slots=True)
class QueryVariant:
    query: str
    variant_type: QueryVariantType | str
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.query = self.query.strip()
        self.variant_type = QueryVariantType(
            self.variant_type
        )
        self.weight = max(
            0.0,
            float(self.weight),
        )


# ============================================================
# Multi-query result
# ============================================================


@dataclass(slots=True)
class MultiQueryResult:
    id: str
    text: str
    score: float
    rank: int

    organization_id: str | None = None
    document_id: str | None = None
    contract_id: str | None = None
    parent_id: str | None = None

    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    matched_queries: list[str] = field(
        default_factory=list
    )

    matched_query_types: list[str] = field(
        default_factory=list
    )

    matched_terms: list[str] = field(
        default_factory=list
    )

    sources: list[str] = field(
        default_factory=list
    )

    source_ranks: dict[str, int] = field(
        default_factory=dict
    )

    source_scores: dict[str, float] = field(
        default_factory=dict
    )

    retrieval_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "organization_id":
                self.organization_id,
            "document_id":
                self.document_id,
            "contract_id":
                self.contract_id,
            "parent_id":
                self.parent_id,
            "title": self.title,
            "page_number":
                self.page_number,
            "chunk_index":
                self.chunk_index,
            "metadata":
                self.metadata,
            "matched_queries":
                self.matched_queries,
            "matched_query_types":
                self.matched_query_types,
            "matched_terms":
                self.matched_terms,
            "sources":
                self.sources,
            "source_ranks":
                self.source_ranks,
            "source_scores":
                self.source_scores,
            "retrieval_count":
                self.retrieval_count,
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

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    matched_queries: list[str] = field(
        default_factory=list
    )

    matched_query_types: list[str] = field(
        default_factory=list
    )

    matched_terms: list[str] = field(
        default_factory=list
    )

    sources: set[str] = field(
        default_factory=set
    )

    source_ranks: dict[str, int] = field(
        default_factory=dict
    )

    source_scores: dict[str, float] = field(
        default_factory=dict
    )

    retrieval_count: int = 0
    fused_score: float = 0.0


# ============================================================
# Multi Query Retriever
# ============================================================


class MultiQueryRetriever:
    """
    Multi-query retrieval for Legal RAG.

    One user query is expanded into multiple search variants.

    Example:

        User:
            "Can the supplier terminate this agreement?"

        Variants:
            1. Can the supplier terminate this agreement?
            2. supplier termination rights
            3. termination clause supplier
            4. termination for convenience or cause
            5. cancellation and early termination provisions

    Each variant is sent through the underlying retriever,
    normally HybridSearch.

    Results are:

        retrieved
            ↓
        deduplicated
            ↓
        Reciprocal Rank Fusion
            ↓
        consensus boosted
            ↓
        top candidates
            ↓
        reranker
            ↓
        Legal LLM + citations

    Tenant restrictions are forwarded to every query.
    """

    DEFAULT_QUERY_WEIGHTS = {
        QueryVariantType.ORIGINAL: 1.20,
        QueryVariantType.REWRITE: 1.00,
        QueryVariantType.SEMANTIC: 0.95,
        QueryVariantType.KEYWORD: 0.90,
        QueryVariantType.LEGAL: 1.00,
        QueryVariantType.BROAD: 0.85,
        QueryVariantType.SPECIFIC: 0.95,
        QueryVariantType.DECOMPOSED: 0.90,
        QueryVariantType.GENERATED: 0.90,
    }

    def __init__(
        self,
        *,
        retriever: Any,
        query_rewriter: Any | None = None,
        legal_llm: Any | None = None,
        max_queries: int = 5,
        candidate_multiplier: int = 3,
        rrf_k: int = 60,
        include_original: bool = True,
        consensus_boost: float = 0.05,
        fail_open: bool = True,
    ) -> None:
        if retriever is None:
            raise MultiQueryValidationError(
                "retriever is required"
            )

        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.legal_llm = legal_llm

        self.max_queries = max(
            1,
            min(int(max_queries), 20),
        )

        self.candidate_multiplier = max(
            1,
            int(candidate_multiplier),
        )

        self.rrf_k = max(
            1,
            int(rrf_k),
        )

        self.include_original = bool(
            include_original
        )

        self.consensus_boost = max(
            0.0,
            float(consensus_boost),
        )

        self.fail_open = bool(fail_open)

    # ========================================================
    # Main retrieval
    # ========================================================

    async def search(
        self,
        query: str,
        *,
        organization_id: UUID | str,
        top_k: int = 10,
        candidate_k: int | None = None,
        document_ids: Sequence[
            UUID | str
        ] | None = None,
        contract_ids: Sequence[
            UUID | str
        ] | None = None,
        filters: Mapping[str, Any] | None = None,
        generated_queries: Sequence[
            str
        ] | None = None,
        max_queries: int | None = None,
        min_score: float = 0.0,
        retriever_kwargs: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> list[MultiQueryResult]:
        query = self._clean_query(query)

        if not query:
            return []

        if organization_id is None:
            raise MultiQueryValidationError(
                "organization_id is required"
            )

        if top_k <= 0:
            return []

        number_of_queries = max(
            1,
            min(
                max_queries or self.max_queries,
                20,
            ),
        )

        candidate_k = (
            candidate_k
            or top_k
            * self.candidate_multiplier
        )

        candidate_k = max(
            top_k,
            candidate_k,
        )

        variants = await self.generate_queries(
            query,
            generated_queries=generated_queries,
            max_queries=number_of_queries,
        )

        if not variants:
            variants = [
                QueryVariant(
                    query=query,
                    variant_type=(
                        QueryVariantType.ORIGINAL
                    ),
                    weight=1.20,
                )
            ]

        tasks = [
            self._retrieve_variant(
                variant,
                organization_id=organization_id,
                top_k=candidate_k,
                document_ids=document_ids,
                contract_ids=contract_ids,
                filters=filters,
                retriever_kwargs=(
                    retriever_kwargs
                ),
            )
            for variant in variants
        ]

        responses = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        successful: list[
            tuple[
                QueryVariant,
                list[Any],
            ]
        ] = []

        errors: list[Exception] = []

        for variant, response in zip(
            variants,
            responses,
            strict=True,
        ):
            if isinstance(
                response,
                BaseException,
            ):
                logger.warning(
                    "Multi-query retrieval failed "
                    "for query variant %r: %s",
                    variant.query,
                    response,
                )

                if isinstance(
                    response,
                    Exception,
                ):
                    errors.append(response)

                continue

            successful.append(
                (
                    variant,
                    list(response or []),
                )
            )

        if not successful:
            if errors and not self.fail_open:
                raise MultiQueryRetrievalError(
                    "All multi-query retrieval "
                    "requests failed"
                ) from errors[0]

            return []

        candidates = self._merge_results(
            successful
        )

        self._calculate_rrf(
            candidates,
            total_queries=len(successful),
        )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                item.fused_score,
                item.retrieval_count,
                max(
                    item.source_scores.values(),
                    default=0.0,
                ),
            ),
            reverse=True,
        )

        output: list[
            MultiQueryResult
        ] = []

        for candidate in ranked:
            if candidate.fused_score < min_score:
                continue

            output.append(
                self._to_result(
                    candidate,
                    rank=len(output) + 1,
                )
            )

            if len(output) >= top_k:
                break

        return output

    # ========================================================
    # Compatibility
    # ========================================================

    async def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[MultiQueryResult]:
        return await self.search(
            query,
            **kwargs,
        )

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
    # Query generation
    # ========================================================

    async def generate_queries(
        self,
        query: str,
        *,
        generated_queries: Sequence[
            str
        ] | None = None,
        max_queries: int | None = None,
    ) -> list[QueryVariant]:
        query = self._clean_query(query)

        if not query:
            return []

        limit = max(
            1,
            min(
                max_queries or self.max_queries,
                20,
            ),
        )

        variants: list[
            QueryVariant
        ] = []

        if self.include_original:
            variants.append(
                self._variant(
                    query,
                    QueryVariantType.ORIGINAL,
                )
            )

        # ----------------------------------------------------
        # Explicit caller-generated variants
        # ----------------------------------------------------

        if generated_queries:
            for generated in generated_queries:
                generated = self._clean_query(
                    generated
                )

                if generated:
                    variants.append(
                        self._variant(
                            generated,
                            QueryVariantType.GENERATED,
                        )
                    )

        # ----------------------------------------------------
        # Query rewriter
        # ----------------------------------------------------

        if (
            len(variants) < limit
            and self.query_rewriter
            is not None
        ):
            try:
                rewritten = (
                    await self._generate_with_rewriter(
                        query,
                        limit=limit,
                    )
                )

                variants.extend(rewritten)

            except Exception:
                logger.exception(
                    "Query rewriter failed during "
                    "multi-query generation"
                )

        # ----------------------------------------------------
        # LLM query expansion
        # ----------------------------------------------------

        if (
            len(variants) < limit
            and self.legal_llm is not None
        ):
            try:
                generated = (
                    await self._generate_with_llm(
                        query,
                        limit=(
                            limit
                            - len(variants)
                        ),
                    )
                )

                variants.extend(generated)

            except Exception:
                logger.exception(
                    "LLM multi-query generation failed"
                )

        # ----------------------------------------------------
        # Deterministic legal expansion fallback
        # ----------------------------------------------------

        if len(variants) < limit:
            variants.extend(
                self._deterministic_expansion(
                    query
                )
            )

        variants = self._deduplicate_queries(
            variants
        )

        if not variants:
            variants.append(
                self._variant(
                    query,
                    QueryVariantType.ORIGINAL,
                )
            )

        return variants[:limit]

    # ========================================================
    # Query rewriter integration
    # ========================================================

    async def _generate_with_rewriter(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[QueryVariant]:
        methods = (
            "generate_queries",
            "multi_query",
            "expand",
            "rewrite_queries",
            "rewrite",
        )

        for method_name in methods:
            method = getattr(
                self.query_rewriter,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = (
                {
                    "query": query,
                    "num_queries": limit,
                },
                {
                    "query": query,
                    "max_queries": limit,
                },
                {
                    "query": query,
                },
            )

            for kwargs in attempts:
                try:
                    result = method(**kwargs)

                    result = (
                        await self._await_if_needed(
                            result
                        )
                    )

                    variants = (
                        self._parse_generated_queries(
                            result,
                            default_type=(
                                QueryVariantType.REWRITE
                            ),
                        )
                    )

                    if variants:
                        return variants

                except TypeError:
                    continue

        return []

    # ========================================================
    # LLM query generation
    # ========================================================

    async def _generate_with_llm(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[QueryVariant]:
        if limit <= 0:
            return []

        prompt = self._query_generation_prompt(
            query,
            limit,
        )

        methods = (
            "generate",
            "complete",
            "invoke",
            "ainvoke",
            "chat",
        )

        for method_name in methods:
            method = getattr(
                self.legal_llm,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = (
                {"prompt": prompt},
                {"query": prompt},
                {"text": prompt},
            )

            for kwargs in attempts:
                try:
                    result = method(**kwargs)

                    result = (
                        await self._await_if_needed(
                            result
                        )
                    )

                    variants = (
                        self._parse_generated_queries(
                            result,
                            default_type=(
                                QueryVariantType.GENERATED
                            ),
                        )
                    )

                    if variants:
                        return variants[:limit]

                except TypeError:
                    continue

        return []

    # ========================================================
    # Deterministic expansion
    # ========================================================

    def _deterministic_expansion(
        self,
        query: str,
    ) -> list[QueryVariant]:
        variants: list[
            QueryVariant
        ] = []

        keyword_query = (
            self._keyword_query(query)
        )

        if (
            keyword_query
            and keyword_query.casefold()
            != query.casefold()
        ):
            variants.append(
                self._variant(
                    keyword_query,
                    QueryVariantType.KEYWORD,
                )
            )

        legal_query = (
            self._legal_expansion(query)
        )

        if (
            legal_query
            and legal_query.casefold()
            != query.casefold()
        ):
            variants.append(
                self._variant(
                    legal_query,
                    QueryVariantType.LEGAL,
                )
            )

        variants.extend(
            self._decompose_query(query)
        )

        return variants

    @staticmethod
    def _keyword_query(
        query: str,
    ) -> str:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "can",
            "could",
            "do",
            "does",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "may",
            "of",
            "on",
            "or",
            "our",
            "please",
            "the",
            "this",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "will",
            "with",
            "would",
        }

        tokens = re.findall(
            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
            query,
        )

        meaningful = [
            token
            for token in tokens
            if token.casefold()
            not in stopwords
        ]

        return " ".join(meaningful)

    @staticmethod
    def _legal_expansion(
        query: str,
    ) -> str | None:
        normalized = query.casefold()

        expansions = {
            "terminate": (
                "termination termination rights "
                "termination for cause termination "
                "for convenience cancellation"
            ),
            "termination": (
                "termination rights termination "
                "for cause termination for "
                "convenience cancellation"
            ),
            "cancel": (
                "cancellation termination "
                "termination rights"
            ),
            "renew": (
                "renewal automatic renewal "
                "renewal notice extension"
            ),
            "renewal": (
                "renewal automatic renewal "
                "renewal notice extension"
            ),
            "expire": (
                "expiration expiry term end date "
                "renewal notice"
            ),
            "expiry": (
                "expiry expiration term end date "
                "renewal notice"
            ),
            "liable": (
                "liability limitation of liability "
                "indemnification damages"
            ),
            "liability": (
                "liability limitation of liability "
                "indemnification damages"
            ),
            "indemnify": (
                "indemnity indemnification "
                "defend hold harmless liability"
            ),
            "indemnification": (
                "indemnification indemnity "
                "hold harmless liability"
            ),
            "confidential": (
                "confidentiality confidential "
                "information nondisclosure NDA"
            ),
            "payment": (
                "payment terms fees invoice "
                "billing late payment"
            ),
            "breach": (
                "breach default material breach "
                "remedies cure period"
            ),
            "notice": (
                "notice requirement notice period "
                "written notice notification"
            ),
            "governing law": (
                "governing law jurisdiction "
                "choice of law venue"
            ),
            "dispute": (
                "dispute resolution arbitration "
                "litigation mediation jurisdiction"
            ),
            "warranty": (
                "warranty warranties representation "
                "representations disclaimer"
            ),
            "obligation": (
                "obligation duty responsibility "
                "requirement covenant"
            ),
            "assign": (
                "assignment assignability transfer "
                "change of control"
            ),
            "force majeure": (
                "force majeure excused performance "
                "uncontrollable event"
            ),
            "non compete": (
                "non-compete restrictive covenant "
                "competition restriction"
            ),
            "data": (
                "data protection privacy personal "
                "data processing security"
            ),
        }

        matched: list[str] = []

        for trigger, expansion in (
            expansions.items()
        ):
            if trigger in normalized:
                matched.append(expansion)

        if not matched:
            return None

        return " ".join(
            dict.fromkeys(
                " ".join(matched).split()
            )
        )

    def _decompose_query(
        self,
        query: str,
    ) -> list[QueryVariant]:
        """
        Split obvious compound legal questions.

        Example:
            "What is the termination period and
             what penalties apply?"

        becomes two retrieval queries.
        """

        text = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        parts = re.split(
            r"""
            \s+
            (?:
                and\s+what
                |
                and\s+when
                |
                and\s+how
                |
                and\s+who
                |
                also
                |
                additionally
            )
            \s+
            """,
            text,
            flags=re.IGNORECASE
            | re.VERBOSE,
        )

        if len(parts) <= 1:
            return []

        variants: list[
            QueryVariant
        ] = []

        for part in parts:
            cleaned = self._clean_query(
                part
            )

            if len(cleaned) < 4:
                continue

            variants.append(
                self._variant(
                    cleaned,
                    QueryVariantType.DECOMPOSED,
                )
            )

        return variants

    # ========================================================
    # Parse generated output
    # ========================================================

    def _parse_generated_queries(
        self,
        result: Any,
        *,
        default_type: QueryVariantType,
    ) -> list[QueryVariant]:
        if result is None:
            return []

        if isinstance(
            result,
            QueryVariant,
        ):
            return [result]

        if isinstance(result, Mapping):
            for key in (
                "queries",
                "rewrites",
                "variants",
                "generated_queries",
                "results",
            ):
                if key in result:
                    return (
                        self._parse_generated_queries(
                            result[key],
                            default_type=default_type,
                        )
                    )

            for key in (
                "query",
                "text",
                "content",
                "answer",
                "response",
            ):
                if key in result:
                    return (
                        self._parse_generated_queries(
                            result[key],
                            default_type=default_type,
                        )
                    )

            return []

        if hasattr(
            result,
            "model_dump",
        ):
            return self._parse_generated_queries(
                result.model_dump(),
                default_type=default_type,
            )

        if isinstance(
            result,
            (list, tuple, set),
        ):
            variants: list[
                QueryVariant
            ] = []

            for item in result:
                if isinstance(
                    item,
                    QueryVariant,
                ):
                    variants.append(item)
                    continue

                if isinstance(
                    item,
                    Mapping,
                ):
                    text = (
                        item.get("query")
                        or item.get("text")
                        or item.get("content")
                    )

                    if text:
                        variant_type = (
                            item.get("type")
                            or item.get(
                                "variant_type"
                            )
                            or default_type
                        )

                        try:
                            parsed_type = (
                                QueryVariantType(
                                    variant_type
                                )
                            )
                        except ValueError:
                            parsed_type = (
                                default_type
                            )

                        variants.append(
                            self._variant(
                                str(text),
                                parsed_type,
                            )
                        )

                    continue

                if isinstance(item, str):
                    variants.append(
                        self._variant(
                            item,
                            default_type,
                        )
                    )

            return variants

        if hasattr(result, "content"):
            return self._parse_generated_queries(
                result.content,
                default_type=default_type,
            )

        if not isinstance(result, str):
            return []

        text = result.strip()

        if not text:
            return []

        lines = text.splitlines()

        variants = []

        for line in lines:
            cleaned = re.sub(
                r"""
                ^\s*
                (?:
                    [-*•]
                    |
                    \d+[\.\)]
                    |
                    query\s*\d*\s*:
                )
                \s*
                """,
                "",
                line,
                flags=(
                    re.IGNORECASE
                    | re.VERBOSE
                ),
            ).strip()

            cleaned = cleaned.strip(
                "\"'`"
            )

            if not cleaned:
                continue

            if cleaned.casefold() in {
                "queries:",
                "query variants:",
                "rewrites:",
            }:
                continue

            variants.append(
                self._variant(
                    cleaned,
                    default_type,
                )
            )

        return variants

    # ========================================================
    # Retrieve one variant
    # ========================================================

    async def _retrieve_variant(
        self,
        variant: QueryVariant,
        *,
        organization_id: UUID | str,
        top_k: int,
        document_ids: Sequence[
            UUID | str
        ] | None,
        contract_ids: Sequence[
            UUID | str
        ] | None,
        filters: Mapping[
            str,
            Any,
        ] | None,
        retriever_kwargs: Mapping[
            str,
            Any,
        ] | None,
    ) -> list[Any]:
        method = self._retriever_method()

        kwargs = dict(
            retriever_kwargs
            or {}
        )

        # Security-sensitive values override arbitrary
        # retriever kwargs.
        kwargs.update(
            {
                "organization_id":
                    organization_id,
                "top_k": top_k,
                "document_ids":
                    document_ids,
                "contract_ids":
                    contract_ids,
                "filters":
                    filters,
            }
        )

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

        try:
            result = method(
                variant.query,
                **kwargs,
            )

            result = (
                await self._await_if_needed(
                    result
                )
            )

            return list(result or [])

        except TypeError:
            # Compatibility with retrievers requiring
            # keyword query.
            try:
                result = method(
                    query=variant.query,
                    **kwargs,
                )

                result = (
                    await self._await_if_needed(
                        result
                    )
                )

                return list(
                    result or []
                )

            except Exception as exc:
                raise MultiQueryRetrievalError(
                    "Retrieval failed for query "
                    f"{variant.query!r}"
                ) from exc

        except Exception as exc:
            raise MultiQueryRetrievalError(
                "Retrieval failed for query "
                f"{variant.query!r}"
            ) from exc

    def _retriever_method(
        self,
    ) -> Any:
        for method_name in (
            "search",
            "retrieve",
            "query",
            "similarity_search",
        ):
            method = getattr(
                self.retriever,
                method_name,
                None,
            )

            if method is not None:
                return method

        raise MultiQueryRetrievalError(
            "Retriever exposes no supported "
            "search method"
        )

    # ========================================================
    # Merge
    # ========================================================

    def _merge_results(
        self,
        query_results: Sequence[
            tuple[
                QueryVariant,
                list[Any],
            ]
        ],
    ) -> dict[str, _Candidate]:
        candidates: dict[
            str,
            _Candidate,
        ] = {}

        for (
            variant_index,
            (
                variant,
                results,
            ),
        ) in enumerate(
            query_results,
            start=1,
        ):
            query_key = (
                f"q{variant_index}"
            )

            for fallback_rank, result in (
                enumerate(
                    results,
                    start=1,
                )
            ):
                data = (
                    self._result_to_mapping(
                        result
                    )
                )

                candidate_id = (
                    self._candidate_id(
                        data
                    )
                )

                if candidate_id is None:
                    continue

                candidate = (
                    candidates.get(
                        candidate_id
                    )
                )

                if candidate is None:
                    candidate = (
                        self._candidate_from_data(
                            candidate_id,
                            data,
                        )
                    )

                    candidates[
                        candidate_id
                    ] = candidate

                self._merge_candidate_data(
                    candidate,
                    data,
                )

                rank = self._rank(
                    data,
                    fallback=fallback_rank,
                )

                score = self._score(
                    data
                )

                candidate.retrieval_count += 1

                if (
                    variant.query
                    not in candidate.matched_queries
                ):
                    candidate.matched_queries.append(
                        variant.query
                    )

                variant_type = (
                    variant.variant_type.value
                )

                if (
                    variant_type
                    not in candidate.matched_query_types
                ):
                    candidate.matched_query_types.append(
                        variant_type
                    )

                candidate.source_ranks[
                    query_key
                ] = rank

                candidate.source_scores[
                    query_key
                ] = score

                candidate.metadata.setdefault(
                    "_multi_query",
                    {},
                )

                candidate.metadata[
                    "_multi_query"
                ][query_key] = {
                    "query":
                        variant.query,
                    "variant_type":
                        variant.variant_type.value,
                    "weight":
                        variant.weight,
                    "rank":
                        rank,
                    "score":
                        score,
                }

        return candidates

    # ========================================================
    # RRF
    # ========================================================

    def _calculate_rrf(
        self,
        candidates: Mapping[
            str,
            _Candidate,
        ],
        *,
        total_queries: int,
    ) -> None:
        if not candidates:
            return

        maximum_rrf = 0.0

        for candidate in candidates.values():
            score = 0.0

            query_metadata = (
                candidate.metadata.get(
                    "_multi_query",
                    {}
                )
            )

            for info in (
                query_metadata.values()
            ):
                rank = max(
                    1,
                    int(
                        info.get(
                            "rank",
                            1,
                        )
                    ),
                )

                weight = max(
                    0.0,
                    float(
                        info.get(
                            "weight",
                            1.0,
                        )
                    ),
                )

                score += (
                    weight
                    / (
                        self.rrf_k
                        + rank
                    )
                )

            candidate.fused_score = score

            maximum_rrf = max(
                maximum_rrf,
                score,
            )

        # Normalize to 0..1.
        if maximum_rrf > 0:
            for candidate in (
                candidates.values()
            ):
                candidate.fused_score /= (
                    maximum_rrf
                )

        # Consensus boost:
        # chunks independently discovered by multiple
        # query variants are more reliable retrieval
        # candidates.
        denominator = max(
            total_queries,
            1,
        )

        for candidate in candidates.values():
            unique_queries = len(
                candidate.matched_queries
            )

            consensus = (
                unique_queries
                / denominator
            )

            boost = (
                1.0
                + self.consensus_boost
                * max(
                    unique_queries - 1,
                    0,
                )
            )

            candidate.fused_score *= (
                boost
            )

            # Small bounded consensus component.
            candidate.fused_score += (
                consensus
                * self.consensus_boost
            )

            candidate.fused_score = min(
                candidate.fused_score,
                1.0,
            )

    # ========================================================
    # Candidate extraction
    # ========================================================

    def _candidate_from_data(
        self,
        candidate_id: str,
        data: Mapping[str, Any],
    ) -> _Candidate:
        metadata = (
            self._metadata(data)
        )

        return _Candidate(
            id=candidate_id,
            text=str(
                data.get("text")
                or data.get("content")
                or data.get(
                    "chunk_text"
                )
                or metadata.get("text")
                or ""
            ),
            organization_id=(
                self._string_or_none(
                    data.get(
                        "organization_id"
                    )
                    or metadata.get(
                        "organization_id"
                    )
                )
            ),
            document_id=(
                self._string_or_none(
                    data.get(
                        "document_id"
                    )
                    or metadata.get(
                        "document_id"
                    )
                )
            ),
            contract_id=(
                self._string_or_none(
                    data.get(
                        "contract_id"
                    )
                    or metadata.get(
                        "contract_id"
                    )
                )
            ),
            parent_id=(
                self._string_or_none(
                    data.get("parent_id")
                    or data.get(
                        "parent_chunk_id"
                    )
                    or metadata.get(
                        "parent_id"
                    )
                    or metadata.get(
                        "parent_chunk_id"
                    )
                )
            ),
            title=(
                data.get("title")
                or metadata.get("title")
            ),
            page_number=(
                self._int_or_none(
                    data.get(
                        "page_number"
                    )
                    or metadata.get(
                        "page_number"
                    )
                )
            ),
            chunk_index=(
                self._int_or_none(
                    data.get(
                        "chunk_index"
                    )
                    if data.get(
                        "chunk_index"
                    )
                    is not None
                    else metadata.get(
                        "chunk_index"
                    )
                )
            ),
            metadata=metadata,
        )

    def _merge_candidate_data(
        self,
        candidate: _Candidate,
        data: Mapping[str, Any],
    ) -> None:
        metadata = self._metadata(
            data
        )

        # Existing metadata wins for conflicting
        # canonical values, while new metadata fills
        # missing fields.
        candidate.metadata = {
            **metadata,
            **candidate.metadata,
        }

        if not candidate.text:
            candidate.text = str(
                data.get("text")
                or data.get("content")
                or data.get(
                    "chunk_text"
                )
                or metadata.get("text")
                or ""
            )

        mappings = (
            (
                "organization_id",
                "organization_id",
            ),
            (
                "document_id",
                "document_id",
            ),
            (
                "contract_id",
                "contract_id",
            ),
            (
                "parent_id",
                "parent_id",
            ),
            (
                "title",
                "title",
            ),
        )

        for attr, key in mappings:
            if getattr(
                candidate,
                attr,
            ) is not None:
                continue

            value = (
                data.get(key)
                or metadata.get(key)
            )

            if value is not None:
                if attr == "title":
                    setattr(
                        candidate,
                        attr,
                        value,
                    )
                else:
                    setattr(
                        candidate,
                        attr,
                        str(value),
                    )

        if candidate.parent_id is None:
            value = (
                data.get(
                    "parent_chunk_id"
                )
                or metadata.get(
                    "parent_chunk_id"
                )
            )

            if value is not None:
                candidate.parent_id = (
                    str(value)
                )

        if candidate.page_number is None:
            candidate.page_number = (
                self._int_or_none(
                    data.get(
                        "page_number"
                    )
                    or metadata.get(
                        "page_number"
                    )
                )
            )

        if candidate.chunk_index is None:
            value = data.get(
                "chunk_index"
            )

            if value is None:
                value = metadata.get(
                    "chunk_index"
                )

            candidate.chunk_index = (
                self._int_or_none(
                    value
                )
            )

        matched_terms = data.get(
            "matched_terms",
            [],
        )

        if isinstance(
            matched_terms,
            (list, tuple, set),
        ):
            candidate.matched_terms = (
                list(
                    dict.fromkeys(
                        [
                            *candidate.matched_terms,
                            *[
                                str(term)
                                for term
                                in matched_terms
                            ],
                        ]
                    )
                )
            )

        sources = data.get(
            "sources",
            [],
        )

        if isinstance(sources, str):
            sources = [sources]

        if isinstance(
            sources,
            (list, tuple, set),
        ):
            candidate.sources.update(
                str(source)
                for source in sources
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

        if isinstance(
            result,
            Mapping,
        ):
            return dict(result)

        if hasattr(
            result,
            "to_dict",
        ):
            value = result.to_dict()

            if isinstance(
                value,
                Mapping,
            ):
                return dict(value)

        if hasattr(
            result,
            "model_dump",
        ):
            value = (
                result.model_dump()
            )

            if isinstance(
                value,
                Mapping,
            ):
                return dict(value)

        output: dict[
            str,
            Any,
        ] = {}

        fields = (
            "id",
            "chunk_id",
            "point_id",
            "text",
            "content",
            "chunk_text",
            "score",
            "rank",
            "organization_id",
            "document_id",
            "contract_id",
            "parent_id",
            "parent_chunk_id",
            "title",
            "page_number",
            "chunk_index",
            "metadata",
            "payload",
            "matched_terms",
            "sources",
        )

        for field_name in fields:
            if hasattr(
                result,
                field_name,
            ):
                output[field_name] = (
                    getattr(
                        result,
                        field_name,
                    )
                )

        return output

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
            metadata = data.get(
                "metadata"
            )

            if isinstance(
                metadata,
                Mapping,
            ):
                value = (
                    metadata.get(
                        "chunk_id"
                    )
                    or metadata.get("id")
                )

        return (
            str(value)
            if value is not None
            else None
        )

    @staticmethod
    def _metadata(
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        output: dict[
            str,
            Any,
        ] = {}

        metadata = data.get(
            "metadata"
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            output.update(metadata)

        payload = data.get(
            "payload"
        )

        if isinstance(
            payload,
            Mapping,
        ):
            output.update(payload)

        return output

    # ========================================================
    # Score / rank
    # ========================================================

    @staticmethod
    def _rank(
        data: Mapping[str, Any],
        *,
        fallback: int,
    ) -> int:
        try:
            value = int(
                data.get(
                    "rank",
                    fallback,
                )
            )

            return max(
                value,
                1,
            )

        except (
            TypeError,
            ValueError,
        ):
            return fallback

    @staticmethod
    def _score(
        data: Mapping[str, Any],
    ) -> float:
        for key in (
            "score",
            "rerank_score",
            "hybrid_score",
            "similarity",
            "relevance_score",
        ):
            value = data.get(key)

            if value is None:
                continue

            try:
                return float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        return 0.0

    # ========================================================
    # Result
    # ========================================================

    @staticmethod
    def _to_result(
        candidate: _Candidate,
        *,
        rank: int,
    ) -> MultiQueryResult:
        metadata = dict(
            candidate.metadata
        )

        # Internal diagnostic metadata is useful for
        # observability but can be removed before
        # generation if desired.
        return MultiQueryResult(
            id=candidate.id,
            text=candidate.text,
            score=round(
                candidate.fused_score,
                6,
            ),
            rank=rank,
            organization_id=(
                candidate.organization_id
            ),
            document_id=(
                candidate.document_id
            ),
            contract_id=(
                candidate.contract_id
            ),
            parent_id=(
                candidate.parent_id
            ),
            title=candidate.title,
            page_number=(
                candidate.page_number
            ),
            chunk_index=(
                candidate.chunk_index
            ),
            metadata=metadata,
            matched_queries=list(
                candidate.matched_queries
            ),
            matched_query_types=list(
                candidate.matched_query_types
            ),
            matched_terms=list(
                candidate.matched_terms
            ),
            sources=sorted(
                candidate.sources
            ),
            source_ranks=dict(
                candidate.source_ranks
            ),
            source_scores=dict(
                candidate.source_scores
            ),
            retrieval_count=(
                candidate.retrieval_count
            ),
        )

    # ========================================================
    # Query utilities
    # ========================================================

    def _variant(
        self,
        query: str,
        variant_type: QueryVariantType,
    ) -> QueryVariant:
        return QueryVariant(
            query=self._clean_query(
                query
            ),
            variant_type=variant_type,
            weight=(
                self.DEFAULT_QUERY_WEIGHTS.get(
                    variant_type,
                    1.0,
                )
            ),
        )

    @staticmethod
    def _clean_query(
        query: Any,
    ) -> str:
        if query is None:
            return ""

        text = str(query)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _deduplicate_queries(
        variants: Sequence[
            QueryVariant
        ],
    ) -> list[QueryVariant]:
        seen: set[str] = set()

        output: list[
            QueryVariant
        ] = []

        for variant in variants:
            normalized = re.sub(
                r"\s+",
                " ",
                variant.query,
            ).strip().casefold()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            output.append(variant)

        return output

    # ========================================================
    # LLM prompt
    # ========================================================

    @staticmethod
    def _query_generation_prompt(
        query: str,
        number: int,
    ) -> str:
        return f"""
You are a legal document retrieval query generator.

Generate {number} alternative search queries for the
user's legal-document question.

Goal:
Improve retrieval recall while preserving the user's
original legal intent.

Generate variants that may include:
- legal terminology
- contract terminology
- concise keyword formulation
- semantic paraphrase
- clause-oriented wording
- alternative terminology commonly used in agreements

Rules:
- Do not answer the question.
- Do not invent facts.
- Do not change parties, dates, amounts, jurisdictions,
  clause numbers, document names, or other constraints.
- Preserve quoted terms.
- Keep each query concise.
- Return one query per line.
- Do not add explanations.

Original query:
{query}
""".strip()

    # ========================================================
    # Async helper
    # ========================================================

    @staticmethod
    async def _await_if_needed(
        value: Any,
    ) -> Any:
        if inspect.isawaitable(
            value
        ):
            return await value

        return value

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(value)

    @staticmethod
    def _int_or_none(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ========================================================
    # Health
    # ========================================================

    async def health_check(
        self,
    ) -> dict[str, Any]:
        retriever_health = None

        method = getattr(
            self.retriever,
            "health_check",
            None,
        )

        if method is not None:
            try:
                retriever_health = method()

                retriever_health = (
                    await self._await_if_needed(
                        retriever_health
                    )
                )

            except Exception as exc:
                retriever_health = {
                    "status": "unhealthy",
                    "error": str(exc),
                }

        return {
            "service":
                "multi_query_retriever",
            "status":
                "healthy",
            "max_queries":
                self.max_queries,
            "candidate_multiplier":
                self.candidate_multiplier,
            "rrf_k":
                self.rrf_k,
            "include_original":
                self.include_original,
            "consensus_boost":
                self.consensus_boost,
            "query_rewriter":
                self.query_rewriter
                is not None,
            "legal_llm":
                self.legal_llm
                is not None,
            "retriever":
                retriever_health,
        }


# ============================================================
# Compatibility aliases
# ============================================================

MultiQuerySearch = MultiQueryRetriever
MultiQueryEngine = MultiQueryRetriever