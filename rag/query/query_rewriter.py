"""
Query Rewriter
==============

Rewrites user queries into retrieval-optimized queries for the
Enterprise Legal Document Intelligence AI platform.

Responsibilities:
- Normalize legal/user queries.
- Resolve lightweight conversational references.
- Expand common legal abbreviations.
- Remove retrieval noise.
- Generate semantic search queries.
- Generate keyword/BM25 queries.
- Generate multiple retrieval variants.
- Preserve important legal entities, dates, clause names, and identifiers.
- Avoid changing the legal meaning of the original query.

This module intentionally does not perform retrieval. It prepares queries
for downstream BM25, vector, hybrid, multi-query, and reranking pipelines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)


class RewriteStrategy(str, Enum):
    NONE = "none"
    NORMALIZE = "normalize"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    MULTI_QUERY = "multi_query"
    AUTO = "auto"


@dataclass(slots=True)
class QueryRewriteRequest:
    query: str
    strategy: RewriteStrategy = RewriteStrategy.AUTO
    conversation_history: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_queries: int = 4


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    keyword_query: str
    semantic_query: str
    alternative_queries: list[str] = field(default_factory=list)
    expanded_terms: dict[str, str] = field(default_factory=dict)
    strategy: RewriteStrategy = RewriteStrategy.AUTO

    @property
    def all_queries(self) -> list[str]:
        values = [
            self.rewritten_query,
            self.semantic_query,
            self.keyword_query,
            *self.alternative_queries,
        ]

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = value.strip()

            if not value:
                continue

            key = value.casefold()

            if key not in seen:
                seen.add(key)
                result.append(value)

        return result


class QueryRewriteLLM(Protocol):
    """Minimal interface required from an optional LLM client."""

    async def generate(self, prompt: str) -> str:
        ...


LEGAL_ABBREVIATIONS: dict[str, str] = {
    "nda": "non-disclosure agreement",
    "msa": "master services agreement",
    "sow": "statement of work",
    "sla": "service level agreement",
    "dpa": "data processing agreement",
    "dpia": "data protection impact assessment",
    "gdpr": "General Data Protection Regulation",
    "ccpa": "California Consumer Privacy Act",
    "ip": "intellectual property",
    "pii": "personally identifiable information",
    "phi": "protected health information",
    "tos": "terms of service",
    "eula": "end-user license agreement",
    "mou": "memorandum of understanding",
    "loi": "letter of intent",
    "po": "purchase order",
    "kpi": "key performance indicator",
}

QUESTION_PREFIXES = (
    "show me",
    "find me",
    "find",
    "search for",
    "search",
    "tell me",
    "give me",
    "can you show",
    "can you find",
    "could you show",
    "could you find",
    "please show",
    "please find",
    "please search",
)

REFERENCE_PATTERNS = (
    r"\bthis contract\b",
    r"\bthis agreement\b",
    r"\bthat contract\b",
    r"\bthat agreement\b",
    r"\bthis document\b",
    r"\bthat document\b",
    r"\bit\b",
    r"\bthe document\b",
)


class QueryRewriter:
    """
    Production-oriented query rewriting service.

    The default implementation is deterministic and therefore works without
    an external LLM. An optional LLM can be injected for richer semantic
    rewriting and multi-query generation.
    """

    def __init__(
        self,
        llm: QueryRewriteLLM | None = None,
        *,
        max_query_length: int = 2_000,
        default_max_queries: int = 4,
    ) -> None:
        if max_query_length <= 0:
            raise ValueError("max_query_length must be positive")

        if default_max_queries <= 0:
            raise ValueError("default_max_queries must be positive")

        self._llm = llm
        self._max_query_length = max_query_length
        self._default_max_queries = default_max_queries

    async def rewrite(
        self,
        query: str,
        *,
        strategy: RewriteStrategy | str = RewriteStrategy.AUTO,
        conversation_history: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        max_queries: int | None = None,
    ) -> QueryRewriteResult:
        """
        Rewrite a query for the legal retrieval pipeline.
        """

        strategy = self._coerce_strategy(strategy)
        original = self._validate_query(query)

        history = tuple(conversation_history or ())
        metadata = metadata or {}

        limit = max_queries or self._default_max_queries
        limit = max(1, min(limit, 10))

        normalized = self.normalize(original)

        if strategy is RewriteStrategy.NONE:
            return QueryRewriteResult(
                original_query=original,
                rewritten_query=original,
                keyword_query=original,
                semantic_query=original,
                strategy=strategy,
            )

        contextual = self._resolve_context(
            normalized,
            history,
            metadata,
        )

        expanded, expansions = self.expand_legal_terms(contextual)

        keyword_query = self.to_keyword_query(expanded)
        semantic_query = self.to_semantic_query(expanded)

        rewritten = semantic_query

        if strategy is RewriteStrategy.NORMALIZE:
            rewritten = normalized
            semantic_query = normalized
            keyword_query = normalized

        elif strategy is RewriteStrategy.KEYWORD:
            rewritten = keyword_query

        elif strategy is RewriteStrategy.SEMANTIC:
            rewritten = semantic_query

        elif strategy in {
            RewriteStrategy.AUTO,
            RewriteStrategy.MULTI_QUERY,
        }:
            rewritten = await self._semantic_rewrite(
                original_query=original,
                deterministic_query=semantic_query,
                history=history,
            )

        alternatives: list[str] = []

        if strategy in {
            RewriteStrategy.AUTO,
            RewriteStrategy.MULTI_QUERY,
        }:
            alternatives = await self.generate_alternatives(
                rewritten,
                keyword_query=keyword_query,
                max_queries=limit,
            )

        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            keyword_query=keyword_query,
            semantic_query=semantic_query,
            alternative_queries=alternatives,
            expanded_terms=expansions,
            strategy=strategy,
        )

    async def rewrite_request(
        self,
        request: QueryRewriteRequest,
    ) -> QueryRewriteResult:
        return await self.rewrite(
            request.query,
            strategy=request.strategy,
            conversation_history=request.conversation_history,
            metadata=request.metadata,
            max_queries=request.max_queries,
        )

    def normalize(self, query: str) -> str:
        """
        Normalize whitespace and harmless punctuation without destroying
        legally meaningful text.
        """

        query = query.replace("\u00a0", " ")
        query = query.replace("\u2018", "'").replace("\u2019", "'")
        query = query.replace("\u201c", '"').replace("\u201d", '"')

        query = re.sub(r"[\r\n\t]+", " ", query)
        query = re.sub(r"\s+", " ", query)

        # Collapse repeated punctuation generated by noisy user input.
        query = re.sub(r"\?{2,}", "?", query)
        query = re.sub(r"!{2,}", "!", query)

        return query.strip()

    def expand_legal_terms(
        self,
        query: str,
    ) -> tuple[str, dict[str, str]]:
        """
        Expand recognized legal abbreviations while retaining the original
        abbreviation to improve both lexical and semantic retrieval.
        """

        result = query
        expansions: dict[str, str] = {}

        for abbreviation, full_form in LEGAL_ABBREVIATIONS.items():
            pattern = re.compile(
                rf"\b{re.escape(abbreviation)}\b",
                flags=re.IGNORECASE,
            )

            if not pattern.search(result):
                continue

            # Do not repeatedly expand a term that is already written as
            # "NDA (non-disclosure agreement)".
            if full_form.casefold() in result.casefold():
                continue

            expansions[abbreviation.upper()] = full_form

            result = pattern.sub(
                lambda match: (
                    f"{match.group(0)} ({full_form})"
                ),
                result,
            )

        return result, expansions

    def to_keyword_query(self, query: str) -> str:
        """
        Produce a concise lexical/BM25-friendly query.
        """

        result = query.strip()

        lowered = result.casefold()

        for prefix in QUESTION_PREFIXES:
            if lowered.startswith(prefix + " "):
                result = result[len(prefix):].strip()
                break

        result = re.sub(
            r"^(what|which|where|when|who|why|how)\s+(is|are|does|do|did|was|were)\s+",
            "",
            result,
            flags=re.IGNORECASE,
        )

        # Remove conversational filler, not legal terms.
        result = re.sub(
            r"\b(?:please|kindly|basically|actually)\b",
            " ",
            result,
            flags=re.IGNORECASE,
        )

        result = re.sub(r"\s+", " ", result)

        return result.strip(" ?!.,") or query

    def to_semantic_query(self, query: str) -> str:
        """
        Convert conversational questions into a cleaner semantic-search form.
        """

        result = query.strip()

        result = re.sub(
            r"^(?:please\s+)?(?:can|could|would)\s+you\s+",
            "",
            result,
            flags=re.IGNORECASE,
        )

        result = re.sub(
            r"^(?:show|find|search|tell|give)\s+(?:me\s+)?",
            "",
            result,
            flags=re.IGNORECASE,
        )

        result = re.sub(r"\s+", " ", result)

        return result.strip(" ?!") or query

    async def generate_alternatives(
        self,
        query: str,
        *,
        keyword_query: str | None = None,
        max_queries: int = 4,
    ) -> list[str]:
        """
        Generate retrieval variants.

        Deterministic variants are always available. If an LLM is configured,
        it can contribute additional legal-search formulations.
        """

        max_queries = max(1, min(max_queries, 10))

        candidates: list[str] = []

        if keyword_query:
            candidates.append(keyword_query)

        candidates.extend(self._rule_based_variants(query))

        if self._llm is not None:
            try:
                llm_queries = await self._llm_multi_query(
                    query,
                    max_queries=max_queries,
                )
                candidates.extend(llm_queries)
            except Exception:
                logger.exception(
                    "LLM multi-query generation failed; "
                    "using deterministic alternatives."
                )

        return self._deduplicate(
            candidates,
            exclude={query.casefold()},
        )[:max_queries]

    def _rule_based_variants(self, query: str) -> list[str]:
        variants: list[str] = []

        clean = query.strip(" ?!.")

        variants.append(clean)

        # Contract-focused variant.
        if not re.search(
            r"\b(contract|agreement|document)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            variants.append(f"{clean} contract agreement")

        # Clause-focused retrieval variant.
        if re.search(
            r"\b("
            r"termination|indemnity|liability|confidentiality|"
            r"renewal|payment|warranty|privacy|governing law|"
            r"intellectual property|obligation"
            r")\b",
            clean,
            flags=re.IGNORECASE,
        ):
            variants.append(f"{clean} clause provision")

        # Risk/compliance variant.
        if re.search(
            r"\b(risk|compliance|violation|exposure|penalty)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            variants.append(
                f"{clean} legal risk compliance obligation"
            )

        return variants

    async def _semantic_rewrite(
        self,
        *,
        original_query: str,
        deterministic_query: str,
        history: Sequence[str],
    ) -> str:
        if self._llm is None:
            return deterministic_query

        history_text = "\n".join(history[-4:])

        prompt = f"""
You rewrite legal-document search queries.

Rules:
1. Preserve the user's legal meaning.
2. Never invent parties, dates, clauses, obligations, or facts.
3. Preserve contract names, document identifiers, dates, monetary values,
   jurisdictions, and named entities.
4. Make the query suitable for semantic and hybrid retrieval.
5. Remove conversational filler.
6. Return only one rewritten query.
7. Do not answer the question.

Conversation context:
{history_text or "None"}

Original query:
{original_query}

Deterministic normalized query:
{deterministic_query}

Rewritten query:
""".strip()

        try:
            response = await self._llm.generate(prompt)

            rewritten = self._clean_llm_output(response)

            if not rewritten:
                return deterministic_query

            return rewritten

        except Exception:
            logger.exception(
                "Query rewrite LLM failed; falling back to deterministic query."
            )
            return deterministic_query

    async def _llm_multi_query(
        self,
        query: str,
        *,
        max_queries: int,
    ) -> list[str]:
        if self._llm is None:
            return []

        prompt = f"""
Generate {max_queries} alternative search queries for a legal document
retrieval system.

Requirements:
- Preserve the exact legal intent.
- Do not invent facts.
- Use different useful retrieval formulations.
- Include legal terminology where appropriate.
- Keep named entities, dates, identifiers and monetary values unchanged.
- One query per line.
- Do not number the lines.
- Do not answer the query.

Query:
{query}
""".strip()

        response = await self._llm.generate(prompt)

        values: list[str] = []

        for line in response.splitlines():
            line = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            if line:
                values.append(line)

        return values[:max_queries]

    def _resolve_context(
        self,
        query: str,
        history: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str:
        """
        Resolve only safe document references.

        We intentionally avoid broad coreference resolution because guessing
        legal entities can materially change retrieval meaning.
        """

        document_name = self._extract_document_context(metadata)

        if not document_name:
            document_name = self._infer_document_from_history(history)

        if not document_name:
            return query

        result = query

        replacements = (
            "this contract",
            "this agreement",
            "that contract",
            "that agreement",
            "this document",
            "that document",
            "the document",
        )

        for phrase in replacements:
            result = re.sub(
                rf"\b{re.escape(phrase)}\b",
                document_name,
                result,
                flags=re.IGNORECASE,
            )

        return result

    @staticmethod
    def _extract_document_context(
        metadata: Mapping[str, Any],
    ) -> str | None:
        for key in (
            "document_name",
            "document_title",
            "contract_name",
            "contract_title",
        ):
            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    @staticmethod
    def _infer_document_from_history(
        history: Sequence[str],
    ) -> str | None:
        """
        Conservative history inference.

        Only explicit quoted document names are accepted.
        """

        pattern = re.compile(
            r"""["']([^"']+\.(?:pdf|docx?|txt))["']""",
            flags=re.IGNORECASE,
        )

        for message in reversed(history[-6:]):
            match = pattern.search(message)

            if match:
                return match.group(1).strip()

        return None

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if len(query) > self._max_query_length:
            raise ValueError(
                f"query exceeds maximum length "
                f"({self._max_query_length} characters)"
            )

        if "\x00" in query:
            raise ValueError("query contains null bytes")

        return query

    @staticmethod
    def _coerce_strategy(
        strategy: RewriteStrategy | str,
    ) -> RewriteStrategy:
        if isinstance(strategy, RewriteStrategy):
            return strategy

        try:
            return RewriteStrategy(strategy.lower().strip())
        except (AttributeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in RewriteStrategy)
            raise ValueError(
                f"Invalid rewrite strategy. Allowed: {allowed}"
            ) from exc

    @staticmethod
    def _clean_llm_output(value: str) -> str:
        if not isinstance(value, str):
            return ""

        value = value.strip()

        value = re.sub(
            r"^(?:rewritten query|query)\s*:\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = value.strip().strip('"').strip("'")
        value = re.sub(r"\s+", " ", value)

        return value

    @staticmethod
    def _deduplicate(
        values: Sequence[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()

        result: list[str] = []
        seen: set[str] = set(exclude)

        for value in values:
            value = value.strip()

            if not value:
                continue

            key = value.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result


_default_rewriter = QueryRewriter()


async def rewrite_query(
    query: str,
    *,
    strategy: RewriteStrategy | str = RewriteStrategy.AUTO,
    conversation_history: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_queries: int = 4,
) -> QueryRewriteResult:
    """Convenience API for callers that do not require dependency injection."""

    return await _default_rewriter.rewrite(
        query,
        strategy=strategy,
        conversation_history=conversation_history,
        metadata=metadata,
        max_queries=max_queries,
    )


__all__ = [
    "LEGAL_ABBREVIATIONS",
    "QueryRewriteLLM",
    "QueryRewriteRequest",
    "QueryRewriteResult",
    "QueryRewriter",
    "RewriteStrategy",
    "rewrite_query",
]