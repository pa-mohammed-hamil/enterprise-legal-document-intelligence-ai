# backend/app/rag/retrieval/bm25.py

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID


# ============================================================
# Exceptions
# ============================================================


class BM25Error(Exception):
    """Base BM25 retrieval error."""


class BM25IndexNotReadyError(BM25Error):
    """Raised when search is attempted before an index exists."""


class BM25ValidationError(BM25Error):
    """Raised when invalid BM25 input is supplied."""


# ============================================================
# Models
# ============================================================


@dataclass(slots=True)
class BM25Document:
    id: str
    text: str

    organization_id: str | None = None
    document_id: str | None = None
    contract_id: str | None = None
    parent_id: str | None = None

    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BM25SearchResult:
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

    metadata: dict[str, Any] = field(default_factory=dict)

    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "organization_id": self.organization_id,
            "document_id": self.document_id,
            "contract_id": self.contract_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "matched_terms": self.matched_terms,
        }


# ============================================================
# Tokenizer
# ============================================================


class LegalTokenizer:
    """
    Lightweight tokenizer optimized for legal/document retrieval.

    Keeps useful legal terms and identifiers such as:

        section 12.4
        clause 7.2
        indemnification
        non-compete
        force majeure
        GDPR
        USD 500000

    A production deployment can replace this tokenizer with a
    Hugging Face or domain-specific tokenizer without changing
    BM25Retriever.
    """

    TOKEN_PATTERN = re.compile(
        r"""
        [A-Za-z]+(?:[-'][A-Za-z]+)*
        |
        \d+(?:\.\d+)*
        |
        [A-Za-z]+\d+
        """,
        re.VERBOSE,
    )

    DEFAULT_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    def __init__(
        self,
        *,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        stopwords: set[str] | None = None,
        min_token_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = (
            stopwords
            if stopwords is not None
            else self.DEFAULT_STOPWORDS
        )
        self.min_token_length = max(1, min_token_length)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        tokens = self.TOKEN_PATTERN.findall(text)

        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        output: list[str] = []

        for token in tokens:
            if len(token) < self.min_token_length:
                continue

            if (
                self.remove_stopwords
                and token in self.stopwords
            ):
                continue

            output.append(token)

        return output


# ============================================================
# BM25 Retriever
# ============================================================


class BM25Retriever:
    """
    Tenant-aware in-memory BM25 lexical retriever.

    Intended architecture:

        legal chunks
            ↓
        BM25Retriever
            ↓
        lexical results
            ┐
            ├── hybrid_search.py
        vector results
            ┘
            ↓
        permission_aware_search.py
            ↓
        multi_query.py
            ↓
        hf_reranker.py
            ↓
        RAG generation + citations

    BM25 formula:

        IDF(q) =
            ln(
                1 +
                (N - n(q) + 0.5)
                /
                (n(q) + 0.5)
            )

        score(D, Q) =
            Σ IDF(q) *
            (
                f(q,D) * (k1 + 1)
            )
            /
            (
                f(q,D) +
                k1 *
                (
                    1 - b +
                    b * |D| / avgdl
                )
            )
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: LegalTokenizer | None = None,
    ) -> None:
        if k1 <= 0:
            raise BM25ValidationError(
                "k1 must be greater than zero"
            )

        if not 0 <= b <= 1:
            raise BM25ValidationError(
                "b must be between 0 and 1"
            )

        self.k1 = float(k1)
        self.b = float(b)

        self.tokenizer = tokenizer or LegalTokenizer()

        self._documents: dict[str, BM25Document] = {}

        self._document_tokens: dict[
            str,
            list[str],
        ] = {}

        self._term_frequencies: dict[
            str,
            Counter[str],
        ] = {}

        self._document_frequencies: Counter[str] = Counter()

        self._document_lengths: dict[str, int] = {}

        self._average_document_length: float = 0.0

        self._lock = threading.RLock()

    # ========================================================
    # Properties
    # ========================================================

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def average_document_length(self) -> float:
        return self._average_document_length

    @property
    def vocabulary_size(self) -> int:
        return len(self._document_frequencies)

    @property
    def is_ready(self) -> bool:
        return bool(self._documents)

    # ========================================================
    # Build index
    # ========================================================

    def build_index(
        self,
        documents: Iterable[
            BM25Document | Mapping[str, Any]
        ],
    ) -> int:
        """
        Replace the current index with supplied documents.

        Returns the number of indexed documents.
        """

        with self._lock:
            self.clear()

            prepared = [
                self._coerce_document(document)
                for document in documents
            ]

            for document in prepared:
                self._add_document_internal(document)

            self._recalculate_statistics()

            return len(self._documents)

    # ========================================================
    # Add documents
    # ========================================================

    def add_documents(
        self,
        documents: Iterable[
            BM25Document | Mapping[str, Any]
        ],
    ) -> int:
        """
        Add or replace documents in the current index.
        """

        count = 0

        with self._lock:
            for raw_document in documents:
                document = self._coerce_document(
                    raw_document
                )

                if document.id in self._documents:
                    self._remove_document_internal(
                        document.id
                    )

                self._add_document_internal(
                    document
                )

                count += 1

            self._recalculate_statistics()

        return count

    def add_document(
        self,
        document: BM25Document | Mapping[str, Any],
    ) -> BM25Document:
        prepared = self._coerce_document(document)

        self.add_documents([prepared])

        return prepared

    # ========================================================
    # Remove
    # ========================================================

    def remove_document(
        self,
        document_id: str | UUID,
    ) -> bool:
        key = str(document_id)

        with self._lock:
            if key not in self._documents:
                return False

            self._remove_document_internal(key)

            self._recalculate_statistics()

            return True

    def remove_documents(
        self,
        document_ids: Iterable[str | UUID],
    ) -> int:
        removed = 0

        with self._lock:
            for document_id in document_ids:
                key = str(document_id)

                if key not in self._documents:
                    continue

                self._remove_document_internal(key)
                removed += 1

            self._recalculate_statistics()

        return removed

    def remove_by_source_document(
        self,
        document_id: str | UUID,
        *,
        organization_id: str | UUID | None = None,
    ) -> int:
        source_id = str(document_id)

        tenant_id = (
            str(organization_id)
            if organization_id is not None
            else None
        )

        targets = []

        for chunk_id, document in self._documents.items():
            if document.document_id != source_id:
                continue

            if (
                tenant_id is not None
                and document.organization_id != tenant_id
            ):
                continue

            targets.append(chunk_id)

        return self.remove_documents(targets)

    # ========================================================
    # Clear
    # ========================================================

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._document_tokens.clear()
            self._term_frequencies.clear()
            self._document_frequencies.clear()
            self._document_lengths.clear()
            self._average_document_length = 0.0

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        organization_id: str | UUID | None = None,
        document_ids: Sequence[str | UUID] | None = None,
        contract_ids: Sequence[str | UUID] | None = None,
        filters: Mapping[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[BM25SearchResult]:
        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        with self._lock:
            if not self._documents:
                return []

            query_tokens = self.tokenizer.tokenize(query)

            if not query_tokens:
                return []

            query_terms = Counter(query_tokens)

            tenant_id = (
                str(organization_id)
                if organization_id is not None
                else None
            )

            allowed_document_ids = (
                {
                    str(value)
                    for value in document_ids
                }
                if document_ids
                else None
            )

            allowed_contract_ids = (
                {
                    str(value)
                    for value in contract_ids
                }
                if contract_ids
                else None
            )

            scored: list[
                tuple[
                    float,
                    BM25Document,
                    list[str],
                ]
            ] = []

            for chunk_id, document in self._documents.items():
                if (
                    tenant_id is not None
                    and document.organization_id
                    != tenant_id
                ):
                    continue

                if (
                    allowed_document_ids is not None
                    and document.document_id
                    not in allowed_document_ids
                ):
                    continue

                if (
                    allowed_contract_ids is not None
                    and document.contract_id
                    not in allowed_contract_ids
                ):
                    continue

                if not self._matches_filters(
                    document,
                    filters,
                ):
                    continue

                score, matched_terms = (
                    self._score_document(
                        chunk_id,
                        query_terms,
                    )
                )

                if score <= min_score:
                    continue

                scored.append(
                    (
                        score,
                        document,
                        matched_terms,
                    )
                )

            scored.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            results: list[BM25SearchResult] = []

            for rank, (
                score,
                document,
                matched_terms,
            ) in enumerate(
                scored[:top_k],
                start=1,
            ):
                results.append(
                    BM25SearchResult(
                        id=document.id,
                        text=document.text,
                        score=round(score, 6),
                        rank=rank,
                        organization_id=(
                            document.organization_id
                        ),
                        document_id=document.document_id,
                        contract_id=document.contract_id,
                        parent_id=document.parent_id,
                        title=document.title,
                        page_number=document.page_number,
                        chunk_index=document.chunk_index,
                        metadata=dict(document.metadata),
                        matched_terms=matched_terms,
                    )
                )

            return results

    # ========================================================
    # Batch search
    # ========================================================

    def batch_search(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 10,
        organization_id: str | UUID | None = None,
        document_ids: Sequence[str | UUID] | None = None,
        contract_ids: Sequence[str | UUID] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, list[BM25SearchResult]]:
        return {
            query: self.search(
                query,
                top_k=top_k,
                organization_id=organization_id,
                document_ids=document_ids,
                contract_ids=contract_ids,
                filters=filters,
            )
            for query in queries
        }

    # ========================================================
    # Search as dictionaries
    # ========================================================

    def search_dicts(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            result.to_dict()
            for result in self.search(
                query,
                **kwargs,
            )
        ]

    # ========================================================
    # Scoring
    # ========================================================

    def _score_document(
        self,
        document_id: str,
        query_terms: Counter[str],
    ) -> tuple[float, list[str]]:
        frequencies = self._term_frequencies.get(
            document_id
        )

        if frequencies is None:
            return 0.0, []

        document_length = self._document_lengths.get(
            document_id,
            0,
        )

        if document_length == 0:
            return 0.0, []

        average_length = (
            self._average_document_length or 1.0
        )

        score = 0.0
        matched_terms: list[str] = []

        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(
                term,
                0,
            )

            if term_frequency <= 0:
                continue

            matched_terms.append(term)

            idf = self._idf(term)

            length_normalization = (
                1.0
                - self.b
                + self.b
                * (
                    document_length
                    / average_length
                )
            )

            denominator = (
                term_frequency
                + self.k1
                * length_normalization
            )

            term_score = (
                idf
                * (
                    term_frequency
                    * (self.k1 + 1.0)
                )
                / denominator
            )

            # Small query-frequency contribution. This keeps
            # repeated meaningful query terms useful without
            # letting them dominate retrieval.
            query_boost = (
                1.0
                + math.log1p(
                    query_frequency - 1
                )
                * 0.15
            )

            score += term_score * query_boost

        return score, matched_terms

    def _idf(self, term: str) -> float:
        n = self._document_frequencies.get(
            term,
            0,
        )

        if n == 0:
            return 0.0

        total = len(self._documents)

        return math.log(
            1.0
            + (
                total
                - n
                + 0.5
            )
            / (
                n
                + 0.5
            )
        )

    # ========================================================
    # Filtering
    # ========================================================

    @staticmethod
    def _matches_filters(
        document: BM25Document,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        standard_fields = {
            "organization_id",
            "document_id",
            "contract_id",
            "parent_id",
            "title",
            "page_number",
            "chunk_index",
        }

        for key, expected in filters.items():
            if key in standard_fields:
                actual = getattr(
                    document,
                    key,
                    None,
                )
            else:
                actual = document.metadata.get(key)

            if isinstance(
                expected,
                (list, tuple, set, frozenset),
            ):
                expected_values = {
                    str(value)
                    for value in expected
                }

                if str(actual) not in expected_values:
                    return False

            elif callable(expected):
                if not expected(actual):
                    return False

            elif actual != expected:
                # UUID/string compatibility.
                if str(actual) != str(expected):
                    return False

        return True

    # ========================================================
    # Index internals
    # ========================================================

    def _add_document_internal(
        self,
        document: BM25Document,
    ) -> None:
        if not document.id:
            raise BM25ValidationError(
                "Document id is required"
            )

        if not document.text:
            return

        tokens = self.tokenizer.tokenize(
            document.text
        )

        if not tokens:
            return

        self._documents[document.id] = document
        self._document_tokens[document.id] = tokens

        frequencies = Counter(tokens)

        self._term_frequencies[
            document.id
        ] = frequencies

        self._document_lengths[
            document.id
        ] = len(tokens)

        for term in frequencies:
            self._document_frequencies[
                term
            ] += 1

    def _remove_document_internal(
        self,
        document_id: str,
    ) -> None:
        frequencies = self._term_frequencies.get(
            document_id,
            Counter(),
        )

        for term in frequencies:
            current = self._document_frequencies.get(
                term,
                0,
            )

            if current <= 1:
                self._document_frequencies.pop(
                    term,
                    None,
                )
            else:
                self._document_frequencies[
                    term
                ] = current - 1

        self._documents.pop(
            document_id,
            None,
        )

        self._document_tokens.pop(
            document_id,
            None,
        )

        self._term_frequencies.pop(
            document_id,
            None,
        )

        self._document_lengths.pop(
            document_id,
            None,
        )

    def _recalculate_statistics(self) -> None:
        if not self._document_lengths:
            self._average_document_length = 0.0
            return

        self._average_document_length = (
            sum(
                self._document_lengths.values()
            )
            / len(self._document_lengths)
        )

    # ========================================================
    # Document conversion
    # ========================================================

    @staticmethod
    def _coerce_document(
        value: BM25Document | Mapping[str, Any],
    ) -> BM25Document:
        if isinstance(value, BM25Document):
            return value

        if not isinstance(value, Mapping):
            raise BM25ValidationError(
                "Document must be BM25Document "
                "or mapping"
            )

        document_id = (
            value.get("id")
            or value.get("chunk_id")
        )

        if document_id is None:
            raise BM25ValidationError(
                "Document requires id or chunk_id"
            )

        text = (
            value.get("text")
            or value.get("content")
            or value.get("chunk_text")
            or ""
        )

        metadata = dict(
            value.get("metadata")
            or value.get("metadata_json")
            or {}
        )

        def string_or_none(
            item: Any,
        ) -> str | None:
            return (
                str(item)
                if item is not None
                else None
            )

        return BM25Document(
            id=str(document_id),
            text=str(text),
            organization_id=string_or_none(
                value.get("organization_id")
            ),
            document_id=string_or_none(
                value.get("document_id")
            ),
            contract_id=string_or_none(
                value.get("contract_id")
            ),
            parent_id=string_or_none(
                value.get("parent_id")
                or value.get("parent_chunk_id")
            ),
            title=value.get("title"),
            page_number=value.get("page_number"),
            chunk_index=value.get(
                "chunk_index"
            ),
            metadata=metadata,
        )

    # ========================================================
    # Index inspection
    # ========================================================

    def get_document(
        self,
        document_id: str | UUID,
    ) -> BM25Document | None:
        return self._documents.get(
            str(document_id)
        )

    def contains(
        self,
        document_id: str | UUID,
    ) -> bool:
        return (
            str(document_id)
            in self._documents
        )

    def document_frequency(
        self,
        term: str,
    ) -> int:
        tokens = self.tokenizer.tokenize(term)

        if not tokens:
            return 0

        return self._document_frequencies.get(
            tokens[0],
            0,
        )

    def idf(
        self,
        term: str,
    ) -> float:
        tokens = self.tokenizer.tokenize(term)

        if not tokens:
            return 0.0

        return self._idf(tokens[0])

    # ========================================================
    # Stats
    # ========================================================

    def statistics(self) -> dict[str, Any]:
        return {
            "documents": self.document_count,
            "vocabulary_size": self.vocabulary_size,
            "average_document_length": round(
                self.average_document_length,
                4,
            ),
            "k1": self.k1,
            "b": self.b,
            "ready": self.is_ready,
        }

    # ========================================================
    # Health
    # ========================================================

    def health_check(self) -> dict[str, Any]:
        return {
            "service": "bm25",
            "status": "healthy",
            "ready": self.is_ready,
            "documents": self.document_count,
            "vocabulary_size": self.vocabulary_size,
            "average_document_length": round(
                self.average_document_length,
                4,
            ),
            "k1": self.k1,
            "b": self.b,
        }


# ============================================================
# Compatibility alias
# ============================================================

BM25Search = BM25Retriever