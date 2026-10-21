# backend/rag/chunking/semantic_chunker.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence
import hashlib
import math
import re
import uuid


class SemanticChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE_GROUP = "sentence_group"
    SECTION = "section"
    TABLE = "table"
    LIST = "list"
    UNKNOWN = "unknown"


class SemanticBoundaryReason(str, Enum):
    DOCUMENT_START = "document_start"
    DOCUMENT_END = "document_end"
    HEADING = "heading"
    PAGE_BOUNDARY = "page_boundary"
    SEMANTIC_SHIFT = "semantic_shift"
    MAX_SIZE = "max_size"
    TABLE_BOUNDARY = "table_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"


@dataclass(slots=True)
class SemanticUnit:
    """
    Atomic unit used before semantic grouping.

    Usually a sentence or a short paragraph.
    """

    text: str
    index: int

    page_number: Optional[int] = None

    character_start: Optional[int] = None
    character_end: Optional[int] = None

    heading: Optional[str] = None

    unit_type: SemanticChunkType = (
        SemanticChunkType.PARAGRAPH
    )

    embedding: Optional[list[float]] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class SemanticChunk:
    chunk_id: str
    document_id: str
    text: str

    chunk_index: int = 0

    chunk_type: SemanticChunkType = (
        SemanticChunkType.UNKNOWN
    )

    title: Optional[str] = None

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    character_start: Optional[int] = None
    character_end: Optional[int] = None

    unit_start: Optional[int] = None
    unit_end: Optional[int] = None

    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    boundary_reason: Optional[
        SemanticBoundaryReason
    ] = None

    semantic_score: Optional[float] = None

    token_count: int = 0
    word_count: int = 0
    character_count: int = 0

    source_filename: Optional[str] = None

    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    document_version: Optional[str] = None

    content_hash: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        self.character_count = len(
            self.text
        )

        self.word_count = len(
            self.text.split()
        )

        if not self.content_hash:
            self.content_hash = (
                hashlib.sha256(
                    self.text.encode(
                        "utf-8"
                    )
                ).hexdigest()
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_type":
                self.chunk_type.value,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "character_start":
                self.character_start,
            "character_end":
                self.character_end,
            "unit_start": self.unit_start,
            "unit_end": self.unit_end,
            "previous_chunk_id":
                self.previous_chunk_id,
            "next_chunk_id":
                self.next_chunk_id,
            "boundary_reason": (
                self.boundary_reason.value
                if self.boundary_reason
                else None
            ),
            "semantic_score":
                self.semantic_score,
            "token_count":
                self.token_count,
            "word_count":
                self.word_count,
            "character_count":
                self.character_count,
            "source_filename":
                self.source_filename,
            "tenant_id":
                self.tenant_id,
            "organization_id":
                self.organization_id,
            "document_version":
                self.document_version,
            "content_hash":
                self.content_hash,
            "metadata":
                self.metadata,
        }


@dataclass(slots=True)
class SemanticChunkingResult:
    chunks: list[SemanticChunk] = field(
        default_factory=list
    )

    document_id: Optional[str] = None

    chunk_count: int = 0
    unit_count: int = 0

    average_chunk_size: float = 0.0

    semantic_boundaries: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.chunk_count = len(
            self.chunks
        )

        if self.chunks:
            self.average_chunk_size = (
                sum(
                    chunk.character_count
                    for chunk in self.chunks
                )
                / len(self.chunks)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id":
                self.document_id,
            "chunk_count":
                self.chunk_count,
            "unit_count":
                self.unit_count,
            "average_chunk_size":
                self.average_chunk_size,
            "semantic_boundaries":
                self.semantic_boundaries,
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "metadata":
                self.metadata,
        }


@dataclass(slots=True)
class SemanticChunkerConfig:
    """
    Semantic chunking configuration.

    Similar adjacent units remain together.

    A chunk boundary is created when adjacent semantic similarity
    drops below semantic_threshold, while size constraints provide
    deterministic safety limits.
    """

    target_chunk_size: int = 1200
    max_chunk_size: int = 1800
    min_chunk_size: int = 200

    overlap_units: int = 1

    semantic_threshold: float = 0.55

    # Avoid creating semantic boundaries from tiny similarity changes.
    semantic_margin: float = 0.05

    max_units_per_chunk: int = 20

    preserve_pages: bool = True
    preserve_headings: bool = True
    preserve_tables: bool = True

    split_sentences: bool = True

    include_heading_context: bool = True

    deterministic_ids: bool = True

    link_neighbors: bool = True

    normalize_embeddings: bool = True

    embedding_batch_size: int = 32

    max_document_characters: int = (
        20_000_000
    )


class SemanticChunkingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = (
            "SEMANTIC_CHUNKING_ERROR"
        ),
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.cause = cause


class SemanticChunker:
    """
    Embedding-aware semantic chunker.

    Pipeline
    --------
    Normalized document
        ↓
    Page / paragraph parsing
        ↓
    Sentence/semantic units
        ↓
    Embeddings
        ↓
    Adjacent cosine similarity
        ↓
    Semantic boundary detection
        ↓
    Size-aware grouping
        ↓
    Semantic chunks

    The embedding model is injected so this module stays independent
    from a specific Hugging Face/OpenAI/local embedding provider.

    Supported embedding interfaces:

        embedder.encode_documents(texts)

        embedder.encode(texts)

        embedder.embed_documents(texts)

        embedder(texts)

    This module does not modify the legal meaning of source text.
    """

    PAGE_MARKER = re.compile(
        r"(?m)^\[PAGE\s+(\d+)\]\s*$",
        re.IGNORECASE,
    )

    TABLE_MARKER = re.compile(
        r"^\[TABLE\s+\d+\]",
        re.IGNORECASE,
    )

    LIST_PATTERN = re.compile(
        r"""
        ^\s*
        (?:
            [-*•]
            |
            \d+[\.\)]
            |
            [a-zA-Z][\.\)]
            |
            \([a-zA-Z0-9]+\)
        )
        \s+
        """,
        re.VERBOSE,
    )

    NUMBERED_HEADING = re.compile(
        r"""
        ^\s*
        (?:
            section\s+|
            article\s+|
            clause\s+
        )?
        (?P<number>
            \d+
            (?:\.\d+)*
        )
        [\.\:\-\s]+
        (?P<title>
            [A-Z][^\n]{1,180}
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        config: Optional[
            SemanticChunkerConfig
        ] = None,
        *,
        embedder: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = (
            config
            or SemanticChunkerConfig()
        )

        self.embedder = embedder
        self.tokenizer = tokenizer

        self._validate_config()

    # ==============================================================
    # Public API
    # ==============================================================

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> SemanticChunkingResult:

        if not document_id:
            raise SemanticChunkingError(
                "document_id is required.",
                code="MISSING_DOCUMENT_ID",
            )

        if not text or not text.strip():
            return SemanticChunkingResult(
                document_id=document_id
            )

        if (
            len(text)
            > self.config
            .max_document_characters
        ):
            raise SemanticChunkingError(
                "Document exceeds configured "
                "semantic chunking limit.",
                code="DOCUMENT_TOO_LARGE",
            )

        units = self._create_units(
            text
        )

        if not units:
            return SemanticChunkingResult(
                document_id=document_id
            )

        embedding_mode = "fallback"

        if self.embedder is not None:
            self._embed_units(
                units
            )

            embedding_mode = (
                "embedding"
            )

        groups = self._group_units(
            units
        )

        chunks: list[
            SemanticChunk
        ] = []

        semantic_boundaries = 0

        document_metadata = (
            metadata or {}
        )

        for chunk_index, group in enumerate(
            groups
        ):
            if not group:
                continue

            chunk = self._create_chunk(
                group,
                document_id=document_id,
                chunk_index=chunk_index,
                source_filename=(
                    source_filename
                ),
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                document_version=(
                    document_version
                ),
                document_metadata=(
                    document_metadata
                ),
            )

            if (
                chunk.boundary_reason
                == SemanticBoundaryReason
                .SEMANTIC_SHIFT
            ):
                semantic_boundaries += 1

            chunks.append(chunk)

        if self.config.link_neighbors:
            self._link_neighbors(
                chunks
            )

        return SemanticChunkingResult(
            chunks=chunks,
            document_id=document_id,
            unit_count=len(units),
            semantic_boundaries=(
                semantic_boundaries
            ),
            metadata={
                "semantic_threshold":
                    self.config
                    .semantic_threshold,
                "target_chunk_size":
                    self.config
                    .target_chunk_size,
                "max_chunk_size":
                    self.config
                    .max_chunk_size,
                "embedding_mode":
                    embedding_mode,
            },
        )

    def chunk_pages(
        self,
        pages: Sequence[Any],
        *,
        document_id: str,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> SemanticChunkingResult:

        parts: list[str] = []

        for index, page in enumerate(
            pages,
            start=1,
        ):
            if isinstance(page, dict):
                page_number = page.get(
                    "page_number",
                    index,
                )

                page_text = page.get(
                    "text",
                    "",
                )

            else:
                page_number = getattr(
                    page,
                    "page_number",
                    index,
                )

                page_text = getattr(
                    page,
                    "text",
                    "",
                )

            if not page_text:
                continue

            parts.append(
                f"[PAGE {page_number}]\n"
                f"{str(page_text).strip()}"
            )

        return self.chunk(
            "\n\n".join(parts),
            document_id=document_id,
            source_filename=(
                source_filename
            ),
            tenant_id=tenant_id,
            organization_id=(
                organization_id
            ),
            document_version=(
                document_version
            ),
            metadata=metadata,
        )

    # ==============================================================
    # Unit creation
    # ==============================================================

    def _create_units(
        self,
        text: str,
    ) -> list[SemanticUnit]:

        pages = self._split_pages(
            text
        )

        units: list[
            SemanticUnit
        ] = []

        unit_index = 0
        global_offset = 0

        current_heading: Optional[
            str
        ] = None

        for page_number, page_text in pages:
            paragraphs = (
                self._paragraphs_with_offsets(
                    page_text
                )
            )

            for paragraph, start, end in paragraphs:
                paragraph = (
                    paragraph.strip()
                )

                if not paragraph:
                    continue

                if self._is_heading(
                    paragraph
                ):
                    current_heading = (
                        paragraph
                    )

                    units.append(
                        SemanticUnit(
                            text=paragraph,
                            index=unit_index,
                            page_number=(
                                page_number
                            ),
                            character_start=(
                                global_offset
                                + start
                            ),
                            character_end=(
                                global_offset
                                + end
                            ),
                            heading=(
                                current_heading
                            ),
                            unit_type=(
                                SemanticChunkType
                                .SECTION
                            ),
                            metadata={
                                "heading": True,
                            },
                        )
                    )

                    unit_index += 1
                    continue

                if (
                    self.config
                    .preserve_tables
                    and self.TABLE_MARKER
                    .match(paragraph)
                ):
                    units.append(
                        SemanticUnit(
                            text=paragraph,
                            index=unit_index,
                            page_number=(
                                page_number
                            ),
                            character_start=(
                                global_offset
                                + start
                            ),
                            character_end=(
                                global_offset
                                + end
                            ),
                            heading=(
                                current_heading
                            ),
                            unit_type=(
                                SemanticChunkType
                                .TABLE
                            ),
                            metadata={
                                "table": True,
                            },
                        )
                    )

                    unit_index += 1
                    continue

                if (
                    self.config
                    .split_sentences
                ):
                    sentence_units = (
                        self._sentence_units(
                            paragraph,
                            base_start=(
                                global_offset
                                + start
                            ),
                        )
                    )

                    for (
                        sentence,
                        sentence_start,
                        sentence_end,
                    ) in sentence_units:

                        unit_type = (
                            SemanticChunkType
                            .LIST
                            if self.LIST_PATTERN
                            .match(sentence)
                            else
                            SemanticChunkType
                            .SENTENCE_GROUP
                        )

                        units.append(
                            SemanticUnit(
                                text=sentence,
                                index=(
                                    unit_index
                                ),
                                page_number=(
                                    page_number
                                ),
                                character_start=(
                                    sentence_start
                                ),
                                character_end=(
                                    sentence_end
                                ),
                                heading=(
                                    current_heading
                                ),
                                unit_type=(
                                    unit_type
                                ),
                            )
                        )

                        unit_index += 1

                else:
                    units.append(
                        SemanticUnit(
                            text=paragraph,
                            index=unit_index,
                            page_number=(
                                page_number
                            ),
                            character_start=(
                                global_offset
                                + start
                            ),
                            character_end=(
                                global_offset
                                + end
                            ),
                            heading=(
                                current_heading
                            ),
                            unit_type=(
                                SemanticChunkType
                                .PARAGRAPH
                            ),
                        )
                    )

                    unit_index += 1

            global_offset += (
                len(page_text) + 1
            )

        return units

    # ==============================================================
    # Embeddings
    # ==============================================================

    def _embed_units(
        self,
        units: list[
            SemanticUnit
        ],
    ) -> None:

        texts = [
            unit.text
            for unit in units
        ]

        batch_size = (
            self.config
            .embedding_batch_size
        )

        embeddings: list[
            list[float]
        ] = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                start:
                start + batch_size
            ]

            batch_embeddings = (
                self._call_embedder(
                    batch
                )
            )

            if (
                len(batch_embeddings)
                != len(batch)
            ):
                raise SemanticChunkingError(
                    "Embedding provider returned "
                    "an unexpected number of "
                    "embeddings.",
                    code=(
                        "EMBEDDING_COUNT_MISMATCH"
                    ),
                )

            embeddings.extend(
                batch_embeddings
            )

        for unit, embedding in zip(
            units,
            embeddings,
        ):
            vector = [
                float(value)
                for value in embedding
            ]

            if (
                self.config
                .normalize_embeddings
            ):
                vector = (
                    self._normalize_vector(
                        vector
                    )
                )

            unit.embedding = vector

    def _call_embedder(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        try:
            if hasattr(
                self.embedder,
                "encode_documents",
            ):
                result = (
                    self.embedder
                    .encode_documents(
                        texts
                    )
                )

            elif hasattr(
                self.embedder,
                "embed_documents",
            ):
                result = (
                    self.embedder
                    .embed_documents(
                        texts
                    )
                )

            elif hasattr(
                self.embedder,
                "encode",
            ):
                result = (
                    self.embedder.encode(
                        texts
                    )
                )

            elif callable(
                self.embedder
            ):
                result = (
                    self.embedder(
                        texts
                    )
                )

            else:
                raise (
                    SemanticChunkingError(
                        "Unsupported embedding "
                        "provider interface.",
                        code=(
                            "INVALID_EMBEDDER"
                        ),
                    )
                )

            # Support your EmbeddingResult object.
            if hasattr(
                result,
                "embeddings",
            ):
                result = (
                    result.embeddings
                )

            if hasattr(
                result,
                "tolist",
            ):
                result = (
                    result.tolist()
                )

            return [
                (
                    embedding.tolist()
                    if hasattr(
                        embedding,
                        "tolist",
                    )
                    else list(embedding)
                )
                for embedding in result
            ]

        except SemanticChunkingError:
            raise

        except Exception as exc:
            raise SemanticChunkingError(
                "Semantic embedding failed.",
                code="EMBEDDING_ERROR",
                cause=exc,
            ) from exc

    # ==============================================================
    # Semantic grouping
    # ==============================================================

    def _group_units(
        self,
        units: list[
            SemanticUnit
        ],
    ) -> list[
        list[SemanticUnit]
    ]:

        if not units:
            return []

        groups: list[
            list[SemanticUnit]
        ] = []

        current: list[
            SemanticUnit
        ] = []

        current_size = 0

        for index, unit in enumerate(
            units
        ):
            if not current:
                current = [unit]
                current_size = len(
                    unit.text
                )
                continue

            previous = current[-1]

            candidate_size = (
                current_size
                + len(unit.text)
                + 2
            )

            reason = (
                self._boundary_reason(
                    previous=previous,
                    current=unit,
                    current_size=(
                        current_size
                    ),
                    candidate_size=(
                        candidate_size
                    ),
                    unit_count=len(
                        current
                    ),
                )
            )

            if reason is not None:
                current[-1].metadata[
                    "boundary_after"
                ] = reason.value

                groups.append(current)

                overlap = (
                    self._overlap_units(
                        current,
                        reason,
                    )
                )

                current = [
                    *overlap,
                    unit,
                ]

                current_size = sum(
                    len(item.text)
                    + 2
                    for item in current
                )

            else:
                current.append(unit)

                current_size = (
                    candidate_size
                )

        if current:
            current[-1].metadata[
                "boundary_after"
            ] = (
                SemanticBoundaryReason
                .DOCUMENT_END.value
            )

            groups.append(current)

        return self._deduplicate_groups(
            groups
        )

    def _boundary_reason(
        self,
        *,
        previous: SemanticUnit,
        current: SemanticUnit,
        current_size: int,
        candidate_size: int,
        unit_count: int,
    ) -> Optional[
        SemanticBoundaryReason
    ]:

        if (
            candidate_size
            > self.config.max_chunk_size
        ):
            return (
                SemanticBoundaryReason
                .MAX_SIZE
            )

        if (
            unit_count
            >= self.config
            .max_units_per_chunk
        ):
            return (
                SemanticBoundaryReason
                .MAX_SIZE
            )

        if (
            self.config.preserve_headings
            and current.metadata.get(
                "heading"
            )
            and current_size
            >= self.config.min_chunk_size
        ):
            return (
                SemanticBoundaryReason
                .HEADING
            )

        if (
            self.config.preserve_tables
            and (
                previous.unit_type
                == SemanticChunkType.TABLE
                or current.unit_type
                == SemanticChunkType.TABLE
            )
            and current_size
            >= self.config.min_chunk_size
        ):
            return (
                SemanticBoundaryReason
                .TABLE_BOUNDARY
            )

        if (
            self.config.preserve_pages
            and previous.page_number
            is not None
            and current.page_number
            is not None
            and previous.page_number
            != current.page_number
            and current_size
            >= self.config.target_chunk_size
        ):
            return (
                SemanticBoundaryReason
                .PAGE_BOUNDARY
            )

        if (
            current_size
            < self.config.min_chunk_size
        ):
            return None

        similarity = (
            self._unit_similarity(
                previous,
                current,
            )
        )

        if similarity is None:
            return None

        previous.metadata[
            "similarity_to_next"
        ] = similarity

        current.metadata[
            "similarity_to_previous"
        ] = similarity

        threshold = (
            self.config
            .semantic_threshold
        )

        # Once target size has been reached, normal threshold applies.
        if (
            current_size
            >= self.config.target_chunk_size
            and similarity < threshold
        ):
            return (
                SemanticBoundaryReason
                .SEMANTIC_SHIFT
            )

        # Before target size, require a stronger semantic shift.
        strong_shift_threshold = max(
            0.0,
            threshold
            - self.config
            .semantic_margin,
        )

        if (
            similarity
            < strong_shift_threshold
            and current_size
            >= self.config.min_chunk_size
        ):
            return (
                SemanticBoundaryReason
                .SEMANTIC_SHIFT
            )

        return None

    # ==============================================================
    # Similarity
    # ==============================================================

    def _unit_similarity(
        self,
        left: SemanticUnit,
        right: SemanticUnit,
    ) -> Optional[float]:

        if (
            left.embedding is not None
            and right.embedding is not None
        ):
            return self._cosine_similarity(
                left.embedding,
                right.embedding,
            )

        # Deterministic lexical fallback keeps this module useful in
        # local tests when an embedding model is unavailable.
        return self._lexical_similarity(
            left.text,
            right.text,
        )

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:

        if (
            not left
            or not right
            or len(left) != len(right)
        ):
            return 0.0

        dot = sum(
            a * b
            for a, b in zip(
                left,
                right,
            )
        )

        left_norm = math.sqrt(
            sum(
                value * value
                for value in left
            )
        )

        right_norm = math.sqrt(
            sum(
                value * value
                for value in right
            )
        )

        denominator = (
            left_norm
            * right_norm
        )

        if denominator == 0:
            return 0.0

        score = (
            dot / denominator
        )

        return max(
            -1.0,
            min(
                1.0,
                score,
            ),
        )

    @staticmethod
    def _lexical_similarity(
        left: str,
        right: str,
    ) -> float:

        left_tokens = {
            token
            for token in re.findall(
                r"[A-Za-z0-9_]+",
                left.lower(),
            )
            if len(token) > 1
        }

        right_tokens = {
            token
            for token in re.findall(
                r"[A-Za-z0-9_]+",
                right.lower(),
            )
            if len(token) > 1
        }

        if (
            not left_tokens
            or not right_tokens
        ):
            return 0.0

        intersection = len(
            left_tokens
            & right_tokens
        )

        denominator = math.sqrt(
            len(left_tokens)
            * len(right_tokens)
        )

        if denominator == 0:
            return 0.0

        return (
            intersection
            / denominator
        )

    # ==============================================================
    # Chunk creation
    # ==============================================================

    def _create_chunk(
        self,
        units: list[
            SemanticUnit
        ],
        *,
        document_id: str,
        chunk_index: int,
        source_filename: Optional[str],
        tenant_id: Optional[str],
        organization_id: Optional[str],
        document_version: Optional[str],
        document_metadata: dict[
            str,
            Any,
        ],
    ) -> SemanticChunk:

        text = self._join_units(
            units
        )

        first = units[0]
        last = units[-1]

        pages = [
            unit.page_number
            for unit in units
            if unit.page_number
            is not None
        ]

        page_start = (
            min(pages)
            if pages
            else None
        )

        page_end = (
            max(pages)
            if pages
            else None
        )

        title = next(
            (
                unit.heading
                for unit in units
                if unit.heading
            ),
            None,
        )

        chunk_type = (
            self._chunk_type(
                units
            )
        )

        boundary = (
            last.metadata.get(
                "boundary_after"
            )
        )

        try:
            boundary_reason = (
                SemanticBoundaryReason(
                    boundary
                )
                if boundary
                else None
            )
        except ValueError:
            boundary_reason = None

        semantic_scores = [
            unit.metadata[
                "similarity_to_next"
            ]
            for unit in units
            if isinstance(
                unit.metadata.get(
                    "similarity_to_next"
                ),
                (int, float),
            )
        ]

        average_score = (
            sum(semantic_scores)
            / len(semantic_scores)
            if semantic_scores
            else None
        )

        chunk_id = self._make_id(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
        )

        return SemanticChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            title=title,
            page_start=page_start,
            page_end=page_end,
            character_start=(
                first.character_start
            ),
            character_end=(
                last.character_end
            ),
            unit_start=first.index,
            unit_end=last.index,
            boundary_reason=(
                boundary_reason
            ),
            semantic_score=(
                average_score
            ),
            token_count=(
                self._token_count(
                    text
                )
            ),
            source_filename=(
                source_filename
            ),
            tenant_id=tenant_id,
            organization_id=(
                organization_id
            ),
            document_version=(
                document_version
            ),
            metadata={
                **document_metadata,
                "unit_count":
                    len(units),
                "semantic_score":
                    average_score,
                "boundary_reason":
                    (
                        boundary_reason.value
                        if boundary_reason
                        else None
                    ),
            },
        )

    @staticmethod
    def _join_units(
        units: list[
            SemanticUnit
        ],
    ) -> str:

        output: list[str] = []

        for unit in units:
            if not output:
                output.append(
                    unit.text
                )
                continue

            if (
                unit.unit_type
                in {
                    SemanticChunkType
                    .SECTION,
                    SemanticChunkType
                    .TABLE,
                    SemanticChunkType
                    .LIST,
                }
            ):
                output.append(
                    "\n\n"
                    + unit.text
                )
            else:
                output.append(
                    " " + unit.text
                )

        return "".join(
            output
        ).strip()

    @staticmethod
    def _chunk_type(
        units: list[
            SemanticUnit
        ],
    ) -> SemanticChunkType:

        types = {
            unit.unit_type
            for unit in units
        }

        if (
            SemanticChunkType.TABLE
            in types
        ):
            return (
                SemanticChunkType.TABLE
            )

        if (
            SemanticChunkType.SECTION
            in types
        ):
            return (
                SemanticChunkType.SECTION
            )

        if (
            SemanticChunkType.LIST
            in types
        ):
            return (
                SemanticChunkType.LIST
            )

        if (
            SemanticChunkType
            .SENTENCE_GROUP
            in types
        ):
            return (
                SemanticChunkType
                .SENTENCE_GROUP
            )

        return (
            SemanticChunkType.PARAGRAPH
        )

    # ==============================================================
    # Page parsing
    # ==============================================================

    def _split_pages(
        self,
        text: str,
    ) -> list[
        tuple[
            Optional[int],
            str,
        ]
    ]:

        matches = list(
            self.PAGE_MARKER
            .finditer(text)
        )

        if not matches:
            return [
                (
                    None,
                    text.strip(),
                )
            ]

        output: list[
            tuple[
                Optional[int],
                str,
            ]
        ] = []

        if matches[0].start() > 0:
            prefix = text[
                : matches[0].start()
            ].strip()

            if prefix:
                output.append(
                    (
                        None,
                        prefix,
                    )
                )

        for index, match in enumerate(
            matches
        ):
            page_number = int(
                match.group(1)
            )

            start = match.end()

            end = (
                matches[
                    index + 1
                ].start()
                if index + 1
                < len(matches)
                else len(text)
            )

            output.append(
                (
                    page_number,
                    text[
                        start:end
                    ].strip(),
                )
            )

        return output

    # ==============================================================
    # Paragraph/sentence parsing
    # ==============================================================

    @staticmethod
    def _paragraphs_with_offsets(
        text: str,
    ) -> list[
        tuple[
            str,
            int,
            int,
        ]
    ]:

        output: list[
            tuple[
                str,
                int,
                int,
            ]
        ] = []

        pattern = re.compile(
            r"\S(?:.*?\S)?"
            r"(?=\n\s*\n|\Z)",
            re.DOTALL,
        )

        for match in pattern.finditer(
            text
        ):
            value = (
                match.group(0)
                .strip()
            )

            if value:
                output.append(
                    (
                        value,
                        match.start(),
                        match.end(),
                    )
                )

        return output

    def _sentence_units(
        self,
        paragraph: str,
        *,
        base_start: int,
    ) -> list[
        tuple[
            str,
            int,
            int,
        ]
    ]:

        # Conservative splitter suitable for contracts. It avoids
        # splitting on every period because legal text contains
        # abbreviations, section numbers and decimal values.
        pattern = re.compile(
            r".+?"
            r"(?:"
            r"(?<=[.!?;])"
            r"(?=\s+[A-Z(\"“])"
            r"|\Z"
            r")",
            re.DOTALL,
        )

        output: list[
            tuple[
                str,
                int,
                int,
            ]
        ] = []

        for match in pattern.finditer(
            paragraph
        ):
            sentence = (
                match.group(0)
                .strip()
            )

            if not sentence:
                continue

            local_start = (
                match.start()
            )

            local_end = (
                match.end()
            )

            output.append(
                (
                    sentence,
                    base_start
                    + local_start,
                    base_start
                    + local_end,
                )
            )

        if not output:
            output.append(
                (
                    paragraph,
                    base_start,
                    base_start
                    + len(paragraph),
                )
            )

        return output

    # ==============================================================
    # Heading detection
    # ==============================================================

    def _is_heading(
        self,
        text: str,
    ) -> bool:

        text = text.strip()

        if (
            not text
            or len(text) > 180
        ):
            return False

        if self.NUMBERED_HEADING.match(
            text
        ):
            return True

        words = text.split()

        if len(words) > 15:
            return False

        letters = [
            character
            for character in text
            if character.isalpha()
        ]

        if not letters:
            return False

        uppercase = sum(
            character.isupper()
            for character in letters
        )

        return (
            uppercase
            / len(letters)
            >= 0.85
        )

    # ==============================================================
    # Overlap
    # ==============================================================

    def _overlap_units(
        self,
        current: list[
            SemanticUnit
        ],
        reason: SemanticBoundaryReason,
    ) -> list[
        SemanticUnit
    ]:

        # Do not leak content across strong structural boundaries.
        if reason in {
            SemanticBoundaryReason.HEADING,
            SemanticBoundaryReason
            .TABLE_BOUNDARY,
        }:
            return []

        count = (
            self.config
            .overlap_units
        )

        if count <= 0:
            return []

        return current[
            -count:
        ]

    # ==============================================================
    # Vector utilities
    # ==============================================================

    @staticmethod
    def _normalize_vector(
        vector: list[float],
    ) -> list[float]:

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        if norm == 0:
            return vector

        return [
            value / norm
            for value in vector
        ]

    # ==============================================================
    # Neighbor relationships
    # ==============================================================

    @staticmethod
    def _link_neighbors(
        chunks: list[
            SemanticChunk
        ],
    ) -> None:

        for index, chunk in enumerate(
            chunks
        ):
            if index > 0:
                chunk.previous_chunk_id = (
                    chunks[
                        index - 1
                    ].chunk_id
                )

            if (
                index + 1
                < len(chunks)
            ):
                chunk.next_chunk_id = (
                    chunks[
                        index + 1
                    ].chunk_id
                )

    # ==============================================================
    # Token counting
    # ==============================================================

    def _token_count(
        self,
        text: str,
    ) -> int:

        if self.tokenizer is None:
            return max(
                1,
                int(
                    len(
                        text.split()
                    )
                    * 1.3
                ),
            )

        try:
            if hasattr(
                self.tokenizer,
                "encode",
            ):
                encoded = (
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                )

                return len(
                    encoded
                )

            if callable(
                self.tokenizer
            ):
                result = (
                    self.tokenizer(
                        text
                    )
                )

                if isinstance(
                    result,
                    dict,
                ):
                    ids = result.get(
                        "input_ids"
                    )

                    if ids is not None:
                        return len(
                            ids
                        )

        except Exception:
            pass

        return max(
            1,
            int(
                len(
                    text.split()
                )
                * 1.3
            ),
        )

    # ==============================================================
    # IDs
    # ==============================================================

    def _make_id(
        self,
        *,
        document_id: str,
        chunk_index: int,
        text: str,
    ) -> str:

        if (
            not self.config
            .deterministic_ids
        ):
            return (
                "semantic_"
                + str(
                    uuid.uuid4()
                )
            )

        content_hash = (
            hashlib.sha256(
                text.encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        identity = (
            f"{document_id}|"
            f"{chunk_index}|"
            f"{content_hash}"
        )

        digest = (
            hashlib.sha256(
                identity.encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        return (
            f"semantic_"
            f"{digest[:24]}"
        )

    # ==============================================================
    # Deduplication
    # ==============================================================

    @staticmethod
    def _deduplicate_groups(
        groups: Iterable[
            list[SemanticUnit]
        ],
    ) -> list[
        list[SemanticUnit]
    ]:

        output: list[
            list[SemanticUnit]
        ] = []

        previous_signature: Optional[
            tuple[int, ...]
        ] = None

        for group in groups:
            if not group:
                continue

            signature = tuple(
                unit.index
                for unit in group
            )

            if (
                signature
                == previous_signature
            ):
                continue

            output.append(
                group
            )

            previous_signature = (
                signature
            )

        return output

    # ==============================================================
    # Validation
    # ==============================================================

    def _validate_config(
        self,
    ) -> None:

        config = self.config

        if (
            config.target_chunk_size
            <= 0
        ):
            raise ValueError(
                "target_chunk_size must "
                "be greater than zero."
            )

        if (
            config.max_chunk_size
            < config.target_chunk_size
        ):
            raise ValueError(
                "max_chunk_size must be >= "
                "target_chunk_size."
            )

        if (
            config.min_chunk_size
            < 0
        ):
            raise ValueError(
                "min_chunk_size cannot "
                "be negative."
            )

        if not (
            -1.0
            <= config.semantic_threshold
            <= 1.0
        ):
            raise ValueError(
                "semantic_threshold must "
                "be between -1 and 1."
            )

        if (
            config.overlap_units < 0
        ):
            raise ValueError(
                "overlap_units cannot "
                "be negative."
            )

        if (
            config.embedding_batch_size
            <= 0
        ):
            raise ValueError(
                "embedding_batch_size must "
                "be greater than zero."
            )


def semantic_chunk(
    text: str,
    *,
    document_id: str,
    embedder: Any = None,
    tokenizer: Any = None,
    source_filename: Optional[str] = None,
    tenant_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    document_version: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> SemanticChunkingResult:
    """
    Convenience API.

    Example:

        embedder = EmbeddingModel()

        result = semantic_chunk(
            text,
            document_id="contract-001",
            embedder=embedder,
        )
    """

    chunker = SemanticChunker(
        embedder=embedder,
        tokenizer=tokenizer,
    )

    return chunker.chunk(
        text,
        document_id=document_id,
        source_filename=source_filename,
        tenant_id=tenant_id,
        organization_id=organization_id,
        document_version=document_version,
        metadata=metadata,
    )