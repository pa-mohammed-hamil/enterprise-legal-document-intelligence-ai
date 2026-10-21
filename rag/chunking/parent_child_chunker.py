vibration_anomaly# backend/rag/chunking/parent_child_chunker.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence
import hashlib
import re
import uuid


class ChunkLevel(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARENT = "parent"
    CHILD = "child"


@dataclass(slots=True)
class ParentChunk:
    chunk_id: str
    document_id: str
    text: str

    chunk_index: int = 0
    level: ChunkLevel = ChunkLevel.PARENT

    title: Optional[str] = None
    section_number: Optional[str] = None
    section_path: list[str] = field(default_factory=list)

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    child_chunk_ids: list[str] = field(default_factory=list)

    previous_parent_id: Optional[str] = None
    next_parent_id: Optional[str] = None

    character_start: Optional[int] = None
    character_end: Optional[int] = None

    token_count: int = 0
    word_count: int = 0
    character_count: int = 0

    source_filename: Optional[str] = None
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    document_version: Optional[str] = None

    content_hash: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        self.character_count = len(self.text)
        self.word_count = len(self.text.split())

        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "level": self.level.value,
            "title": self.title,
            "section_number": self.section_number,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "child_chunk_ids": self.child_chunk_ids,
            "previous_parent_id": self.previous_parent_id,
            "next_parent_id": self.next_parent_id,
            "character_start": self.character_start,
            "character_end": self.character_end,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "source_filename": self.source_filename,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "document_version": self.document_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ChildChunk:
    chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str

    chunk_index: int = 0
    child_index: int = 0

    level: ChunkLevel = ChunkLevel.CHILD

    title: Optional[str] = None
    section_number: Optional[str] = None
    section_path: list[str] = field(default_factory=list)

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    character_start: Optional[int] = None
    character_end: Optional[int] = None

    token_count: int = 0
    word_count: int = 0
    character_count: int = 0

    source_filename: Optional[str] = None
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    document_version: Optional[str] = None

    content_hash: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        self.character_count = len(self.text)
        self.word_count = len(self.text.split())

        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "child_index": self.child_index,
            "level": self.level.value,
            "title": self.title,
            "section_number": self.section_number,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "character_start": self.character_start,
            "character_end": self.character_end,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "source_filename": self.source_filename,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "document_version": self.document_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ParentChildChunkingResult:
    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)

    document_id: Optional[str] = None

    parent_count: int = 0
    child_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parent_count = len(self.parents)
        self.child_count = len(self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "parents": [
                parent.to_dict()
                for parent in self.parents
            ],
            "children": [
                child.to_dict()
                for child in self.children
            ],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ParentChildChunkerConfig:
    """
    Parent chunks contain broader legal context.

    Child chunks are smaller units embedded/indexed for retrieval.

    Typical flow:

        query
          ↓
        retrieve child
          ↓
        rerank child
          ↓
        resolve parent_chunk_id
          ↓
        send broader parent context to LLM
    """

    parent_target_size: int = 4000
    parent_max_size: int = 6000

    child_target_size: int = 800
    child_max_size: int = 1200

    child_overlap: int = 100

    min_child_size: int = 80

    preserve_pages: bool = True
    preserve_sections: bool = True

    include_parent_heading_in_children: bool = True

    deterministic_ids: bool = True

    link_neighbors: bool = True

    max_children_per_parent: int = 50


@dataclass(slots=True)
class _SourceBlock:
    text: str

    page_number: Optional[int] = None

    title: Optional[str] = None
    section_number: Optional[str] = None

    section_path: list[str] = field(default_factory=list)

    start: Optional[int] = None
    end: Optional[int] = None

    metadata: dict[str, Any] = field(default_factory=dict)


class ParentChildChunkingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "PARENT_CHILD_CHUNKING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class ParentChildChunker:
    """
    Parent-child chunker designed for enterprise/legal RAG.

    Parent chunks:
        Larger context units stored in PostgreSQL/document store.

    Child chunks:
        Smaller retrieval units stored in Qdrant/BM25.

    Example:

        Contract
          │
          ├── Parent: Section 8 - Liability
          │      │
          │      ├── Child 8.1
          │      ├── Child 8.2
          │      └── Child 8.3
          │
          └── Parent: Section 9 - Indemnification
                 │
                 ├── Child 9.1
                 └── Child 9.2

    This allows high-precision retrieval without losing the surrounding
    legal context required for grounded answers.
    """

    PAGE_MARKER = re.compile(
        r"(?m)^\[PAGE\s+(\d+)\]\s*$",
        re.IGNORECASE,
    )

    SECTION_PREFIX = re.compile(
        r"""
        ^\s*
        (?:
            section|
            article|
            clause
        )
        \s+
        (?P<number>
            [A-Za-z0-9]+
            (?:\.[A-Za-z0-9]+)*
        )
        [\.\:\-\s]+
        (?P<title>.+?)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    NUMBERED_HEADING = re.compile(
        r"""
        ^\s*
        (?P<number>
            \d+(?:\.\d+){0,6}
        )
        [\.\)]?
        \s+
        (?P<title>
            [A-Z][^\n]{1,180}
        )
        \s*$
        """,
        re.VERBOSE,
    )

    SPECIAL_HEADING = re.compile(
        r"""
        ^\s*
        (?P<type>
            schedule|
            exhibit|
            annex|
            appendix
        )
        \s*
        (?P<number>[A-Za-z0-9\-]*)?
        [\.\:\-\s]*
        (?P<title>.*?)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        config: Optional[
            ParentChildChunkerConfig
        ] = None,
        *,
        tokenizer: Any = None,
    ) -> None:
        self.config = (
            config
            or ParentChildChunkerConfig()
        )

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
    ) -> ParentChildChunkingResult:

        if not document_id:
            raise ParentChildChunkingError(
                "document_id is required.",
                code="MISSING_DOCUMENT_ID",
            )

        if not text or not text.strip():
            return ParentChildChunkingResult(
                document_id=document_id
            )

        blocks = self._parse_blocks(
            text
        )

        parent_sources = (
            self._build_parent_sources(
                blocks
            )
        )

        parents: list[
            ParentChunk
        ] = []

        children: list[
            ChildChunk
        ] = []

        document_metadata = (
            metadata or {}
        )

        global_child_index = 0

        for parent_index, source in enumerate(
            parent_sources
        ):
            parent_id = self._make_id(
                prefix="parent",
                document_id=document_id,
                index=parent_index,
                text=source.text,
            )

            parent = ParentChunk(
                chunk_id=parent_id,
                document_id=document_id,
                text=source.text,
                chunk_index=parent_index,
                title=source.title,
                section_number=(
                    source.section_number
                ),
                section_path=list(
                    source.section_path
                ),
                page_start=(
                    source.metadata.get(
                        "page_start",
                        source.page_number,
                    )
                ),
                page_end=(
                    source.metadata.get(
                        "page_end",
                        source.page_number,
                    )
                ),
                character_start=source.start,
                character_end=source.end,
                token_count=self._token_count(
                    source.text
                ),
                source_filename=source_filename,
                tenant_id=tenant_id,
                organization_id=organization_id,
                document_version=(
                    document_version
                ),
                metadata={
                    **document_metadata,
                    **source.metadata,
                },
            )

            child_texts = (
                self._split_parent(
                    source.text,
                    heading=source.title,
                )
            )

            if (
                len(child_texts)
                > self.config
                .max_children_per_parent
            ):
                child_texts = child_texts[
                    : self.config
                    .max_children_per_parent
                ]

            parent_children: list[
                ChildChunk
            ] = []

            search_cursor = 0

            for local_index, child_text in enumerate(
                child_texts
            ):
                child_id = self._make_id(
                    prefix="child",
                    document_id=document_id,
                    index=global_child_index,
                    text=child_text,
                    parent_id=parent_id,
                )

                relative_start = (
                    source.text.find(
                        child_text,
                        search_cursor,
                    )
                )

                if relative_start < 0:
                    relative_start = None
                    relative_end = None
                else:
                    relative_end = (
                        relative_start
                        + len(child_text)
                    )

                    search_cursor = max(
                        relative_start + 1,
                        0,
                    )

                character_start = None
                character_end = None

                if (
                    source.start is not None
                    and relative_start is not None
                ):
                    character_start = (
                        source.start
                        + relative_start
                    )

                    character_end = (
                        source.start
                        + relative_end
                    )

                child = ChildChunk(
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    document_id=document_id,
                    text=child_text,
                    chunk_index=(
                        global_child_index
                    ),
                    child_index=local_index,
                    title=source.title,
                    section_number=(
                        source.section_number
                    ),
                    section_path=list(
                        source.section_path
                    ),
                    page_start=(
                        parent.page_start
                    ),
                    page_end=parent.page_end,
                    character_start=(
                        character_start
                    ),
                    character_end=(
                        character_end
                    ),
                    token_count=(
                        self._token_count(
                            child_text
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
                        "parent_index":
                            parent_index,
                        "child_index":
                            local_index,
                    },
                )

                parent.child_chunk_ids.append(
                    child_id
                )

                parent_children.append(
                    child
                )

                children.append(child)

                global_child_index += 1

            if self.config.link_neighbors:
                self._link_child_neighbors(
                    parent_children
                )

            parents.append(parent)

        if self.config.link_neighbors:
            self._link_parent_neighbors(
                parents
            )

            self._link_global_children(
                children
            )

        return ParentChildChunkingResult(
            parents=parents,
            children=children,
            document_id=document_id,
            metadata={
                "parent_target_size":
                    self.config
                    .parent_target_size,
                "parent_max_size":
                    self.config
                    .parent_max_size,
                "child_target_size":
                    self.config
                    .child_target_size,
                "child_max_size":
                    self.config
                    .child_max_size,
                "child_overlap":
                    self.config
                    .child_overlap,
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
    ) -> ParentChildChunkingResult:

        output: list[str] = []

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

            output.append(
                f"[PAGE {page_number}]\n"
                f"{str(page_text).strip()}"
            )

        return self.chunk(
            "\n\n".join(output),
            document_id=document_id,
            source_filename=source_filename,
            tenant_id=tenant_id,
            organization_id=organization_id,
            document_version=document_version,
            metadata=metadata,
        )

    # ==============================================================
    # Parsing
    # ==============================================================

    def _parse_blocks(
        self,
        text: str,
    ) -> list[_SourceBlock]:

        pages = self._split_pages(
            text
        )

        blocks: list[
            _SourceBlock
        ] = []

        offset = 0

        current_title = None
        current_number = None
        current_path: list[str] = []

        for page_number, page_text in pages:
            paragraphs = (
                self._paragraphs(
                    page_text
                )
            )

            for paragraph, start, end in paragraphs:
                heading = (
                    self._parse_heading(
                        paragraph
                    )
                )

                if heading:
                    (
                        current_number,
                        current_title,
                    ) = heading

                    current_path = (
                        self._section_path(
                            current_path,
                            current_number,
                            current_title,
                        )
                    )

                blocks.append(
                    _SourceBlock(
                        text=paragraph,
                        page_number=(
                            page_number
                        ),
                        title=current_title,
                        section_number=(
                            current_number
                        ),
                        section_path=list(
                            current_path
                        ),
                        start=offset + start,
                        end=offset + end,
                        metadata={
                            "heading":
                                bool(heading),
                        },
                    )
                )

            offset += (
                len(page_text) + 1
            )

        return blocks

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
            self.PAGE_MARKER.finditer(
                text
            )
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
                matches[index + 1].start()
                if index + 1
                < len(matches)
                else len(text)
            )

            page_text = text[
                start:end
            ].strip()

            output.append(
                (
                    page_number,
                    page_text,
                )
            )

        return output

    # ==============================================================
    # Parent construction
    # ==============================================================

    def _build_parent_sources(
        self,
        blocks: list[_SourceBlock],
    ) -> list[_SourceBlock]:

        if not blocks:
            return []

        parents: list[
            _SourceBlock
        ] = []

        current: list[
            _SourceBlock
        ] = []

        current_size = 0

        current_section: Optional[
            tuple[
                Optional[str],
                tuple[str, ...],
            ]
        ] = None

        def flush() -> None:
            nonlocal current
            nonlocal current_size
            nonlocal current_section

            if not current:
                return

            text = "\n\n".join(
                block.text
                for block in current
            ).strip()

            if not text:
                current = []
                current_size = 0
                return

            page_numbers = [
                block.page_number
                for block in current
                if block.page_number
                is not None
            ]

            first = current[0]

            parents.append(
                _SourceBlock(
                    text=text,
                    page_number=(
                        min(page_numbers)
                        if page_numbers
                        else None
                    ),
                    title=first.title,
                    section_number=(
                        first.section_number
                    ),
                    section_path=list(
                        first.section_path
                    ),
                    start=first.start,
                    end=current[-1].end,
                    metadata={
                        "page_start": (
                            min(page_numbers)
                            if page_numbers
                            else None
                        ),
                        "page_end": (
                            max(page_numbers)
                            if page_numbers
                            else None
                        ),
                        "source_block_count":
                            len(current),
                    },
                )
            )

            current = []
            current_size = 0
            current_section = None

        for block in blocks:
            section_key = (
                block.section_number,
                tuple(
                    block.section_path
                ),
            )

            block_size = len(
                block.text
            )

            # Start a new parent when a new legal
            # section begins and there is already
            # meaningful content in the parent.
            section_changed = (
                self.config
                .preserve_sections
                and current
                and current_section
                is not None
                and section_key
                != current_section
            )

            would_exceed = (
                current
                and (
                    current_size
                    + block_size
                    + 2
                    > self.config
                    .parent_max_size
                )
            )

            target_reached = (
                current
                and current_size
                >= self.config
                .parent_target_size
                and block.metadata.get(
                    "heading"
                )
            )

            if (
                section_changed
                or would_exceed
                or target_reached
            ):
                flush()

            if not current:
                current_section = (
                    section_key
                )

            # A single paragraph can itself exceed
            # parent_max_size.
            if (
                block_size
                > self.config
                .parent_max_size
            ):
                flush()

                pieces = self._hard_split(
                    block.text,
                    max_size=(
                        self.config
                        .parent_max_size
                    ),
                    overlap=0,
                )

                for piece in pieces:
                    parents.append(
                        _SourceBlock(
                            text=piece,
                            page_number=(
                                block.page_number
                            ),
                            title=block.title,
                            section_number=(
                                block
                                .section_number
                            ),
                            section_path=list(
                                block
                                .section_path
                            ),
                            start=block.start,
                            end=block.end,
                            metadata={
                                "page_start":
                                    block
                                    .page_number,
                                "page_end":
                                    block
                                    .page_number,
                                "split_large_block":
                                    True,
                            },
                        )
                    )

                continue

            current.append(block)

            current_size += (
                block_size
                + (
                    2
                    if len(current) > 1
                    else 0
                )
            )

        flush()

        return parents

    # ==============================================================
    # Child splitting
    # ==============================================================

    def _split_parent(
        self,
        text: str,
        *,
        heading: Optional[str],
    ) -> list[str]:

        if (
            len(text)
            <= self.config.child_max_size
        ):
            return [text.strip()]

        sentences = (
            self._split_sentences(
                text
            )
        )

        if not sentences:
            return self._hard_split(
                text,
                max_size=(
                    self.config
                    .child_max_size
                ),
                overlap=(
                    self.config
                    .child_overlap
                ),
            )

        output: list[str] = []

        current: list[str] = []
        size = 0

        for sentence in sentences:
            if (
                len(sentence)
                > self.config
                .child_max_size
            ):
                if current:
                    output.append(
                        " ".join(
                            current
                        ).strip()
                    )

                    current = []
                    size = 0

                output.extend(
                    self._hard_split(
                        sentence,
                        max_size=(
                            self.config
                            .child_max_size
                        ),
                        overlap=(
                            self.config
                            .child_overlap
                        ),
                    )
                )

                continue

            candidate = (
                size
                + len(sentence)
                + (
                    1
                    if current
                    else 0
                )
            )

            if (
                current
                and candidate
                > self.config
                .child_target_size
            ):
                completed = (
                    " ".join(
                        current
                    ).strip()
                )

                output.append(
                    completed
                )

                overlap = (
                    self._overlap(
                        completed
                    )
                )

                current = (
                    [overlap]
                    if overlap
                    else []
                )

                size = len(
                    overlap
                )

            current.append(
                sentence
            )

            size += (
                len(sentence)
                + (
                    1
                    if size
                    else 0
                )
            )

        if current:
            output.append(
                " ".join(
                    current
                ).strip()
            )

        output = (
            self._deduplicate(
                output
            )
        )

        if (
            self.config
            .include_parent_heading_in_children
            and heading
        ):
            output = [
                self._add_heading(
                    chunk,
                    heading,
                )
                for chunk in output
            ]

        return output

    @staticmethod
    def _add_heading(
        text: str,
        heading: str,
    ) -> str:

        normalized_text = (
            text.lower()
        )

        normalized_heading = (
            heading.lower()
        )

        if (
            normalized_heading
            in normalized_text[:200]
        ):
            return text

        return (
            f"{heading}\n\n{text}"
        )

    # ==============================================================
    # Heading parsing
    # ==============================================================

    def _parse_heading(
        self,
        text: str,
    ) -> Optional[
        tuple[
            Optional[str],
            str,
        ]
    ]:

        stripped = text.strip()

        special = (
            self.SPECIAL_HEADING
            .match(stripped)
        )

        if special:
            kind = (
                special.group("type")
                .title()
            )

            number = (
                special.group(
                    "number"
                )
                or None
            )

            title = (
                special.group(
                    "title"
                )
                or ""
            ).strip()

            full_title = (
                f"{kind} {number or ''} "
                f"{title}"
            ).strip()

            return (
                number,
                full_title,
            )

        prefixed = (
            self.SECTION_PREFIX
            .match(stripped)
        )

        if prefixed:
            return (
                prefixed.group(
                    "number"
                ),
                prefixed.group(
                    "title"
                ).strip(),
            )

        numbered = (
            self.NUMBERED_HEADING
            .match(stripped)
        )

        if numbered:
            title = numbered.group(
                "title"
            ).strip()

            if (
                len(title) <= 180
                and not title.endswith(
                    (".", ";")
                )
            ):
                return (
                    numbered.group(
                        "number"
                    ),
                    title,
                )

        if self._all_caps_heading(
            stripped
        ):
            return (
                None,
                stripped,
            )

        return None

    @staticmethod
    def _all_caps_heading(
        text: str,
    ) -> bool:

        if (
            not text
            or len(text) > 120
        ):
            return False

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
    # Section hierarchy
    # ==============================================================

    @staticmethod
    def _section_path(
        current: list[str],
        number: Optional[str],
        title: str,
    ) -> list[str]:

        label = (
            f"{number} {title}"
            if number
            else title
        ).strip()

        if (
            number
            and re.fullmatch(
                r"\d+(?:\.\d+)*",
                number,
            )
        ):
            depth = (
                number.count(".")
                + 1
            )

            return [
                *current[
                    : max(
                        0,
                        depth - 1,
                    )
                ],
                label,
            ]

        return [label]

    # ==============================================================
    # Paragraph parsing
    # ==============================================================

    @staticmethod
    def _paragraphs(
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

    # ==============================================================
    # Sentence splitting
    # ==============================================================

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        parts = re.split(
            r"(?<=[.!?;])\s+"
            r"(?=[A-Z(\"“])",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ==============================================================
    # Hard splitting
    # ==============================================================

    @staticmethod
    def _hard_split(
        text: str,
        *,
        max_size: int,
        overlap: int,
    ) -> list[str]:

        if not text:
            return []

        output: list[str] = []

        start = 0

        overlap = min(
            overlap,
            max_size // 3,
        )

        while start < len(text):
            end = min(
                start + max_size,
                len(text),
            )

            if end < len(text):
                boundary = text.rfind(
                    " ",
                    start,
                    end,
                )

                if boundary > start:
                    end = boundary

            piece = text[
                start:end
            ].strip()

            if piece:
                output.append(piece)

            if end >= len(text):
                break

            start = max(
                end - overlap,
                start + 1,
            )

        return output

    def _overlap(
        self,
        text: str,
    ) -> str:

        size = (
            self.config
            .child_overlap
        )

        if size <= 0:
            return ""

        if len(text) <= size:
            return text

        value = text[
            -size:
        ]

        boundary = value.find(
            " "
        )

        if boundary >= 0:
            value = value[
                boundary + 1:
            ]

        return value.strip()

    # ==============================================================
    # Neighbor linking
    # ==============================================================

    @staticmethod
    def _link_parent_neighbors(
        parents: list[
            ParentChunk
        ],
    ) -> None:

        for index, parent in enumerate(
            parents
        ):
            if index > 0:
                parent.previous_parent_id = (
                    parents[
                        index - 1
                    ].chunk_id
                )

            if index + 1 < len(
                parents
            ):
                parent.next_parent_id = (
                    parents[
                        index + 1
                    ].chunk_id
                )

    @staticmethod
    def _link_child_neighbors(
        children: list[
            ChildChunk
        ],
    ) -> None:

        for index, child in enumerate(
            children
        ):
            if index > 0:
                child.previous_chunk_id = (
                    children[
                        index - 1
                    ].chunk_id
                )

            if index + 1 < len(
                children
            ):
                child.next_chunk_id = (
                    children[
                        index + 1
                    ].chunk_id
                )

    @staticmethod
    def _link_global_children(
        children: list[
            ChildChunk
        ],
    ) -> None:
        """
        Preserve document-level neighboring relationships while
        respecting existing parent-local relationships.
        """

        for index, child in enumerate(
            children
        ):
            if (
                index > 0
                and child.previous_chunk_id
                is None
            ):
                child.previous_chunk_id = (
                    children[
                        index - 1
                    ].chunk_id
                )

            if (
                index + 1
                < len(children)
                and child.next_chunk_id
                is None
            ):
                child.next_chunk_id = (
                    children[
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
                return len(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
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
                        return len(ids)

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
        prefix: str,
        document_id: str,
        index: int,
        text: str,
        parent_id: Optional[
            str
        ] = None,
    ) -> str:

        if not self.config.deterministic_ids:
            return (
                f"{prefix}_"
                f"{uuid.uuid4()}"
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
            f"{parent_id or ''}|"
            f"{index}|"
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
            f"{prefix}_"
            f"{digest[:24]}"
        )

    # ==============================================================
    # Deduplication
    # ==============================================================

    @staticmethod
    def _deduplicate(
        chunks: Iterable[str],
    ) -> list[str]:

        output: list[str] = []

        previous = None

        for chunk in chunks:
            chunk = chunk.strip()

            if not chunk:
                continue

            if chunk == previous:
                continue

            output.append(chunk)

            previous = chunk

        return output

    # ==============================================================
    # Validation
    # ==============================================================

    def _validate_config(
        self,
    ) -> None:

        config = self.config

        if (
            config.parent_target_size
            <= 0
        ):
            raise ValueError(
                "parent_target_size must "
                "be greater than zero."
            )

        if (
            config.parent_max_size
            < config.parent_target_size
        ):
            raise ValueError(
                "parent_max_size must be >= "
                "parent_target_size."
            )

        if (
            config.child_target_size
            <= 0
        ):
            raise ValueError(
                "child_target_size must be "
                "greater than zero."
            )

        if (
            config.child_max_size
            < config.child_target_size
        ):
            raise ValueError(
                "child_max_size must be >= "
                "child_target_size."
            )

        if (
            config.child_overlap < 0
            or config.child_overlap
            >= config.child_max_size
        ):
            raise ValueError(
                "child_overlap must be >= 0 "
                "and < child_max_size."
            )


def parent_child_chunk(
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
) -> ParentChildChunkingResult:

    chunker = ParentChildChunker()

    return chunker.chunk(
        text,
        document_id=document_id,
        source_filename=source_filename,
        tenant_id=tenant_id,
        organization_id=organization_id,
        document_version=document_version,
        metadata=metadata,
    )