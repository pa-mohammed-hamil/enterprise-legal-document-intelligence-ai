# backend/rag/chunking/legal_chunker.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence
import hashlib
import re
import uuid


class LegalChunkType(str, Enum):
    TITLE = "title"
    PREAMBLE = "preamble"
    RECITAL = "recital"
    DEFINITION = "definition"
    SECTION = "section"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    SIGNATURE = "signature"
    SCHEDULE = "schedule"
    EXHIBIT = "exhibit"
    ANNEX = "annex"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


class LegalSectionType(str, Enum):
    DEFINITIONS = "definitions"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    PRICING = "pricing"
    TERM = "term"
    TERMINATION = "termination"
    RENEWAL = "renewal"
    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    WARRANTY = "warranty"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    AUDIT = "audit"
    INSURANCE = "insurance"
    SERVICE_LEVEL = "service_level"
    FORCE_MAJEURE = "force_majeure"
    ASSIGNMENT = "assignment"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    NOTICE = "notice"
    OBLIGATION = "obligation"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class LegalChunk:
    chunk_id: str
    document_id: str
    text: str

    chunk_index: int = 0

    chunk_type: LegalChunkType = LegalChunkType.UNKNOWN
    section_type: LegalSectionType = LegalSectionType.UNKNOWN

    title: Optional[str] = None
    section_number: Optional[str] = None
    section_path: list[str] = field(default_factory=list)

    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    parent_chunk_id: Optional[str] = None

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
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "section_type": self.section_type.value,
            "title": self.title,
            "section_number": self.section_number,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "parent_chunk_id": self.parent_chunk_id,
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
class LegalChunkerConfig:
    """
    Legal-aware chunking configuration.

    The goal is semantic/legal boundary preservation rather than
    blindly splitting every N characters.
    """

    target_chunk_size: int = 1200
    max_chunk_size: int = 1800
    min_chunk_size: int = 150

    overlap_characters: int = 150

    preserve_sections: bool = True
    preserve_clauses: bool = True
    preserve_page_numbers: bool = True
    preserve_tables: bool = True

    include_heading_in_chunk: bool = True

    create_parent_chunks: bool = True

    max_parent_size: int = 6000

    split_large_clauses: bool = True

    generate_neighbors: bool = True

    deterministic_ids: bool = True


@dataclass(slots=True)
class _LegalBlock:
    text: str

    chunk_type: LegalChunkType

    title: Optional[str] = None
    section_number: Optional[str] = None

    page_number: Optional[int] = None

    section_path: list[str] = field(default_factory=list)

    start: Optional[int] = None
    end: Optional[int] = None

    metadata: dict[str, Any] = field(default_factory=dict)


class LegalChunkingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "LEGAL_CHUNKING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class LegalChunker:
    """
    Legal-aware document chunker.

    Key properties
    --------------
    1. Preserves numbered legal sections where possible.
    2. Preserves physical page provenance.
    3. Understands schedules/exhibits/annexes.
    4. Detects common legal section categories.
    5. Supports parent-child retrieval.
    6. Produces deterministic metadata suitable for vector stores.
    7. Adds neighboring chunk references.
    8. Avoids changing legal wording.

    This module performs structural chunking only.

    It does NOT decide:
        - whether a clause is valid
        - whether a clause is risky
        - whether a contract is compliant
        - whether a legal obligation exists

    Those belong to downstream classifiers/agents.
    """

    PAGE_MARKER = re.compile(
        r"(?m)^\[PAGE\s+(\d+)\]\s*$",
        re.IGNORECASE,
    )

    NUMBERED_SECTION = re.compile(
        r"""
        ^\s*
        (?P<number>
            (?:\d+(?:\.\d+){0,6})
            |
            (?:[IVXLCDM]+)
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

    SECTION_PREFIX = re.compile(
        r"""
        ^\s*
        (?P<prefix>
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

    SUBCLAUSE = re.compile(
        r"""
        ^\s*
        (?:
            \([a-zA-Z0-9ivxIVX]+\)
            |
            [a-zA-Z]\)
            |
            \d+\.\d+(?:\.\d+)*
        )
        \s+
        """,
        re.VERBOSE,
    )

    LIST_ITEM = re.compile(
        r"""
        ^\s*
        (?:
            [-•*]
            |
            \([a-zA-Z0-9]+\)
            |
            [a-zA-Z]\.
            |
            \d+\)
        )
        \s+
        """,
        re.VERBOSE,
    )

    TABLE_MARKER = re.compile(
        r"(?m)^\[TABLE\s+\d+\]",
        re.IGNORECASE,
    )

    SIGNATURE_MARKERS = (
        "IN WITNESS WHEREOF",
        "SIGNATURES",
        "SIGNED BY",
        "EXECUTED BY",
        "AUTHORIZED SIGNATORY",
    )

    RECITAL_MARKERS = (
        "WHEREAS",
        "RECITALS",
        "BACKGROUND",
    )

    SECTION_KEYWORDS: dict[
        LegalSectionType,
        tuple[str, ...],
    ] = {
        LegalSectionType.DEFINITIONS: (
            "definition",
            "definitions",
            "defined terms",
            "interpretation",
        ),
        LegalSectionType.CONFIDENTIALITY: (
            "confidentiality",
            "confidential information",
            "non-disclosure",
            "nondisclosure",
        ),
        LegalSectionType.PAYMENT: (
            "payment",
            "payment terms",
            "invoicing",
            "invoice",
        ),
        LegalSectionType.PRICING: (
            "pricing",
            "fees",
            "charges",
            "price",
        ),
        LegalSectionType.TERM: (
            "term",
            "duration",
            "effective date",
        ),
        LegalSectionType.TERMINATION: (
            "termination",
            "terminate",
            "termination rights",
        ),
        LegalSectionType.RENEWAL: (
            "renewal",
            "automatic renewal",
            "auto renewal",
            "extension",
        ),
        LegalSectionType.LIABILITY: (
            "limitation of liability",
            "liability",
            "liability cap",
        ),
        LegalSectionType.INDEMNIFICATION: (
            "indemnification",
            "indemnity",
            "indemnification obligations",
        ),
        LegalSectionType.WARRANTY: (
            "warranty",
            "warranties",
            "representations and warranties",
        ),
        LegalSectionType.INTELLECTUAL_PROPERTY: (
            "intellectual property",
            "ownership",
            "ip rights",
            "proprietary rights",
        ),
        LegalSectionType.DATA_PROTECTION: (
            "data protection",
            "personal data",
            "data processing",
            "dpa",
        ),
        LegalSectionType.PRIVACY: (
            "privacy",
            "privacy obligations",
        ),
        LegalSectionType.SECURITY: (
            "information security",
            "security",
            "cybersecurity",
            "security requirements",
        ),
        LegalSectionType.COMPLIANCE: (
            "compliance",
            "regulatory compliance",
            "laws and regulations",
        ),
        LegalSectionType.AUDIT: (
            "audit",
            "audit rights",
            "inspection rights",
        ),
        LegalSectionType.INSURANCE: (
            "insurance",
            "insurance requirements",
        ),
        LegalSectionType.SERVICE_LEVEL: (
            "service level",
            "service levels",
            "sla",
            "service level agreement",
        ),
        LegalSectionType.FORCE_MAJEURE: (
            "force majeure",
        ),
        LegalSectionType.ASSIGNMENT: (
            "assignment",
            "assignability",
        ),
        LegalSectionType.GOVERNING_LAW: (
            "governing law",
            "applicable law",
        ),
        LegalSectionType.DISPUTE_RESOLUTION: (
            "dispute resolution",
            "arbitration",
            "jurisdiction",
            "venue",
        ),
        LegalSectionType.NOTICE: (
            "notice",
            "notices",
        ),
        LegalSectionType.OBLIGATION: (
            "obligations",
            "responsibilities",
            "duties",
        ),
    }

    def __init__(
        self,
        config: Optional[LegalChunkerConfig] = None,
        *,
        tokenizer: Any = None,
    ) -> None:
        self.config = config or LegalChunkerConfig()
        self.tokenizer = tokenizer

        self._validate_config()

    # ================================================================
    # Public API
    # ================================================================

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[LegalChunk]:

        if not document_id:
            raise LegalChunkingError(
                "document_id is required.",
                code="MISSING_DOCUMENT_ID",
            )

        if not text or not text.strip():
            return []

        document_metadata = metadata or {}

        blocks = self._parse_blocks(text)

        child_chunks: list[LegalChunk] = []

        for block in blocks:
            pieces = self._split_block(block)

            for piece_index, piece in enumerate(pieces):
                chunk = self._create_chunk(
                    document_id=document_id,
                    block=block,
                    text=piece,
                    chunk_index=len(child_chunks),
                    piece_index=piece_index,
                    source_filename=source_filename,
                    tenant_id=tenant_id,
                    organization_id=organization_id,
                    document_version=document_version,
                    document_metadata=document_metadata,
                )

                child_chunks.append(chunk)

        if self.config.generate_neighbors:
            self._link_neighbors(child_chunks)

        if self.config.create_parent_chunks:
            self._assign_parent_ids(
                child_chunks,
                document_id=document_id,
            )

        return child_chunks

    def chunk_pages(
        self,
        pages: Sequence[Any],
        *,
        document_id: str,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[LegalChunk]:
        """
        Chunk page objects produced by PageExtractor/PDFProcessor.

        Objects can expose:
            page.page_number
            page.text

        or be dictionaries with those keys.
        """

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
                f"{page_text.strip()}"
            )

        return self.chunk(
            "\n\n".join(parts),
            document_id=document_id,
            source_filename=source_filename,
            tenant_id=tenant_id,
            organization_id=organization_id,
            document_version=document_version,
            metadata=metadata,
        )

    # ================================================================
    # Structural parsing
    # ================================================================

    def _parse_blocks(
        self,
        text: str,
    ) -> list[_LegalBlock]:

        pages = self._split_pages(text)

        blocks: list[_LegalBlock] = []

        current_path: list[str] = []

        global_offset = 0

        for page_number, page_text in pages:
            page_blocks = self._parse_page(
                page_text,
                page_number=page_number,
                base_offset=global_offset,
                current_path=current_path,
            )

            blocks.extend(page_blocks)

            if page_blocks:
                last = page_blocks[-1]

                if last.section_path:
                    current_path = list(
                        last.section_path
                    )

            global_offset += len(page_text) + 1

        return blocks

    def _split_pages(
        self,
        text: str,
    ) -> list[
        tuple[Optional[int], str]
    ]:

        matches = list(
            self.PAGE_MARKER.finditer(text)
        )

        if not matches:
            return [(None, text)]

        output: list[
            tuple[Optional[int], str]
        ] = []

        # Preserve content before first page marker.
        if matches[0].start() > 0:
            prefix = text[
                : matches[0].start()
            ].strip()

            if prefix:
                output.append(
                    (None, prefix)
                )

        for index, match in enumerate(matches):
            page_number = int(
                match.group(1)
            )

            start = match.end()

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
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

    def _parse_page(
        self,
        text: str,
        *,
        page_number: Optional[int],
        base_offset: int,
        current_path: list[str],
    ) -> list[_LegalBlock]:

        paragraphs = self._paragraphs_with_offsets(
            text
        )

        blocks: list[_LegalBlock] = []

        active_title: Optional[str] = None
        active_number: Optional[str] = None

        section_path = list(
            current_path
        )

        for paragraph, start, end in paragraphs:
            stripped = paragraph.strip()

            if not stripped:
                continue

            heading = self._parse_heading(
                stripped
            )

            if heading is not None:
                (
                    chunk_type,
                    number,
                    title,
                ) = heading

                active_title = title
                active_number = number

                path_label = (
                    f"{number} {title}".strip()
                    if number
                    else title
                )

                section_path = self._update_section_path(
                    section_path,
                    number,
                    path_label,
                )

                blocks.append(
                    _LegalBlock(
                        text=stripped,
                        chunk_type=chunk_type,
                        title=title,
                        section_number=number,
                        page_number=page_number,
                        section_path=list(
                            section_path
                        ),
                        start=base_offset + start,
                        end=base_offset + end,
                        metadata={
                            "heading": True,
                        },
                    )
                )

                continue

            chunk_type = self._detect_block_type(
                stripped
            )

            blocks.append(
                _LegalBlock(
                    text=stripped,
                    chunk_type=chunk_type,
                    title=active_title,
                    section_number=active_number,
                    page_number=page_number,
                    section_path=list(
                        section_path
                    ),
                    start=base_offset + start,
                    end=base_offset + end,
                )
            )

        return self._merge_related_blocks(
            blocks
        )

    # ================================================================
    # Heading detection
    # ================================================================

    def _parse_heading(
        self,
        text: str,
    ) -> Optional[
        tuple[
            LegalChunkType,
            Optional[str],
            str,
        ]
    ]:

        special = self.SPECIAL_HEADING.match(
            text
        )

        if special:
            type_name = (
                special.group("type")
                .lower()
            )

            mapping = {
                "schedule":
                    LegalChunkType.SCHEDULE,
                "exhibit":
                    LegalChunkType.EXHIBIT,
                "annex":
                    LegalChunkType.ANNEX,
                "appendix":
                    LegalChunkType.APPENDIX,
            }

            number = (
                special.group("number")
                or None
            )

            title = (
                special.group("title")
                or type_name.title()
            ).strip()

            return (
                mapping[type_name],
                number,
                title,
            )

        prefixed = self.SECTION_PREFIX.match(
            text
        )

        if prefixed:
            return (
                LegalChunkType.SECTION,
                prefixed.group("number"),
                prefixed.group("title").strip(),
            )

        numbered = self.NUMBERED_SECTION.match(
            text
        )

        if numbered:
            title = numbered.group(
                "title"
            ).strip()

            # Avoid treating long sentences as headings.
            if (
                len(title) <= 180
                and not title.endswith(
                    (".", ";")
                )
            ):
                return (
                    LegalChunkType.SECTION,
                    numbered.group(
                        "number"
                    ),
                    title,
                )

        # ALL-CAPS headings are common in contracts.
        if self._looks_like_heading(text):
            return (
                LegalChunkType.SECTION,
                None,
                text.strip(),
            )

        return None

    @staticmethod
    def _looks_like_heading(
        text: str,
    ) -> bool:

        cleaned = text.strip()

        if not cleaned:
            return False

        if len(cleaned) > 120:
            return False

        words = cleaned.split()

        if len(words) > 14:
            return False

        letters = [
            character
            for character in cleaned
            if character.isalpha()
        ]

        if not letters:
            return False

        uppercase = sum(
            character.isupper()
            for character in letters
        )

        ratio = uppercase / len(letters)

        return ratio >= 0.85

    # ================================================================
    # Block classification
    # ================================================================

    def _detect_block_type(
        self,
        text: str,
    ) -> LegalChunkType:

        upper = text.upper()

        if self.TABLE_MARKER.search(text):
            return LegalChunkType.TABLE

        if any(
            marker in upper
            for marker
            in self.SIGNATURE_MARKERS
        ):
            return LegalChunkType.SIGNATURE

        if any(
            upper.startswith(marker)
            for marker
            in self.RECITAL_MARKERS
        ):
            return LegalChunkType.RECITAL

        if self.SUBCLAUSE.match(text):
            return LegalChunkType.SUBCLAUSE

        if self.LIST_ITEM.match(text):
            return LegalChunkType.LIST

        if self._looks_like_definition(
            text
        ):
            return LegalChunkType.DEFINITION

        return LegalChunkType.PARAGRAPH

    @staticmethod
    def _looks_like_definition(
        text: str,
    ) -> bool:

        patterns = (
            r'^["“][^"”]{1,80}["”]\s+'
            r'(?:means|shall mean|refers to)\b',

            r"^[A-Z][A-Za-z0-9 _\-]{1,60}"
            r"\s+(?:means|shall mean)\b",
        )

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    # ================================================================
    # Merge blocks
    # ================================================================

    def _merge_related_blocks(
        self,
        blocks: list[_LegalBlock],
    ) -> list[_LegalBlock]:

        if not blocks:
            return []

        output: list[_LegalBlock] = []

        index = 0

        while index < len(blocks):
            block = blocks[index]

            # A heading should normally travel with the text that
            # immediately follows it.
            if (
                block.metadata.get("heading")
                and index + 1 < len(blocks)
            ):
                next_block = blocks[
                    index + 1
                ]

                combined = (
                    f"{block.text}\n\n"
                    f"{next_block.text}"
                )

                if (
                    len(combined)
                    <= self.config.max_chunk_size
                    and not next_block
                    .metadata.get("heading")
                ):
                    output.append(
                        _LegalBlock(
                            text=combined,
                            chunk_type=(
                                next_block.chunk_type
                                if next_block.chunk_type
                                != LegalChunkType
                                .PARAGRAPH
                                else LegalChunkType
                                .CLAUSE
                            ),
                            title=block.title,
                            section_number=(
                                block.section_number
                            ),
                            page_number=(
                                block.page_number
                            ),
                            section_path=list(
                                block.section_path
                            ),
                            start=block.start,
                            end=next_block.end,
                            metadata={
                                "heading_included":
                                    True,
                            },
                        )
                    )

                    index += 2
                    continue

            output.append(block)
            index += 1

        return output

    # ================================================================
    # Splitting
    # ================================================================

    def _split_block(
        self,
        block: _LegalBlock,
    ) -> list[str]:

        text = block.text.strip()

        if not text:
            return []

        if len(text) <= self.config.max_chunk_size:
            return [text]

        # Tables should be preserved as much as possible.
        if (
            block.chunk_type
            == LegalChunkType.TABLE
        ):
            return self._split_table_text(
                text
            )

        return self._semantic_split(
            text
        )

    def _semantic_split(
        self,
        text: str,
    ) -> list[str]:

        sentences = self._split_sentences(
            text
        )

        if not sentences:
            return self._hard_split(text)

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if (
                sentence_size
                > self.config.max_chunk_size
            ):
                if current:
                    chunks.append(
                        " ".join(current).strip()
                    )
                    current = []
                    current_size = 0

                chunks.extend(
                    self._hard_split(
                        sentence
                    )
                )
                continue

            candidate_size = (
                current_size
                + sentence_size
                + (1 if current else 0)
            )

            if (
                current
                and candidate_size
                > self.config.target_chunk_size
            ):
                completed = (
                    " ".join(current).strip()
                )

                chunks.append(completed)

                overlap = self._make_overlap(
                    completed
                )

                current = (
                    [overlap]
                    if overlap
                    else []
                )

                current_size = len(overlap)

            current.append(sentence)

            current_size += (
                sentence_size
                + (1 if current_size else 0)
            )

        if current:
            final = " ".join(
                current
            ).strip()

            if final:
                chunks.append(final)

        return self._remove_duplicate_chunks(
            chunks
        )

    def _split_table_text(
        self,
        text: str,
    ) -> list[str]:

        lines = [
            line
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return []

        chunks: list[str] = []
        current: list[str] = []
        size = 0

        # Keep the table marker/header for each continuation.
        header = lines[0]

        for line in lines:
            line_size = len(line) + 1

            if (
                current
                and size + line_size
                > self.config.max_chunk_size
            ):
                chunks.append(
                    "\n".join(current)
                )

                current = [header]
                size = len(header) + 1

            current.append(line)
            size += line_size

        if current:
            chunks.append(
                "\n".join(current)
            )

        return self._remove_duplicate_chunks(
            chunks
        )

    def _hard_split(
        self,
        text: str,
    ) -> list[str]:

        max_size = self.config.max_chunk_size
        overlap = min(
            self.config.overlap_characters,
            max_size // 3,
        )

        output: list[str] = []

        start = 0

        while start < len(text):
            end = min(
                start + max_size,
                len(text),
            )

            # Prefer a whitespace boundary.
            if end < len(text):
                boundary = text.rfind(
                    " ",
                    start,
                    end,
                )

                if (
                    boundary
                    > start
                    + self.config.min_chunk_size
                ):
                    end = boundary

            piece = text[
                start:end
            ].strip()

            if piece:
                output.append(piece)

            if end >= len(text):
                break

            next_start = max(
                end - overlap,
                start + 1,
            )

            start = next_start

        return output

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Conservative sentence splitter.

        Legal documents contain section references, decimal numbers,
        abbreviations and enumerations, so avoid an overly aggressive
        NLP sentence split here.
        """

        parts = re.split(
            r"(?<=[.!?;])\s+(?=[A-Z(\"“])",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    def _make_overlap(
        self,
        text: str,
    ) -> str:

        size = (
            self.config.overlap_characters
        )

        if size <= 0:
            return ""

        if len(text) <= size:
            return text

        overlap = text[-size:]

        first_space = overlap.find(" ")

        if first_space >= 0:
            overlap = overlap[
                first_space + 1:
            ]

        return overlap.strip()

    # ================================================================
    # Chunk construction
    # ================================================================

    def _create_chunk(
        self,
        *,
        document_id: str,
        block: _LegalBlock,
        text: str,
        chunk_index: int,
        piece_index: int,
        source_filename: Optional[str],
        tenant_id: Optional[str],
        organization_id: Optional[str],
        document_version: Optional[str],
        document_metadata: dict[str, Any],
    ) -> LegalChunk:

        chunk_id = self._chunk_id(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
        )

        section_type = (
            self._detect_section_type(
                block.title,
                text,
            )
        )

        page_number = block.page_number

        metadata = {
            **document_metadata,
            **block.metadata,
            "piece_index": piece_index,
            "legal_section_type":
                section_type.value,
        }

        return LegalChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            chunk_type=block.chunk_type,
            section_type=section_type,
            title=block.title,
            section_number=(
                block.section_number
            ),
            section_path=list(
                block.section_path
            ),
            page_number=page_number,
            page_start=page_number,
            page_end=page_number,
            character_start=block.start,
            character_end=block.end,
            token_count=self._token_count(
                text
            ),
            source_filename=source_filename,
            tenant_id=tenant_id,
            organization_id=organization_id,
            document_version=document_version,
            metadata=metadata,
        )

    # ================================================================
    # Section classification
    # ================================================================

    def _detect_section_type(
        self,
        title: Optional[str],
        text: str,
    ) -> LegalSectionType:

        search_text = (
            f"{title or ''} {text[:500]}"
        ).lower()

        scores: dict[
            LegalSectionType,
            int,
        ] = {}

        for section_type, keywords in (
            self.SECTION_KEYWORDS.items()
        ):
            score = 0

            for keyword in keywords:
                if keyword in search_text:
                    score += (
                        3
                        if title
                        and keyword
                        in title.lower()
                        else 1
                    )

            if score:
                scores[
                    section_type
                ] = score

        if not scores:
            return LegalSectionType.UNKNOWN

        return max(
            scores,
            key=scores.get,
        )

    # ================================================================
    # Parent-child support
    # ================================================================

    def _assign_parent_ids(
        self,
        chunks: list[LegalChunk],
        *,
        document_id: str,
    ) -> None:

        groups: dict[
            tuple[
                Optional[int],
                Optional[str],
                tuple[str, ...],
            ],
            list[LegalChunk],
        ] = {}

        for chunk in chunks:
            key = (
                chunk.page_number,
                chunk.section_number,
                tuple(chunk.section_path),
            )

            groups.setdefault(
                key,
                [],
            ).append(chunk)

        for key, children in groups.items():
            combined = "\n\n".join(
                child.text
                for child in children
            )

            if (
                len(children) <= 1
                and len(combined)
                <= self.config.target_chunk_size
            ):
                continue

            identity = (
                f"{document_id}|"
                f"{key[0]}|"
                f"{key[1]}|"
                f"{'|'.join(key[2])}"
            )

            parent_id = (
                "parent_"
                + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
            )

            for child in children:
                child.parent_chunk_id = (
                    parent_id
                )

                child.metadata[
                    "parent_text_length"
                ] = min(
                    len(combined),
                    self.config.max_parent_size,
                )

    # ================================================================
    # Neighbor relationships
    # ================================================================

    @staticmethod
    def _link_neighbors(
        chunks: list[LegalChunk],
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

            if index + 1 < len(chunks):
                chunk.next_chunk_id = (
                    chunks[
                        index + 1
                    ].chunk_id
                )

    # ================================================================
    # Section hierarchy
    # ================================================================

    def _update_section_path(
        self,
        current: list[str],
        number: Optional[str],
        label: str,
    ) -> list[str]:

        if not number:
            return [label]

        # 4.2.1 => hierarchy depth 3.
        if re.fullmatch(
            r"\d+(?:\.\d+)*",
            number,
        ):
            depth = (
                number.count(".") + 1
            )

            parent = current[
                : max(0, depth - 1)
            ]

            return [
                *parent,
                label,
            ]

        # ARTICLE IV / Schedule A etc.
        return [label]

    # ================================================================
    # Offset parsing
    # ================================================================

    @staticmethod
    def _paragraphs_with_offsets(
        text: str,
    ) -> list[
        tuple[str, int, int]
    ]:

        output: list[
            tuple[str, int, int]
        ] = []

        pattern = re.compile(
            r"\S(?:.*?\S)?"
            r"(?=\n\s*\n|\Z)",
            re.DOTALL,
        )

        for match in pattern.finditer(text):
            value = match.group(0).strip()

            if value:
                output.append(
                    (
                        value,
                        match.start(),
                        match.end(),
                    )
                )

        return output

    # ================================================================
    # Token counting
    # ================================================================

    def _token_count(
        self,
        text: str,
    ) -> int:

        if self.tokenizer is None:
            # Lightweight approximation only.
            return max(
                1,
                int(len(text.split()) * 1.3),
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

                return len(encoded)

            if callable(self.tokenizer):
                result = self.tokenizer(
                    text
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
            int(len(text.split()) * 1.3),
        )

    # ================================================================
    # IDs
    # ================================================================

    def _chunk_id(
        self,
        *,
        document_id: str,
        chunk_index: int,
        text: str,
    ) -> str:

        if not self.config.deterministic_ids:
            return str(
                uuid.uuid4()
            )

        identity = (
            f"{document_id}|"
            f"{chunk_index}|"
            f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return f"chunk_{digest[:24]}"

    # ================================================================
    # Utilities
    # ================================================================

    @staticmethod
    def _remove_duplicate_chunks(
        chunks: Iterable[str],
    ) -> list[str]:

        output: list[str] = []

        previous: Optional[str] = None

        for chunk in chunks:
            chunk = chunk.strip()

            if not chunk:
                continue

            if chunk == previous:
                continue

            output.append(chunk)
            previous = chunk

        return output

    def _validate_config(
        self,
    ) -> None:

        if self.config.target_chunk_size <= 0:
            raise ValueError(
                "target_chunk_size must be > 0"
            )

        if (
            self.config.max_chunk_size
            < self.config.target_chunk_size
        ):
            raise ValueError(
                "max_chunk_size must be >= "
                "target_chunk_size"
            )

        if self.config.min_chunk_size < 0:
            raise ValueError(
                "min_chunk_size cannot be negative"
            )

        if (
            self.config.overlap_characters
            >= self.config.max_chunk_size
        ):
            raise ValueError(
                "overlap_characters must be smaller "
                "than max_chunk_size"
            )


def chunk_legal_document(
    text: str,
    *,
    document_id: str,
    source_filename: Optional[str] = None,
    tenant_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    document_version: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> list[LegalChunk]:
    """
    Convenience entry point for the RAG ingestion pipeline.
    """

    chunker = LegalChunker()

    return chunker.chunk(
        text,
        document_id=document_id,
        source_filename=source_filename,
        tenant_id=tenant_id,
        organization_id=organization_id,
        document_version=document_version,
        metadata=metadata,
    )