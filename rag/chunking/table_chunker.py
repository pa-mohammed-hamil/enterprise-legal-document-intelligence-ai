# backend/rag/chunking/table_chunker.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence
import hashlib
import json
import re
import uuid


class TableFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    TSV = "tsv"
    ROWS = "rows"
    TEXT = "text"
    UNKNOWN = "unknown"


class TableChunkType(str, Enum):
    WHOLE_TABLE = "whole_table"
    ROW_GROUP = "row_group"
    SINGLE_ROW = "single_row"
    HEADER_ONLY = "header_only"
    CONTINUATION = "continuation"


@dataclass(slots=True)
class TableCell:
    value: str

    row_index: int
    column_index: int

    column_name: Optional[str] = None

    is_header: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "column_name": self.column_name,
            "is_header": self.is_header,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TableRow:
    row_index: int

    values: list[str]

    cells: list[TableCell] = field(
        default_factory=list
    )

    page_number: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "values": self.values,
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TableData:
    table_id: str

    headers: list[str]

    rows: list[TableRow]

    title: Optional[str] = None

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    source_format: TableFormat = (
        TableFormat.UNKNOWN
    )

    source_filename: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max(
            [len(self.headers)]
            + [
                len(row.values)
                for row in self.rows
            ],
            default=0,
        )


@dataclass(slots=True)
class TableChunk:
    chunk_id: str
    document_id: str
    table_id: str

    text: str

    chunk_index: int = 0

    chunk_type: TableChunkType = (
        TableChunkType.ROW_GROUP
    )

    title: Optional[str] = None

    headers: list[str] = field(
        default_factory=list
    )

    row_start: Optional[int] = None
    row_end: Optional[int] = None

    row_indices: list[int] = field(
        default_factory=list
    )

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

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
                    self.text.encode("utf-8")
                ).hexdigest()
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "table_id": self.table_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_type":
                self.chunk_type.value,
            "title": self.title,
            "headers": self.headers,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "row_indices": self.row_indices,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "previous_chunk_id":
                self.previous_chunk_id,
            "next_chunk_id":
                self.next_chunk_id,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "character_count":
                self.character_count,
            "source_filename":
                self.source_filename,
            "tenant_id": self.tenant_id,
            "organization_id":
                self.organization_id,
            "document_version":
                self.document_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TableChunkingResult:
    chunks: list[TableChunk] = field(
        default_factory=list
    )

    document_id: Optional[str] = None
    table_id: Optional[str] = None

    row_count: int = 0
    column_count: int = 0
    chunk_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.chunk_count = len(
            self.chunks
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "table_id": self.table_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "chunk_count": self.chunk_count,
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TableChunkerConfig:
    """
    Table-aware chunking configuration.

    Important rule:

        Repeat the table title + headers in every chunk.

    This makes each retrieved table chunk understandable without
    requiring the LLM to see the previous table chunk.
    """

    target_chunk_size: int = 1200
    max_chunk_size: int = 1800

    max_rows_per_chunk: int = 20
    min_rows_per_chunk: int = 1

    row_overlap: int = 1

    repeat_headers: bool = True
    repeat_title: bool = True

    include_row_numbers: bool = True

    preserve_single_row: bool = True

    link_neighbors: bool = True

    deterministic_ids: bool = True

    normalize_cells: bool = True

    max_cell_characters: int = 5000

    max_columns: int = 200

    max_rows: int = 100_000


class TableChunkingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "TABLE_CHUNKING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.cause = cause


class TableChunker:
    """
    Structure-aware table chunker for enterprise/legal RAG.

    Example legal table:

        Schedule A - Pricing

        | Service | Price | Renewal |
        |---------|-------|---------|
        | Basic   | $100  | Annual  |
        | Pro     | $250  | Annual  |

    Instead of blindly splitting characters, every chunk preserves:

        table title
        headers
        row boundaries
        row numbers
        page provenance
        table ID
        document ID

    This is especially important for:

        pricing schedules
        obligation matrices
        SLA tables
        renewal tables
        compliance tables
        liability matrices
        risk tables
        payment schedules
        contract comparison tables
    """

    MARKDOWN_SEPARATOR = re.compile(
        r"^\s*\|?"
        r"\s*:?-{3,}:?"
        r"(?:\s*\|\s*:?-{3,}:?)+"
        r"\s*\|?\s*$"
    )

    TABLE_MARKER = re.compile(
        r"^\[TABLE\s+(?P<number>\d+)\]"
        r"(?:\s*[-:]\s*(?P<title>.*))?$",
        re.IGNORECASE,
    )

    PAGE_MARKER = re.compile(
        r"^\[PAGE\s+(?P<number>\d+)\]$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        config: Optional[
            TableChunkerConfig
        ] = None,
        *,
        tokenizer: Any = None,
    ) -> None:
        self.config = (
            config
            or TableChunkerConfig()
        )

        self.tokenizer = tokenizer

        self._validate_config()

    # ==========================================================
    # Public API
    # ==========================================================

    def chunk(
        self,
        table: (
            TableData
            | Sequence[Sequence[Any]]
            | Sequence[dict[str, Any]]
        ),
        *,
        document_id: str,
        table_id: Optional[str] = None,
        headers: Optional[
            Sequence[str]
        ] = None,
        title: Optional[str] = None,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> TableChunkingResult:

        if not document_id:
            raise TableChunkingError(
                "document_id is required.",
                code="MISSING_DOCUMENT_ID",
            )

        normalized = self._normalize_input(
            table,
            document_id=document_id,
            table_id=table_id,
            headers=headers,
            title=title,
            source_filename=(
                source_filename
            ),
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )

        if not normalized.rows:
            return TableChunkingResult(
                document_id=document_id,
                table_id=normalized.table_id,
                row_count=0,
                column_count=(
                    normalized.column_count
                ),
                metadata={
                    **normalized.metadata,
                    "empty_table": True,
                },
            )

        groups = self._group_rows(
            normalized
        )

        chunks: list[
            TableChunk
        ] = []

        for chunk_index, rows in enumerate(
            groups
        ):
            if not rows:
                continue

            chunk = self._create_chunk(
                table=normalized,
                rows=rows,
                document_id=document_id,
                chunk_index=chunk_index,
                source_filename=(
                    source_filename
                    or normalized
                    .source_filename
                ),
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                document_version=(
                    document_version
                ),
                metadata=metadata or {},
            )

            chunks.append(chunk)

        if self.config.link_neighbors:
            self._link_neighbors(
                chunks
            )

        return TableChunkingResult(
            chunks=chunks,
            document_id=document_id,
            table_id=normalized.table_id,
            row_count=normalized.row_count,
            column_count=(
                normalized.column_count
            ),
            metadata={
                **normalized.metadata,
                "title": normalized.title,
                "headers":
                    normalized.headers,
                "source_format":
                    normalized
                    .source_format.value,
            },
        )

    def chunk_markdown(
        self,
        text: str,
        *,
        document_id: str,
        table_id: Optional[str] = None,
        title: Optional[str] = None,
        source_filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        document_version: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> TableChunkingResult:

        table = self.parse_markdown(
            text,
            table_id=table_id,
            title=title,
            page_number=page_number,
            source_filename=(
                source_filename
            ),
        )

        return self.chunk(
            table,
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

    # ==========================================================
    # Markdown parsing
    # ==========================================================

    def parse_markdown(
        self,
        text: str,
        *,
        table_id: Optional[str] = None,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        source_filename: Optional[str] = None,
    ) -> TableData:

        if not text or not text.strip():
            raise TableChunkingError(
                "Table text is empty.",
                code="EMPTY_TABLE",
            )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        detected_title = title
        detected_id = table_id
        detected_page = page_number

        table_lines: list[str] = []

        for line in lines:
            marker = self.TABLE_MARKER.match(
                line
            )

            if marker:
                detected_id = (
                    detected_id
                    or f"table_{marker.group('number')}"
                )

                if (
                    not detected_title
                    and marker.group("title")
                ):
                    detected_title = (
                        marker.group(
                            "title"
                        ).strip()
                    )

                continue

            page = self.PAGE_MARKER.match(
                line
            )

            if page:
                detected_page = int(
                    page.group("number")
                )
                continue

            if "|" in line:
                table_lines.append(line)
            elif (
                not detected_title
                and not table_lines
            ):
                detected_title = line

        if len(table_lines) < 2:
            raise TableChunkingError(
                "Unable to identify a valid "
                "Markdown table.",
                code="INVALID_MARKDOWN_TABLE",
            )

        raw_rows = [
            self._parse_markdown_row(
                line
            )
            for line in table_lines
            if not self.MARKDOWN_SEPARATOR.match(
                line
            )
        ]

        if not raw_rows:
            raise TableChunkingError(
                "Markdown table contains "
                "no rows.",
                code="EMPTY_TABLE",
            )

        headers = raw_rows[0]
        data_rows = raw_rows[1:]

        rows = self._create_rows(
            data_rows,
            headers=headers,
            page_number=detected_page,
        )

        return TableData(
            table_id=(
                detected_id
                or self._table_id(
                    headers,
                    data_rows,
                )
            ),
            headers=headers,
            rows=rows,
            title=detected_title,
            page_start=detected_page,
            page_end=detected_page,
            source_format=(
                TableFormat.MARKDOWN
            ),
            source_filename=(
                source_filename
            ),
        )

    @staticmethod
    def _parse_markdown_row(
        line: str,
    ) -> list[str]:

        line = line.strip()

        if line.startswith("|"):
            line = line[1:]

        if line.endswith("|"):
            line = line[:-1]

        # Preserve escaped pipes.
        placeholder = (
            "__ESCAPED_PIPE__"
        )

        line = line.replace(
            r"\|",
            placeholder,
        )

        values = [
            value.strip().replace(
                placeholder,
                "|",
            )
            for value in line.split("|")
        ]

        return values

    # ==========================================================
    # Input normalization
    # ==========================================================

    def _normalize_input(
        self,
        table: (
            TableData
            | Sequence[Sequence[Any]]
            | Sequence[dict[str, Any]]
        ),
        *,
        document_id: str,
        table_id: Optional[str],
        headers: Optional[
            Sequence[str]
        ],
        title: Optional[str],
        source_filename: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> TableData:

        if isinstance(
            table,
            TableData,
        ):
            self._validate_table(
                table
            )

            return table

        values = list(table)

        if not values:
            return TableData(
                table_id=(
                    table_id
                    or self._table_id(
                        [],
                        [],
                    )
                ),
                headers=list(
                    headers or []
                ),
                rows=[],
                title=title,
                page_start=page_start,
                page_end=page_end,
                source_format=(
                    TableFormat.ROWS
                ),
                source_filename=(
                    source_filename
                ),
                metadata=metadata or {},
            )

        if isinstance(
            values[0],
            dict,
        ):
            return (
                self._normalize_dict_rows(
                    values,
                    table_id=table_id,
                    headers=headers,
                    title=title,
                    source_filename=(
                        source_filename
                    ),
                    page_start=page_start,
                    page_end=page_end,
                    metadata=metadata,
                )
            )

        return (
            self._normalize_sequence_rows(
                values,
                table_id=table_id,
                headers=headers,
                title=title,
                source_filename=(
                    source_filename
                ),
                page_start=page_start,
                page_end=page_end,
                metadata=metadata,
            )
        )

    def _normalize_dict_rows(
        self,
        values: Sequence[
            dict[str, Any]
        ],
        *,
        table_id: Optional[str],
        headers: Optional[
            Sequence[str]
        ],
        title: Optional[str],
        source_filename: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> TableData:

        if headers:
            normalized_headers = [
                str(value)
                for value in headers
            ]
        else:
            normalized_headers = []

            seen: set[str] = set()

            for row in values:
                for key in row.keys():
                    key = str(key)

                    if key not in seen:
                        seen.add(key)
                        normalized_headers.append(
                            key
                        )

        raw_rows = [
            [
                self._normalize_cell(
                    row.get(
                        header,
                        "",
                    )
                )
                for header
                in normalized_headers
            ]
            for row in values
        ]

        rows = self._create_rows(
            raw_rows,
            headers=normalized_headers,
        )

        result = TableData(
            table_id=(
                table_id
                or self._table_id(
                    normalized_headers,
                    raw_rows,
                )
            ),
            headers=normalized_headers,
            rows=rows,
            title=title,
            page_start=page_start,
            page_end=page_end,
            source_format=(
                TableFormat.ROWS
            ),
            source_filename=(
                source_filename
            ),
            metadata=metadata or {},
        )

        self._validate_table(
            result
        )

        return result

    def _normalize_sequence_rows(
        self,
        values: Sequence[
            Sequence[Any]
        ],
        *,
        table_id: Optional[str],
        headers: Optional[
            Sequence[str]
        ],
        title: Optional[str],
        source_filename: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> TableData:

        raw_rows = [
            [
                self._normalize_cell(
                    value
                )
                for value in row
            ]
            for row in values
        ]

        if headers:
            normalized_headers = [
                self._normalize_cell(
                    value
                )
                for value in headers
            ]

        else:
            width = max(
                (
                    len(row)
                    for row in raw_rows
                ),
                default=0,
            )

            normalized_headers = [
                f"Column {index + 1}"
                for index in range(width)
            ]

        rows = self._create_rows(
            raw_rows,
            headers=normalized_headers,
        )

        result = TableData(
            table_id=(
                table_id
                or self._table_id(
                    normalized_headers,
                    raw_rows,
                )
            ),
            headers=normalized_headers,
            rows=rows,
            title=title,
            page_start=page_start,
            page_end=page_end,
            source_format=(
                TableFormat.ROWS
            ),
            source_filename=(
                source_filename
            ),
            metadata=metadata or {},
        )

        self._validate_table(
            result
        )

        return result

    def _create_rows(
        self,
        rows: Sequence[
            Sequence[Any]
        ],
        *,
        headers: Sequence[str],
        page_number: Optional[int] = None,
    ) -> list[TableRow]:

        output: list[
            TableRow
        ] = []

        for row_index, values in enumerate(
            rows
        ):
            normalized = [
                self._normalize_cell(
                    value
                )
                for value in values
            ]

            cells: list[
                TableCell
            ] = []

            for column_index, value in enumerate(
                normalized
            ):
                column_name = (
                    headers[column_index]
                    if column_index
                    < len(headers)
                    else (
                        f"Column "
                        f"{column_index + 1}"
                    )
                )

                cells.append(
                    TableCell(
                        value=value,
                        row_index=row_index,
                        column_index=(
                            column_index
                        ),
                        column_name=(
                            column_name
                        ),
                    )
                )

            output.append(
                TableRow(
                    row_index=row_index,
                    values=normalized,
                    cells=cells,
                    page_number=(
                        page_number
                    ),
                )
            )

        return output

    # ==========================================================
    # Row grouping
    # ==========================================================

    def _group_rows(
        self,
        table: TableData,
    ) -> list[list[TableRow]]:

        groups: list[
            list[TableRow]
        ] = []

        current: list[
            TableRow
        ] = []

        for row in table.rows:
            candidate = [
                *current,
                row,
            ]

            candidate_text = (
                self._render_rows(
                    table,
                    candidate,
                )
            )

            exceeds_size = (
                current
                and len(candidate_text)
                > self.config
                .max_chunk_size
            )

            exceeds_rows = (
                current
                and len(candidate)
                > self.config
                .max_rows_per_chunk
            )

            target_reached = (
                current
                and len(
                    self._render_rows(
                        table,
                        current,
                    )
                )
                >= self.config
                .target_chunk_size
                and len(current)
                >= self.config
                .min_rows_per_chunk
            )

            if (
                exceeds_size
                or exceeds_rows
                or target_reached
            ):
                groups.append(
                    current
                )

                overlap = (
                    self._row_overlap(
                        current
                    )
                )

                current = [
                    *overlap,
                    row,
                ]

                # If one row is huge, don't allow
                # the overlap to make it worse.
                if (
                    len(
                        self._render_rows(
                            table,
                            current,
                        )
                    )
                    > self.config
                    .max_chunk_size
                    and len(current) > 1
                ):
                    current = [row]

            else:
                current.append(row)

        if current:
            groups.append(current)

        return self._deduplicate_groups(
            groups
        )

    def _row_overlap(
        self,
        rows: list[TableRow],
    ) -> list[TableRow]:

        count = self.config.row_overlap

        if count <= 0:
            return []

        return rows[-count:]

    # ==========================================================
    # Chunk construction
    # ==========================================================

    def _create_chunk(
        self,
        *,
        table: TableData,
        rows: list[TableRow],
        document_id: str,
        chunk_index: int,
        source_filename: Optional[str],
        tenant_id: Optional[str],
        organization_id: Optional[str],
        document_version: Optional[str],
        metadata: dict[str, Any],
    ) -> TableChunk:

        text = self._render_rows(
            table,
            rows,
        )

        row_indices = [
            row.row_index
            for row in rows
        ]

        pages = [
            row.page_number
            for row in rows
            if row.page_number
            is not None
        ]

        page_start = (
            min(pages)
            if pages
            else table.page_start
        )

        page_end = (
            max(pages)
            if pages
            else table.page_end
        )

        if (
            len(rows)
            == table.row_count
            and chunk_index == 0
        ):
            chunk_type = (
                TableChunkType
                .WHOLE_TABLE
            )

        elif len(rows) == 1:
            chunk_type = (
                TableChunkType
                .SINGLE_ROW
            )

        elif chunk_index > 0:
            chunk_type = (
                TableChunkType
                .CONTINUATION
            )

        else:
            chunk_type = (
                TableChunkType
                .ROW_GROUP
            )

        chunk_id = self._chunk_id(
            document_id=document_id,
            table_id=table.table_id,
            chunk_index=chunk_index,
            text=text,
        )

        return TableChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            table_id=table.table_id,
            text=text,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            title=table.title,
            headers=list(
                table.headers
            ),
            row_start=(
                min(row_indices)
                if row_indices
                else None
            ),
            row_end=(
                max(row_indices)
                if row_indices
                else None
            ),
            row_indices=row_indices,
            page_start=page_start,
            page_end=page_end,
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
                **table.metadata,
                **metadata,
                "table_title":
                    table.title,
                "source_format":
                    table
                    .source_format.value,
                "row_count":
                    len(rows),
                "column_count":
                    table.column_count,
            },
        )

    # ==========================================================
    # Rendering
    # ==========================================================

    def _render_rows(
        self,
        table: TableData,
        rows: Sequence[
            TableRow
        ],
    ) -> str:

        output: list[str] = []

        if (
            self.config.repeat_title
            and table.title
        ):
            output.append(
                f"Table: {table.title}"
            )

        if self.config.repeat_headers:
            output.append(
                self._render_header(
                    table.headers
                )
            )

        for row in rows:
            output.append(
                self._render_row(
                    row,
                    table.headers,
                )
            )

        return "\n".join(
            value
            for value in output
            if value
        ).strip()

    @staticmethod
    def _render_header(
        headers: Sequence[str],
    ) -> str:

        if not headers:
            return ""

        values = [
            TableChunker
            ._escape_markdown(
                value
            )
            for value in headers
        ]

        header = (
            "| "
            + " | ".join(values)
            + " |"
        )

        separator = (
            "| "
            + " | ".join(
                "---"
                for _ in values
            )
            + " |"
        )

        return (
            f"{header}\n"
            f"{separator}"
        )

    def _render_row(
        self,
        row: TableRow,
        headers: Sequence[str],
    ) -> str:

        width = max(
            len(headers),
            len(row.values),
        )

        values = list(
            row.values
        )

        while len(values) < width:
            values.append("")

        values = [
            self._escape_markdown(
                value
            )
            for value in values
        ]

        rendered = (
            "| "
            + " | ".join(values)
            + " |"
        )

        if (
            self.config
            .include_row_numbers
        ):
            return (
                f"Row {row.row_index + 1}: "
                f"{rendered}"
            )

        return rendered

    @staticmethod
    def _escape_markdown(
        value: Any,
    ) -> str:

        text = str(
            value
            if value is not None
            else ""
        )

        text = text.replace(
            "\n",
            " ",
        )

        text = text.replace(
            "|",
            r"\|",
        )

        return text.strip()

    # ==========================================================
    # Cell normalization
    # ==========================================================

    def _normalize_cell(
        self,
        value: Any,
    ) -> str:

        if value is None:
            return ""

        if isinstance(
            value,
            (dict, list, tuple),
        ):
            try:
                text = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                text = str(value)
        else:
            text = str(value)

        if self.config.normalize_cells:
            text = text.replace(
                "\r\n",
                "\n",
            )

            text = text.replace(
                "\r",
                "\n",
            )

            text = re.sub(
                r"[ \t]+",
                " ",
                text,
            )

            text = re.sub(
                r"\n{3,}",
                "\n\n",
                text,
            )

            text = text.strip()

        if (
            len(text)
            > self.config
            .max_cell_characters
        ):
            text = (
                text[
                    : self.config
                    .max_cell_characters
                ]
                + " …"
            )

        return text

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_table(
        self,
        table: TableData,
    ) -> None:

        if (
            table.column_count
            > self.config.max_columns
        ):
            raise TableChunkingError(
                "Table exceeds maximum "
                "column count.",
                code="TOO_MANY_COLUMNS",
            )

        if (
            table.row_count
            > self.config.max_rows
        ):
            raise TableChunkingError(
                "Table exceeds maximum "
                "row count.",
                code="TOO_MANY_ROWS",
            )

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
                "be > 0."
            )

        if (
            config.max_chunk_size
            < config.target_chunk_size
        ):
            raise ValueError(
                "max_chunk_size must be "
                ">= target_chunk_size."
            )

        if (
            config.max_rows_per_chunk
            <= 0
        ):
            raise ValueError(
                "max_rows_per_chunk must "
                "be > 0."
            )

        if (
            config.min_rows_per_chunk
            <= 0
        ):
            raise ValueError(
                "min_rows_per_chunk must "
                "be > 0."
            )

        if (
            config.row_overlap < 0
        ):
            raise ValueError(
                "row_overlap cannot "
                "be negative."
            )

        if (
            config.row_overlap
            >= config.max_rows_per_chunk
        ):
            raise ValueError(
                "row_overlap must be less "
                "than max_rows_per_chunk."
            )

    # ==========================================================
    # Neighbors
    # ==========================================================

    @staticmethod
    def _link_neighbors(
        chunks: list[
            TableChunk
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

            if index + 1 < len(chunks):
                chunk.next_chunk_id = (
                    chunks[
                        index + 1
                    ].chunk_id
                )

    # ==========================================================
    # Token counting
    # ==========================================================

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
                result = (
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                )

                return len(result)

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

    # ==========================================================
    # IDs
    # ==========================================================

    def _chunk_id(
        self,
        *,
        document_id: str,
        table_id: str,
        chunk_index: int,
        text: str,
    ) -> str:

        if (
            not self.config
            .deterministic_ids
        ):
            return (
                "table_chunk_"
                + str(uuid.uuid4())
            )

        content_hash = (
            hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
        )

        identity = (
            f"{document_id}|"
            f"{table_id}|"
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
            f"table_chunk_"
            f"{digest[:24]}"
        )

    @staticmethod
    def _table_id(
        headers: Sequence[Any],
        rows: Sequence[
            Sequence[Any]
        ],
    ) -> str:

        sample = {
            "headers": [
                str(value)
                for value in headers
            ],
            "rows": [
                [
                    str(value)
                    for value in row
                ]
                for row in rows[:10]
            ],
        }

        encoded = json.dumps(
            sample,
            sort_keys=True,
            ensure_ascii=False,
        )

        digest = hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

        return (
            f"table_{digest[:20]}"
        )

    # ==========================================================
    # Deduplication
    # ==========================================================

    @staticmethod
    def _deduplicate_groups(
        groups: Iterable[
            list[TableRow]
        ],
    ) -> list[
        list[TableRow]
    ]:

        output: list[
            list[TableRow]
        ] = []

        previous: Optional[
            tuple[int, ...]
        ] = None

        for group in groups:
            signature = tuple(
                row.row_index
                for row in group
            )

            if not signature:
                continue

            if signature == previous:
                continue

            output.append(group)
            previous = signature

        return output


def chunk_table(
    table: (
        TableData
        | Sequence[Sequence[Any]]
        | Sequence[dict[str, Any]]
    ),
    *,
    document_id: str,
    table_id: Optional[str] = None,
    headers: Optional[
        Sequence[str]
    ] = None,
    title: Optional[str] = None,
    source_filename: Optional[str] = None,
    tenant_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    document_version: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> TableChunkingResult:
    """
    Convenience entry point.
    """

    chunker = TableChunker()

    return chunker.chunk(
        table,
        document_id=document_id,
        table_id=table_id,
        headers=headers,
        title=title,
        source_filename=source_filename,
        tenant_id=tenant_id,
        organization_id=organization_id,
        document_version=document_version,
        page_start=page_start,
        page_end=page_end,
        metadata=metadata,
    )