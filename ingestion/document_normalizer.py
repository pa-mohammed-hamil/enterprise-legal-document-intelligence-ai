"""
Legal document normalization.

Path:
    backend/ingestion/document_normalizer.py

Responsibilities:
    - Normalize extracted document text.
    - Preserve paragraph and section boundaries.
    - Normalize whitespace and Unicode artifacts.
    - Normalize common PDF/OCR extraction problems.
    - Preserve legally meaningful punctuation and numbers.
    - Produce deterministic normalized output.

Non-responsibilities:
    - OCR.
    - PDF parsing.
    - Clause classification.
    - Legal interpretation.
    - Risk detection.
    - Chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any, Optional


class NormalizationLevel(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass(slots=True)
class NormalizationResult:
    """Normalized document output."""

    text: str

    original_length: int
    normalized_length: int

    level: NormalizationLevel

    changed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DocumentNormalizer:
    """
    Deterministic document text normalizer.

    The normalizer is intentionally conservative because legal
    documents contain information where seemingly harmless changes
    can alter meaning.

    Examples of content that must be preserved:

        - section numbers
        - dates
        - monetary values
        - percentages
        - punctuation
        - contract identifiers
        - legal citations
        - paragraph boundaries
        - capitalization where useful
    """

    # Unicode characters frequently introduced by PDF extraction.
    UNICODE_REPLACEMENTS: dict[str, str] = {
        "\u2018": "'",       # left single quote
        "\u2019": "'",       # right single quote
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',

        "\u2013": "-",       # en dash
        "\u2014": "-",       # em dash
        "\u2212": "-",       # mathematical minus

        "\u00a0": " ",       # non-breaking space
        "\u202f": " ",       # narrow NBSP
        "\u2007": " ",

        "\u2026": "...",     # ellipsis

        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
    }

    # Characters that often appear as extraction noise.
    ZERO_WIDTH_CHARS = (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    )

    # Common OCR/PDF ligatures and artifacts.
    OCR_REPLACEMENTS: dict[str, str] = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    def __init__(
        self,
        level: NormalizationLevel = (
            NormalizationLevel.STANDARD
        ),
        *,
        preserve_page_markers: bool = True,
        preserve_line_breaks: bool = True,
        normalize_unicode: bool = True,
        fix_hyphenated_line_breaks: bool = True,
        collapse_spaces: bool = True,
        remove_repeated_headers: bool = False,
        remove_repeated_footers: bool = False,
    ) -> None:
        self.level = level

        self.preserve_page_markers = (
            preserve_page_markers
        )

        self.preserve_line_breaks = (
            preserve_line_breaks
        )

        self.normalize_unicode = (
            normalize_unicode
        )

        self.fix_hyphenated_line_breaks = (
            fix_hyphenated_line_breaks
        )

        self.collapse_spaces = (
            collapse_spaces
        )

        self.remove_repeated_headers = (
            remove_repeated_headers
        )

        self.remove_repeated_footers = (
            remove_repeated_footers
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        text: str,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> NormalizationResult:
        """
        Normalize extracted document text.

        Processing order is deliberately deterministic.
        """

        self._validate_text(text)

        original = text
        warnings: list[str] = []

        if not text:
            return self._result(
                original,
                "",
                warnings,
                metadata,
            )

        normalized = text

        if self.normalize_unicode:
            normalized = (
                self._normalize_unicode(
                    normalized
                )
            )

        normalized = (
            self._remove_zero_width_chars(
                normalized
            )
        )

        normalized = (
            self._normalize_line_endings(
                normalized
            )
        )

        normalized = (
            self._remove_extraction_artifacts(
                normalized
            )
        )

        if self.fix_hyphenated_line_breaks:
            normalized = (
                self._fix_hyphenated_line_breaks(
                    normalized
                )
            )

        if self.collapse_spaces:
            normalized = (
                self._normalize_spaces(
                    normalized
                )
            )

        normalized = (
            self._normalize_blank_lines(
                normalized
            )
        )

        if self.level in {
            NormalizationLevel.STANDARD,
            NormalizationLevel.AGGRESSIVE,
        }:
            normalized = (
                self._normalize_paragraphs(
                    normalized
                )
            )

        if self.level == (
            NormalizationLevel.AGGRESSIVE
        ):
            normalized = (
                self._aggressive_cleanup(
                    normalized
                )
            )

        normalized = normalized.strip()

        if original != normalized:
            warnings.append(
                "Document text was modified "
                "during normalization."
            )

        return self._result(
            original,
            normalized,
            warnings,
            metadata,
        )

    def normalize_pages(
        self,
        pages: list[str],
        *,
        page_marker_template: str = (
            "[PAGE {page}]"
        ),
    ) -> NormalizationResult:
        """
        Normalize individual pages while preserving
        page boundaries.
        """

        if not pages:
            return self.normalize("")

        normalized_pages: list[str] = []

        for index, page in enumerate(
            pages,
            start=1,
        ):
            result = self.normalize(page)

            page_text = result.text

            if self.preserve_page_markers:
                page_text = (
                    f"{page_marker_template.format(page=index)}\n"
                    f"{page_text}"
                )

            normalized_pages.append(
                page_text
            )

        combined = "\n\n".join(
            normalized_pages
        )

        return self.normalize(
            combined,
            metadata={
                "page_count": len(pages),
            },
        )

    def normalize_lines(
        self,
        lines: list[str],
    ) -> list[str]:
        """
        Normalize lines independently.

        Useful before layout-aware processing.
        """

        normalized: list[str] = []

        for line in lines:
            result = self.normalize(line)
            normalized.append(result.text)

        return normalized

    # ------------------------------------------------------------------
    # Unicode
    # ------------------------------------------------------------------

    def _normalize_unicode(
        self,
        text: str,
    ) -> str:
        """
        Normalize Unicode without aggressively changing
        meaningful legal text.
        """

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        for source, target in (
            self.UNICODE_REPLACEMENTS.items()
        ):
            text = text.replace(
                source,
                target,
            )

        for source, target in (
            self.OCR_REPLACEMENTS.items()
        ):
            text = text.replace(
                source,
                target,
            )

        return text

    def _remove_zero_width_chars(
        self,
        text: str,
    ) -> str:
        for char in self.ZERO_WIDTH_CHARS:
            text = text.replace(
                char,
                "",
            )

        return text

    # ------------------------------------------------------------------
    # Line handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_line_endings(
        text: str,
    ) -> str:
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        return text

    def _fix_hyphenated_line_breaks(
        self,
        text: str,
    ) -> str:
        """
        Convert:

            confi-
            dential

        into:

            confidential

        but preserve cases such as:

            well-
            known

        only when the pattern strongly resembles
        a line-wrapped word.

        """

        if not text:
            return text

        pattern = re.compile(
            r"(?<=[A-Za-z])-\n"
            r"(?=[a-z])"
        )

        return pattern.sub(
            "",
            text,
        )

    def _normalize_spaces(
        self,
        text: str,
    ) -> str:
        """
        Collapse repeated horizontal whitespace while
        preserving newlines.
        """

        text = re.sub(
            r"[ \t\f\v]+",
            " ",
            text,
        )

        # Remove spaces directly before punctuation.
        text = re.sub(
            r"[ \t]+([,.;:!?])",
            r"\1",
            text,
        )

        # Normalize spaces after punctuation.
        text = re.sub(
            r"([,;:]) {2,}",
            r"\1 ",
            text,
        )

        return text

    def _normalize_blank_lines(
        self,
        text: str,
    ) -> str:
        # Maximum two consecutive newlines.
        return re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

    # ------------------------------------------------------------------
    # Extraction artifacts
    # ------------------------------------------------------------------

    def _remove_extraction_artifacts(
        self,
        text: str,
    ) -> str:
        """
        Remove common extraction artifacts while preserving
        legal content.
        """

        # Remove obvious form-feed/page separators.
        text = text.replace(
            "\f",
            "\n",
        )

        # Normalize repeated decorative bullets.
        text = re.sub(
            r"[•·]{2,}",
            "•",
            text,
        )

        # Remove isolated control characters.
        text = "".join(
            char
            for char in text
            if (
                char == "\n"
                or char == "\t"
                or ord(char) >= 32
            )
        )

        return text

    # ------------------------------------------------------------------
    # Paragraph normalization
    # ------------------------------------------------------------------

    def _normalize_paragraphs(
        self,
        text: str,
    ) -> str:
        """
        Preserve paragraphs while removing accidental
        line wrapping.
        """

        lines = text.split("\n")

        result: list[str] = []
        buffer: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if buffer:
                    result.append(
                        " ".join(buffer)
                    )
                    buffer = []

                result.append("")
                continue

            if self._is_page_marker(stripped):
                if buffer:
                    result.append(
                        " ".join(buffer)
                    )
                    buffer = []

                result.append(stripped)
                continue

            if self._looks_like_heading(stripped):
                if buffer:
                    result.append(
                        " ".join(buffer)
                    )
                    buffer = []

                result.append(stripped)
                continue

            if self._looks_like_list_item(
                stripped
            ):
                if buffer:
                    result.append(
                        " ".join(buffer)
                    )
                    buffer = []

                result.append(stripped)
                continue

            buffer.append(stripped)

        if buffer:
            result.append(
                " ".join(buffer)
            )

        return "\n".join(result)

    @staticmethod
    def _looks_like_heading(
        line: str,
    ) -> bool:
        """
        Conservative heading detection.
        """

        if len(line) > 160:
            return False

        if re.match(
            r"^\d+(?:\.\d+)*[\s.)-]+",
            line,
        ):
            return True

        if re.match(
            r"^(section|article|schedule|"
            r"appendix|exhibit|annex)\b",
            line,
            flags=re.IGNORECASE,
        ):
            return True

        # Short all-uppercase lines are often headings.
        if (
            len(line) <= 100
            and line.upper() == line
            and any(char.isalpha() for char in line)
        ):
            return True

        return False

    @staticmethod
    def _looks_like_list_item(
        line: str,
    ) -> bool:
        return bool(
            re.match(
                r"^(?:"
                r"[•●▪◦]"
                r"|[-*]"
                r"|\(?[a-z]\)"
                r"|\(?[ivxlcdm]+\)"
                r"|\(?\d+\)"
                r")\s+",
                line,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_page_marker(
        line: str,
    ) -> bool:
        return bool(
            re.match(
                r"^\[PAGE\s+\d+\]$",
                line,
                flags=re.IGNORECASE,
            )
        )

    # ------------------------------------------------------------------
    # Aggressive mode
    # ------------------------------------------------------------------

    def _aggressive_cleanup(
        self,
        text: str,
    ) -> str:
        """
        Additional cleanup.

        Use carefully for legal documents because aggressive
        normalization can remove layout information.
        """

        # Remove whitespace surrounding page markers.
        text = re.sub(
            r"\n+\[PAGE\s+(\d+)\]\n+",
            r"\n[PAGE \1]\n",
            text,
            flags=re.IGNORECASE,
        )

        # Remove trailing spaces from every line.
        text = "\n".join(
            line.rstrip()
            for line in text.splitlines()
        )

        return text

    # ------------------------------------------------------------------
    # Validation / result
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if len(text) > 20_000_000:
            raise ValueError(
                "Document text exceeds the "
                "20 MB normalization limit"
            )

    @staticmethod
    def _result(
        original: str,
        normalized: str,
        warnings: list[str],
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> NormalizationResult:
        return NormalizationResult(
            text=normalized,
            original_length=len(original),
            normalized_length=len(normalized),
            level=NormalizationLevel.STANDARD,
            changed=original != normalized,
            warnings=warnings,
            metadata=metadata or {},
        )


__all__ = [
    "NormalizationLevel",
    "NormalizationResult",
    "DocumentNormalizer",
]