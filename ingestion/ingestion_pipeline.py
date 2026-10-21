"""
Enterprise legal document ingestion pipeline.

Path:
    backend/ingestion/ingestion_pipeline.py

Pipeline:

    Input
      │
      ▼
    Validation
      │
      ▼
    Format Detection
      │
      ▼
    Document Parser
      │
      ├── PDF
      ├── DOCX
      ├── DOC
      ├── TXT
      ├── Markdown
      ├── HTML
      └── JSON
      │
      ▼
    OCR fallback (when configured)
      │
      ▼
    Document Normalization
      │
      ▼
    Metadata Extraction
      │
      ▼
    Language Detection
      │
      ▼
    Ingestion Result
      │
      ├── text
      ├── pages
      ├── metadata
      ├── warnings
      ├── provenance
      └── processing metrics

Non-responsibilities:
    - Chunking
    - Embeddings
    - Vector database writes
    - LLM generation
    - Legal advice
    - Risk decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import hashlib
import logging
import mimetypes
import time


from .document_parser import (
    DocumentFormat,
    DocumentPage,
    DocumentParser,
    ParsedDocument,
    ParserConfig,
    ParserError,
    ParseStatus,
)

from .document_normalizer import (
    DocumentNormalizer,
    NormalizationLevel,
    NormalizationResult,
)


logger = logging.getLogger(__name__)


# ======================================================================
# Enums
# ======================================================================


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class IngestionStage(str, Enum):
    VALIDATION = "validation"
    PARSING = "parsing"
    OCR = "ocr"
    NORMALIZATION = "normalization"
    METADATA = "metadata"
    LANGUAGE = "language"
    FINALIZATION = "finalization"


class IngestionErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_FORMAT = "unsupported_format"
    FILE_TOO_LARGE = "file_too_large"
    PARSE_ERROR = "parse_error"
    OCR_ERROR = "ocr_error"
    NORMALIZATION_ERROR = "normalization_error"
    METADATA_ERROR = "metadata_error"
    LANGUAGE_ERROR = "language_error"
    PIPELINE_ERROR = "pipeline_error"


# ======================================================================
# Data structures
# ======================================================================


@dataclass(slots=True)
class IngestionError:
    code: IngestionErrorCode
    stage: IngestionStage
    message: str
    recoverable: bool = False
    details: Optional[str] = None


@dataclass(slots=True)
class IngestionMetrics:
    total_ms: float = 0.0
    parsing_ms: float = 0.0
    ocr_ms: float = 0.0
    normalization_ms: float = 0.0
    metadata_ms: float = 0.0
    language_ms: float = 0.0

    input_bytes: int = 0
    input_characters: int = 0
    output_characters: int = 0
    word_count: int = 0

    page_count: int = 0

    ocr_used: bool = False
    normalization_changed: bool = False


@dataclass(slots=True)
class IngestionProvenance:
    document_id: str

    filename: str

    content_hash: str

    format: DocumentFormat

    mime_type: Optional[str]

    parser_version: str = "1.0"

    pipeline_version: str = "1.0"

    source: str = "upload"

    created_at: float = field(
        default_factory=time.time
    )


@dataclass(slots=True)
class IngestionResult:
    document_id: str

    filename: str

    status: IngestionStatus

    format: DocumentFormat

    mime_type: Optional[str]

    text: str

    pages: list[DocumentPage] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[IngestionError] = field(
        default_factory=list
    )

    provenance: Optional[
        IngestionProvenance
    ] = None

    metrics: IngestionMetrics = field(
        default_factory=IngestionMetrics
    )


@dataclass(slots=True)
class IngestionConfig:
    max_file_size_bytes: int = (
        50 * 1024 * 1024
    )

    normalization_level: (
        NormalizationLevel
    ) = NormalizationLevel.STANDARD

    enable_ocr_fallback: bool = True

    minimum_text_characters: int = 20

    ocr_min_text_characters: int = 50

    extract_metadata: bool = True

    detect_language: bool = True

    preserve_pages: bool = True

    calculate_hash: bool = True

    fail_on_normalization_error: bool = False

    fail_on_metadata_error: bool = False

    fail_on_language_error: bool = False


# ======================================================================
# Ingestion Pipeline
# ======================================================================


class IngestionPipeline:
    """
    Orchestrates the complete legal-document ingestion process.

    Specialized components are injected so the pipeline remains
    independently testable and does not own OCR, metadata, or
    language-detection implementations.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        config: Optional[IngestionConfig] = None,
        parser: Optional[DocumentParser] = None,
        normalizer: Optional[
            DocumentNormalizer
        ] = None,
        ocr_processor: Optional[
            Callable[
                [bytes, str],
                Any,
            ]
        ] = None,
        metadata_extractor: Optional[
            Callable[
                [ParsedDocument],
                dict[str, Any],
            ]
        ] = None,
        language_detector: Optional[
            Callable[
                [str],
                Any,
            ]
        ] = None,
    ) -> None:
        self.config = (
            config
            or IngestionConfig()
        )

        self.parser = (
            parser
            or DocumentParser(
                config=ParserConfig(
                    max_file_size_bytes=(
                        self.config.max_file_size_bytes
                    )
                )
            )
        )

        self.normalizer = (
            normalizer
            or DocumentNormalizer(
                level=(
                    self.config.normalization_level
                )
            )
        )

        self.ocr_processor = (
            ocr_processor
        )

        self.metadata_extractor = (
            metadata_extractor
        )

        self.language_detector = (
            language_detector
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        content: bytes,
        *,
        filename: str,
        document_id: Optional[str] = None,
        mime_type: Optional[str] = None,
        source: str = "upload",
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> IngestionResult:
        """
        Execute the complete ingestion pipeline.
        """

        started_at = time.perf_counter()

        metrics = IngestionMetrics(
            input_bytes=len(content)
            if isinstance(content, bytes)
            else 0
        )

        warnings: list[str] = []
        errors: list[
            IngestionError
        ] = []

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        try:
            self._validate_input(
                content,
                filename,
            )
        except Exception as exc:
            error = IngestionError(
                code=(
                    IngestionErrorCode.INVALID_INPUT
                ),
                stage=IngestionStage.VALIDATION,
                message=str(exc),
                recoverable=False,
            )

            errors.append(error)

            metrics.total_ms = (
                self._elapsed_ms(
                    started_at
                )
            )

            return IngestionResult(
                document_id=(
                    document_id
                    or self._fallback_document_id(
                        content,
                        filename,
                    )
                ),
                filename=filename,
                status=IngestionStatus.FAILED,
                format=DocumentFormat.UNKNOWN,
                mime_type=mime_type,
                text="",
                errors=errors,
                metrics=metrics,
            )

        # --------------------------------------------------------------
        # Document identity
        # --------------------------------------------------------------

        content_hash = (
            self._hash(content)
            if self.config.calculate_hash
            else ""
        )

        document_id = (
            document_id
            or self._generate_document_id(
                content,
                filename,
            )
        )

        # --------------------------------------------------------------
        # Parsing
        # --------------------------------------------------------------

        parse_started = (
            time.perf_counter()
        )

        try:
            parsed = self.parser.parse_bytes(
                content,
                filename=filename,
                document_id=document_id,
                mime_type=mime_type,
                metadata=metadata,
            )

            metrics.parsing_ms = (
                self._elapsed_ms(
                    parse_started
                )
            )

        except ParserError as exc:
            metrics.parsing_ms = (
                self._elapsed_ms(
                    parse_started
                )
            )

            if (
                exc.code.value
                == "unsupported_format"
            ):
                code = (
                    IngestionErrorCode
                    .UNSUPPORTED_FORMAT
                )
            elif (
                exc.code.value
                == "file_too_large"
            ):
                code = (
                    IngestionErrorCode
                    .FILE_TOO_LARGE
                )
            else:
                code = (
                    IngestionErrorCode
                    .PARSE_ERROR
                )

            error = IngestionError(
                code=code,
                stage=IngestionStage.PARSING,
                message=str(exc),
                recoverable=False,
                details=exc.cause,
            )

            errors.append(error)

            metrics.total_ms = (
                self._elapsed_ms(
                    started_at
                )
            )

            return IngestionResult(
                document_id=document_id,
                filename=filename,
                status=IngestionStatus.FAILED,
                format=DocumentFormat.UNKNOWN,
                mime_type=mime_type,
                text="",
                errors=errors,
                metrics=metrics,
            )

        # --------------------------------------------------------------
        # OCR fallback
        # --------------------------------------------------------------

        ocr_used = False

        if self._should_use_ocr(
            parsed
        ):
            if (
                self.config.enable_ocr_fallback
                and self.ocr_processor
            ):
                ocr_started = (
                    time.perf_counter()
                )

                try:
                    ocr_result = (
                        self.ocr_processor(
                            content,
                            filename,
                        )
                    )

                    parsed = (
                        self._merge_ocr_result(
                            parsed,
                            ocr_result,
                        )
                    )

                    ocr_used = True

                except Exception as exc:
                    warnings.append(
                        (
                            "OCR fallback failed: "
                            f"{exc}"
                        )
                    )

                    errors.append(
                        IngestionError(
                            code=(
                                IngestionErrorCode
                                .OCR_ERROR
                            ),
                            stage=(
                                IngestionStage.OCR
                            ),
                            message=(
                                "OCR fallback failed."
                            ),
                            recoverable=True,
                            details=str(exc),
                        )
                    )

                finally:
                    metrics.ocr_ms = (
                        self._elapsed_ms(
                            ocr_started
                        )
                    )

            elif (
                self.config.enable_ocr_fallback
            ):
                warnings.append(
                    (
                        "Document appears to require "
                        "OCR, but no OCR processor "
                        "is configured."
                    )
                )

        metrics.ocr_used = ocr_used

        # --------------------------------------------------------------
        # Normalization
        # --------------------------------------------------------------

        normalized_text = parsed.text

        normalization_result: Optional[
            NormalizationResult
        ] = None

        normalization_started = (
            time.perf_counter()
        )

        try:
            normalization_result = (
                self._normalize(
                    parsed.text
                )
            )

            normalized_text = (
                normalization_result.text
            )

            metrics.normalization_changed = (
                normalization_result.changed
            )

            warnings.extend(
                normalization_result.warnings
            )

        except Exception as exc:
            error = IngestionError(
                code=(
                    IngestionErrorCode
                    .NORMALIZATION_ERROR
                ),
                stage=(
                    IngestionStage.NORMALIZATION
                ),
                message=(
                    "Document normalization failed."
                ),
                recoverable=(
                    not self.config
                    .fail_on_normalization_error
                ),
                details=str(exc),
            )

            errors.append(error)

            if (
                self.config
                .fail_on_normalization_error
            ):
                metrics.total_ms = (
                    self._elapsed_ms(
                        started_at
                    )
                )

                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    status=(
                        IngestionStatus.FAILED
                    ),
                    format=parsed.format,
                    mime_type=parsed.mime_type,
                    text="",
                    pages=parsed.pages,
                    warnings=warnings,
                    errors=errors,
                    metrics=metrics,
                )

        finally:
            metrics.normalization_ms = (
                self._elapsed_ms(
                    normalization_started
                )
            )

        # --------------------------------------------------------------
        # Page synchronization
        # --------------------------------------------------------------

        pages = self._normalize_pages(
            parsed.pages
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        extracted_metadata: dict[
            str,
            Any,
        ] = {}

        metadata_started = (
            time.perf_counter()
        )

        if self.config.extract_metadata:
            try:
                extracted_metadata = (
                    self._extract_metadata(
                        parsed
                    )
                )

            except Exception as exc:
                error = IngestionError(
                    code=(
                        IngestionErrorCode
                        .METADATA_ERROR
                    ),
                    stage=(
                        IngestionStage.METADATA
                    ),
                    message=(
                        "Metadata extraction failed."
                    ),
                    recoverable=(
                        not self.config
                        .fail_on_metadata_error
                    ),
                    details=str(exc),
                )

                errors.append(error)

                if (
                    self.config
                    .fail_on_metadata_error
                ):
                    metrics.total_ms = (
                        self._elapsed_ms(
                            started_at
                        )
                    )

                    return IngestionResult(
                        document_id=document_id,
                        filename=filename,
                        status=(
                            IngestionStatus.FAILED
                        ),
                        format=parsed.format,
                        mime_type=parsed.mime_type,
                        text=normalized_text,
                        pages=pages,
                        warnings=warnings,
                        errors=errors,
                        metrics=metrics,
                    )

        metrics.metadata_ms = (
            self._elapsed_ms(
                metadata_started
            )
        )

        # --------------------------------------------------------------
        # Language detection
        # --------------------------------------------------------------

        language_data: dict[
            str,
            Any,
        ] = {}

        language_started = (
            time.perf_counter()
        )

        if self.config.detect_language:
            try:
                language_data = (
                    self._detect_language(
                        normalized_text
                    )
                )

            except Exception as exc:
                error = IngestionError(
                    code=(
                        IngestionErrorCode
                        .LANGUAGE_ERROR
                    ),
                    stage=(
                        IngestionStage.LANGUAGE
                    ),
                    message=(
                        "Language detection failed."
                    ),
                    recoverable=(
                        not self.config
                        .fail_on_language_error
                    ),
                    details=str(exc),
                )

                errors.append(error)

                if (
                    self.config
                    .fail_on_language_error
                ):
                    metrics.total_ms = (
                        self._elapsed_ms(
                            started_at
                        )
                    )

                    return IngestionResult(
                        document_id=document_id,
                        filename=filename,
                        status=(
                            IngestionStatus.FAILED
                        ),
                        format=parsed.format,
                        mime_type=parsed.mime_type,
                        text=normalized_text,
                        pages=pages,
                        warnings=warnings,
                        errors=errors,
                        metrics=metrics,
                    )

        metrics.language_ms = (
            self._elapsed_ms(
                language_started
            )
        )

        # --------------------------------------------------------------
        # Final metadata
        # --------------------------------------------------------------

        final_metadata = {
            **parsed.metadata,
            **(metadata or {}),
            **extracted_metadata,
            **language_data,
            "ingestion": {
                "pipeline_version": self.VERSION,
                "parser_status": (
                    parsed.status.value
                ),
                "ocr_used": ocr_used,
                "normalization_level": (
                    self.config
                    .normalization_level.value
                ),
                "normalization_changed": (
                    metrics.normalization_changed
                ),
            },
        }

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metrics.input_characters = len(
            parsed.text
        )

        metrics.output_characters = len(
            normalized_text
        )

        metrics.word_count = len(
            normalized_text.split()
        )

        metrics.page_count = len(
            pages
        )

        metrics.total_ms = (
            self._elapsed_ms(
                started_at
            )
        )

        # --------------------------------------------------------------
        # Provenance
        # --------------------------------------------------------------

        provenance = IngestionProvenance(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            format=parsed.format,
            mime_type=(
                parsed.mime_type
                or mime_type
            ),
            source=source,
        )

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        status = self._calculate_status(
            errors=errors,
            warnings=warnings,
            text=normalized_text,
        )

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            status=status,
            format=parsed.format,
            mime_type=(
                parsed.mime_type
                or mime_type
            ),
            text=normalized_text,
            pages=(
                pages
                if self.config.preserve_pages
                else []
            ),
            metadata=final_metadata,
            warnings=warnings,
            errors=errors,
            provenance=provenance,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # File API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str | Path,
        *,
        document_id: Optional[str] = None,
        source: str = "file",
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> IngestionResult:
        """
        Ingest a document from disk.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

        content = path.read_bytes()

        mime_type, _ = (
            mimetypes.guess_type(
                path.name
            )
        )

        return self.ingest(
            content,
            filename=path.name,
            document_id=document_id,
            mime_type=mime_type,
            source=source,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        content: bytes,
        filename: str,
    ) -> None:
        if not filename:
            raise ValueError(
                "Filename is required."
            )

        if not isinstance(
            content,
            bytes,
        ):
            raise TypeError(
                "Document content must be bytes."
            )

        if not content:
            raise ValueError(
                "Document content is empty."
            )

        if len(content) > (
            self.config.max_file_size_bytes
        ):
            raise ValueError(
                (
                    "Document exceeds maximum "
                    f"size of "
                    f"{self.config.max_file_size_bytes} bytes."
                )
            )

    # ------------------------------------------------------------------
    # OCR decision
    # ------------------------------------------------------------------

    def _should_use_ocr(
        self,
        parsed: ParsedDocument,
    ) -> bool:
        """
        OCR is considered when extracted text is very small.

        This is intentionally conservative so normal text PDFs do not
        unnecessarily enter the OCR path.
        """

        text_length = len(
            parsed.text.strip()
        )

        if text_length == 0:
            return True

        if (
            parsed.format
            == DocumentFormat.PDF
            and text_length
            < self.config.ocr_min_text_characters
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # OCR merge
    # ------------------------------------------------------------------

    def _merge_ocr_result(
        self,
        parsed: ParsedDocument,
        ocr_result: Any,
    ) -> ParsedDocument:
        """
        Merge OCR output without discarding original parser metadata.
        """

        if isinstance(
            ocr_result,
            str,
        ):
            ocr_text = ocr_result
            ocr_pages = []

        elif isinstance(
            ocr_result,
            dict,
        ):
            ocr_text = str(
                ocr_result.get(
                    "text",
                    "",
                )
            )

            ocr_pages = (
                ocr_result.get(
                    "pages",
                    [],
                )
            )

        else:
            raise TypeError(
                "Unsupported OCR result type."
            )

        if not ocr_text.strip():
            raise ValueError(
                "OCR returned no text."
            )

        parsed.text = ocr_text

        if ocr_pages:
            parsed.pages = (
                self._coerce_pages(
                    ocr_pages
                )
            )

        parsed.metadata[
            "ocr_used"
        ] = True

        return parsed

    @staticmethod
    def _coerce_pages(
        raw_pages: Any,
    ) -> list[DocumentPage]:
        pages: list[
            DocumentPage
        ] = []

        if not isinstance(
            raw_pages,
            list,
        ):
            return pages

        for index, page in enumerate(
            raw_pages,
            start=1,
        ):
            if isinstance(
                page,
                DocumentPage,
            ):
                pages.append(page)

            elif isinstance(
                page,
                str,
            ):
                pages.append(
                    DocumentPage(
                        page_number=index,
                        text=page,
                    )
                )

            elif isinstance(
                page,
                dict,
            ):
                pages.append(
                    DocumentPage(
                        page_number=int(
                            page.get(
                                "page_number",
                                index,
                            )
                        ),
                        text=str(
                            page.get(
                                "text",
                                "",
                            )
                        ),
                        metadata=dict(
                            page.get(
                                "metadata",
                                {},
                            )
                        ),
                    )
                )

        return pages

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(
        self,
        text: str,
    ) -> NormalizationResult:
        """
        Use the existing DocumentNormalizer.

        The normalizer should remain conservative for legal text.
        """

        if hasattr(
            self.normalizer,
            "normalize",
        ):
            return self.normalizer.normalize(
                text
            )

        if hasattr(
            self.normalizer,
            "normalize_text",
        ):
            return (
                self.normalizer
                .normalize_text(
                    text
                )
            )

        raise AttributeError(
            "DocumentNormalizer does not expose "
            "normalize() or normalize_text()."
        )

    # ------------------------------------------------------------------
    # Page normalization
    # ------------------------------------------------------------------

    def _normalize_pages(
        self,
        pages: list[DocumentPage],
    ) -> list[DocumentPage]:
        if not pages:
            return []

        normalized_pages: list[
            DocumentPage
        ] = []

        for page in pages:
            text = page.text or ""

            try:
                result = (
                    self.normalizer.normalize(
                        text
                    )
                )

                text = result.text

            except Exception:
                # Keep original page text if page-level
                # normalization fails.
                pass

            normalized_pages.append(
                DocumentPage(
                    page_number=page.page_number,
                    text=text,
                    metadata=dict(
                        page.metadata
                    ),
                )
            )

        return normalized_pages

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(
        self,
        parsed: ParsedDocument,
    ) -> dict[str, Any]:
        if self.metadata_extractor:
            result = (
                self.metadata_extractor(
                    parsed
                )
            )

            if result is None:
                return {}

            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "Metadata extractor must return dict."
                )

            return result

        return {
            "filename": parsed.filename,
            "format": parsed.format.value,
            "mime_type": parsed.mime_type,
            "page_count": len(
                parsed.pages
            ),
            "character_count": len(
                parsed.text
            ),
            "word_count": len(
                parsed.text.split()
            ),
        }

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def _detect_language(
        self,
        text: str,
    ) -> dict[str, Any]:
        if not text.strip():
            return {
                "language": None,
                "language_confidence": 0.0,
            }

        if self.language_detector:
            result = (
                self.language_detector(
                    text
                )
            )

            if isinstance(
                result,
                str,
            ):
                return {
                    "language": result,
                    "language_confidence": None,
                }

            if isinstance(
                result,
                dict,
            ):
                return result

            raise TypeError(
                "Language detector must return str or dict."
            )

        # Lightweight fallback.
        # The real application can inject fastText,
        # lingua, langdetect, or another validated detector.
        return {
            "language": "unknown",
            "language_confidence": 0.0,
            "language_detection_method": (
                "fallback"
            ),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _calculate_status(
        self,
        *,
        errors: list[IngestionError],
        warnings: list[str],
        text: str,
    ) -> IngestionStatus:
        if not text.strip():
            return IngestionStatus.FAILED

        fatal_errors = [
            error
            for error in errors
            if not error.recoverable
        ]

        if fatal_errors:
            return IngestionStatus.FAILED

        if errors or warnings:
            return IngestionStatus.PARTIAL

        return IngestionStatus.SUCCESS

    # ------------------------------------------------------------------
    # Identity / hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(
        content: bytes,
    ) -> str:
        return hashlib.sha256(
            content
        ).hexdigest()

    @classmethod
    def _generate_document_id(
        cls,
        content: bytes,
        filename: str,
    ) -> str:
        digest = hashlib.sha256()

        digest.update(content)

        digest.update(
            filename.encode(
                "utf-8",
                errors="replace",
            )
        )

        return digest.hexdigest()[:32]

    @classmethod
    def _fallback_document_id(
        cls,
        content: bytes,
        filename: str,
    ) -> str:
        return cls._generate_document_id(
            content,
            filename,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )


# ======================================================================
# Default pipeline
# ======================================================================


_default_pipeline = IngestionPipeline()


def ingest_document(
    content: bytes,
    *,
    filename: str,
    document_id: Optional[str] = None,
    mime_type: Optional[str] = None,
    source: str = "upload",
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> IngestionResult:
    """
    Convenience API for document ingestion.
    """

    return _default_pipeline.ingest(
        content,
        filename=filename,
        document_id=document_id,
        mime_type=mime_type,
        source=source,
        metadata=metadata,
    )


__all__ = [
    "IngestionStatus",
    "IngestionStage",
    "IngestionErrorCode",
    "IngestionError",
    "IngestionMetrics",
    "IngestionProvenance",
    "IngestionResult",
    "IngestionConfig",
    "IngestionPipeline",
    "ingest_document",
]