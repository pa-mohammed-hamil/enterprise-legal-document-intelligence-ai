# backend/ingestion/pdf_processor.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
import hashlib
import re


class PDFProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_TEXT = "no_text"
    ENCRYPTED = "encrypted"
    FAILED = "failed"


class PDFPageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass(slots=True)
class PDFPage:
    page_number: int
    text: str = ""

    width: Optional[float] = None
    height: Optional[float] = None
    rotation: int = 0

    has_text: bool = False
    has_images: bool = False
    image_count: int = 0

    page_type: PDFPageType = PDFPageType.EMPTY

    character_count: int = 0
    word_count: int = 0

    needs_ocr: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = self.text or ""

        stripped = self.text.strip()

        self.character_count = len(stripped)
        self.word_count = len(stripped.split())
        self.has_text = bool(stripped)

        if self.has_text and self.has_images:
            self.page_type = PDFPageType.MIXED
        elif self.has_text:
            self.page_type = PDFPageType.TEXT
        elif self.has_images:
            self.page_type = PDFPageType.IMAGE
        else:
            self.page_type = PDFPageType.EMPTY

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "image_count": self.image_count,
            "page_type": self.page_type.value,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "needs_ocr": self.needs_ocr,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class PDFMetadata:
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None

    creation_date: Optional[str] = None
    modification_date: Optional[str] = None

    keywords: Optional[str] = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "keywords": self.keywords,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PDFProcessingResult:
    status: PDFProcessingStatus

    text: str = ""
    pages: list[PDFPage] = field(default_factory=list)

    filename: Optional[str] = None
    mime_type: str = "application/pdf"

    page_count: int = 0
    processed_pages: int = 0
    failed_pages: int = 0

    text_page_count: int = 0
    scanned_page_count: int = 0

    is_encrypted: bool = False
    is_scanned: bool = False
    is_mixed: bool = False

    sha256: Optional[str] = None

    metadata: PDFMetadata = field(
        default_factory=PDFMetadata
    )

    warnings: list[str] = field(default_factory=list)

    processing_metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "text": self.text,
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "filename": self.filename,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "processed_pages": self.processed_pages,
            "failed_pages": self.failed_pages,
            "text_page_count": self.text_page_count,
            "scanned_page_count": self.scanned_page_count,
            "is_encrypted": self.is_encrypted,
            "is_scanned": self.is_scanned,
            "is_mixed": self.is_mixed,
            "sha256": self.sha256,
            "metadata": self.metadata.to_dict(),
            "warnings": self.warnings,
            "processing_metadata": self.processing_metadata,
        }


@dataclass(slots=True)
class PDFProcessorConfig:
    preserve_page_boundaries: bool = True

    normalize_whitespace: bool = True

    extract_metadata: bool = True
    detect_images: bool = True
    detect_scanned_pages: bool = True

    # Pages below this native-text threshold become OCR candidates.
    ocr_text_threshold: int = 20

    max_pages: Optional[int] = None

    max_document_bytes: int = 100 * 1024 * 1024

    calculate_sha256: bool = True

    strict_pdf_header: bool = True

    extract_links: bool = True

    extract_annotations: bool = False


class PDFProcessingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "PDF_PROCESSING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class PDFProcessor:
    """
    Native PDF extraction layer.

    Responsibilities
    ----------------
    - validate PDFs
    - extract page-level text
    - preserve physical page numbers
    - inspect page dimensions and rotation
    - detect image/scanned pages
    - identify pages requiring OCR
    - extract technical PDF metadata
    - preserve citation provenance

    This class does NOT perform OCR itself.

    OCR should be delegated to OCRProcessor so native extraction and
    OCR remain independently testable.
    """

    MIME_TYPE = "application/pdf"

    def __init__(
        self,
        config: Optional[PDFProcessorConfig] = None,
    ) -> None:
        self.config = config or PDFProcessorConfig()

    # ==============================================================
    # Public API
    # ==============================================================

    def process(
        self,
        source: str | Path | bytes | bytearray,
        *,
        filename: Optional[str] = None,
    ) -> PDFProcessingResult:

        content = self._load_content(source)

        self._validate(content)

        resolved_filename = self._resolve_filename(
            source,
            filename,
        )

        sha256 = (
            hashlib.sha256(content).hexdigest()
            if self.config.calculate_sha256
            else None
        )

        try:
            import fitz
        except ImportError as exc:
            raise PDFProcessingError(
                "PyMuPDF is required for PDF processing.",
                code="PYMUPDF_NOT_INSTALLED",
                cause=exc,
            ) from exc

        document = None

        try:
            document = fitz.open(
                stream=content,
                filetype="pdf",
            )

            if document.needs_pass:
                return PDFProcessingResult(
                    status=PDFProcessingStatus.ENCRYPTED,
                    filename=resolved_filename,
                    page_count=document.page_count,
                    is_encrypted=True,
                    sha256=sha256,
                    warnings=[
                        "PDF is encrypted and requires a password."
                    ],
                )

            metadata = (
                self._extract_metadata(document)
                if self.config.extract_metadata
                else PDFMetadata()
            )

            total_pages = document.page_count

            limit = total_pages

            if self.config.max_pages is not None:
                limit = min(
                    total_pages,
                    self.config.max_pages,
                )

            pages: list[PDFPage] = []
            warnings: list[str] = []

            failed_pages = 0

            for index in range(limit):
                page_number = index + 1

                try:
                    page = document.load_page(index)

                    extracted = self._extract_page(
                        page,
                        page_number=page_number,
                    )

                    pages.append(extracted)

                except Exception as exc:
                    failed_pages += 1

                    warnings.append(
                        f"Page {page_number} extraction failed: {exc}"
                    )

            return self._build_result(
                pages=pages,
                filename=resolved_filename,
                total_pages=total_pages,
                failed_pages=failed_pages,
                sha256=sha256,
                metadata=metadata,
                warnings=warnings,
            )

        except PDFProcessingError:
            raise

        except Exception as exc:
            raise PDFProcessingError(
                "Unable to process PDF.",
                code="PDF_EXTRACTION_ERROR",
                cause=exc,
            ) from exc

        finally:
            if document is not None:
                try:
                    document.close()
                except Exception:
                    pass

    def process_bytes(
        self,
        content: bytes,
        *,
        filename: Optional[str] = None,
    ) -> PDFProcessingResult:
        return self.process(
            content,
            filename=filename,
        )

    def process_file(
        self,
        path: str | Path,
    ) -> PDFProcessingResult:
        return self.process(path)

    # ==============================================================
    # Page extraction
    # ==============================================================

    def _extract_page(
        self,
        page: Any,
        *,
        page_number: int,
    ) -> PDFPage:

        raw_text = page.get_text("text") or ""

        text = self._normalize_text(raw_text)

        image_count = 0

        if self.config.detect_images:
            try:
                image_count = len(
                    page.get_images(full=True)
                )
            except Exception:
                image_count = 0

        rect = page.rect

        try:
            rotation = int(page.rotation)
        except Exception:
            rotation = 0

        page_metadata: dict[str, Any] = {
            "pdf_page_index": page_number - 1,
        }

        if self.config.extract_links:
            page_metadata["links"] = (
                self._extract_links(page)
            )

        if self.config.extract_annotations:
            page_metadata["annotations"] = (
                self._extract_annotations(page)
            )

        native_text_length = len(
            text.strip()
        )

        needs_ocr = (
            self.config.detect_scanned_pages
            and native_text_length
            < self.config.ocr_text_threshold
        )

        result = PDFPage(
            page_number=page_number,
            text=text,
            width=float(rect.width),
            height=float(rect.height),
            rotation=rotation,
            has_images=image_count > 0,
            image_count=image_count,
            needs_ocr=needs_ocr,
            metadata=page_metadata,
        )

        # __post_init__ determines page type and native text stats.
        # needs_ocr remains an independent signal.
        return result

    # ==============================================================
    # Metadata
    # ==============================================================

    def _extract_metadata(
        self,
        document: Any,
    ) -> PDFMetadata:

        raw = document.metadata or {}

        normalized_raw = {
            str(key): self._safe_value(value)
            for key, value in raw.items()
        }

        return PDFMetadata(
            title=self._clean_metadata(
                raw.get("title")
            ),
            author=self._clean_metadata(
                raw.get("author")
            ),
            subject=self._clean_metadata(
                raw.get("subject")
            ),
            creator=self._clean_metadata(
                raw.get("creator")
            ),
            producer=self._clean_metadata(
                raw.get("producer")
            ),
            creation_date=self._clean_metadata(
                raw.get("creationDate")
            ),
            modification_date=self._clean_metadata(
                raw.get("modDate")
            ),
            keywords=self._clean_metadata(
                raw.get("keywords")
            ),
            raw=normalized_raw,
        )

    # ==============================================================
    # Links
    # ==============================================================

    @staticmethod
    def _extract_links(
        page: Any,
    ) -> list[dict[str, Any]]:

        output: list[dict[str, Any]] = []

        try:
            links = page.get_links()
        except Exception:
            return output

        for link in links:
            entry: dict[str, Any] = {}

            if "uri" in link:
                entry["uri"] = link.get("uri")

            if "page" in link:
                entry["target_page"] = link.get("page")

            rect = link.get("from")

            if rect is not None:
                try:
                    entry["bbox"] = [
                        float(rect.x0),
                        float(rect.y0),
                        float(rect.x1),
                        float(rect.y1),
                    ]
                except Exception:
                    pass

            if entry:
                output.append(entry)

        return output

    # ==============================================================
    # Annotations
    # ==============================================================

    @staticmethod
    def _extract_annotations(
        page: Any,
    ) -> list[dict[str, Any]]:

        output: list[dict[str, Any]] = []

        try:
            annotations = page.annots()

            if annotations is None:
                return output

            for annotation in annotations:
                item: dict[str, Any] = {}

                try:
                    item["type"] = (
                        annotation.type[1]
                        if annotation.type
                        else None
                    )
                except Exception:
                    pass

                try:
                    info = annotation.info or {}

                    item["content"] = info.get(
                        "content"
                    )

                    item["title"] = info.get(
                        "title"
                    )

                    item["subject"] = info.get(
                        "subject"
                    )
                except Exception:
                    pass

                try:
                    rect = annotation.rect

                    item["bbox"] = [
                        float(rect.x0),
                        float(rect.y0),
                        float(rect.x1),
                        float(rect.y1),
                    ]
                except Exception:
                    pass

                output.append(item)

        except Exception:
            pass

        return output

    # ==============================================================
    # Result construction
    # ==============================================================

    def _build_result(
        self,
        *,
        pages: list[PDFPage],
        filename: Optional[str],
        total_pages: int,
        failed_pages: int,
        sha256: Optional[str],
        metadata: PDFMetadata,
        warnings: list[str],
    ) -> PDFProcessingResult:

        pages.sort(
            key=lambda item: item.page_number
        )

        text_pages = [
            page
            for page in pages
            if page.has_text
        ]

        scanned_pages = [
            page
            for page in pages
            if page.needs_ocr
        ]

        native_pages = [
            page
            for page in pages
            if not page.needs_ocr
        ]

        text = self._join_pages(
            pages
        )

        if failed_pages and pages:
            status = PDFProcessingStatus.PARTIAL

        elif failed_pages and not pages:
            status = PDFProcessingStatus.FAILED

        elif not text.strip():
            status = PDFProcessingStatus.NO_TEXT

        else:
            status = PDFProcessingStatus.SUCCESS

        processed_count = len(pages)

        scanned_count = len(scanned_pages)

        is_scanned = (
            processed_count > 0
            and scanned_count == processed_count
        )

        is_mixed = bool(
            scanned_pages
            and native_pages
        )

        if (
            self.config.max_pages is not None
            and total_pages > self.config.max_pages
        ):
            warnings.append(
                "PDF page processing was limited by max_pages."
            )

        return PDFProcessingResult(
            status=status,
            text=text,
            pages=pages,
            filename=filename,
            page_count=total_pages,
            processed_pages=processed_count,
            failed_pages=failed_pages,
            text_page_count=len(text_pages),
            scanned_page_count=scanned_count,
            is_encrypted=False,
            is_scanned=is_scanned,
            is_mixed=is_mixed,
            sha256=sha256,
            metadata=metadata,
            warnings=warnings,
            processing_metadata={
                "native_text_characters": sum(
                    page.character_count
                    for page in pages
                ),
                "native_text_words": sum(
                    page.word_count
                    for page in pages
                ),
                "ocr_candidate_pages": [
                    page.page_number
                    for page in scanned_pages
                ],
                "preserve_page_boundaries": (
                    self.config.preserve_page_boundaries
                ),
            },
        )

    def _join_pages(
        self,
        pages: list[PDFPage],
    ) -> str:

        output: list[str] = []

        for page in pages:
            text = page.text.strip()

            if not text:
                continue

            if self.config.preserve_page_boundaries:
                output.append(
                    f"[PAGE {page.page_number}]\n{text}"
                )
            else:
                output.append(text)

        return "\n\n".join(output)

    # ==============================================================
    # OCR integration helpers
    # ==============================================================

    def get_ocr_candidates(
        self,
        result: PDFProcessingResult,
    ) -> list[int]:
        """
        Return physical page numbers that should be passed to the
        OCR layer.
        """

        return [
            page.page_number
            for page in result.pages
            if page.needs_ocr
        ]

    def merge_ocr_text(
        self,
        result: PDFProcessingResult,
        ocr_pages: dict[int, str],
    ) -> PDFProcessingResult:
        """
        Merge page-level OCR output back into the native PDF result.

        ocr_pages:
            {
                1: "OCR text...",
                4: "OCR text..."
            }

        This preserves the original physical page numbers.
        """

        for page in result.pages:
            ocr_text = ocr_pages.get(
                page.page_number
            )

            if not ocr_text:
                continue

            cleaned = self._normalize_text(
                ocr_text
            )

            if not cleaned:
                continue

            # Replace only when native text is insufficient.
            if (
                page.needs_ocr
                or not page.text.strip()
            ):
                page.text = cleaned
                page.has_text = True
                page.character_count = len(cleaned)
                page.word_count = len(
                    cleaned.split()
                )

                page.needs_ocr = False

                page.metadata["ocr_used"] = True

                if page.has_images:
                    page.page_type = PDFPageType.MIXED
                else:
                    page.page_type = PDFPageType.TEXT

        result.text = self._join_pages(
            result.pages
        )

        result.text_page_count = sum(
            1
            for page in result.pages
            if page.text.strip()
        )

        remaining_ocr = [
            page
            for page in result.pages
            if page.needs_ocr
        ]

        result.scanned_page_count = len(
            remaining_ocr
        )

        result.is_scanned = (
            bool(result.pages)
            and len(remaining_ocr)
            == len(result.pages)
        )

        result.is_mixed = (
            bool(remaining_ocr)
            and len(remaining_ocr)
            < len(result.pages)
        )

        if result.text.strip():
            result.status = PDFProcessingStatus.SUCCESS

        return result

    # ==============================================================
    # Validation
    # ==============================================================

    def _validate(
        self,
        content: bytes,
    ) -> None:

        if not content:
            raise PDFProcessingError(
                "PDF content is empty.",
                code="EMPTY_DOCUMENT",
            )

        if (
            len(content)
            > self.config.max_document_bytes
        ):
            raise PDFProcessingError(
                "PDF exceeds configured size limit.",
                code="DOCUMENT_TOO_LARGE",
            )

        if (
            self.config.strict_pdf_header
            and not content.lstrip().startswith(b"%PDF-")
        ):
            raise PDFProcessingError(
                "Content does not contain a valid PDF header.",
                code="INVALID_PDF",
            )

    # ==============================================================
    # Source helpers
    # ==============================================================

    @staticmethod
    def _load_content(
        source: str | Path | bytes | bytearray,
    ) -> bytes:

        if isinstance(
            source,
            (bytes, bytearray),
        ):
            return bytes(source)

        path = Path(source)

        if not path.exists():
            raise PDFProcessingError(
                f"PDF does not exist: {path}",
                code="FILE_NOT_FOUND",
            )

        if not path.is_file():
            raise PDFProcessingError(
                f"Source is not a file: {path}",
                code="INVALID_SOURCE",
            )

        try:
            return path.read_bytes()

        except OSError as exc:
            raise PDFProcessingError(
                f"Unable to read PDF: {path}",
                code="FILE_READ_ERROR",
                cause=exc,
            ) from exc

    @staticmethod
    def _resolve_filename(
        source: str | Path | bytes | bytearray,
        filename: Optional[str],
    ) -> Optional[str]:

        if filename:
            return Path(filename).name

        if isinstance(
            source,
            (str, Path),
        ):
            return Path(source).name

        return None

    # ==============================================================
    # Text helpers
    # ==============================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        # Remove null characters introduced by malformed extraction.
        text = text.replace(
            "\x00",
            "",
        )

        if not self.config.normalize_whitespace:
            return text.strip()

        # Normalize horizontal whitespace without destroying legal
        # paragraph/section boundaries.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"[ \t]+\n",
            "\n",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{4,}",
            "\n\n\n",
            text,
        )

        return text.strip()

    # ==============================================================
    # Metadata helpers
    # ==============================================================

    @staticmethod
    def _clean_metadata(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        text = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
            "",
            text,
        )

        return text[:10_000] or None

    @staticmethod
    def _safe_value(
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        try:
            return str(value)
        except Exception:
            return repr(value)


def process_pdf(
    source: str | Path | bytes | bytearray,
    *,
    filename: Optional[str] = None,
) -> PDFProcessingResult:
    """
    Convenience entry point used by DocumentParser/IngestionPipeline.
    """

    processor = PDFProcessor()

    return processor.process(
        source,
        filename=filename,
    )