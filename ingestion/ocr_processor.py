# backend/ingestion/ocr_processor.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, Optional, Sequence
import re


class OCRStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_TEXT = "no_text"
    FAILED = "failed"
    SKIPPED = "skipped"


class OCRProvider(str, Enum):
    AUTO = "auto"
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    INJECTED = "injected"


class OCRDocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class OCRPageResult:
    page_number: int
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    word_count: int = 0
    character_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = self.text or ""
        self.word_count = len(self.text.split())
        self.character_count = len(self.text)


@dataclass(slots=True)
class OCRResult:
    status: OCRStatus
    text: str
    pages: list[OCRPageResult] = field(default_factory=list)

    provider: str = OCRProvider.AUTO.value
    language: Optional[str] = None
    confidence: Optional[float] = None

    page_count: int = 0
    processed_pages: int = 0
    failed_pages: int = 0

    is_scanned: bool = False
    has_text: bool = False

    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "text": self.text,
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text,
                    "confidence": page.confidence,
                    "language": page.language,
                    "width": page.width,
                    "height": page.height,
                    "word_count": page.word_count,
                    "character_count": page.character_count,
                    "metadata": page.metadata,
                }
                for page in self.pages
            ],
            "provider": self.provider,
            "language": self.language,
            "confidence": self.confidence,
            "page_count": self.page_count,
            "processed_pages": self.processed_pages,
            "failed_pages": self.failed_pages,
            "is_scanned": self.is_scanned,
            "has_text": self.has_text,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class OCRConfig:
    provider: OCRProvider = OCRProvider.AUTO

    languages: tuple[str, ...] = ("eng",)

    dpi: int = 300

    confidence_threshold: float = 0.0

    max_pages: Optional[int] = None

    preprocess_image: bool = True

    detect_orientation: bool = True

    preserve_page_boundaries: bool = True

    min_text_length_for_page: int = 2

    # OCR is normally a fallback after native PDF/DOCX parsing.
    only_if_no_text: bool = True

    # Prevent accidentally processing enormous documents.
    max_document_bytes: int = 100 * 1024 * 1024


class OCRProcessingError(Exception):
    """Raised for fatal OCR processing failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "OCR_PROCESSING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class OCRProcessor:
    """
    Production-oriented OCR abstraction for legal-document ingestion.

    Responsibilities:
        - convert PDF/image pages into OCR text
        - preserve page boundaries
        - capture confidence and OCR metadata
        - support multiple OCR providers
        - provide an injectable provider for testing/cloud OCR
        - avoid modifying the legal source document

    Non-responsibilities:
        - document classification
        - clause classification
        - legal interpretation
        - risk detection
        - RAG
        - hallucination detection
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        *,
        provider: Any = None,
    ) -> None:
        self.config = config or OCRConfig()
        self.provider = provider

        self._tesseract = None
        self._paddle = None

    # ================================================================
    # Public API
    # ================================================================

    def process(
        self,
        content: bytes,
        *,
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> OCRResult:
        """
        OCR a PDF or image.

        For PDFs, OCR is performed page-by-page so page numbers can
        remain aligned with the document ingestion pipeline.
        """

        self._validate_content(content)

        extension = self._extension(filename)

        if extension == ".pdf":
            return self.process_pdf(
                content,
                language=language,
            )

        if extension in self.IMAGE_EXTENSIONS:
            return self.process_image(
                content,
                page_number=1,
                language=language,
            )

        return OCRResult(
            status=OCRStatus.SKIPPED,
            text="",
            provider=self.config.provider.value,
            warnings=[
                f"Unsupported OCR format: {extension or 'unknown'}"
            ],
            metadata={
                "filename": filename,
                "extension": extension,
            },
        )

    def process_pdf(
        self,
        content: bytes,
        *,
        language: Optional[str] = None,
    ) -> OCRResult:
        pages = self._render_pdf_pages(content)

        if not pages:
            return OCRResult(
                status=OCRStatus.NO_TEXT,
                text="",
                provider=self._provider_name(),
                is_scanned=True,
                has_text=False,
                warnings=["PDF contains no renderable pages."],
            )

        max_pages = self.config.max_pages

        if max_pages is not None:
            pages = pages[:max_pages]

        results: list[OCRPageResult] = []
        warnings: list[str] = []
        failed_pages = 0

        for page_number, image in pages:
            try:
                page_result = self._ocr_image(
                    image,
                    page_number=page_number,
                    language=language,
                )

                results.append(page_result)

            except Exception as exc:
                failed_pages += 1
                warnings.append(
                    f"OCR failed on page {page_number}: {exc}"
                )

        return self._build_result(
            results,
            total_pages=len(pages),
            failed_pages=failed_pages,
            warnings=warnings,
        )

    def process_image(
        self,
        content: bytes,
        *,
        page_number: int = 1,
        language: Optional[str] = None,
    ) -> OCRResult:
        try:
            image = self._load_image(content)

            page = self._ocr_image(
                image,
                page_number=page_number,
                language=language,
            )

            return self._build_result(
                [page],
                total_pages=1,
                failed_pages=0,
                warnings=[],
            )

        except Exception as exc:
            return OCRResult(
                status=OCRStatus.FAILED,
                text="",
                provider=self._provider_name(),
                page_count=1,
                processed_pages=0,
                failed_pages=1,
                is_scanned=True,
                has_text=False,
                warnings=[f"Image OCR failed: {exc}"],
            )

    def should_ocr(
        self,
        *,
        extracted_text: Optional[str],
        is_scanned: Optional[bool] = None,
    ) -> bool:
        """
        Decide whether OCR should be invoked after native parsing.
        """

        if not self.config.only_if_no_text:
            return True

        if is_scanned is True:
            return True

        text = self._clean_text(extracted_text or "")

        return len(text) < self.config.min_text_length_for_page

    # ================================================================
    # Provider dispatch
    # ================================================================

    def _ocr_image(
        self,
        image: Any,
        *,
        page_number: int,
        language: Optional[str],
    ) -> OCRPageResult:
        provider = self.config.provider

        if provider == OCRProvider.INJECTED:
            return self._ocr_injected(
                image,
                page_number=page_number,
                language=language,
            )

        if provider == OCRProvider.PADDLEOCR:
            return self._ocr_paddle(
                image,
                page_number=page_number,
                language=language,
            )

        if provider == OCRProvider.TESSERACT:
            return self._ocr_tesseract(
                image,
                page_number=page_number,
                language=language,
            )

        # AUTO:
        # Prefer an injected provider, then PaddleOCR, then Tesseract.
        if self.provider is not None:
            return self._ocr_injected(
                image,
                page_number=page_number,
                language=language,
            )

        try:
            return self._ocr_paddle(
                image,
                page_number=page_number,
                language=language,
            )
        except Exception:
            return self._ocr_tesseract(
                image,
                page_number=page_number,
                language=language,
            )

    # ================================================================
    # Injected provider
    # ================================================================

    def _ocr_injected(
        self,
        image: Any,
        *,
        page_number: int,
        language: Optional[str],
    ) -> OCRPageResult:
        if self.provider is None:
            raise OCRProcessingError(
                "No injected OCR provider configured.",
                code="OCR_PROVIDER_NOT_CONFIGURED",
            )

        if callable(self.provider):
            response = self.provider(
                image,
                language=language,
                page_number=page_number,
            )
        elif hasattr(self.provider, "ocr"):
            response = self.provider.ocr(
                image,
                language=language,
                page_number=page_number,
            )
        elif hasattr(self.provider, "process"):
            response = self.provider.process(
                image,
                language=language,
                page_number=page_number,
            )
        else:
            raise OCRProcessingError(
                "Injected OCR provider must be callable or expose "
                "`ocr()`/`process()`.",
                code="INVALID_OCR_PROVIDER",
            )

        return self._normalize_provider_response(
            response,
            page_number=page_number,
            language=language,
        )

    def _normalize_provider_response(
        self,
        response: Any,
        *,
        page_number: int,
        language: Optional[str],
    ) -> OCRPageResult:
        if isinstance(response, OCRPageResult):
            return response

        if isinstance(response, str):
            text = self._clean_text(response)

            return OCRPageResult(
                page_number=page_number,
                text=text,
                language=language,
            )

        if isinstance(response, dict):
            text = self._clean_text(
                response.get("text", "")
            )

            confidence = self._safe_float(
                response.get("confidence")
            )

            return OCRPageResult(
                page_number=page_number,
                text=text,
                confidence=confidence,
                language=response.get("language") or language,
                width=response.get("width"),
                height=response.get("height"),
                metadata=response.get("metadata", {}),
            )

        raise OCRProcessingError(
            "Unsupported OCR provider response type.",
            code="INVALID_OCR_RESPONSE",
        )

    # ================================================================
    # Tesseract
    # ================================================================

    def _ocr_tesseract(
        self,
        image: Any,
        *,
        page_number: int,
        language: Optional[str],
    ) -> OCRPageResult:
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRProcessingError(
                "pytesseract is not installed.",
                code="TESSERACT_NOT_INSTALLED",
                cause=exc,
            ) from exc

        processed_image = self._preprocess(image)

        lang = language or "+".join(
            self.config.languages
        )

        text = pytesseract.image_to_string(
            processed_image,
            lang=lang,
            config=self._tesseract_config(),
        )

        confidence = self._tesseract_confidence(
            pytesseract,
            processed_image,
            lang,
        )

        width, height = self._image_dimensions(
            processed_image
        )

        return OCRPageResult(
            page_number=page_number,
            text=self._clean_text(text),
            confidence=confidence,
            language=language or self._primary_language(),
            width=width,
            height=height,
            metadata={
                "engine": "tesseract",
                "dpi": self.config.dpi,
            },
        )

    def _tesseract_confidence(
        self,
        pytesseract: Any,
        image: Any,
        language: str,
    ) -> Optional[float]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=self._tesseract_config(),
                output_type=pytesseract.Output.DICT,
            )

            values: list[float] = []

            for value, text in zip(
                data.get("conf", []),
                data.get("text", []),
            ):
                if not str(text).strip():
                    continue

                try:
                    score = float(value)

                    if score >= 0:
                        values.append(score)
                except (TypeError, ValueError):
                    continue

            if not values:
                return None

            return round(
                sum(values) / len(values) / 100.0,
                4,
            )

        except Exception:
            return None

    def _tesseract_config(self) -> str:
        # PSM 3 is suitable for general legal-document pages.
        return "--psm 3"

    # ================================================================
    # PaddleOCR
    # ================================================================

    def _ocr_paddle(
        self,
        image: Any,
        *,
        page_number: int,
        language: Optional[str],
    ) -> OCRPageResult:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRProcessingError(
                "PaddleOCR is not installed.",
                code="PADDLEOCR_NOT_INSTALLED",
                cause=exc,
            ) from exc

        if self._paddle is None:
            lang = self._paddle_language(language)

            self._paddle = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=self.config.detect_orientation,
                use_doc_unwarping=self.config.preprocess_image,
                use_textline_orientation=self.config.detect_orientation,
            )

        processed_image = self._preprocess(image)

        result = self._paddle.predict(
            processed_image
        )

        text_parts: list[str] = []
        confidences: list[float] = []

        for item in result:
            item_json = getattr(
                item,
                "json",
                None,
            )

            if callable(item_json):
                item_json = item_json()

            if not isinstance(item_json, dict):
                continue

            data = item_json.get(
                "res",
                item_json,
            )

            texts = (
                data.get("rec_texts", [])
                if isinstance(data, dict)
                else []
            )

            scores = (
                data.get("rec_scores", [])
                if isinstance(data, dict)
                else []
            )

            text_parts.extend(
                str(value)
                for value in texts
                if str(value).strip()
            )

            for score in scores:
                try:
                    confidences.append(float(score))
                except (TypeError, ValueError):
                    pass

        text = "\n".join(text_parts)

        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        width, height = self._image_dimensions(
            processed_image
        )

        return OCRPageResult(
            page_number=page_number,
            text=self._clean_text(text),
            confidence=confidence,
            language=language or self._primary_language(),
            width=width,
            height=height,
            metadata={
                "engine": "paddleocr",
                "dpi": self.config.dpi,
            },
        )

    @staticmethod
    def _paddle_language(
        language: Optional[str],
    ) -> str:
        mapping = {
            "eng": "en",
            "en": "en",
            "deu": "german",
            "de": "german",
            "fra": "fr",
            "fr": "fr",
        }

        return mapping.get(
            language or "",
            "en",
        )

    # ================================================================
    # PDF rendering
    # ================================================================

    def _render_pdf_pages(
        self,
        content: bytes,
    ) -> list[tuple[int, Any]]:
        try:
            import fitz
        except ImportError as exc:
            raise OCRProcessingError(
                "PyMuPDF is required for PDF OCR rendering.",
                code="PYMUPDF_NOT_INSTALLED",
                cause=exc,
            ) from exc

        try:
            document = fitz.open(
                stream=content,
                filetype="pdf",
            )

            if document.needs_pass:
                raise OCRProcessingError(
                    "Encrypted PDF requires a password.",
                    code="ENCRYPTED_PDF",
                )

            matrix = fitz.Matrix(
                self.config.dpi / 72.0,
                self.config.dpi / 72.0,
            )

            pages: list[tuple[int, Any]] = []

            page_limit = self.config.max_pages

            for index, page in enumerate(document):
                if (
                    page_limit is not None
                    and index >= page_limit
                ):
                    break

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image_bytes = pixmap.tobytes("png")

                image = self._load_image(
                    image_bytes
                )

                pages.append(
                    (
                        index + 1,
                        image,
                    )
                )

            document.close()

            return pages

        except OCRProcessingError:
            raise

        except Exception as exc:
            raise OCRProcessingError(
                "Failed to render PDF pages for OCR.",
                code="PDF_RENDER_ERROR",
                cause=exc,
            ) from exc

    # ================================================================
    # Image processing
    # ================================================================

    def _load_image(
        self,
        content: bytes,
    ) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise OCRProcessingError(
                "Pillow is required for image OCR.",
                code="PILLOW_NOT_INSTALLED",
                cause=exc,
            ) from exc

        try:
            image = Image.open(
                BytesIO(content)
            )

            # Materialize the image so the BytesIO lifetime does not
            # matter to downstream OCR providers.
            return image.convert("RGB")

        except Exception as exc:
            raise OCRProcessingError(
                "Unable to decode image.",
                code="INVALID_IMAGE",
                cause=exc,
            ) from exc

    def _preprocess(
        self,
        image: Any,
    ) -> Any:
        if not self.config.preprocess_image:
            return image

        try:
            from PIL import ImageOps

            processed = ImageOps.exif_transpose(
                image
            )

            # Keep preprocessing conservative. Legal documents can
            # contain signatures, tables, numbers, and fine print.
            processed = ImageOps.grayscale(
                processed
            )

            return processed

        except Exception:
            return image

    @staticmethod
    def _image_dimensions(
        image: Any,
    ) -> tuple[Optional[int], Optional[int]]:
        try:
            width, height = image.size
            return width, height
        except Exception:
            return None, None

    # ================================================================
    # Result construction
    # ================================================================

    def _build_result(
        self,
        pages: Sequence[OCRPageResult],
        *,
        total_pages: int,
        failed_pages: int,
        warnings: list[str],
    ) -> OCRResult:
        ordered_pages = sorted(
            pages,
            key=lambda page: page.page_number,
        )

        text_parts: list[str] = []

        for page in ordered_pages:
            page_text = page.text.strip()

            if not page_text:
                continue

            if self.config.preserve_page_boundaries:
                text_parts.append(
                    f"[PAGE {page.page_number}]\n"
                    f"{page_text}"
                )
            else:
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        confidences = [
            page.confidence
            for page in ordered_pages
            if page.confidence is not None
        ]

        overall_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        has_text = bool(
            self._clean_text(full_text)
        )

        if failed_pages == total_pages:
            status = OCRStatus.FAILED
        elif failed_pages > 0:
            status = OCRStatus.PARTIAL
        elif not has_text:
            status = OCRStatus.NO_TEXT
        else:
            status = OCRStatus.SUCCESS

        return OCRResult(
            status=status,
            text=full_text,
            pages=list(ordered_pages),
            provider=self._provider_name(),
            language=self._primary_language(),
            confidence=overall_confidence,
            page_count=total_pages,
            processed_pages=len(ordered_pages),
            failed_pages=failed_pages,
            is_scanned=True,
            has_text=has_text,
            warnings=warnings,
            metadata={
                "ocr_engine": self._provider_name(),
                "dpi": self.config.dpi,
                "preserve_page_boundaries": (
                    self.config.preserve_page_boundaries
                ),
            },
        )

    # ================================================================
    # Utility
    # ================================================================

    def _provider_name(self) -> str:
        if self.config.provider != OCRProvider.AUTO:
            return self.config.provider.value

        if self.provider is not None:
            return OCRProvider.INJECTED.value

        return OCRProvider.AUTO.value

    def _primary_language(self) -> Optional[str]:
        if not self.config.languages:
            return None

        return self.config.languages[0]

    @staticmethod
    def _extension(
        filename: Optional[str],
    ) -> str:
        if not filename:
            return ""

        import os

        return os.path.splitext(
            filename
        )[1].lower()

    def _validate_content(
        self,
        content: bytes,
    ) -> None:
        if not isinstance(content, bytes):
            raise OCRProcessingError(
                "OCR content must be bytes.",
                code="INVALID_CONTENT",
            )

        if not content:
            raise OCRProcessingError(
                "OCR content is empty.",
                code="EMPTY_CONTENT",
            )

        if (
            len(content)
            > self.config.max_document_bytes
        ):
            raise OCRProcessingError(
                "Document exceeds OCR size limit.",
                code="DOCUMENT_TOO_LARGE",
            )

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace(
            "\x00",
            "",
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

        return text.strip()

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def process_ocr(
    content: bytes,
    *,
    filename: Optional[str] = None,
    language: Optional[str] = None,
) -> OCRResult:
    """
    Convenience function for the ingestion pipeline.
    """

    processor = OCRProcessor()

    return processor.process(
        content,
        filename=filename,
        language=language,
    )