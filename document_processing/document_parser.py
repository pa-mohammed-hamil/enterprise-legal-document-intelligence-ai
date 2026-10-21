"""
Document Parser

Multi-format document parsing with OCR support for legal documents.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import PyPDF2
import pytesseract
from docx import Document as DocxDocument
from striprtf.striprtf import rtf_to_text
import pdfplumber

from backend.core.exceptions import DocumentProcessingError
from backend.document_processing.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Multi-format document parser with OCR support.
    
    Supported formats:
    - PDF (text and scanned)
    - DOCX, DOC
    - TXT
    - RTF
    - ODT
    - Images (PNG, JPG, TIFF)
    """

    SUPPORTED_FORMATS = {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".rtf",
        ".odt",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }

    def __init__(self, enable_ocr: bool = True, ocr_languages: list[str] = None):
        """
        Initialize document parser.

        Args:
            enable_ocr: Whether to enable OCR for scanned documents
            ocr_languages: List of languages for OCR (e.g., ['eng', 'fra'])
        """
        self.enable_ocr = enable_ocr
        self.ocr_languages = ocr_languages or ["eng"]
        
        if enable_ocr:
            self.ocr_engine = OCREngine(languages=self.ocr_languages)

    def parse(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
    ) -> dict:
        """
        Parse a document and extract text content.

        Args:
            file_path: Path to the document file
            file_content: Optional file content as bytes

        Returns:
            Dictionary with extracted text and metadata:
            {
                "text": str,
                "pages": int,
                "format": str,
                "is_scanned": bool,
                "metadata": dict,
            }

        Raises:
            DocumentProcessingError: If parsing fails
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise DocumentProcessingError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        try:
            # Parse based on file type
            if extension == ".pdf":
                return self._parse_pdf(file_path, file_content)
            elif extension in [".docx", ".doc"]:
                return self._parse_docx(file_path, file_content)
            elif extension == ".txt":
                return self._parse_txt(file_path, file_content)
            elif extension == ".rtf":
                return self._parse_rtf(file_path, file_content)
            elif extension == ".odt":
                return self._parse_odt(file_path, file_content)
            elif extension in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]:
                return self._parse_image(file_path, file_content)
            else:
                raise DocumentProcessingError(f"Parser not implemented for {extension}")

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to parse document: {str(e)}") from e

    def _parse_pdf(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse PDF document with OCR fallback for scanned PDFs."""
        # Try text extraction first
        text_content = self._extract_pdf_text(file_path, file_content)
        
        metadata = {
            "format": "pdf",
            "is_scanned": False,
            "pages": 0,
        }

        # If extracted text is minimal, try OCR
        if len(text_content.strip()) < 100 and self.enable_ocr:
            logger.info(f"PDF appears to be scanned, attempting OCR: {file_path}")
            text_content = self._extract_pdf_with_ocr(file_path, file_content)
            metadata["is_scanned"] = True

        # Get page count
        try:
            if file_content:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            else:
                pdf_reader = PyPDF2.PdfReader(str(file_path))
            metadata["pages"] = len(pdf_reader.pages)
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")

        return {
            "text": text_content,
            "metadata": metadata,
        }

    def _extract_pdf_text(self, file_path: Path, file_content: Optional[bytes]) -> str:
        """Extract text from PDF using pdfplumber."""
        text_parts = []

        try:
            if file_content:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
            else:
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}, trying PyPDF2")
            return self._extract_pdf_text_pypdf2(file_path, file_content)

    def _extract_pdf_text_pypdf2(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> str:
        """Fallback PDF text extraction using PyPDF2."""
        text_parts = []

        try:
            if file_content:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            else:
                pdf_reader = PyPDF2.PdfReader(str(file_path))

            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""

    def _extract_pdf_with_ocr(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> str:
        """Extract text from scanned PDF using OCR."""
        if not self.enable_ocr:
            return ""

        try:
            # Convert PDF pages to images and OCR each
            import pdf2image

            if file_content:
                images = pdf2image.convert_from_bytes(file_content)
            else:
                images = pdf2image.convert_from_path(str(file_path))

            text_parts = []
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i + 1}/{len(images)}")
                page_text = self.ocr_engine.extract_text(image)
                if page_text:
                    text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return ""

    def _parse_docx(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse DOCX document."""
        try:
            if file_content:
                doc = DocxDocument(io.BytesIO(file_content))
            else:
                doc = DocxDocument(str(file_path))

            # Extract text from paragraphs
            text_parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text:
                            text_parts.append(cell.text)

            text = "\n\n".join(text_parts)

            return {
                "text": text,
                "metadata": {
                    "format": "docx",
                    "is_scanned": False,
                    "pages": len(doc.sections),
                    "paragraphs": len(doc.paragraphs),
                    "tables": len(doc.tables),
                },
            }

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse DOCX: {str(e)}") from e

    def _parse_txt(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse plain text document."""
        try:
            if file_content:
                text = file_content.decode("utf-8", errors="ignore")
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            return {
                "text": text,
                "metadata": {
                    "format": "txt",
                    "is_scanned": False,
                    "lines": len(text.split("\n")),
                },
            }

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse TXT: {str(e)}") from e

    def _parse_rtf(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse RTF document."""
        try:
            if file_content:
                rtf_content = file_content.decode("utf-8", errors="ignore")
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    rtf_content = f.read()

            text = rtf_to_text(rtf_content)

            return {
                "text": text,
                "metadata": {
                    "format": "rtf",
                    "is_scanned": False,
                },
            }

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse RTF: {str(e)}") from e

    def _parse_odt(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse ODT (OpenDocument Text) document."""
        try:
            from odfdo import Document as ODFDocument

            if file_content:
                doc = ODFDocument(io.BytesIO(file_content))
            else:
                doc = ODFDocument(str(file_path))

            # Extract text from body
            body = doc.body
            text_parts = []

            for paragraph in body.get_paragraphs():
                text = paragraph.get_text()
                if text:
                    text_parts.append(text)

            text = "\n\n".join(text_parts)

            return {
                "text": text,
                "metadata": {
                    "format": "odt",
                    "is_scanned": False,
                },
            }

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse ODT: {str(e)}") from e

    def _parse_image(self, file_path: Path, file_content: Optional[bytes]) -> dict:
        """Parse image file using OCR."""
        if not self.enable_ocr:
            raise DocumentProcessingError("OCR is disabled, cannot process images")

        try:
            if file_content:
                image = Image.open(io.BytesIO(file_content))
            else:
                image = Image.open(str(file_path))

            text = self.ocr_engine.extract_text(image)

            return {
                "text": text,
                "metadata": {
                    "format": file_path.suffix.lower()[1:],
                    "is_scanned": True,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                },
            }

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse image: {str(e)}") from e

    @staticmethod
    def is_supported(file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if supported, False otherwise
        """
        extension = Path(file_path).suffix.lower()
        return extension in DocumentParser.SUPPORTED_FORMATS
