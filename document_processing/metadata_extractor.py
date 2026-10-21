"""
Metadata Extractor

Extract metadata from various document formats.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import PyPDF2
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract metadata from documents including:
    - Author, creator, producer
    - Creation and modification dates
    - Title, subject, keywords
    - Page count, word count
    - File properties
    """

    def extract(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
    ) -> dict:
        """
        Extract metadata from a document.

        Args:
            file_path: Path to document
            file_content: Optional file content bytes

        Returns:
            Dictionary with extracted metadata
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        metadata = {
            "file_name": file_path.name,
            "file_extension": extension,
            "file_size_bytes": 0,
            "extracted_at": datetime.utcnow().isoformat(),
        }

        try:
            # Get file size
            if file_content:
                metadata["file_size_bytes"] = len(file_content)
            elif file_path.exists():
                metadata["file_size_bytes"] = file_path.stat().st_size

            # Extract format-specific metadata
            if extension == ".pdf":
                pdf_metadata = self._extract_pdf_metadata(file_path, file_content)
                metadata.update(pdf_metadata)
            elif extension in [".docx", ".doc"]:
                docx_metadata = self._extract_docx_metadata(file_path, file_content)
                metadata.update(docx_metadata)
            elif extension in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
                image_metadata = self._extract_image_metadata(file_path, file_content)
                metadata.update(image_metadata)

            return metadata

        except Exception as e:
            logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            return metadata

    def _extract_pdf_metadata(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> dict:
        """Extract metadata from PDF."""
        metadata = {}

        try:
            if file_content:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            else:
                pdf_reader = PyPDF2.PdfReader(str(file_path))

            # Get document info
            doc_info = pdf_reader.metadata

            if doc_info:
                # Standard PDF metadata fields
                metadata["author"] = doc_info.get("/Author", "")
                metadata["creator"] = doc_info.get("/Creator", "")
                metadata["producer"] = doc_info.get("/Producer", "")
                metadata["title"] = doc_info.get("/Title", "")
                metadata["subject"] = doc_info.get("/Subject", "")
                metadata["keywords"] = doc_info.get("/Keywords", "")

                # Dates
                creation_date = doc_info.get("/CreationDate")
                if creation_date:
                    metadata["creation_date"] = self._parse_pdf_date(creation_date)

                mod_date = doc_info.get("/ModDate")
                if mod_date:
                    metadata["modification_date"] = self._parse_pdf_date(mod_date)

            # Page count
            metadata["page_count"] = len(pdf_reader.pages)

            # Check if encrypted
            metadata["is_encrypted"] = pdf_reader.is_encrypted

            # PDF version
            if hasattr(pdf_reader, "pdf_header"):
                metadata["pdf_version"] = pdf_reader.pdf_header

        except Exception as e:
            logger.warning(f"PDF metadata extraction failed: {e}")

        return metadata

    def _extract_docx_metadata(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> dict:
        """Extract metadata from DOCX."""
        metadata = {}

        try:
            if file_content:
                doc = DocxDocument(io.BytesIO(file_content))
            else:
                doc = DocxDocument(str(file_path))

            # Core properties
            core_props = doc.core_properties

            metadata["author"] = core_props.author or ""
            metadata["title"] = core_props.title or ""
            metadata["subject"] = core_props.subject or ""
            metadata["keywords"] = core_props.keywords or ""
            metadata["comments"] = core_props.comments or ""
            metadata["category"] = core_props.category or ""
            metadata["last_modified_by"] = core_props.last_modified_by or ""

            # Dates
            if core_props.created:
                metadata["creation_date"] = core_props.created.isoformat()
            if core_props.modified:
                metadata["modification_date"] = core_props.modified.isoformat()

            # Document statistics
            metadata["page_count"] = len(doc.sections)
            metadata["paragraph_count"] = len(doc.paragraphs)
            metadata["table_count"] = len(doc.tables)

            # Word count (approximate)
            text = "\n".join([p.text for p in doc.paragraphs])
            metadata["word_count"] = len(text.split())

        except Exception as e:
            logger.warning(f"DOCX metadata extraction failed: {e}")

        return metadata

    def _extract_image_metadata(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> dict:
        """Extract metadata from image."""
        metadata = {}

        try:
            if file_content:
                image = Image.open(io.BytesIO(file_content))
            else:
                image = Image.open(str(file_path))

            # Basic image properties
            metadata["width"] = image.width
            metadata["height"] = image.height
            metadata["format"] = image.format
            metadata["mode"] = image.mode

            # DPI
            if "dpi" in image.info:
                metadata["dpi"] = image.info["dpi"]

            # EXIF data (for JPEG)
            if hasattr(image, "_getexif") and image._getexif():
                exif = image._getexif()
                if exif:
                    # Common EXIF tags
                    exif_tags = {
                        271: "make",
                        272: "model",
                        274: "orientation",
                        306: "datetime",
                        36867: "datetime_original",
                        36868: "datetime_digitized",
                    }

                    for tag_id, tag_name in exif_tags.items():
                        if tag_id in exif:
                            metadata[f"exif_{tag_name}"] = exif[tag_id]

        except Exception as e:
            logger.warning(f"Image metadata extraction failed: {e}")

        return metadata

    @staticmethod
    def _parse_pdf_date(pdf_date: str) -> str:
        """
        Parse PDF date format to ISO format.

        PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
        Example: D:20240115120000+05'00'
        """
        try:
            # Remove D: prefix
            if pdf_date.startswith("D:"):
                pdf_date = pdf_date[2:]

            # Extract date components
            year = pdf_date[0:4]
            month = pdf_date[4:6]
            day = pdf_date[6:8]
            hour = pdf_date[8:10] if len(pdf_date) > 8 else "00"
            minute = pdf_date[10:12] if len(pdf_date) > 10 else "00"
            second = pdf_date[12:14] if len(pdf_date) > 12 else "00"

            # Create ISO format
            iso_date = f"{year}-{month}-{day}T{hour}:{minute}:{second}"

            # Add timezone if present
            if len(pdf_date) > 14:
                tz = pdf_date[14:]
                iso_date += tz.replace("'", ":")

            return iso_date

        except Exception as e:
            logger.warning(f"Failed to parse PDF date {pdf_date}: {e}")
            return pdf_date
