"""
Document Processor Service

High-level document processing service for legal documents.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

from backend.core.exceptions import DocumentProcessingError
from backend.document_processing.document_parser import DocumentParser
from backend.document_processing.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    High-level document processing service.
    
    Features:
    - Multi-format document parsing
    - OCR for scanned documents
    - Metadata extraction
    - Text chunking for RAG
    - Document validation
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_languages: list[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize document processor.

        Args:
            enable_ocr: Whether to enable OCR
            ocr_languages: Languages for OCR
            chunk_size: Size of text chunks for RAG
            chunk_overlap: Overlap between chunks
        """
        self.parser = DocumentParser(
            enable_ocr=enable_ocr,
            ocr_languages=ocr_languages or ["eng"]
        )
        self.metadata_extractor = MetadataExtractor()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
        extract_metadata: bool = True,
        create_chunks: bool = True,
    ) -> dict:
        """
        Process a document end-to-end.

        Args:
            file_path: Path to document
            file_content: Optional file content bytes
            extract_metadata: Whether to extract metadata
            create_chunks: Whether to create text chunks

        Returns:
            Dictionary with:
            {
                "text": str,
                "chunks": list[dict],
                "metadata": dict,
                "document_hash": str,
                "word_count": int,
                "char_count": int,
            }
        """
        file_path = Path(file_path)

        try:
            # Validate file
            self._validate_file(file_path, file_content)

            # Parse document
            logger.info(f"Parsing document: {file_path}")
            parsed = self.parser.parse(file_path, file_content)

            text = parsed["text"]
            parse_metadata = parsed.get("metadata", {})

            # Extract additional metadata
            if extract_metadata:
                logger.info(f"Extracting metadata from: {file_path}")
                doc_metadata = self.metadata_extractor.extract(file_path, file_content)
                parse_metadata.update(doc_metadata)

            # Create chunks for RAG
            chunks = []
            if create_chunks and text:
                logger.info(f"Creating text chunks for: {file_path}")
                chunks = self._create_chunks(text)

            # Calculate document hash
            document_hash = self._calculate_hash(
                file_content if file_content else file_path.read_bytes()
            )

            # Word and character counts
            word_count = len(text.split())
            char_count = len(text)

            result = {
                "text": text,
                "chunks": chunks,
                "metadata": parse_metadata,
                "document_hash": document_hash,
                "word_count": word_count,
                "char_count": char_count,
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
            }

            logger.info(
                f"Document processed successfully: {file_path.name} "
                f"({word_count} words, {len(chunks)} chunks)"
            )

            return result

        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            raise DocumentProcessingError(
                f"Failed to process document: {str(e)}"
            ) from e

    def _validate_file(
        self, file_path: Path, file_content: Optional[bytes]
    ) -> None:
        """
        Validate file before processing.

        Args:
            file_path: Path to file
            file_content: Optional file content

        Raises:
            DocumentProcessingError: If validation fails
        """
        # Check if file exists (if not using file_content)
        if not file_content and not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        # Check if format is supported
        if not DocumentParser.is_supported(file_path):
            raise DocumentProcessingError(
                f"Unsupported file format: {file_path.suffix}"
            )

        # Check file size
        max_size_mb = 100
        if file_content:
            size_mb = len(file_content) / (1024 * 1024)
        else:
            size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            raise DocumentProcessingError(
                f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
            )

    def _create_chunks(self, text: str) -> list[dict]:
        """
        Split text into overlapping chunks for RAG.

        Args:
            text: Full document text

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Find the last sentence/paragraph boundary within chunk
            if end < len(text):
                # Try to break at paragraph
                break_point = text.rfind("\n\n", start, end)
                if break_point == -1:
                    # Try to break at sentence
                    break_point = text.rfind(". ", start, end)
                if break_point != -1 and break_point > start:
                    end = break_point + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                })
                chunk_index += 1

            # Move start forward with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks

    @staticmethod
    def _calculate_hash(content: bytes) -> str:
        """
        Calculate SHA-256 hash of file content.

        Args:
            content: File content bytes

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content).hexdigest()

    def batch_process(
        self,
        file_paths: list[Union[str, Path]],
        continue_on_error: bool = True,
    ) -> list[dict]:
        """
        Process multiple documents in batch.

        Args:
            file_paths: List of file paths
            continue_on_error: Whether to continue if one document fails

        Returns:
            List of processing results (or errors)
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append({
                    "file_path": str(file_path),
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                
                if continue_on_error:
                    results.append({
                        "file_path": str(file_path),
                        "success": False,
                        "error": str(e),
                    })
                else:
                    raise

        successful = sum(1 for r in results if r["success"])
        logger.info(
            f"Batch processing complete: {successful}/{len(results)} successful"
        )

        return results

    def get_supported_formats(self) -> set[str]:
        """
        Get list of supported file formats.

        Returns:
            Set of file extensions
        """
        return DocumentParser.SUPPORTED_FORMATS

    def is_format_supported(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """
        return DocumentParser.is_supported(file_path)
