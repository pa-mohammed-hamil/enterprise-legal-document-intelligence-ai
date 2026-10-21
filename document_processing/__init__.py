"""
Document Processing Module

Multi-format document parsing with OCR support for legal documents.
"""

from backend.document_processing.document_parser import DocumentParser
from backend.document_processing.document_processor import DocumentProcessor
from backend.document_processing.metadata_extractor import MetadataExtractor
from backend.document_processing.ocr_engine import OCREngine

__all__ = [
    "DocumentParser",
    "DocumentProcessor",
    "MetadataExtractor",
    "OCREngine",
]
