"""
Build / rebuild the vector index for legal documents.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --source data/sample
    python scripts/build_index.py --rebuild
    python scripts/build_index.py --dry-run

The script:
    1. Discovers supported legal documents.
    2. Parses and normalizes documents.
    3. Chunks documents using the configured chunking strategy.
    4. Generates embeddings in batches.
    5. Stores vectors and metadata in the configured vector database.
    6. Preserves tenant/document/version metadata for permission-aware retrieval.
    7. Supports idempotent indexing and full rebuilds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover
    QdrantClient = None
    qdrant_models = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("legal-index-builder")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".html",
    ".htm",
}

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "sample"

DEFAULT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "legal_documents",
)

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "32")
)

DEFAULT_CHUNK_SIZE = int(
    os.getenv("RAG_CHUNK_SIZE", "900")
)

DEFAULT_CHUNK_OVERLAP = int(
    os.getenv("RAG_CHUNK_OVERLAP", "150")
)

DEFAULT_VECTOR_SIZE = int(
    os.getenv("EMBEDDING_DIMENSION", "384")
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)

TENANT_ID = os.getenv(
    "DEFAULT_TENANT_ID",
    "demo-tenant",
)

DEFAULT_ORGANIZATION_ID = os.getenv(
    "DEFAULT_ORGANIZATION_ID",
    "demo-organization",
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceDocument:
    """A source legal document."""

    path: Path
    document_id: str
    version_id: str
    filename: str
    extension: str
    content_hash: str
    tenant_id: str
    organization_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class DocumentChunk:
    """A chunk ready for embedding/indexing."""

    chunk_id: str
    document_id: str
    version_id: str
    tenant_id: str
    organization_id: str

    text: str
    chunk_index: int

    filename: str
    source_path: str
    content_hash: str

    metadata: dict[str, Any]


@dataclass(slots=True)
class IndexStats:
    """Indexing execution statistics."""

    started_at: str
    finished_at: str | None = None

    documents_discovered: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0

    chunks_created: int = 0
    chunks_indexed: int = 0

    batches_processed: int = 0

    elapsed_seconds: float = 0.0

    errors: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate a stable SHA-256 hash for a document."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def stable_uuid(value: str) -> str:
    """Generate deterministic UUID from a stable string."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def normalize_text(text: str) -> str:
    """Normalize extracted document text."""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
    ]

    return "\n".join(
        line for line in lines if line
    ).strip()


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------


def discover_documents(source: Path) -> list[Path]:
    """Recursively discover supported documents."""

    if not source.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source}"
        )

    if source.is_file():
        if source.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [source]

        return []

    documents = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(documents)


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------


def extract_pdf(path: Path) -> str:
    """Extract text from PDF."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF indexing."
        ) from exc

    reader = PdfReader(str(path))

    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def extract_docx(path: Path) -> str:
    """Extract text from DOCX."""

    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for DOCX indexing."
        ) from exc

    document = Document(str(path))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    # Include table content because legal contracts frequently
    # contain schedules, pricing, obligations and signatures in tables.
    tables: list[str] = []

    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            if any(cells):
                tables.append(" | ".join(cells))

    return "\n".join(
        paragraphs + tables
    )


def extract_html(path: Path) -> str:
    """Extract readable text from HTML."""

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required for HTML indexing."
        ) from exc

    html = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    return soup.get_text(
        separator="\n"
    )


def extract_text(path: Path) -> str:
    """Extract text according to document type."""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_pdf(path)

    elif suffix == ".docx":
        text = extract_docx(path)

    elif suffix in {
        ".html",
        ".htm",
    }:
        text = extract_html(path)

    else:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    return normalize_text(text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text using paragraph/sentence-aware boundaries.

    This is intentionally dependency-light. The production application can
    replace this with the project's advanced chunking implementation while
    retaining the same DocumentChunk contract.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current

        if current.strip():
            chunks.append(current.strip())

        current = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current
            else f"{current}\n{paragraph}"
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        flush()

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        # Long paragraph fallback.
        start = 0

        while start < len(paragraph):
            end = min(
                start + chunk_size,
                len(paragraph),
            )

            piece = paragraph[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= len(paragraph):
                break

            start = max(
                end - overlap,
                start + 1,
            )

    flush()

    return chunks


def create_chunks(
    document: SourceDocument,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Create metadata-rich chunks."""

    raw_chunks = split_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks: list[DocumentChunk] = []

    for index, chunk_text in enumerate(raw_chunks):
        chunk_id = stable_uuid(
            f"{document.document_id}:"
            f"{document.version_id}:"
            f"{index}:"
            f"{chunk_text}"
        )

        metadata = {
            **document.metadata,
            "chunk_index": index,
            "chunk_count": len(raw_chunks),
            "document_id": document.document_id,
            "version_id": document.version_id,
            "tenant_id": document.tenant_id,
            "organization_id": document.organization_id,
            "filename": document.filename,
            "source_path": str(document.path),
            "content_hash": document.content_hash,
            "indexed_at": utc_now(),
        }

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                version_id=document.version_id,
                tenant_id=document.tenant_id,
                organization_id=document.organization_id,
                text=chunk_text,
                chunk_index=index,
                filename=document.filename,
                source_path=str(document.path),
                content_hash=document.content_hash,
                metadata=metadata,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------


def build_source_document(
    path: Path,
    tenant_id: str,
    organization_id: str,
) -> SourceDocument:
    """Build deterministic document identity."""

    content_hash = sha256_file(path)

    relative_path = (
        path.relative_to(PROJECT_ROOT)
        if path.is_relative_to(PROJECT_ROOT)
        else path
    )

    document_id = stable_uuid(
        f"{tenant_id}:{relative_path}"
    )

    version_id = stable_uuid(
        f"{document_id}:{content_hash}"
    )

    metadata = {
        "file_type": path.suffix.lower(),
        "relative_path": str(relative_path),
        "document_name": path.stem,
        "document_hash": content_hash,
    }

    return SourceDocument(
        path=path,
        document_id=document_id,
        version_id=version_id,
        filename=path.name,
        extension=path.suffix.lower(),
        content_hash=content_hash,
        tenant_id=tenant_id,
        organization_id=organization_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Batch embedding wrapper."""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install project requirements first."
            )

        logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

        self.dimension = int(dimension)

        logger.info(
            "Embedding dimension: %s",
            self.dimension,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        return embeddings.tolist()


# ---------------------------------------------------------------------------
# Qdrant index
# ---------------------------------------------------------------------------


class QdrantIndex:
    """Qdrant vector index abstraction."""

    def __init__(
        self,
        url: str,
        collection: str,
        vector_size: int,
        api_key: str | None = None,
    ) -> None:
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is required."
            )

        self.collection = collection

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
        )

        self.vector_size = vector_size

    def ensure_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """Create collection if needed."""

        exists = self.client.collection_exists(
            self.collection
        )

        if recreate and exists:
            logger.warning(
                "Recreating Qdrant collection: %s",
                self.collection,
            )

            self.client.delete_collection(
                self.collection
            )

            exists = False

        if exists:
            logger.info(
                "Using existing collection: %s",
                self.collection,
            )
            return

        logger.info(
            "Creating collection: %s",
            self.collection,
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        # Payload indexes improve filtered legal retrieval.
        payload_fields = {
            "tenant_id": "keyword",
            "organization_id": "keyword",
            "document_id": "keyword",
            "version_id": "keyword",
            "content_hash": "keyword",
            "file_type": "keyword",
        }

        for field_name, field_type in payload_fields.items():
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_type,
            )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Upsert vectors and metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings length mismatch."
            )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "version_id": chunk.version_id,
                "tenant_id": chunk.tenant_id,
                "organization_id": chunk.organization_id,
                "filename": chunk.filename,
                "source_path": chunk.source_path,
                "content_hash": chunk.content_hash,
                **chunk.metadata,
            }

            points.append(
                qdrant_models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(embedding),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )


# ---------------------------------------------------------------------------
# Dry-run index
# ---------------------------------------------------------------------------


class DryRunIndex:
    """No-op index implementation for validation/testing."""

    def ensure_collection(
        self,
        recreate: bool = False,
    ) -> None:
        logger.info(
            "Dry-run: collection initialization skipped."
        )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        logger.info(
            "Dry-run: would index %d chunks.",
            len(chunks),
        )


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------


def batch_items(
    items: Sequence[Any],
    batch_size: int,
) -> Iterable[Sequence[Any]]:
    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start:start + batch_size
        ]


def index_documents(
    documents: Sequence[Path],
    index: QdrantIndex | DryRunIndex,
    embedder: EmbeddingService | None,
    tenant_id: str,
    organization_id: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    dry_run: bool,
    stats: IndexStats,
) -> None:
    """Run the complete indexing pipeline."""

    for path in documents:
        try:
            logger.info(
                "Processing document: %s",
                path,
            )

            document = build_source_document(
                path=path,
                tenant_id=tenant_id,
                organization_id=organization_id,
            )

            text = extract_text(path)

            if not text.strip():
                logger.warning(
                    "Skipping empty document: %s",
                    path,
                )

                stats.documents_skipped += 1
                continue

            chunks = create_chunks(
                document=document,
                text=text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            if not chunks:
                logger.warning(
                    "No chunks generated: %s",
                    path,
                )

                stats.documents_skipped += 1
                continue

            stats.chunks_created += len(chunks)

            logger.info(
                "Created %d chunks from %s",
                len(chunks),
                path.name,
            )

            if dry_run:
                index.upsert(
                    chunks=chunks,
                    embeddings=[
                        []
                        for _ in chunks
                    ],
                )

                stats.chunks_indexed += len(chunks)
                stats.documents_processed += 1
                continue

            if embedder is None:
                raise RuntimeError(
                    "Embedding service is required "
                    "when not running in dry-run mode."
                )

            for chunk_batch in batch_items(
                chunks,
                batch_size,
            ):
                embeddings = embedder.encode(
                    [
                        chunk.text
                        for chunk in chunk_batch
                    ],
                    batch_size=batch_size,
                )

                index.upsert(
                    chunks=chunk_batch,
                    embeddings=embeddings,
                )

                stats.chunks_indexed += len(
                    chunk_batch
                )

                stats.batches_processed += 1

            stats.documents_processed += 1

        except Exception:
            stats.errors += 1

            logger.exception(
                "Failed to index document: %s",
                path,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Enterprise Legal Document "
            "Intelligence AI vector index."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Document directory or file.",
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size in characters.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding/index batch size.",
    )

    parser.add_argument(
        "--tenant-id",
        default=TENANT_ID,
        help="Tenant identifier.",
    )

    parser.add_argument(
        "--organization-id",
        default=DEFAULT_ORGANIZATION_ID,
        help="Organization identifier.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the vector collection.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without connecting to Qdrant.",
    )

    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero exit code if any document fails.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    started = time.perf_counter()

    stats = IndexStats(
        started_at=utc_now()
    )

    logger.info(
        "Starting legal document index build."
    )

    logger.info(
        "Source: %s",
        args.source,
    )

    logger.info(
        "Collection: %s",
        args.collection,
    )

    documents = discover_documents(
        args.source
    )

    stats.documents_discovered = len(
        documents
    )

    logger.info(
        "Discovered %d supported documents.",
        len(documents),
    )

    if not documents:
        logger.warning(
            "No supported documents found."
        )

        stats.finished_at = utc_now()
        stats.elapsed_seconds = (
            time.perf_counter() - started
        )

        return 0

    if args.dry_run:
        index: QdrantIndex | DryRunIndex = (
            DryRunIndex()
        )

        embedder = None

    else:
        embedder = EmbeddingService(
            model_name=args.embedding_model
        )

        index = QdrantIndex(
            url=QDRANT_URL,
            collection=args.collection,
            vector_size=embedder.dimension,
            api_key=QDRANT_API_KEY,
        )

    index.ensure_collection(
        recreate=args.rebuild
    )

    index_documents(
        documents=documents,
        index=index,
        embedder=embedder,
        tenant_id=args.tenant_id,
        organization_id=args.organization_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        stats=stats,
    )

    stats.finished_at = utc_now()
    stats.elapsed_seconds = (
        time.perf_counter() - started
    )

    report_dir = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "index_reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        / "latest_index_build.json"
    )

    report_path.write_text(
        json.dumps(
            asdict(stats),
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Index build completed."
    )

    logger.info(
        "Documents discovered: %d",
        stats.documents_discovered,
    )

    logger.info(
        "Documents processed: %d",
        stats.documents_processed,
    )

    logger.info(
        "Documents skipped: %d",
        stats.documents_skipped,
    )

    logger.info(
        "Chunks created: %d",
        stats.chunks_created,
    )

    logger.info(
        "Chunks indexed: %d",
        stats.chunks_indexed,
    )

    logger.info(
        "Errors: %d",
        stats.errors,
    )

    logger.info(
        "Elapsed: %.2f seconds",
        stats.elapsed_seconds,
    )

    logger.info(
        "Report: %s",
        report_path,
    )

    if args.fail_on_error and stats.errors:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())