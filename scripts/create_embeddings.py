"""
Create embeddings for legal-document chunks.

Usage:
    python scripts/create_embeddings.py
    python scripts/create_embeddings.py --source data/processed/chunks.jsonl
    python scripts/create_embeddings.py --output data/processed/embeddings.jsonl
    python scripts/create_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
    python scripts/create_embeddings.py --batch-size 64
    python scripts/create_embeddings.py --resume

Features:
    - Batch embedding generation
    - L2-normalized embeddings
    - Deterministic chunk/document identifiers
    - Tenant-aware metadata preservation
    - Resume support
    - Duplicate detection
    - JSONL output suitable for RAG/vector indexing
    - Embedding manifest and statistics
    - Atomic output writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "legal-create-embeddings"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.jsonl"
)

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_manifest.json"
)

DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)

DEFAULT_MAX_LENGTH = int(
    os.getenv(
        "EMBEDDING_MAX_LENGTH",
        "512",
    )
)

DEFAULT_NORMALIZE = True

SUPPORTED_INPUT_EXTENSIONS = {
    ".jsonl",
    ".json",
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmbeddingRecord:
    """One embedded legal-document chunk."""

    embedding_id: str
    chunk_id: str

    document_id: str
    version_id: str

    tenant_id: str
    organization_id: str

    text: str

    embedding: list[float]

    embedding_model: str
    embedding_dimension: int

    normalized: bool

    content_hash: str
    text_hash: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass(slots=True)
class EmbeddingStats:
    """Embedding job statistics."""

    started_at: str

    finished_at: str | None = None

    input_records: int = 0
    valid_records: int = 0
    skipped_records: int = 0
    duplicate_records: int = 0
    embedded_records: int = 0

    batches_processed: int = 0

    embedding_dimension: int = 0

    elapsed_seconds: float = 0.0

    errors: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


def stable_uuid(
    value: str,
) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )
    )


def atomic_write_text(
    destination: Path,
    content: str,
) -> None:
    """
    Atomically write a file.

    This prevents partially-written embedding files
    if the process crashes during output.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass

        raise


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


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def read_jsonl(
    path: Path,
) -> Iterator[dict[str, Any]]:
    """Stream JSONL records."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid JSON at line %d: %s",
                    line_number,
                    exc,
                )
                continue

            if not isinstance(record, dict):
                logger.warning(
                    "Skipping non-object at line %d.",
                    line_number,
                )
                continue

            yield record


def read_json(
    path: Path,
) -> Iterator[dict[str, Any]]:
    """Read a JSON array or common wrapped JSON structure."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record

        return

    if isinstance(payload, dict):
        for key in (
            "chunks",
            "records",
            "documents",
            "data",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                for record in value:
                    if isinstance(record, dict):
                        yield record

                return

        # Allow one-record JSON files.
        yield payload


def load_records(
    source: Path,
) -> list[dict[str, Any]]:
    """Load chunk records from JSON or JSONL."""

    if not source.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {source}"
        )

    suffix = source.suffix.lower()

    if suffix == ".jsonl":
        records = list(
            read_jsonl(source)
        )

    elif suffix == ".json":
        records = list(
            read_json(source)
        )

    else:
        raise ValueError(
            "Input must be .json or .jsonl"
        )

    logger.info(
        "Loaded %d records from %s",
        len(records),
        source,
    )

    return records


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------


def normalize_record(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Normalize different chunk formats into a stable schema.
    """

    text = (
        record.get("text")
        or record.get("content")
        or record.get("chunk_text")
        or ""
    )

    if not isinstance(text, str):
        text = str(text)

    text = " ".join(
        text.split()
    ).strip()

    if not text:
        return None

    chunk_id = (
        record.get("chunk_id")
        or record.get("id")
    )

    document_id = (
        record.get("document_id")
        or record.get("doc_id")
    )

    version_id = (
        record.get("version_id")
        or record.get("document_version_id")
        or ""
    )

    tenant_id = (
        record.get("tenant_id")
        or "default-tenant"
    )

    organization_id = (
        record.get("organization_id")
        or "default-organization"
    )

    if not chunk_id:
        chunk_id = stable_uuid(
            f"{document_id or 'document'}:"
            f"{version_id}:"
            f"{text}"
        )

    if not document_id:
        document_id = stable_uuid(
            f"document:{chunk_id}"
        )

    text_hash = sha256_text(text)

    content_hash = (
        record.get("content_hash")
        or record.get("document_hash")
        or text_hash
    )

    metadata = record.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    # Preserve useful top-level fields that may not
    # have been nested under metadata.
    passthrough_fields = {
        "filename",
        "source_path",
        "file_type",
        "page_number",
        "page",
        "chunk_index",
        "chunk_count",
        "section",
        "title",
        "language",
        "jurisdiction",
        "document_type",
        "contract_type",
        "clause_type",
        "access_level",
    }

    for field_name in passthrough_fields:
        if (
            field_name in record
            and field_name not in metadata
        ):
            metadata[field_name] = (
                record[field_name]
            )

    return {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "version_id": str(version_id),
        "tenant_id": str(tenant_id),
        "organization_id": str(
            organization_id
        ),
        "text": text,
        "content_hash": str(
            content_hash
        ),
        "text_hash": text_hash,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Sentence-transformers embedding service."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        normalize: bool,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install the project requirements first."
            )

        logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model_name = model_name
        self.normalize = normalize

        self.model = SentenceTransformer(
            model_name
        )

        # Prevent unexpectedly long legal passages
        # from consuming excessive memory.
        try:
            self.model.max_seq_length = (
                max_length
            )
        except Exception:
            logger.warning(
                "Unable to set model max_seq_length."
            )

        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        self.dimension = int(
            dimension
        )

        logger.info(
            "Embedding dimension: %d",
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
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings.tolist()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_existing_ids(
    output: Path,
) -> set[str]:
    """
    Load already embedded chunk IDs.

    Used by --resume to avoid recomputing embeddings.
    """

    if not output.exists():
        return set()

    ids: set[str] = set()

    try:
        for record in read_jsonl(output):
            chunk_id = record.get(
                "chunk_id"
            )

            if chunk_id:
                ids.add(str(chunk_id))

    except Exception:
        logger.warning(
            "Unable to read existing output for resume.",
            exc_info=True,
        )

    logger.info(
        "Found %d existing embedded chunks.",
        len(ids),
    )

    return ids


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def create_embedding_records(
    records: Sequence[dict[str, Any]],
    service: EmbeddingService,
    batch_size: int,
    stats: EmbeddingStats,
    existing_ids: set[str],
    resume: bool,
) -> list[EmbeddingRecord]:
    """Generate embeddings in batches."""

    output: list[EmbeddingRecord] = []

    for batch in batch_items(
        records,
        batch_size,
    ):
        texts = [
            record["text"]
            for record in batch
        ]

        try:
            embeddings = service.encode(
                texts=texts,
                batch_size=batch_size,
            )

        except Exception:
            stats.errors += len(batch)

            logger.exception(
                "Embedding batch failed."
            )

            continue

        if len(embeddings) != len(batch):
            stats.errors += len(batch)

            logger.error(
                "Embedding count mismatch: "
                "expected=%d actual=%d",
                len(batch),
                len(embeddings),
            )

            continue

        for record, embedding in zip(
            batch,
            embeddings,
        ):
            chunk_id = record["chunk_id"]

            if (
                resume
                and chunk_id in existing_ids
            ):
                stats.skipped_records += 1
                continue

            embedding_id = stable_uuid(
                f"{chunk_id}:"
                f"{service.model_name}"
            )

            output.append(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    chunk_id=chunk_id,
                    document_id=record[
                        "document_id"
                    ],
                    version_id=record[
                        "version_id"
                    ],
                    tenant_id=record[
                        "tenant_id"
                    ],
                    organization_id=record[
                        "organization_id"
                    ],
                    text=record["text"],
                    embedding=embedding,
                    embedding_model=(
                        service.model_name
                    ),
                    embedding_dimension=(
                        service.dimension
                    ),
                    normalized=(
                        service.normalize
                    ),
                    content_hash=record[
                        "content_hash"
                    ],
                    text_hash=record[
                        "text_hash"
                    ],
                    metadata=record[
                        "metadata"
                    ],
                )
            )

        stats.batches_processed += 1

        logger.info(
            "Processed embedding batch: %d records.",
            len(batch),
        )

    return output


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def serialize_record(
    record: EmbeddingRecord,
) -> str:
    return json.dumps(
        asdict(record),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_embeddings(
    records: Sequence[EmbeddingRecord],
    output: Path,
    resume: bool,
) -> None:
    """
    Write embeddings to JSONL.

    In normal mode the output is replaced atomically.
    In resume mode existing records are preserved and
    new records are appended safely.
    """

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resume and output.exists():
        existing_content = output.read_text(
            encoding="utf-8"
        )

        new_content = "".join(
            serialize_record(record) + "\n"
            for record in records
        )

        atomic_write_text(
            output,
            existing_content + new_content,
        )

    else:
        content = "".join(
            serialize_record(record) + "\n"
            for record in records
        )

        atomic_write_text(
            output,
            content,
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    manifest_path: Path,
    source: Path,
    output: Path,
    model_name: str,
    dimension: int,
    stats: EmbeddingStats,
    normalize: bool,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "source": str(source),
        "output": str(output),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "normalized": normalize,
        "distance_recommendation": (
            "cosine"
            if normalize
            else "cosine/dot-product"
        ),
        "statistics": asdict(stats),
    }

    atomic_write_text(
        manifest_path,
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_records(
    records: Sequence[dict[str, Any]],
    stats: EmbeddingStats,
) -> list[dict[str, Any]]:
    """Validate and normalize chunk records."""

    normalized: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for record in records:
        stats.input_records += 1

        item = normalize_record(
            record
        )

        if item is None:
            stats.skipped_records += 1
            continue

        chunk_id = item["chunk_id"]

        if chunk_id in seen:
            stats.duplicate_records += 1
            continue

        seen.add(chunk_id)

        normalized.append(item)

        stats.valid_records += 1

    return normalized


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create embeddings for legal-document "
            "RAG chunks."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Input chunks JSON/JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output embeddings JSONL file.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Embedding manifest JSON file.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-transformers embedding model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum model sequence length.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip chunks already present "
            "in the output file."
        ),
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly allow replacing an existing "
            "embedding output."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.max_length <= 0:
        raise ValueError(
            "--max-length must be greater than zero."
        )

    if (
        args.output.exists()
        and not args.resume
        and not args.overwrite
    ):
        logger.error(
            "Output already exists: %s",
            args.output,
        )

        logger.error(
            "Use --overwrite or --resume."
        )

        return 2

    started = time.perf_counter()

    stats = EmbeddingStats(
        started_at=utc_now()
    )

    logger.info(
        "Starting legal embedding generation."
    )

    logger.info(
        "Source: %s",
        args.source,
    )

    logger.info(
        "Output: %s",
        args.output,
    )

    logger.info(
        "Model: %s",
        args.model,
    )

    logger.info(
        "Batch size: %d",
        args.batch_size,
    )

    normalize = not args.no_normalize

    try:
        raw_records = load_records(
            args.source
        )

        records = validate_records(
            raw_records,
            stats,
        )

        if not records:
            logger.warning(
                "No valid chunk records found."
            )

            stats.finished_at = utc_now()
            stats.elapsed_seconds = (
                time.perf_counter() - started
            )

            write_manifest(
                manifest_path=args.manifest,
                source=args.source,
                output=args.output,
                model_name=args.model,
                dimension=0,
                stats=stats,
                normalize=normalize,
            )

            return 0

        existing_ids: set[str] = set()

        if args.resume:
            existing_ids = load_existing_ids(
                args.output
            )

        service = EmbeddingService(
            model_name=args.model,
            max_length=args.max_length,
            normalize=normalize,
        )

        stats.embedding_dimension = (
            service.dimension
        )

        embedding_records = (
            create_embedding_records(
                records=records,
                service=service,
                batch_size=args.batch_size,
                stats=stats,
                existing_ids=existing_ids,
                resume=args.resume,
            )
        )

        write_embeddings(
            records=embedding_records,
            output=args.output,
            resume=args.resume,
        )

        stats.embedded_records = len(
            embedding_records
        )

        stats.finished_at = utc_now()

        stats.elapsed_seconds = (
            time.perf_counter() - started
        )

        write_manifest(
            manifest_path=args.manifest,
            source=args.source,
            output=args.output,
            model_name=args.model,
            dimension=service.dimension,
            stats=stats,
            normalize=normalize,
        )

    except Exception:
        logger.exception(
            "Embedding generation failed."
        )
        return 1

    logger.info(
        "Embedding generation completed."
    )

    logger.info(
        "Input records: %d",
        stats.input_records,
    )

    logger.info(
        "Valid records: %d",
        stats.valid_records,
    )

    logger.info(
        "Skipped records: %d",
        stats.skipped_records,
    )

    logger.info(
        "Duplicate records: %d",
        stats.duplicate_records,
    )

    logger.info(
        "Embedded records: %d",
        stats.embedded_records,
    )

    logger.info(
        "Embedding dimension: %d",
        stats.embedding_dimension,
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
        "Embeddings: %s",
        args.output,
    )

    logger.info(
        "Manifest: %s",
        args.manifest,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())