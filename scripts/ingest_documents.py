"""
Ingest legal documents into the document-intelligence pipeline.

Responsibilities:
    - Discover documents from files/directories
    - Validate supported file types
    - Calculate SHA-256 content hashes
    - Extract basic metadata
    - Detect duplicate files
    - Create normalized ingestion records
    - Preserve tenant/organization metadata
    - Write JSONL ingestion manifests
    - Optionally copy files into the processed data area
    - Support incremental ingestion
    - Avoid unsafe path traversal
    - Provide deterministic document IDs

Usage:
    python scripts/ingest_documents.py

    python scripts/ingest_documents.py \
        --source data/sample/contracts

    python scripts/ingest_documents.py \
        --source data/sample/contracts \
        --output data/processed/ingestion_manifest.jsonl

    python scripts/ingest_documents.py \
        --source data/sample/contracts \
        --copy-to data/processed/documents

    python scripts/ingest_documents.py \
        --source data/sample/contracts \
        --recursive

    python scripts/ingest_documents.py \
        --source data/sample/contracts \
        --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    "legal-document-ingestion"
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "contracts"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ingestion_manifest.jsonl"
)

DEFAULT_COPY_TO = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents"
)

DEFAULT_TENANT_ID = os.getenv(
    "DEFAULT_TENANT_ID",
    "default-tenant",
)

DEFAULT_ORGANIZATION_ID = os.getenv(
    "DEFAULT_ORGANIZATION_ID",
    "default-organization",
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".rtf",
    ".json",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

MAX_FILE_SIZE_BYTES = int(
    os.getenv(
        "INGEST_MAX_FILE_SIZE_BYTES",
        str(100 * 1024 * 1024),
    )
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DocumentMetadata:
    """Normalized metadata for an ingested legal document."""

    document_id: str
    version_id: str

    filename: str
    original_filename: str

    source_path: str
    relative_path: str

    file_extension: str
    mime_type: str

    size_bytes: int

    sha256: str

    tenant_id: str
    organization_id: str

    document_type: str

    language: str | None = None

    title: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    modified_at: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class IngestionStats:
    discovered: int = 0
    supported: int = 0
    ingested: int = 0
    skipped: int = 0
    duplicates: int = 0
    too_large: int = 0
    unsupported: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a streaming SHA-256 hash."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def stable_uuid(
    value: str,
) -> str:
    """Generate a deterministic UUID."""

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )
    )


def atomic_write(
    path: Path,
    content: str,
) -> None:
    """Atomically replace a text file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass

        raise


def safe_relative_path(
    path: Path,
    root: Path,
) -> Path:
    """
    Return a path relative to root and reject traversal.
    """

    resolved_path = path.resolve()
    resolved_root = root.resolve()

    try:
        relative = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsafe path outside source root: {path}"
        ) from exc

    if any(
        part == ".."
        for part in relative.parts
    ):
        raise ValueError(
            f"Path traversal detected: {path}"
        )

    return relative


def detect_mime_type(
    path: Path,
) -> str:
    mime_type, _ = mimetypes.guess_type(
        path.name
    )

    if mime_type:
        return mime_type

    extension_map = {
        ".pdf": "application/pdf",
        ".docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".rtf": "application/rtf",
        ".json": "application/json",
    }

    return extension_map.get(
        path.suffix.lower(),
        "application/octet-stream",
    )


def infer_document_type(
    filename: str,
) -> str:
    """
    Infer a legal document type from its filename.

    This is intentionally conservative. The authoritative
    document classifier should run later in the pipeline.
    """

    name = filename.lower()

    patterns = {
        "contract": (
            "contract",
            "agreement",
            "msa",
            "master_service",
            "service_agreement",
        ),
        "nda": (
            "nda",
            "non_disclosure",
            "nondisclosure",
            "confidentiality",
        ),
        "employment": (
            "employment",
            "offer_letter",
            "joining",
            "employee_agreement",
        ),
        "lease": (
            "lease",
            "rental",
            "tenancy",
        ),
        "purchase": (
            "purchase",
            "procurement",
            "purchase_order",
            "po_",
        ),
        "privacy": (
            "privacy",
            "data_protection",
            "dpa",
        ),
        "terms_of_service": (
            "terms_of_service",
            "terms_conditions",
            "terms-and-conditions",
            "tos",
        ),
        "policy": (
            "policy",
            "compliance",
            "governance",
        ),
        "invoice": (
            "invoice",
            "bill",
        ),
        "court_filing": (
            "court",
            "filing",
            "petition",
            "complaint",
            "motion",
        ),
    }

    for document_type, keywords in patterns.items():
        if any(
            keyword in name
            for keyword in keywords
        ):
            return document_type

    return "unknown"


def detect_language(
    path: Path,
) -> str | None:
    """
    Lightweight language hint.

    Full language detection belongs to the ingestion
    pipeline. This function avoids adding an expensive
    dependency to a file-discovery script.
    """

    if path.suffix.lower() not in {
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".rtf",
    }:
        return None

    try:
        sample = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )[:10_000]

    except OSError:
        return None

    if not sample.strip():
        return None

    # Conservative default for this project.
    # Do not claim a language when detection is uncertain.
    ascii_ratio = sum(
        1
        for char in sample
        if ord(char) < 128
    ) / max(
        len(sample),
        1,
    )

    if ascii_ratio > 0.85:
        return "en"

    return None


def extract_title(
    path: Path,
) -> str:
    """Extract a simple title from text-like documents."""

    if path.suffix.lower() not in {
        ".txt",
        ".md",
        ".html",
        ".htm",
    }:
        return path.stem.replace(
            "_",
            " ",
        ).replace(
            "-",
            " ",
        ).strip().title()

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except OSError:
        return path.stem

    for line in text.splitlines():
        cleaned = line.strip()

        if not cleaned:
            continue

        if cleaned.startswith("#"):
            return cleaned.lstrip(
                "#"
            ).strip()

        if len(cleaned) <= 200:
            return cleaned

        break

    return path.stem


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def should_ignore(
    path: Path,
) -> bool:
    return any(
        part in IGNORED_DIRECTORIES
        for part in path.parts
    )


def iter_files(
    source: Path,
    recursive: bool,
) -> Iterator[Path]:
    """Yield files from a source file or directory."""

    if source.is_file():
        yield source
        return

    if not source.is_dir():
        raise FileNotFoundError(
            f"Source does not exist: {source}"
        )

    iterator = (
        source.rglob("*")
        if recursive
        else source.glob("*")
    )

    for path in iterator:
        if not path.is_file():
            continue

        if should_ignore(path):
            continue

        yield path


# ---------------------------------------------------------------------------
# Existing manifest
# ---------------------------------------------------------------------------


def load_existing_hashes(
    output: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load previously ingested documents.

    Hash is the primary duplicate key.
    """

    existing: dict[
        str,
        dict[str, Any],
    ] = {}

    if not output.exists():
        return existing

    try:
        with output.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                content_hash = record.get(
                    "sha256"
                )

                if content_hash:
                    existing[
                        str(content_hash)
                    ] = record

    except OSError:
        logger.warning(
            "Unable to load existing manifest.",
            exc_info=True,
        )

    return existing


# ---------------------------------------------------------------------------
# Document creation
# ---------------------------------------------------------------------------


def build_document_metadata(
    path: Path,
    source_root: Path,
    tenant_id: str,
    organization_id: str,
) -> DocumentMetadata:
    stat = path.stat()

    relative_path = safe_relative_path(
        path,
        source_root,
    )

    content_hash = sha256_file(
        path
    )

    document_id = stable_uuid(
        f"document:"
        f"{tenant_id}:"
        f"{organization_id}:"
        f"{content_hash}"
    )

    version_id = stable_uuid(
        f"version:"
        f"{document_id}:"
        f"{content_hash}"
    )

    extension = (
        path.suffix.lower()
    )

    mime_type = detect_mime_type(
        path
    )

    document_type = infer_document_type(
        path.name
    )

    language = detect_language(
        path
    )

    title = extract_title(
        path
    )

    modified_at = (
        datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat()
    )

    return DocumentMetadata(
        document_id=document_id,
        version_id=version_id,
        filename=path.name,
        original_filename=path.name,
        source_path=str(
            path.resolve()
        ),
        relative_path=str(
            relative_path
        ),
        file_extension=extension,
        mime_type=mime_type,
        size_bytes=stat.st_size,
        sha256=content_hash,
        tenant_id=tenant_id,
        organization_id=organization_id,
        document_type=document_type,
        language=language,
        title=title,
        modified_at=modified_at,
        metadata={
            "ingestion_source": "script",
            "ingested_at": utc_now(),
            "source_root": str(
                source_root.resolve()
            ),
        },
    )


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


def copy_document(
    source: Path,
    destination_root: Path,
    relative_path: Path,
) -> Path:
    """
    Copy an ingested document while preserving
    its relative directory structure.
    """

    destination = (
        destination_root
        / relative_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Avoid overwriting an unrelated file.
    if destination.exists():
        source_hash = sha256_file(
            source
        )

        destination_hash = sha256_file(
            destination
        )

        if source_hash != destination_hash:
            stem = destination.stem
            suffix = destination.suffix

            destination = (
                destination.parent
                / (
                    f"{stem}"
                    f"-{source_hash[:12]}"
                    f"{suffix}"
                )
            )

    shutil.copy2(
        source,
        destination,
    )

    return destination


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_file(
    path: Path,
    max_size_bytes: int,
) -> tuple[bool, str]:
    extension = (
        path.suffix.lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            False,
            "unsupported_extension",
        )

    try:
        size = path.stat().st_size
    except OSError:
        return (
            False,
            "stat_failed",
        )

    if size <= 0:
        return (
            False,
            "empty_file",
        )

    if size > max_size_bytes:
        return (
            False,
            "file_too_large",
        )

    return True, "valid"


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def append_records(
    output: Path,
    records: Sequence[DocumentMetadata],
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "a",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
            )
            file.write("\n")

        file.flush()
        os.fsync(file.fileno())


def rewrite_manifest(
    output: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    content = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )

    atomic_write(
        output,
        content,
    )


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------


def ingest_documents(
    source: Path,
    output: Path,
    copy_to: Path | None,
    recursive: bool,
    resume: bool,
    tenant_id: str,
    organization_id: str,
    max_size_bytes: int,
) -> IngestionStats:
    stats = IngestionStats()

    existing = (
        load_existing_hashes(output)
        if resume
        else {}
    )

    current_hashes: set[str] = set()

    records_to_write: list[
        DocumentMetadata
    ] = []

    files = list(
        iter_files(
            source,
            recursive,
        )
    )

    stats.discovered = len(files)

    logger.info(
        "Discovered %d files.",
        stats.discovered,
    )

    source_root = (
        source
        if source.is_dir()
        else source.parent
    )

    for path in files:
        try:
            valid, reason = validate_file(
                path,
                max_size_bytes,
            )

            if not valid:
                stats.skipped += 1

                if reason == (
                    "unsupported_extension"
                ):
                    stats.unsupported += 1

                elif reason == (
                    "file_too_large"
                ):
                    stats.too_large += 1

                logger.warning(
                    "Skipping %s: %s",
                    path,
                    reason,
                )

                continue

            stats.supported += 1

            metadata = build_document_metadata(
                path=path,
                source_root=source_root,
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
            )

            content_hash = (
                metadata.sha256
            )

            if content_hash in current_hashes:
                stats.duplicates += 1

                logger.info(
                    "Duplicate in current batch: %s",
                    path,
                )

                continue

            current_hashes.add(
                content_hash
            )

            if (
                resume
                and content_hash in existing
            ):
                stats.duplicates += 1

                logger.info(
                    "Already ingested: %s",
                    path,
                )

                continue

            if copy_to is not None:
                copied_path = copy_document(
                    source=path,
                    destination_root=copy_to,
                    relative_path=Path(
                        metadata.relative_path
                    ),
                )

                metadata.metadata[
                    "processed_path"
                ] = str(
                    copied_path.resolve()
                )

            records_to_write.append(
                metadata
            )

            stats.ingested += 1

        except Exception:
            stats.errors += 1

            logger.exception(
                "Failed to ingest document: %s",
                path,
            )

    if records_to_write:
        append_records(
            output,
            records_to_write,
        )

    return stats


# ---------------------------------------------------------------------------
# Summary manifest
# ---------------------------------------------------------------------------


def write_summary(
    output: Path,
    stats: IngestionStats,
) -> Path:
    summary_path = output.with_name(
        f"{output.stem}_summary.json"
    )

    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "manifest": str(
            output
        ),
        "statistics": asdict(
            stats
        ),
    }

    atomic_write(
        summary_path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
    )

    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest legal documents into the "
            "enterprise document-intelligence pipeline."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Source file or directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL ingestion manifest.",
    )

    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help=(
            "Optional destination for normalized "
            "document copies."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip files whose content hash already "
            "exists in the manifest."
        ),
    )

    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Tenant identifier.",
    )

    parser.add_argument(
        "--organization-id",
        default=DEFAULT_ORGANIZATION_ID,
        help="Organization identifier.",
    )

    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=(
            MAX_FILE_SIZE_BYTES
            / 1024
            / 1024
        ),
        help="Maximum accepted file size in MB.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if args.max_file_size_mb <= 0:
        raise ValueError(
            "--max-file-size-mb must be greater than zero."
        )

    max_size_bytes = int(
        args.max_file_size_mb
        * 1024
        * 1024
    )

    logger.info(
        "Starting legal document ingestion."
    )

    logger.info(
        "Source: %s",
        args.source,
    )

    logger.info(
        "Output manifest: %s",
        args.output,
    )

    logger.info(
        "Tenant: %s",
        args.tenant_id,
    )

    logger.info(
        "Organization: %s",
        args.organization_id,
    )

    try:
        stats = ingest_documents(
            source=args.source,
            output=args.output,
            copy_to=args.copy_to,
            recursive=args.recursive,
            resume=args.resume,
            tenant_id=args.tenant_id,
            organization_id=(
                args.organization_id
            ),
            max_size_bytes=max_size_bytes,
        )

        summary_path = write_summary(
            args.output,
            stats,
        )

    except Exception:
        logger.exception(
            "Document ingestion failed."
        )
        return 1

    logger.info(
        "Document ingestion completed."
    )

    logger.info(
        "Discovered: %d",
        stats.discovered,
    )

    logger.info(
        "Supported: %d",
        stats.supported,
    )

    logger.info(
        "Ingested: %d",
        stats.ingested,
    )

    logger.info(
        "Skipped: %d",
        stats.skipped,
    )

    logger.info(
        "Duplicates: %d",
        stats.duplicates,
    )

    logger.info(
        "Too large: %d",
        stats.too_large,
    )

    logger.info(
        "Unsupported: %d",
        stats.unsupported,
    )

    logger.info(
        "Errors: %d",
        stats.errors,
    )

    logger.info(
        "Manifest: %s",
        args.output,
    )

    logger.info(
        "Summary: %s",
        summary_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())