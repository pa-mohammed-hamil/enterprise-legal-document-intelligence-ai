"""
Clean, normalize, validate, and optionally quarantine legal documents.

Usage:
    python scripts/cleanup_documents.py
    python scripts/cleanup_documents.py --source data/sample
    python scripts/cleanup_documents.py --source data/sample --apply
    python scripts/cleanup_documents.py --source data/sample --apply --quarantine
    python scripts/cleanup_documents.py --source data/sample --deduplicate
    python scripts/cleanup_documents.py --source data/sample --report-only

Safety:
    - Default mode is report-only.
    - Files are never deleted unless --apply is explicitly supplied.
    - Duplicate files are identified using SHA-256 hashes.
    - Suspicious/unsupported files can be quarantined instead of deleted.
    - Original legal documents are never modified in-place.
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
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("legal-document-cleanup")

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

MAX_DOCUMENT_SIZE_MB = int(
    os.getenv("MAX_DOCUMENT_SIZE_MB", "100")
)

MAX_DOCUMENT_SIZE_BYTES = (
    MAX_DOCUMENT_SIZE_MB * 1024 * 1024
)

DEFAULT_SOURCE = (
    PROJECT_ROOT / "data" / "sample"
)

DEFAULT_QUARANTINE = (
    PROJECT_ROOT
    / "data"
    / "quarantine"
    / "documents"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "cleanup_reports"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DocumentInfo:
    path: str
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    mime_type: str | None
    modified_at: str
    readable: bool
    supported: bool
    empty: bool
    suspicious: bool
    reason: str | None = None


@dataclass(slots=True)
class CleanupAction:
    action: str
    source: str
    destination: str | None = None
    reason: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass(slots=True)
class CleanupReport:
    started_at: str
    finished_at: str | None = None

    scanned: int = 0
    valid: int = 0
    unsupported: int = 0
    empty: int = 0
    oversized: int = 0
    duplicates: int = 0
    unreadable: int = 0
    suspicious: int = 0

    actions: list[CleanupAction] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def calculate_sha256(
    path: Path,
) -> str:
    """Calculate a streaming SHA-256 hash."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def safe_relative_path(
    path: Path,
    root: Path,
) -> str:
    """Return a safe relative path for reports."""

    try:
        return str(
            path.resolve().relative_to(
                root.resolve()
            )
        )
    except ValueError:
        return path.name


def is_hidden(path: Path) -> bool:
    """Detect hidden files."""

    return path.name.startswith(".")


def is_supported(path: Path) -> bool:
    return (
        path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )


def detect_mime_type(
    path: Path,
) -> str | None:
    mime, _ = mimetypes.guess_type(
        str(path)
    )
    return mime


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_files(
    source: Path,
) -> list[Path]:
    """Recursively discover files."""

    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if source.is_file():
        return [source]

    files: list[Path] = []

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        # Ignore temporary/editor files.
        if path.name.startswith("~$"):
            continue

        if path.name.endswith(
            (
                ".tmp",
                ".temp",
                ".swp",
                ".part",
                ".crdownload",
            )
        ):
            continue

        files.append(path)

    return sorted(files)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_file(
    path: Path,
) -> DocumentInfo:
    """Inspect and validate a document."""

    extension = path.suffix.lower()

    try:
        stat = path.stat()
        size_bytes = stat.st_size
        modified_at = datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat()
    except OSError as exc:
        return DocumentInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            size_bytes=0,
            sha256="",
            mime_type=None,
            modified_at="",
            readable=False,
            supported=False,
            empty=False,
            suspicious=True,
            reason=f"stat_failed:{exc}",
        )

    supported = is_supported(path)

    if size_bytes == 0:
        return DocumentInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            size_bytes=0,
            sha256="",
            mime_type=detect_mime_type(path),
            modified_at=modified_at,
            readable=True,
            supported=supported,
            empty=True,
            suspicious=False,
            reason="empty_file",
        )

    if size_bytes > MAX_DOCUMENT_SIZE_BYTES:
        return DocumentInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            size_bytes=size_bytes,
            sha256="",
            mime_type=detect_mime_type(path),
            modified_at=modified_at,
            readable=True,
            supported=supported,
            empty=False,
            suspicious=False,
            reason="file_too_large",
        )

    if not supported:
        return DocumentInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            size_bytes=size_bytes,
            sha256="",
            mime_type=detect_mime_type(path),
            modified_at=modified_at,
            readable=True,
            supported=False,
            empty=False,
            suspicious=False,
            reason="unsupported_extension",
        )

    try:
        with path.open("rb") as file:
            file.read(4096)

        sha256 = calculate_sha256(path)

    except (OSError, PermissionError) as exc:
        return DocumentInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            size_bytes=size_bytes,
            sha256="",
            mime_type=detect_mime_type(path),
            modified_at=modified_at,
            readable=False,
            supported=supported,
            empty=False,
            suspicious=True,
            reason=f"read_failed:{exc}",
        )

    return DocumentInfo(
        path=str(path),
        filename=path.name,
        extension=extension,
        size_bytes=size_bytes,
        sha256=sha256,
        mime_type=detect_mime_type(path),
        modified_at=modified_at,
        readable=True,
        supported=True,
        empty=False,
        suspicious=False,
    )


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------


def validate_pdf_signature(
    path: Path,
) -> bool:
    """Check the basic PDF magic header."""

    try:
        with path.open("rb") as file:
            header = file.read(5)

        return header == b"%PDF-"

    except OSError:
        return False


def validate_zip_container(
    path: Path,
) -> bool:
    """
    DOCX files are ZIP containers.

    This validates the container without extracting
    potentially untrusted contents.
    """

    try:
        import zipfile

        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None

    except (
        OSError,
        zipfile.BadZipFile,
    ):
        return False


def content_integrity_check(
    info: DocumentInfo,
) -> tuple[bool, str | None]:
    """Perform lightweight format-specific integrity checks."""

    path = Path(info.path)

    if not info.supported:
        return True, None

    if info.extension == ".pdf":
        if not validate_pdf_signature(path):
            return False, "invalid_pdf_signature"

    elif info.extension == ".docx":
        if not validate_zip_container(path):
            return False, "invalid_docx_container"

    return True, None


# ---------------------------------------------------------------------------
# Filename normalization
# ---------------------------------------------------------------------------


def normalize_filename(
    filename: str,
) -> str:
    """
    Normalize filename without changing its extension.

    Legal documents are copied rather than renamed in-place.
    """

    filename = filename.strip()

    # Replace unsafe path characters.
    unsafe = '<>:"/\\|?*'

    for character in unsafe:
        filename = filename.replace(
            character,
            "_",
        )

    # Remove control characters.
    filename = "".join(
        character
        for character in filename
        if ord(character) >= 32
    )

    # Avoid reserved Windows device names.
    stem = Path(filename).stem.upper()

    if stem in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }:
        filename = f"document_{filename}"

    return filename[:240]


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def find_duplicates(
    documents: Iterable[DocumentInfo],
) -> dict[str, list[DocumentInfo]]:
    """Group documents by SHA-256."""

    groups: dict[
        str,
        list[DocumentInfo],
    ] = {}

    for document in documents:
        if not document.sha256:
            continue

        groups.setdefault(
            document.sha256,
            [],
        ).append(document)

    return {
        digest: items
        for digest, items in groups.items()
        if len(items) > 1
    }


def choose_duplicate_keeper(
    documents: list[DocumentInfo],
) -> DocumentInfo:
    """
    Pick the canonical duplicate.

    Preference:
        1. Supported document.
        2. Non-hidden filename.
        3. Shorter path.
        4. Lexicographically stable path.
    """

    return sorted(
        documents,
        key=lambda item: (
            not item.supported,
            item.filename.startswith("."),
            len(item.path),
            item.path.lower(),
        ),
    )[0]


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def quarantine_file(
    path: Path,
    quarantine_root: Path,
    reason: str,
) -> Path:
    """
    Move a suspicious/invalid file to quarantine.

    A UUID suffix prevents collisions.
    """

    quarantine_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = normalize_filename(
        path.name
    )

    destination = (
        quarantine_root
        / f"{path.stem}_{uuid.uuid4().hex[:10]}"
        f"{path.suffix.lower()}"
    )

    shutil.move(
        str(path),
        str(destination),
    )

    metadata_path = destination.with_suffix(
        destination.suffix + ".json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "original_path": str(path),
                "reason": reason,
                "quarantined_at": utc_now(),
                "filename": safe_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return destination


# ---------------------------------------------------------------------------
# Safe deletion
# ---------------------------------------------------------------------------


def delete_file(
    path: Path,
) -> None:
    """Delete one file after explicit --apply."""

    path.unlink()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    report: CleanupReport,
    report_dir: Path,
) -> Path:
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    report_path = (
        report_dir
        / f"cleanup_{timestamp}.json"
    )

    report_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_path = (
        report_dir
        / "latest_cleanup.json"
    )

    latest_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    return report_path


# ---------------------------------------------------------------------------
# Cleanup engine
# ---------------------------------------------------------------------------


class DocumentCleanup:
    """Safe legal-document cleanup engine."""

    def __init__(
        self,
        source: Path,
        quarantine_root: Path,
        apply: bool,
        quarantine: bool,
        remove_duplicates: bool,
    ) -> None:
        self.source = source
        self.quarantine_root = (
            quarantine_root
        )
        self.apply = apply
        self.quarantine = quarantine
        self.remove_duplicates = (
            remove_duplicates
        )

        self.report = CleanupReport(
            started_at=utc_now()
        )

    def record(
        self,
        action: str,
        source: Path,
        destination: Path | None = None,
        reason: str | None = None,
    ) -> None:
        self.report.actions.append(
            CleanupAction(
                action=action,
                source=str(source),
                destination=(
                    str(destination)
                    if destination
                    else None
                ),
                reason=reason,
            )
        )

    def handle_problem(
        self,
        info: DocumentInfo,
    ) -> None:
        """Handle invalid/problematic documents."""

        path = Path(info.path)
        reason = info.reason or "validation_failed"

        if self.quarantine and self.apply:
            destination = quarantine_file(
                path=path,
                quarantine_root=self.quarantine_root,
                reason=reason,
            )

            self.record(
                action="quarantine",
                source=path,
                destination=destination,
                reason=reason,
            )

            logger.warning(
                "Quarantined %s -> %s (%s)",
                path,
                destination,
                reason,
            )

        elif self.apply and self.remove_duplicates:
            delete_file(path)

            self.record(
                action="delete",
                source=path,
                reason=reason,
            )

            logger.warning(
                "Deleted %s (%s)",
                path,
                reason,
            )

        else:
            self.record(
                action="report",
                source=path,
                reason=reason,
            )

            logger.warning(
                "Would handle %s (%s)",
                path,
                reason,
            )

    def scan(self) -> list[DocumentInfo]:
        """Scan and validate all files."""

        files = discover_files(
            self.source
        )

        self.report.scanned = len(files)

        logger.info(
            "Discovered %d files.",
            len(files),
        )

        results: list[DocumentInfo] = []

        for path in files:
            info = validate_file(path)

            results.append(info)

            if not info.supported:
                self.report.unsupported += 1

            elif info.empty:
                self.report.empty += 1

            elif not info.readable:
                self.report.unreadable += 1

            elif info.suspicious:
                self.report.suspicious += 1

            elif info.reason == "file_too_large":
                self.report.oversized += 1

            else:
                valid, reason = (
                    content_integrity_check(info)
                )

                if not valid:
                    info.suspicious = True
                    info.reason = reason
                    self.report.suspicious += 1

                else:
                    self.report.valid += 1

        return results

    def process_duplicates(
        self,
        documents: list[DocumentInfo],
    ) -> None:
        """Detect and optionally remove duplicates."""

        duplicate_groups = find_duplicates(
            documents
        )

        for digest, group in duplicate_groups.items():
            keeper = choose_duplicate_keeper(
                group
            )

            duplicates = [
                item
                for item in group
                if item.path != keeper.path
            ]

            self.report.duplicates += len(
                duplicates
            )

            logger.info(
                "Duplicate group %s: keeper=%s duplicates=%d",
                digest[:12],
                keeper.filename,
                len(duplicates),
            )

            for duplicate in duplicates:
                path = Path(
                    duplicate.path
                )

                reason = (
                    f"duplicate_of:{keeper.path}"
                )

                if self.quarantine and self.apply:
                    destination = quarantine_file(
                        path=path,
                        quarantine_root=(
                            self.quarantine_root
                            / "duplicates"
                        ),
                        reason=reason,
                    )

                    self.record(
                        action="quarantine_duplicate",
                        source=path,
                        destination=destination,
                        reason=reason,
                    )

                elif (
                    self.apply
                    and self.remove_duplicates
                ):
                    delete_file(path)

                    self.record(
                        action="delete_duplicate",
                        source=path,
                        reason=reason,
                    )

                else:
                    self.record(
                        action="report_duplicate",
                        source=path,
                        reason=reason,
                    )

    def run(self) -> CleanupReport:
        """Run cleanup."""

        documents = self.scan()

        for info in documents:
            if (
                not info.supported
                or info.empty
                or not info.readable
                or info.suspicious
                or info.reason == "file_too_large"
            ):
                self.handle_problem(info)

        valid_documents = [
            info
            for info in documents
            if (
                info.supported
                and info.readable
                and not info.empty
                and not info.suspicious
                and info.reason != "file_too_large"
                and info.sha256
            )
        ]

        self.process_duplicates(
            valid_documents
        )

        self.report.finished_at = utc_now()

        return self.report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely clean and validate legal "
            "documents before RAG ingestion."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Document file or directory.",
    )

    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=DEFAULT_QUARANTINE,
        help="Quarantine destination.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Cleanup report directory.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually perform cleanup actions. "
            "Without this flag the script is report-only."
        ),
    )

    parser.add_argument(
        "--quarantine",
        action="store_true",
        help=(
            "Move problematic/duplicate documents "
            "to quarantine instead of deleting them."
        ),
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help=(
            "Enable duplicate cleanup when --apply "
            "is supplied."
        ),
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Explicitly force report-only mode.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Explicit report-only always wins.
    apply = (
        args.apply
        and not args.report_only
    )

    logger.info(
        "Enterprise Legal Document cleanup started."
    )

    logger.info(
        "Source: %s",
        args.source,
    )

    logger.info(
        "Mode: %s",
        "APPLY" if apply else "REPORT-ONLY",
    )

    cleanup = DocumentCleanup(
        source=args.source,
        quarantine_root=args.quarantine_dir,
        apply=apply,
        quarantine=args.quarantine,
        remove_duplicates=args.deduplicate,
    )

    try:
        report = cleanup.run()

    except Exception:
        logger.exception(
            "Document cleanup failed."
        )
        return 1

    report_path = write_report(
        report=report,
        report_dir=args.report_dir,
    )

    logger.info(
        "Cleanup completed."
    )

    logger.info(
        "Files scanned: %d",
        report.scanned,
    )

    logger.info(
        "Valid documents: %d",
        report.valid,
    )

    logger.info(
        "Unsupported: %d",
        report.unsupported,
    )

    logger.info(
        "Empty: %d",
        report.empty,
    )

    logger.info(
        "Oversized: %d",
        report.oversized,
    )

    logger.info(
        "Unreadable: %d",
        report.unreadable,
    )

    logger.info(
        "Suspicious: %d",
        report.suspicious,
    )

    logger.info(
        "Duplicates: %d",
        report.duplicates,
    )

    logger.info(
        "Actions recorded: %d",
        len(report.actions),
    )

    logger.info(
        "Report: %s",
        report_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())