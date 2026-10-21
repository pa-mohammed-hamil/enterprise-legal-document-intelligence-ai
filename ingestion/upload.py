# backend/ingestion/upload.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Optional
import hashlib
import mimetypes
import os
import re
import tempfile
import uuid


class UploadStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class UploadErrorCode(str, Enum):
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNSUPPORTED_MIME_TYPE = "unsupported_mime_type"
    MIME_MISMATCH = "mime_mismatch"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_FILENAME = "invalid_filename"
    MALICIOUS_FILENAME = "malicious_filename"
    STORAGE_ERROR = "storage_error"
    INVALID_CONTENT = "invalid_content"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class UploadConfig:
    max_file_size_bytes: int = 50 * 1024 * 1024

    allowed_extensions: frozenset[str] = frozenset(
        {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".csv",
            ".xlsx",
            ".pptx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
        }
    )

    allowed_mime_types: frozenset[str] = frozenset(
        {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation",
            "text/plain",
            "text/markdown",
            "text/html",
            "text/csv",
            "image/png",
            "image/jpeg",
            "image/tiff",
            "application/octet-stream",
        }
    )

    sanitize_filename: bool = True
    validate_signature: bool = True
    validate_mime_type: bool = True

    calculate_sha256: bool = True

    preserve_original_filename: bool = True

    temporary_storage: bool = True

    storage_directory: Optional[Path] = None

    quarantine_suspicious_files: bool = True

    max_filename_length: int = 255


@dataclass(slots=True)
class UploadedDocument:
    upload_id: str

    original_filename: str
    safe_filename: str

    extension: str
    mime_type: Optional[str]

    size_bytes: int

    sha256: Optional[str]

    status: UploadStatus

    content: Optional[bytes] = None

    storage_path: Optional[str] = None

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:

        data = {
            "upload_id": self.upload_id,
            "original_filename":
                self.original_filename,
            "safe_filename":
                self.safe_filename,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status.value,
            "storage_path": self.storage_path,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

        if include_content:
            data["content"] = self.content

        return data


class UploadError(Exception):

    def __init__(
        self,
        message: str,
        *,
        code: UploadErrorCode = (
            UploadErrorCode.UNKNOWN
        ),
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.cause = cause


class UploadValidator:
    """
    Performs validation before a document enters the ingestion
    pipeline.

    This is intentionally separated from DocumentParser.

    UploadValidator
        -> validates transport/file safety

    DocumentParser
        -> understands document structure/content
    """

    MIME_BY_EXTENSION = {
        ".pdf": {
            "application/pdf",
        },
        ".docx": {
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
        },
        ".xlsx": {
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet",
            "application/zip",
            "application/octet-stream",
        },
        ".pptx": {
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation",
            "application/zip",
            "application/octet-stream",
        },
        ".txt": {
            "text/plain",
            "application/octet-stream",
        },
        ".md": {
            "text/markdown",
            "text/plain",
            "application/octet-stream",
        },
        ".html": {
            "text/html",
            "application/octet-stream",
        },
        ".htm": {
            "text/html",
            "application/octet-stream",
        },
        ".csv": {
            "text/csv",
            "text/plain",
            "application/octet-stream",
        },
        ".png": {
            "image/png",
        },
        ".jpg": {
            "image/jpeg",
        },
        ".jpeg": {
            "image/jpeg",
        },
        ".tif": {
            "image/tiff",
        },
        ".tiff": {
            "image/tiff",
        },
    }

    SIGNATURES = {
        ".pdf": (
            b"%PDF-",
        ),
        ".png": (
            b"\x89PNG\r\n\x1a\n",
        ),
        ".jpg": (
            b"\xff\xd8\xff",
        ),
        ".jpeg": (
            b"\xff\xd8\xff",
        ),
        ".tif": (
            b"II*\x00",
            b"MM\x00*",
        ),
        ".tiff": (
            b"II*\x00",
            b"MM\x00*",
        ),
        # OOXML formats are ZIP containers.
        ".docx": (
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
        ),
        ".xlsx": (
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
        ),
        ".pptx": (
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
        ),
    }

    def __init__(
        self,
        config: Optional[UploadConfig] = None,
    ) -> None:
        self.config = config or UploadConfig()

    def validate(
        self,
        content: bytes,
        *,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> list[str]:

        warnings: list[str] = []

        self._validate_content(content)

        safe_name = self.sanitize_filename(
            filename
        )

        extension = Path(
            safe_name
        ).suffix.lower()

        self._validate_extension(
            extension
        )

        if mime_type:
            normalized_mime = (
                mime_type
                .split(";", 1)[0]
                .strip()
                .lower()
            )

            if self.config.validate_mime_type:
                self._validate_mime(
                    extension,
                    normalized_mime,
                )

        else:
            warnings.append(
                "Client MIME type was not provided."
            )

        if self.config.validate_signature:
            self._validate_signature(
                content,
                extension,
            )

        return warnings

    def _validate_content(
        self,
        content: bytes,
    ) -> None:

        if not isinstance(
            content,
            bytes,
        ):
            raise UploadError(
                "Uploaded content must be bytes.",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
            )

        if not content:
            raise UploadError(
                "Uploaded file is empty.",
                code=(
                    UploadErrorCode
                    .EMPTY_FILE
                ),
            )

        if (
            len(content)
            > self.config.max_file_size_bytes
        ):
            raise UploadError(
                "Uploaded file exceeds the "
                "configured size limit.",
                code=(
                    UploadErrorCode
                    .FILE_TOO_LARGE
                ),
            )

    def _validate_extension(
        self,
        extension: str,
    ) -> None:

        if (
            extension
            not in self.config.allowed_extensions
        ):
            raise UploadError(
                f"Unsupported file extension: "
                f"{extension or 'none'}",
                code=(
                    UploadErrorCode
                    .UNSUPPORTED_EXTENSION
                ),
            )

    def _validate_mime(
        self,
        extension: str,
        mime_type: str,
    ) -> None:

        if (
            mime_type
            not in self.config.allowed_mime_types
        ):
            raise UploadError(
                f"Unsupported MIME type: "
                f"{mime_type}",
                code=(
                    UploadErrorCode
                    .UNSUPPORTED_MIME_TYPE
                ),
            )

        expected = (
            self.MIME_BY_EXTENSION.get(
                extension
            )
        )

        if (
            expected
            and mime_type not in expected
        ):
            raise UploadError(
                "Uploaded MIME type does not "
                "match the file extension.",
                code=(
                    UploadErrorCode
                    .MIME_MISMATCH
                ),
            )

    def _validate_signature(
        self,
        content: bytes,
        extension: str,
    ) -> None:

        signatures = self.SIGNATURES.get(
            extension
        )

        # Text formats intentionally do not have
        # reliable magic signatures.
        if not signatures:
            return

        if not any(
            content.startswith(signature)
            for signature in signatures
        ):
            raise UploadError(
                "File signature does not match "
                f"extension {extension}.",
                code=(
                    UploadErrorCode
                    .INVALID_SIGNATURE
                ),
            )

    def sanitize_filename(
        self,
        filename: str,
    ) -> str:

        if not filename:
            raise UploadError(
                "Filename is required.",
                code=(
                    UploadErrorCode
                    .INVALID_FILENAME
                ),
            )

        # Remove path traversal.
        filename = (
            filename
            .replace("\\", "/")
            .split("/")[-1]
        )

        filename = filename.strip()

        if not filename:
            raise UploadError(
                "Invalid filename.",
                code=(
                    UploadErrorCode
                    .INVALID_FILENAME
                ),
            )

        if "\x00" in filename:
            raise UploadError(
                "Filename contains a null byte.",
                code=(
                    UploadErrorCode
                    .MALICIOUS_FILENAME
                ),
            )

        if not self.config.sanitize_filename:
            return filename

        # Keep useful filename characters while
        # removing shell/path/control characters.
        filename = re.sub(
            r"[\x00-\x1f\x7f]",
            "",
            filename,
        )

        filename = re.sub(
            r"[^A-Za-z0-9._()\- ]",
            "_",
            filename,
        )

        filename = re.sub(
            r"\s+",
            " ",
            filename,
        ).strip()

        # Avoid hidden filenames.
        filename = filename.lstrip(".")

        if not filename:
            raise UploadError(
                "Filename became empty after "
                "sanitization.",
                code=(
                    UploadErrorCode
                    .INVALID_FILENAME
                ),
            )

        if (
            len(filename)
            > self.config.max_filename_length
        ):
            extension = Path(
                filename
            ).suffix

            stem_limit = (
                self.config.max_filename_length
                - len(extension)
            )

            filename = (
                Path(filename).stem[
                    :stem_limit
                ]
                + extension
            )

        return filename


class UploadService:
    """
    Secure upload entry point for document ingestion.

    Flow:

        HTTP upload
             ↓
        UploadService
             ↓
        UploadValidator
             ↓
        SHA-256 / metadata
             ↓
        optional storage
             ↓
        DocumentParser
             ↓
        IngestionPipeline
    """

    def __init__(
        self,
        config: Optional[UploadConfig] = None,
        *,
        duplicate_checker: Any = None,
    ) -> None:

        self.config = config or UploadConfig()

        self.validator = UploadValidator(
            self.config
        )

        self.duplicate_checker = (
            duplicate_checker
        )

    def upload(
        self,
        source: bytes | bytearray | BinaryIO,
        *,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        store: bool = False,
    ) -> UploadedDocument:

        upload_id = str(
            uuid.uuid4()
        )

        try:
            content = self._read_content(
                source
            )

            safe_filename = (
                self.validator
                .sanitize_filename(
                    filename
                )
            )

            warnings = (
                self.validator.validate(
                    content,
                    filename=safe_filename,
                    mime_type=mime_type,
                )
            )

            extension = (
                Path(safe_filename)
                .suffix
                .lower()
            )

            detected_mime = (
                self._resolve_mime_type(
                    safe_filename,
                    mime_type,
                )
            )

            sha256 = (
                hashlib.sha256(
                    content
                ).hexdigest()
                if self.config.calculate_sha256
                else None
            )

            if (
                sha256
                and self._is_duplicate(
                    sha256
                )
            ):
                return UploadedDocument(
                    upload_id=upload_id,
                    original_filename=filename,
                    safe_filename=safe_filename,
                    extension=extension,
                    mime_type=detected_mime,
                    size_bytes=len(content),
                    sha256=sha256,
                    status=(
                        UploadStatus.DUPLICATE
                    ),
                    content=content,
                    warnings=[
                        *warnings,
                        "A document with the same "
                        "SHA-256 already exists.",
                    ],
                    metadata=metadata or {},
                )

            storage_path = None

            if store:
                storage_path = (
                    self._store_file(
                        content,
                        safe_filename,
                        upload_id,
                    )
                )

            return UploadedDocument(
                upload_id=upload_id,
                original_filename=filename,
                safe_filename=safe_filename,
                extension=extension,
                mime_type=detected_mime,
                size_bytes=len(content),
                sha256=sha256,
                status=UploadStatus.ACCEPTED,
                content=content,
                storage_path=storage_path,
                warnings=warnings,
                metadata={
                    **(metadata or {}),
                    "upload_id": upload_id,
                },
            )

        except UploadError:
            raise

        except Exception as exc:
            raise UploadError(
                "Document upload failed.",
                code=(
                    UploadErrorCode
                    .UNKNOWN
                ),
                cause=exc,
            ) from exc

    def upload_file(
        self,
        path: str | Path,
        *,
        mime_type: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        store: bool = False,
    ) -> UploadedDocument:

        path = Path(path)

        if not path.exists():
            raise UploadError(
                f"File does not exist: {path}",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
            )

        if not path.is_file():
            raise UploadError(
                f"Source is not a file: {path}",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
            )

        try:
            content = path.read_bytes()

        except OSError as exc:
            raise UploadError(
                f"Unable to read file: {path}",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
                cause=exc,
            ) from exc

        return self.upload(
            content,
            filename=path.name,
            mime_type=mime_type,
            metadata=metadata,
            store=store,
        )

    # ==============================================================
    # Storage
    # ==============================================================

    def _store_file(
        self,
        content: bytes,
        filename: str,
        upload_id: str,
    ) -> str:

        try:
            if self.config.storage_directory:
                root = Path(
                    self.config.storage_directory
                )

                root.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            else:
                root = Path(
                    tempfile.gettempdir()
                ) / "legal-ai-uploads"

                root.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            extension = Path(
                filename
            ).suffix.lower()

            # Never use the client filename as the
            # physical storage filename.
            stored_filename = (
                f"{upload_id}{extension}"
            )

            destination = (
                root / stored_filename
            )

            # Defense-in-depth check.
            resolved_root = root.resolve()

            resolved_destination = (
                destination.resolve()
            )

            if (
                resolved_root
                not in resolved_destination.parents
            ):
                raise UploadError(
                    "Unsafe storage path.",
                    code=(
                        UploadErrorCode
                        .STORAGE_ERROR
                    ),
                )

            with destination.open(
                "xb"
            ) as file_obj:
                file_obj.write(content)

            return str(destination)

        except UploadError:
            raise

        except OSError as exc:
            raise UploadError(
                "Unable to store uploaded file.",
                code=(
                    UploadErrorCode
                    .STORAGE_ERROR
                ),
                cause=exc,
            ) from exc

    # ==============================================================
    # Duplicate detection
    # ==============================================================

    def _is_duplicate(
        self,
        sha256: str,
    ) -> bool:

        if self.duplicate_checker is None:
            return False

        try:
            if callable(
                self.duplicate_checker
            ):
                return bool(
                    self.duplicate_checker(
                        sha256
                    )
                )

            if hasattr(
                self.duplicate_checker,
                "exists",
            ):
                return bool(
                    self.duplicate_checker
                    .exists(sha256)
                )

            if hasattr(
                self.duplicate_checker,
                "exists_by_hash",
            ):
                return bool(
                    self.duplicate_checker
                    .exists_by_hash(
                        sha256
                    )
                )

        except Exception:
            # Duplicate-check infrastructure
            # should not silently reject uploads.
            return False

        return False

    # ==============================================================
    # Content handling
    # ==============================================================

    @staticmethod
    def _read_content(
        source: bytes | bytearray | BinaryIO,
    ) -> bytes:

        if isinstance(
            source,
            (bytes, bytearray),
        ):
            return bytes(source)

        if not hasattr(
            source,
            "read",
        ):
            raise UploadError(
                "Upload source must be bytes "
                "or a binary stream.",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
            )

        try:
            position = source.tell()
        except Exception:
            position = None

        try:
            data = source.read()

            if isinstance(
                data,
                str,
            ):
                data = data.encode(
                    "utf-8"
                )

            return bytes(data)

        except Exception as exc:
            raise UploadError(
                "Unable to read uploaded stream.",
                code=(
                    UploadErrorCode
                    .INVALID_CONTENT
                ),
                cause=exc,
            ) from exc

        finally:
            if position is not None:
                try:
                    source.seek(
                        position
                    )
                except Exception:
                    pass

    # ==============================================================
    # MIME
    # ==============================================================

    @staticmethod
    def _resolve_mime_type(
        filename: str,
        supplied_mime: Optional[str],
    ) -> Optional[str]:

        if supplied_mime:
            return (
                supplied_mime
                .split(";", 1)[0]
                .strip()
                .lower()
            )

        guessed, _ = (
            mimetypes.guess_type(
                filename
            )
        )

        return guessed


def upload_document(
    source: bytes | bytearray | BinaryIO,
    *,
    filename: str,
    mime_type: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> UploadedDocument:
    """
    Convenience API for the ingestion layer.
    """

    service = UploadService()

    return service.upload(
        source,
        filename=filename,
        mime_type=mime_type,
        metadata=metadata,
    )