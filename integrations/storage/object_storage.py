"""
Object storage integration.

Provides an S3-compatible object storage abstraction for:
- Legal document files
- Document versions
- Extracted artifacts
- OCR outputs
- Generated reports
- Temporary downloads

Business logic should remain in the service layer.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import BinaryIO, Mapping

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from backend.config import settings

logger = logging.getLogger(__name__)


class ObjectStorageError(Exception):
    """Base exception for object-storage failures."""


class ObjectStorageConfigurationError(ObjectStorageError):
    """Raised when storage configuration is invalid."""


class ObjectNotFoundError(ObjectStorageError):
    """Raised when an object does not exist."""


class ObjectStorageUploadError(ObjectStorageError):
    """Raised when an object cannot be uploaded."""


class ObjectStorageDownloadError(ObjectStorageError):
    """Raised when an object cannot be downloaded."""


@dataclass(frozen=True)
class StoredObject:
    """Metadata describing an object in storage."""

    key: str
    bucket: str
    content_type: str
    size: int | None = None
    etag: str | None = None
    metadata: Mapping[str, str] | None = None


class ObjectStorage:
    """
    Provider-agnostic object-storage client.

    Current implementation uses an S3-compatible backend.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self.bucket = bucket or getattr(
            settings,
            "OBJECT_STORAGE_BUCKET",
            None,
        )

        self.region = region or getattr(
            settings,
            "AWS_REGION",
            "us-east-1",
        )

        self.endpoint_url = endpoint_url or getattr(
            settings,
            "OBJECT_STORAGE_ENDPOINT",
            None,
        )

        self.access_key = access_key or getattr(
            settings,
            "AWS_ACCESS_KEY_ID",
            None,
        )

        self.secret_key = secret_key or getattr(
            settings,
            "AWS_SECRET_ACCESS_KEY",
            None,
        )

        self.client = self._create_client()

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(
        self,
        *,
        key: str,
        file: BinaryIO | bytes,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        """
        Upload an object.

        Args:
            key: Storage object key.
            file: Bytes or binary file-like object.
            content_type: MIME type.
            metadata: Optional object metadata.

        Returns:
            StoredObject describing the uploaded object.
        """
        self._validate_key(key)

        if isinstance(file, bytes):
            file = io.BytesIO(file)

        extra_args: dict[str, object] = {
            "ContentType": content_type,
        }

        if metadata:
            extra_args["Metadata"] = dict(metadata)

        try:
            self.client.upload_fileobj(
                Fileobj=file,
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra_args,
            )

            logger.info(
                "Object uploaded",
                extra={
                    "bucket": self.bucket,
                    "key": key,
                    "content_type": content_type,
                },
            )

            return StoredObject(
                key=key,
                bucket=self.bucket,
                content_type=content_type,
                metadata=metadata,
            )

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "Object upload failed",
                extra={"key": key},
            )

            raise ObjectStorageUploadError(
                f"Failed to upload object: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        *,
        key: str,
    ) -> bytes:
        """Download an object as bytes."""
        self._validate_key(key)

        buffer = io.BytesIO()

        try:
            self.client.download_fileobj(
                Bucket=self.bucket,
                Key=key,
                Fileobj=buffer,
            )

            return buffer.getvalue()

        except ClientError as exc:
            if self._is_not_found(exc):
                raise ObjectNotFoundError(
                    f"Object not found: {key}"
                ) from exc

            logger.exception(
                "Object download failed",
                extra={"key": key},
            )

            raise ObjectStorageDownloadError(
                f"Failed to download object: {key}"
            ) from exc

        except BotoCoreError as exc:
            logger.exception(
                "Object download failed",
                extra={"key": key},
            )

            raise ObjectStorageDownloadError(
                f"Failed to download object: {key}"
            ) from exc

    def download_to_file(
        self,
        *,
        key: str,
        destination: BinaryIO,
    ) -> None:
        """Stream an object directly into a file-like object."""
        self._validate_key(key)

        try:
            self.client.download_fileobj(
                Bucket=self.bucket,
                Key=key,
                Fileobj=destination,
            )

        except ClientError as exc:
            if self._is_not_found(exc):
                raise ObjectNotFoundError(
                    f"Object not found: {key}"
                ) from exc

            raise ObjectStorageDownloadError(
                f"Failed to download object: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def head(
        self,
        *,
        key: str,
    ) -> StoredObject:
        """Retrieve object metadata without downloading its contents."""
        self._validate_key(key)

        try:
            response = self.client.head_object(
                Bucket=self.bucket,
                Key=key,
            )

            return StoredObject(
                key=key,
                bucket=self.bucket,
                content_type=response.get(
                    "ContentType",
                    "application/octet-stream",
                ),
                size=response.get("ContentLength"),
                etag=response.get("ETag"),
                metadata=response.get("Metadata"),
            )

        except ClientError as exc:
            if self._is_not_found(exc):
                raise ObjectNotFoundError(
                    f"Object not found: {key}"
                ) from exc

            raise ObjectStorageError(
                f"Failed to retrieve metadata: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def exists(
        self,
        *,
        key: str,
    ) -> bool:
        """Return whether an object exists."""
        self._validate_key(key)

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=key,
            )
            return True

        except ClientError as exc:
            if self._is_not_found(exc):
                return False

            raise ObjectStorageError(
                f"Unable to check object: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        *,
        key: str,
    ) -> bool:
        """Delete an object."""
        self._validate_key(key)

        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=key,
            )

            logger.info(
                "Object deleted",
                extra={
                    "bucket": self.bucket,
                    "key": key,
                },
            )

            return True

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "Object deletion failed",
                extra={"key": key},
            )

            raise ObjectStorageError(
                f"Failed to delete object: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Presigned URLs
    # ------------------------------------------------------------------

    def generate_download_url(
        self,
        *,
        key: str,
        expires_in: int = 900,
    ) -> str:
        """
        Generate a temporary presigned download URL.

        Default expiration: 15 minutes.
        """
        self._validate_key(key)

        try:
            return self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": key,
                },
                ExpiresIn=expires_in,
            )

        except (BotoCoreError, ClientError) as exc:
            raise ObjectStorageError(
                f"Failed to generate download URL: {key}"
            ) from exc

    def generate_upload_url(
        self,
        *,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 900,
    ) -> str:
        """
        Generate a temporary presigned upload URL.
        """
        self._validate_key(key)

        try:
            return self.client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )

        except (BotoCoreError, ClientError) as exc:
            raise ObjectStorageError(
                f"Failed to generate upload URL: {key}"
            ) from exc

    # ------------------------------------------------------------------
    # Copy / move
    # ------------------------------------------------------------------

    def copy(
        self,
        *,
        source_key: str,
        destination_key: str,
    ) -> StoredObject:
        """Copy an object within the configured bucket."""
        self._validate_key(source_key)
        self._validate_key(destination_key)

        try:
            self.client.copy_object(
                Bucket=self.bucket,
                CopySource={
                    "Bucket": self.bucket,
                    "Key": source_key,
                },
                Key=destination_key,
            )

            metadata = self.head(key=destination_key)

            return metadata

        except (BotoCoreError, ClientError) as exc:
            raise ObjectStorageError(
                f"Failed to copy object: {source_key}"
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify that the configured bucket is accessible."""
        try:
            self.client.head_bucket(
                Bucket=self.bucket,
            )
            return True

        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "Object storage health check failed"
            )

            raise ObjectStorageError(
                "Object storage is unavailable."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_client(self) -> BaseClient:
        """Create the S3-compatible client."""
        if not self.bucket:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_BUCKET is not configured."
            )

        client_kwargs: dict[str, object] = {
            "region_name": self.region,
        }

        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        if self.access_key:
            client_kwargs["aws_access_key_id"] = self.access_key

        if self.secret_key:
            client_kwargs["aws_secret_access_key"] = self.secret_key

        return boto3.client(
            "s3",
            **client_kwargs,
        )

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validate an object key."""
        if not key or not key.strip():
            raise ObjectStorageError(
                "Object key cannot be empty."
            )

        if key.startswith("/"):
            raise ObjectStorageError(
                "Object key must not start with '/'."
            )

    @staticmethod
    def _is_not_found(exc: ClientError) -> bool:
        """Check whether an AWS error represents a missing object."""
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))

        return code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }


def get_object_storage() -> ObjectStorage:
    """Return the application object-storage client."""
    return ObjectStorage()