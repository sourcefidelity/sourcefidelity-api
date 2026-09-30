"""Storage backend abstraction.

Supports S3-compatible storage (AWS S3, MinIO, DigitalOcean Spaces).
Seafile backend is deferred — interface exists for future implementation.
"""

import logging
from abc import ABC, abstractmethod

import boto3
from botocore.exceptions import ClientError

from app.config import secret_value, settings
from app.log_safety import private_value_id
from app.services.upload_completion import UploadReceipt

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    def upload(self, file_bytes: bytes, key: str) -> str:
        """Upload a file. Return the storage key."""
        ...

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download a file. Return raw bytes.

        Raises FileNotFoundError if key does not exist.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a file. Return True if deleted (or already absent)."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """List all keys with the given prefix."""
        ...

    def list_keys_page(self, prefix: str, *, start_after: str = "", limit: int = 100) -> list[str]:
        """Bounded lexical page; adapters may override to avoid full enumeration."""
        return [key for key in sorted(self.list_keys(prefix)) if key > start_after][:limit]


class S3Backend(StorageBackend):
    """S3-compatible storage backend (AWS S3, MinIO, etc.)."""

    def __init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=secret_value(settings.S3_ACCESS_KEY),
            aws_secret_access_key=secret_value(settings.S3_SECRET_KEY),
            region_name=settings.S3_REGION,
        )
        self._bucket = settings.S3_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)
            logger.info("Created S3 bucket (%s)", private_value_id("bucket", self._bucket))

    def upload(self, file_bytes: bytes, key: str) -> str:
        response = self._client.put_object(Bucket=self._bucket, Key=key, Body=file_bytes)
        logger.info(
            "Uploaded object (%s, %d bytes)",
            private_value_id("storage_key", key),
            len(file_bytes),
        )
        retries = response.get("ResponseMetadata", {}).get("RetryAttempts")
        return UploadReceipt(key, confirmed=type(retries) is int and retries == 0)

    def download(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError("Stored object was not found") from e
            raise

    def delete(self, key: str) -> bool:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            # Permission/transport failures do not establish object absence.
            raise

    def list_keys(self, prefix: str) -> list[str]:
        response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def list_keys_page(self, prefix: str, *, start_after: str = "", limit: int = 100) -> list[str]:
        response = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix, StartAfter=start_after,
            MaxKeys=max(1, min(limit, 1000)),
        )
        return [obj["Key"] for obj in response.get("Contents", [])]


# ── Singleton ──────────────────────────────────────────────

_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend singleton."""
    global _backend
    if _backend is None:
        backend_type = settings.STORAGE_BACKEND.lower()
        if backend_type == "s3":
            _backend = S3Backend()
        elif backend_type == "seafile":
            raise NotImplementedError(
                "Seafile backend is not yet implemented. Use STORAGE_BACKEND='s3'."
            )
        else:
            raise ValueError(f"Unknown storage backend: {settings.STORAGE_BACKEND}")
    return _backend
