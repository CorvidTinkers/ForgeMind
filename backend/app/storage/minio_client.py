"""
ForgeMind MinIO Storage Client.

S3-compatible object storage abstraction. All dataset files
are stored in MinIO, never on local disk.
"""

import io
import logging
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MinioStorageClient:
    """Abstraction over MinIO for dataset file storage."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_use_ssl,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        """Create the default bucket if it does not exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created MinIO bucket: %s", self.bucket)
        except S3Error as e:
            logger.error("Failed to ensure MinIO bucket: %s", e)
            raise

    def upload_file(
        self,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload a file to MinIO.

        Args:
            object_name: The destination path in the bucket.
            data: File-like binary stream.
            length: Size in bytes.
            content_type: MIME type of the file.

        Returns:
            The full storage path (bucket/object_name).
        """
        self.ensure_bucket()
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=data,
            length=length,
            content_type=content_type,
        )
        storage_path = f"{self.bucket}/{object_name}"
        logger.info("Uploaded file to MinIO: %s", storage_path)
        return storage_path

    def download_file(self, object_name: str) -> bytes:
        """
        Download a file from MinIO and return its content as bytes.

        Args:
            object_name: The object path in the bucket.

        Returns:
            File content as bytes.
        """
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error("Failed to download from MinIO: %s", e)
            raise

    def get_file_stream(self, object_name: str) -> io.BytesIO:
        """
        Download a file and return as a seekable BytesIO stream.

        Args:
            object_name: The object path in the bucket.

        Returns:
            BytesIO stream of the file content.
        """
        data = self.download_file(object_name)
        return io.BytesIO(data)


def get_minio_client() -> MinioStorageClient:
    """Factory function for dependency injection."""
    return MinioStorageClient()
