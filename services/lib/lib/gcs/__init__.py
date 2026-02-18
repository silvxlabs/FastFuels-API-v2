"""
lib.gcs - Synchronous Google Cloud Storage operations

Provides blob operations and signed URL generation for background jobs.
"""

from lib.gcs.blobs import (
    delete_directory,
    delete_file,
    download_file,
    exists,
    gcsfs_client,
    upload_file,
)
from lib.gcs.signed_urls import (
    gcs_client,
    generate_download_signed_url,
    generate_upload_signed_url,
)

__all__ = [
    "gcs_client",
    "gcsfs_client",
    "upload_file",
    "download_file",
    "delete_file",
    "delete_directory",
    "exists",
    "generate_upload_signed_url",
    "generate_download_signed_url",
]
