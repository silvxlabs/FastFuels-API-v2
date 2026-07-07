"""
lib.gcs - Synchronous Google Cloud Storage operations

Provides blob operations and signed URL generation for background jobs.
"""

from lib.gcs.blobs import (
    delete_directory,
    delete_file,
    download_file,
    exists,
    get_gcsfs_client,
    storage_size,
    upload_file,
)
from lib.gcs.signed_urls import (
    gcs_client,
    generate_download_signed_url,
    generate_upload_signed_url,
    upload_required_headers,
)

__all__ = [
    "gcs_client",
    "get_gcsfs_client",
    "upload_file",
    "download_file",
    "delete_file",
    "delete_directory",
    "exists",
    "storage_size",
    "generate_upload_signed_url",
    "generate_download_signed_url",
    "upload_required_headers",
]
