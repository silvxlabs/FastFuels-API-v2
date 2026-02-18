"""
Synchronous GCS blob operations.

Provides file upload, download, delete, and existence checking.
"""

import gcsfs

gcsfs_client: gcsfs.GCSFileSystem = gcsfs.GCSFileSystem()


def _normalize_path(gcs_path: str) -> str:
    """Remove gs:// prefix if present."""
    if gcs_path.startswith("gs://"):
        return gcs_path[5:]
    return gcs_path


def upload_file(local_path: str, gcs_path: str) -> None:
    """
    Upload a local file to GCS.

    Args:
        local_path: Path to local file.
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
    """
    normalized = _normalize_path(gcs_path)
    gcsfs_client.put(local_path, normalized)


def download_file(gcs_path: str, local_path: str, recursive: bool = False) -> None:
    """
    Download a file or directory from GCS to local filesystem.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
        local_path: Destination path on local filesystem.
        recursive: If True, download directory recursively.
    """
    normalized = _normalize_path(gcs_path)
    gcsfs_client.get(normalized, local_path, recursive=recursive)


def delete_file(gcs_path: str) -> None:
    """
    Delete a single file from GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
    """
    normalized = _normalize_path(gcs_path)
    gcsfs_client.rm(normalized)


def delete_directory(gcs_path: str) -> None:
    """
    Delete a directory and all its contents from GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/ or bucket/path/).
    """
    normalized = _normalize_path(gcs_path)
    gcsfs_client.rm(normalized, recursive=True)


def exists(gcs_path: str) -> bool:
    """
    Check if a file or directory exists in GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path or bucket/path).

    Returns:
        True if the path exists, False otherwise.
    """
    normalized = _normalize_path(gcs_path)
    return gcsfs_client.exists(normalized)
