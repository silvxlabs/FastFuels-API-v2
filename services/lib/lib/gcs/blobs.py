"""
Synchronous GCS blob operations.

Provides file upload, download, delete, and existence checking.
"""

import functools
import json

import gcsfs


@functools.cache
def get_gcsfs_client() -> gcsfs.GCSFileSystem:
    """Return a process-wide GCS filesystem client, built lazily on first use.

    fsspec's async filesystems pin their event loop to the PID that created
    them and raise ``RuntimeError: This class is not fork-safe`` if used from a
    forked child. functions-framework imports this module in the gunicorn
    master *before* it forks workers, so the client must not be constructed at
    import time — otherwise it is created in the master and poisoned in every
    worker (#333). Building it lazily means each worker constructs its own
    client on its first request, after the fork.

    For the same reason, do not call this at import / module scope.

    ``requests_timeout`` bounds each HTTP request. Without it, a request sent
    on a stale pooled keep-alive connection waits forever — observed as a
    worker hanging indefinitely in fsspec ``sync()`` with the socket in
    CLOSE_WAIT. With it, the dead connection raises a timeout and gcsfs's
    built-in retry re-issues the request on a fresh connection.
    """
    return gcsfs.GCSFileSystem(requests_timeout=90)


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
    get_gcsfs_client().put(local_path, normalized)


def download_file(gcs_path: str, local_path: str, recursive: bool = False) -> None:
    """
    Download a file or directory from GCS to local filesystem.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
        local_path: Destination path on local filesystem.
        recursive: If True, download directory recursively.
    """
    normalized = _normalize_path(gcs_path)
    get_gcsfs_client().get(normalized, local_path, recursive=recursive)


def delete_file(gcs_path: str) -> None:
    """
    Delete a single file from GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
    """
    normalized = _normalize_path(gcs_path)
    get_gcsfs_client().rm(normalized)


def delete_directory(gcs_path: str) -> None:
    """
    Delete a directory and all its contents from GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/ or bucket/path/).
    """
    normalized = _normalize_path(gcs_path)
    get_gcsfs_client().rm(normalized, recursive=True)


def storage_size(gcs_path: str) -> int:
    """
    Total bytes stored under a GCS path — its absolute artifact footprint.

    Sums the size of every object under ``gcs_path``, so a single object (a
    ``.laz`` point cloud, an export archive) and a multi-object store (a zarr
    grid, a partitioned-parquet inventory) are both handled the same way. The
    listing cache is invalidated first so the count reflects objects just
    written, and the result is the absolute current footprint — an in-place
    rewrite that replaced the prefix reads as a replacement, never an
    accumulation.

    Args:
        gcs_path: Full GCS path (gs://bucket/path or bucket/path).

    Returns:
        Total size in bytes of every object under the path.
    """
    normalized = _normalize_path(gcs_path)
    client = get_gcsfs_client()
    client.invalidate_cache(normalized)
    return client.du(normalized, total=True)


def exists(gcs_path: str) -> bool:
    """
    Check if a file or directory exists in GCS.

    Args:
        gcs_path: Full GCS path (gs://bucket/path or bucket/path).

    Returns:
        True if the path exists, False otherwise.
    """
    normalized = _normalize_path(gcs_path)
    return get_gcsfs_client().exists(normalized)


def upload_json(gcs_path: str, data: dict) -> None:
    """
    Upload a JSON dictionary directly to GCS from memory.

    Args:
        gcs_path: Full GCS path (gs://bucket/path/file or bucket/path/file).
        data: The dictionary to upload as JSON.
    """
    normalized = _normalize_path(gcs_path)
    with get_gcsfs_client().open(normalized, "w") as f:
        json.dump(data, f)
