"""
Synchronous GCS blob operations.

Provides file upload, download, delete, and existence checking.
"""

import json
import os

import gcsfs

# fsspec async filesystems pin their event loop to the PID that created them and
# raise ``RuntimeError: This class is not fork-safe`` when used from a forked
# child (fsspec/asyn.py: the ``loop`` property compares the instance's creation
# PID against the current PID). functions-framework imports this module in the
# gunicorn master before it forks its workers, so a single module-level client
# would be created in the master and poisoned in every worker. Build the client
# lazily and rebuild it whenever the PID changes, so each forked worker gets its
# own instance bound to its own loop. ``skip_instance_cache=True`` is required:
# fsspec's instance cache is inherited across the fork, so a plain
# ``GCSFileSystem()`` in the child could hand back the master's cached (poisoned)
# instance.
_gcsfs_client: gcsfs.GCSFileSystem | None = None
_gcsfs_client_pid: int | None = None


def get_gcsfs_client() -> gcsfs.GCSFileSystem:
    """Return a GCS filesystem client safe to use in the current process.

    The client is created on first use and rebuilt whenever the process ID
    changes (i.e. after a fork), so it is never shared across a fork boundary.
    """
    global _gcsfs_client, _gcsfs_client_pid
    pid = os.getpid()
    if _gcsfs_client is None or _gcsfs_client_pid != pid:
        _gcsfs_client = gcsfs.GCSFileSystem(skip_instance_cache=True)
        _gcsfs_client_pid = pid
    return _gcsfs_client


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
