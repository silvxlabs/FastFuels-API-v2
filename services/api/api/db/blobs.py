"""
Async GCS blob operations for FastAPI routes.

Provides async file operations using gcsfs with asynchronous=True.

Note: gcsfs async methods are prefixed with underscore (e.g., _exists, _rm).
This is the official async API, not private methods.
"""

import json
import logging

import gcsfs

logger = logging.getLogger(__name__)

gcsfs_client: gcsfs.GCSFileSystem = gcsfs.GCSFileSystem(asynchronous=True)


async def upload_json(gcs_path: str, data: dict) -> None:
    """Upload a JSON dict directly to GCS from memory."""
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    await gcsfs_client._pipe_file(gcs_path, json.dumps(data).encode())


async def delete_directory(bucket_name: str, directory_path: str) -> None:
    """Delete a directory and all its contents from GCS."""
    full_path = f"{bucket_name}/{directory_path}"
    if await gcsfs_client._exists(full_path):
        await gcsfs_client._rm(full_path, recursive=True)


async def copy_directory(bucket_name: str, source_path: str, dest_path: str) -> None:
    """Copy a directory and all its contents within GCS.

    For same-bucket copies this is a server-side rewrite (no egress).
    """
    source = f"{bucket_name}/{source_path}"
    dest = f"{bucket_name}/{dest_path}"
    await gcsfs_client._copy(source, dest, recursive=True)


async def copy_directory_verified(
    bucket_name: str, source_path: str, dest_path: str
) -> None:
    """Copy a directory within GCS and verify the copy against a pre-copy
    snapshot of the source listing.

    The source's object names and sizes are snapshotted before the copy; the
    destination listing must match that snapshot exactly afterwards. This
    catches a source that is deleted or rewritten while the copy is running,
    which a plain recursive copy would turn into a silently incomplete clone.

    Raises:
        FileNotFoundError: The source contains no objects.
        RuntimeError: The destination listing does not match the snapshot.
    """
    source = f"{bucket_name}/{source_path}"
    dest = f"{bucket_name}/{dest_path}"

    # Both listings must reflect live bucket state, not the shared client's
    # dircache — a stale snapshot would make the verification vacuous.
    gcsfs_client.invalidate_cache(source)
    snapshot = await gcsfs_client._find(source, detail=True)
    if not snapshot:
        raise FileNotFoundError(f"No objects to copy at {source}")
    expected = {
        path.removeprefix(f"{source}/"): info["size"] for path, info in snapshot.items()
    }

    await gcsfs_client._copy(source, dest, recursive=True)

    gcsfs_client.invalidate_cache(dest)
    dest_listing = await gcsfs_client._find(dest, detail=True)
    copied = {
        path.removeprefix(f"{dest}/"): info["size"]
        for path, info in dest_listing.items()
    }
    if copied != expected:
        missing = sorted(set(expected) - set(copied))
        raise RuntimeError(
            f"Copy verification failed for {source} -> {dest}: expected "
            f"{len(expected)} objects, found {len(copied)}"
            + (f"; missing {missing[:5]}" if missing else "")
        )


async def delete_file(bucket_name: str, file_path: str) -> None:
    """Delete a single file from GCS."""
    full_path = f"{bucket_name}/{file_path}"
    await gcsfs_client._rm(full_path)


async def check_exists(bucket_name: str, path: str) -> bool:
    """Check if a file or directory exists in GCS."""
    full_path = f"{bucket_name}/{path}"
    return await gcsfs_client._exists(full_path)


async def download_file(gcs_path: str, local_path: str) -> None:
    """Download a file from GCS to local filesystem."""
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    await gcsfs_client._get(gcs_path, local_path)


async def delete_directory_safe(bucket_name: str, directory_path: str) -> None:
    """Best-effort async delete. For use as a BackgroundTasks callback."""
    try:
        await delete_directory(bucket_name, directory_path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(
            "Failed to delete GCS data at %s/%s: %s",
            bucket_name,
            directory_path,
            e,
        )


async def delete_file_safe(bucket_name: str, file_path: str) -> None:
    """Best-effort async file delete. For use as a BackgroundTasks callback."""
    try:
        await delete_file(bucket_name, file_path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(
            "Failed to delete GCS file at %s/%s: %s",
            bucket_name,
            file_path,
            e,
        )
