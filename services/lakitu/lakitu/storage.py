"""
GCS storage for point clouds.

Each point cloud owns a directory keyed by its id, holding a single LAZ. The
directory layout is what walle sweeps when a resource is deleted, so it must
stay in step with the uploader, which writes the same path for uploads.
"""

import io

from lib.config import POINT_CLOUDS_BUCKET
from lib.gcs import delete_directory, exists, get_gcsfs_client, storage_size

CLOUD_FILENAME = "cloud.laz"


def cloud_path(point_cloud_id: str) -> str:
    """Return the GCS path of a point cloud's LAZ."""
    return f"{POINT_CLOUDS_BUCKET}/{point_cloud_id}/{CLOUD_FILENAME}"


def write_cloud(point_cloud_id: str, buffer: io.BytesIO) -> int:
    """Write a point cloud's LAZ to GCS and return its size in bytes."""
    path = cloud_path(point_cloud_id)
    with get_gcsfs_client().open(path, "wb") as out:
        out.write(buffer.getbuffer())
    return storage_size(path)


def delete_cloud(point_cloud_id: str) -> None:
    """Remove a point cloud's directory, ignoring one that was never written."""
    path = f"{POINT_CLOUDS_BUCKET}/{point_cloud_id}"
    if exists(path):
        delete_directory(path)
