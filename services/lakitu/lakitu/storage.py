"""
GCS storage for point clouds.

Each point cloud owns a directory keyed by its id. The directory layout is what
walle sweeps when a resource is deleted, so it must stay in step with the
uploader, which writes into the same directory.

A 3DEP cloud is a partitioned Parquet dataset written in place by
``lakitu.parquet_writer``, not a single object, so there is no upload step here
-- only the path it is written under and the size it came to.
"""

from lib.config import POINT_CLOUDS_BUCKET
from lib.gcs import delete_directory, exists, storage_size

CLOUD_DIRNAME = "cloud.parquet"


def cloud_prefix(point_cloud_id: str) -> str:
    """Return the GCS prefix of a point cloud's Parquet dataset."""
    return f"{POINT_CLOUDS_BUCKET}/{point_cloud_id}/{CLOUD_DIRNAME}"


def cloud_location(point_cloud_id: str) -> tuple[str, str]:
    """Return ``(bucket, prefix)`` for a point cloud's dataset.

    The writer talks to the storage client directly rather than through a path
    string, so it needs the two halves apart.
    """
    return POINT_CLOUDS_BUCKET, f"{point_cloud_id}/{CLOUD_DIRNAME}"


def cloud_size(point_cloud_id: str) -> int:
    """Return the total size of a point cloud's dataset in bytes."""
    return storage_size(cloud_prefix(point_cloud_id))


def delete_cloud(point_cloud_id: str) -> None:
    """Remove a point cloud's directory, ignoring one that was never written."""
    path = f"{POINT_CLOUDS_BUCKET}/{point_cloud_id}"
    if exists(path):
        delete_directory(path)
