"""Cached indexed datasets and off-event-loop point-cloud reads."""

import asyncio

import gcsfs
import pyarrow as pa
from ring import lru

from lib.config import POINT_CLOUD_READ_THREADS, POINT_CLOUDS_BUCKET
from lib.pointcloud.reader import PointCloudStorage, open_point_cloud, read_tile
from lib.pointcloud.schema import cloud_prefix

# PyArrow otherwise sizes this from the Cloud Run host rather than the vCPU
# quota. This process-wide pool is shared by the cached Dataset scanners.
pa.set_cpu_count(POINT_CLOUD_READ_THREADS)

_gcsfs: gcsfs.GCSFileSystem | None = None
_gcsfs_loop: asyncio.AbstractEventLoop | None = None


def _gcsfs_on_running_loop() -> gcsfs.GCSFileSystem:
    """Return a sync gcsfs client whose I/O runs on this worker's event loop.

    The default `lib.gcs` client starts fsspec's own event-loop thread, and a
    second loop in the process trips grpcio's `PollerCompletionQueue` race:
    every later Firestore RPC logs `BlockingIOError` (#265, #703). Bound to the
    running loop, the sync calls the reader makes from `asyncio.to_thread` are
    dispatched onto that loop instead. Built on first use, never at import, so
    it is not created in the `--preload` gunicorn master (#702).
    """
    global _gcsfs, _gcsfs_loop
    loop = asyncio.get_running_loop()
    if _gcsfs is None or _gcsfs_loop is not loop:
        _gcsfs = gcsfs.GCSFileSystem(
            loop=loop, requests_timeout=90, skip_instance_cache=True
        )
        _gcsfs_loop = loop
    return _gcsfs


@lru(maxsize=128, force_asyncio=True)
async def get_point_cloud_storage(
    point_cloud_id: str, checksum: str
) -> PointCloudStorage:
    """Open and cache one immutable manifest plus its Parquet `_metadata`."""
    prefix = cloud_prefix(POINT_CLOUDS_BUCKET, point_cloud_id)
    return await asyncio.to_thread(open_point_cloud, prefix, _gcsfs_on_running_loop())


async def get_point_cloud_tile(
    storage: PointCloudStorage,
    tile_x: int,
    tile_y: int,
    columns: list[str],
    classes: list[int] | None,
    lod: int,
) -> pa.Table:
    """Scan one indexed tile off the event loop with Arrow predicate pushdown."""
    return await asyncio.to_thread(
        read_tile,
        storage,
        tile_x,
        tile_y,
        columns,
        classes,
        lod,
    )
