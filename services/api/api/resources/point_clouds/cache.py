"""Cached indexed datasets and off-event-loop point-cloud reads."""

import asyncio

import pyarrow as pa
from ring import lru

from lib.config import POINT_CLOUD_READ_THREADS, POINT_CLOUDS_BUCKET
from lib.pointcloud.reader import PointCloudStorage, open_point_cloud, read_tile
from lib.pointcloud.schema import cloud_prefix

# PyArrow otherwise sizes this from the Cloud Run host rather than the vCPU
# quota. This process-wide pool is shared by the cached Dataset scanners.
pa.set_cpu_count(POINT_CLOUD_READ_THREADS)


@lru(maxsize=128, force_asyncio=True)
async def get_point_cloud_storage(
    point_cloud_id: str, checksum: str
) -> PointCloudStorage:
    """Open and cache one immutable manifest plus its Parquet `_metadata`."""
    prefix = cloud_prefix(POINT_CLOUDS_BUCKET, point_cloud_id)
    return await asyncio.to_thread(open_point_cloud, prefix)


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
