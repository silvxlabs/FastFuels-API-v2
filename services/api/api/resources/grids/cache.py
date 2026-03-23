"""
api/v2/resources/grids/cache.py

Cached async zarr group/array access for grid data streaming.
"""

import zarr.api.asynchronous as zarr_async
from ring import lru
from zarr import AsyncArray, AsyncGroup

from lib.config import GRIDS_BUCKET


@lru(maxsize=128, force_asyncio=True)
async def get_grid_group(grid_id: str) -> AsyncGroup:
    return await zarr_async.open_group(store=f"gs://{GRIDS_BUCKET}/{grid_id}", mode="r")


@lru(maxsize=128, force_asyncio=True)
async def get_grid_array(grid_id: str, band_key: str) -> AsyncArray:
    group = await get_grid_group(grid_id)
    return await group.getitem(band_key)
