"""
api/v2/resources/inventories/cache.py

Cached async access to an inventory's partitioned-Parquet metadata and data.

Counts come from the dask DataFrame, which reads each partition's own footer (and
ignores any aggregated ``_metadata``). This stays correct after standgen rewrites
only the partitions a modification changed — an in-place overwrite leaves the
``_metadata`` aggregate stale, but ``len(ddf)`` does not trust it.
"""

import asyncio
from dataclasses import dataclass

import dask.dataframe as dd
import pandas as pd
from ring import lru

from lib.config import INVENTORIES_BUCKET


@dataclass
class InventoryMeta:
    num_partitions: int
    total_rows: int
    columns: list[str]


def _inventory_path(inventory_id: str) -> str:
    return f"gs://{INVENTORIES_BUCKET}/{inventory_id}"


def _read_metadata_sync(inventory_id: str) -> InventoryMeta:
    ddf = dd.read_parquet(_inventory_path(inventory_id))
    return InventoryMeta(
        num_partitions=ddf.npartitions,
        total_rows=len(ddf),  # footer-level row count; ignores stale _metadata
        columns=list(ddf.columns),
    )


@lru(maxsize=128, force_asyncio=True)
async def get_inventory_metadata(
    inventory_id: str, checksum: str | None
) -> InventoryMeta:
    """Read an inventory's partition metadata, cached per content.

    ``checksum`` participates only in the cache key (the body ignores it). An
    in-place modification rewrites the changed partitions under the same
    ``inventory_id`` and re-assigns the inventory's ``checksum``, so a new
    checksum bypasses the stale entry. ``None`` (legacy inventories without a
    checksum) is a valid, stable key.
    """
    return await asyncio.to_thread(_read_metadata_sync, inventory_id)


def _read_partition_sync(
    inventory_id: str, partition_index: int, columns: list[str] | None
) -> pd.DataFrame:
    ddf = dd.read_parquet(_inventory_path(inventory_id), columns=columns)
    return ddf.partitions[partition_index].compute().reset_index(drop=True)


async def read_partition(
    inventory_id: str,
    partition_index: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return await asyncio.to_thread(
        _read_partition_sync, inventory_id, partition_index, columns
    )
