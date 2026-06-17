"""
api/v2/resources/inventories/cache.py

Cached async parquet metadata access for inventory data streaming.
"""

import asyncio
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
from ring import lru

from lib.config import INVENTORIES_BUCKET


@dataclass
class PartitionMeta:
    index: int
    num_rows: int
    path: str


@dataclass
class InventoryMeta:
    num_partitions: int
    total_rows: int
    columns: list[str]
    partitions: list[PartitionMeta]


def _read_metadata_sync(inventory_id: str) -> InventoryMeta:
    metadata_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}/_metadata"
    pf = pq.ParquetFile(metadata_path)
    return _parse_metadata(pf.metadata)


def _parse_metadata(metadata: pq.FileMetaData) -> InventoryMeta:
    partitions_by_path: dict[str, int] = {}
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        path = rg.column(0).file_path
        partitions_by_path[path] = partitions_by_path.get(path, 0) + rg.num_rows

    columns = metadata.schema.to_arrow_schema().names

    sorted_paths = sorted(partitions_by_path.keys())
    partitions = [
        PartitionMeta(index=idx, num_rows=partitions_by_path[path], path=path)
        for idx, path in enumerate(sorted_paths)
    ]

    return InventoryMeta(
        num_partitions=len(partitions),
        total_rows=sum(p.num_rows for p in partitions),
        columns=columns,
        partitions=partitions,
    )


@lru(maxsize=128, force_asyncio=True)
async def get_inventory_metadata(
    inventory_id: str, checksum: str | None
) -> InventoryMeta:
    """Read an inventory's partitioned-Parquet metadata, cached per content.

    ``checksum`` participates only in the cache key (the body ignores it). An
    in-place modification (POST .../modifications) rewrites the data under the
    same ``inventory_id`` and re-assigns the inventory's ``checksum``, so a new
    checksum bypasses the stale entry. Without it, this LRU would keep serving
    pre-modification partition paths and row counts. ``None`` (legacy
    inventories without a checksum) is a valid, stable key — such inventories
    predate in-place edits, and any in-place edit assigns a fresh checksum.
    """
    return await asyncio.to_thread(_read_metadata_sync, inventory_id)


async def read_partition(
    inventory_id: str,
    partition_path: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}/{partition_path}"
    return await asyncio.to_thread(pd.read_parquet, path, columns=columns)
