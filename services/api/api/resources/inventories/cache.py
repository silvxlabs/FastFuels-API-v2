"""
api/v2/resources/inventories/cache.py

Cached async parquet metadata access for inventory data streaming.

A data partition is one Parquet row group. Writers cap row groups at 50k rows
(#763); older inventories have one row group per file, so their partitions are
the same as when a partition was a whole file.
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
    row_group: int


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
    """Enumerate ``_metadata`` row groups as partitions, ordered by file path
    and then by row-group order within the file.

    The aggregated ``_metadata`` appends each file's row groups in file order,
    so the n-th entry for a path is row group n of that file.
    """
    row_groups_by_path: dict[str, list[int]] = {}
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        row_groups_by_path.setdefault(rg.column(0).file_path, []).append(rg.num_rows)

    columns = metadata.schema.to_arrow_schema().names

    ordered = [
        (path, row_group, num_rows)
        for path in sorted(row_groups_by_path)
        for row_group, num_rows in enumerate(row_groups_by_path[path])
    ]
    partitions = [
        PartitionMeta(index=idx, num_rows=num_rows, path=path, row_group=row_group)
        for idx, (path, row_group, num_rows) in enumerate(ordered)
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


def _read_row_group_sync(
    path: str, row_group: int, columns: list[str] | None
) -> pd.DataFrame:
    # pyarrow's native GCS filesystem, not a sync gcsfs client (#265).
    # pre_buffer coalesces column-chunk reads, as pd.read_parquet does.
    with pq.ParquetFile(path, pre_buffer=True) as pf:
        table = pf.read_row_group(row_group, columns=columns, use_pandas_metadata=True)
    return table.to_pandas()


async def read_partition(
    inventory_id: str,
    partition: PartitionMeta,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one partition (a single row group of one file) from GCS."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}/{partition.path}"
    return await asyncio.to_thread(
        _read_row_group_sync, path, partition.row_group, columns
    )
