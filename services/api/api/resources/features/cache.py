"""
api/v2/resources/features/cache.py

Partitioned GeoJSON page fetch for feature data blobs.

Feature blobs are GeoParquet written with row groups sized at
``PARTITION_SIZE``. Each ``partition_index`` maps to one row group:
``/data/metadata`` reads only the file footer, and
``/data/{partition_index}`` range-reads the single requested row group.
No whole-file load, no in-process cache.
"""

import asyncio
from dataclasses import dataclass

import gcsfs
import geopandas as gpd
import pyarrow.parquet as pq

from lib.config import FEATURES_BUCKET

# Writer-configured row group size for feature data. The two writers
# (``services/etcher/etcher/storage.py`` and
# ``services/api/api/resources/features/layerset/router.py``) must pass
# ``row_group_size=PARTITION_SIZE`` so ``partition_index`` aligns with
# row group index. Surfaced in the /data/metadata response as the
# documented maximum partition size.
PARTITION_SIZE = 1000
MAX_RESPONSE_BYTES = 30 * 1024 * 1024
GEOJSON_MEDIA_TYPE = "application/geo+json"


@dataclass(frozen=True)
class FeatureMetadata:
    """Partition layout for a feature blob."""

    total_features: int
    partition_size: int
    partition_count: int

    def to_dict(self) -> dict:
        return {
            "total_features": self.total_features,
            "partition_size": self.partition_size,
            "partition_count": self.partition_count,
        }


class InvalidFeatureParquet(Exception):
    """The blob exists but is not parseable as GeoParquet."""


class PartitionOutOfRange(Exception):
    """``partition_index`` is outside ``[0, partition_count)``."""

    def __init__(self, partition_index: int, partition_count: int):
        super().__init__(
            f"partition {partition_index} out of range "
            f"(file has {partition_count} partition(s))"
        )
        self.partition_index = partition_index
        self.partition_count = partition_count


class PartitionTooLarge(Exception):
    """The serialized GeoJSON partition exceeds ``MAX_RESPONSE_BYTES``."""

    def __init__(self, payload_bytes: int, limit: int):
        super().__init__(
            f"partition payload {payload_bytes} bytes exceeds {limit} bytes"
        )
        self.payload_bytes = payload_bytes
        self.limit = limit


def _blob_path(domain_id: str, feature_id: str) -> str:
    return f"{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"


def _open_parquet_file(domain_id: str, feature_id: str) -> pq.ParquetFile:
    """Open the GeoParquet file's footer. No row-group data is read.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob exists but is not parseable as GeoParquet.
    """
    fs = gcsfs.GCSFileSystem()
    try:
        return pq.ParquetFile(_blob_path(domain_id, feature_id), filesystem=fs)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise InvalidFeatureParquet(str(exc)) from exc


def _read_metadata_sync(domain_id: str, feature_id: str) -> FeatureMetadata:
    pf = _open_parquet_file(domain_id, feature_id)
    return FeatureMetadata(
        total_features=pf.metadata.num_rows,
        partition_size=PARTITION_SIZE,
        partition_count=pf.num_row_groups,
    )


async def get_feature_metadata(domain_id: str, feature_id: str) -> FeatureMetadata:
    """Read the GeoParquet footer and return its partition layout.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob exists but is not parseable as GeoParquet.
    """
    return await asyncio.to_thread(_read_metadata_sync, domain_id, feature_id)


def _read_partition_sync(
    domain_id: str, feature_id: str, partition_index: int
) -> tuple[bytes, int, int]:
    pf = _open_parquet_file(domain_id, feature_id)
    partition_count = pf.num_row_groups
    total = pf.metadata.num_rows
    if partition_index < 0 or partition_index >= partition_count:
        raise PartitionOutOfRange(partition_index, partition_count)
    try:
        table = pf.read_row_group(partition_index)
    except Exception as exc:
        raise InvalidFeatureParquet(str(exc)) from exc
    gdf = gpd.GeoDataFrame.from_arrow(table)
    # `from_arrow` produces a fresh RangeIndex starting at 0 per row group.
    # Shift it by the sum of preceding row group sizes so the GeoJSON "id"
    # field stays continuous across partitions and matches what
    # ``gpd.read_parquet(<whole file>).to_json()`` would emit.
    start_offset = sum(
        pf.metadata.row_group(j).num_rows for j in range(partition_index)
    )
    gdf.index = range(start_offset, start_offset + len(gdf))
    payload = gdf.to_json().encode("utf-8")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise PartitionTooLarge(len(payload), MAX_RESPONSE_BYTES)
    return payload, len(gdf), total


async def fetch_partition_geojson(
    domain_id: str, feature_id: str, partition_index: int
) -> tuple[bytes, int, int]:
    """Return ``(payload_bytes, num_features_in_partition, total_features)``.

    ``payload_bytes`` is a self-contained GeoJSON FeatureCollection for the
    requested row group, range-read from GCS.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob is malformed Parquet.
        PartitionOutOfRange: ``partition_index`` is past the last partition.
        PartitionTooLarge: serialized payload exceeds ``MAX_RESPONSE_BYTES``.
    """
    return await asyncio.to_thread(
        _read_partition_sync, domain_id, feature_id, partition_index
    )
