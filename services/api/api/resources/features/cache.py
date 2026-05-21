"""
api/v2/resources/features/cache.py

Partitioned GeoJSON page fetch for feature data blobs.

Feature blobs are GeoParquet written with row groups (see
``services/etcher/etcher/storage.py`` and
``services/api/api/resources/features/layerset/router.py``). Each
``partition_index`` maps to one row group: ``/data/metadata`` reads only the
file footer and reports actual per-row-group row counts, and
``/data/{partition_index}`` range-reads the single requested row group.

The opened ``ParquetFile`` is held in an in-process LRU keyed on
``(domain_id, feature_id)`` so the footer is fetched at most once per
feature per process. Features are immutable after completion (only
``name``/``description``/``tags`` are PATCHable), so cache invalidation is
unnecessary.
"""

import asyncio
import functools

import gcsfs
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq

from api.resources.features.schema import (
    FeatureDataMetadata,
    FeaturePartitionInfo,
)
from lib.config import FEATURES_BUCKET

MAX_RESPONSE_BYTES = 30 * 1024 * 1024
GEOJSON_MEDIA_TYPE = "application/geo+json"
_PARQUET_FILE_CACHE_SIZE = 128


class InvalidFeatureParquet(Exception):
    """The blob exists but is not parseable as GeoParquet."""


class PartitionOutOfRange(Exception):
    """``partition_index`` is outside ``[0, partition_count)``."""

    def __init__(self, partition_index: int, partition_count: int):
        if partition_count == 0:
            message = (
                f"partition {partition_index} out of range: "
                "feature has 0 partitions, no /data/{i} calls are valid"
            )
        else:
            message = (
                f"partition {partition_index} out of range "
                f"(file has {partition_count} partition(s))"
            )
        super().__init__(message)
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


@functools.lru_cache(maxsize=_PARQUET_FILE_CACHE_SIZE)
def _open_parquet_file(domain_id: str, feature_id: str) -> pq.ParquetFile:
    """Open the GeoParquet file's footer. No row-group data is read.

    LRU-cached on ``(domain_id, feature_id)`` so a client doing
    ``1 × /data/metadata + N × /data/{i}`` issues a single footer fetch per
    process instead of ``N + 1``. ``functools.lru_cache`` is thread-safe and
    does not cache exceptions, so failed opens (missing blob, transient GCS
    error) re-try on the next call.

    Only ``FileNotFoundError`` (missing blob) and ``pyarrow.lib.ArrowInvalid``
    (corrupt Parquet) are mapped to typed exceptions; transient gcsfs / auth /
    network failures propagate untouched so they surface as 500s rather than a
    misleading "malformed blob" 422.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob exists but is not parseable as GeoParquet.
    """
    fs = gcsfs.GCSFileSystem()
    try:
        return pq.ParquetFile(_blob_path(domain_id, feature_id), filesystem=fs)
    except pa.lib.ArrowInvalid as exc:
        raise InvalidFeatureParquet(str(exc)) from exc


def _read_metadata_sync(domain_id: str, feature_id: str) -> FeatureDataMetadata:
    pf = _open_parquet_file(domain_id, feature_id)
    partitions = [
        FeaturePartitionInfo(index=i, num_features=pf.metadata.row_group(i).num_rows)
        for i in range(pf.num_row_groups)
    ]
    return FeatureDataMetadata(
        total_features=pf.metadata.num_rows,
        partition_count=pf.num_row_groups,
        partitions=partitions,
    )


async def get_feature_metadata(domain_id: str, feature_id: str) -> FeatureDataMetadata:
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
    except pa.lib.ArrowInvalid as exc:
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
