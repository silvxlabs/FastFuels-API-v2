"""
api/v2/resources/features/cache.py

Per-feature GeoDataFrame cache + partitioned GeoJSON page fetch.

Feature blobs are GeoParquet and immutable post-completion. The cache
reads the entire file once via ``geopandas.read_parquet`` and slices
the resulting GeoDataFrame into fixed-size partitions in memory.
"""

import asyncio
from dataclasses import dataclass

import geopandas as gpd
from ring import lru

from lib.config import FEATURES_BUCKET

# Fixed partition size for the streaming endpoint. Matches the writer's
# ``row_group_size`` and is exposed in the /data/metadata response so
# clients know what to expect from each /data/{partition_index} page.
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
    return f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"


def _read_sync(domain_id: str, feature_id: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_parquet(_blob_path(domain_id, feature_id))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise InvalidFeatureParquet(str(exc)) from exc


@lru(maxsize=64, force_asyncio=True)
async def get_feature_gdf(domain_id: str, feature_id: str) -> gpd.GeoDataFrame:
    """Cached whole-file GeoDataFrame read.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob exists but is not parseable as GeoParquet.
    """
    return await asyncio.to_thread(_read_sync, domain_id, feature_id)


def _partition_count(total_features: int) -> int:
    return (total_features + PARTITION_SIZE - 1) // PARTITION_SIZE


async def get_feature_metadata(domain_id: str, feature_id: str) -> FeatureMetadata:
    gdf = await get_feature_gdf(domain_id, feature_id)
    n = len(gdf)
    return FeatureMetadata(
        total_features=n,
        partition_size=PARTITION_SIZE,
        partition_count=_partition_count(n),
    )


async def fetch_partition_geojson(
    domain_id: str, feature_id: str, partition_index: int
) -> tuple[bytes, int, int]:
    """Return ``(payload_bytes, num_features_in_partition, total_features)``.

    ``payload_bytes`` is a self-contained GeoJSON FeatureCollection for one
    partition, ready to stream to the client.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureParquet: blob is malformed Parquet.
        PartitionOutOfRange: ``partition_index`` is past the last partition.
        PartitionTooLarge: serialized payload exceeds ``MAX_RESPONSE_BYTES``.
    """
    gdf = await get_feature_gdf(domain_id, feature_id)
    n = len(gdf)
    partition_count = _partition_count(n)
    if partition_index < 0 or partition_index >= partition_count:
        raise PartitionOutOfRange(partition_index, partition_count)
    start = partition_index * PARTITION_SIZE
    end = min(start + PARTITION_SIZE, n)
    payload = gdf.iloc[start:end].to_json().encode("utf-8")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise PartitionTooLarge(len(payload), MAX_RESPONSE_BYTES)
    return payload, end - start, n
