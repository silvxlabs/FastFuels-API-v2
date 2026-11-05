"""Storage utilities for Features."""

import json
import logging

import geopandas as gpd
import pyarrow.parquet as pq
from geopandas.io.arrow import _geopandas_to_arrow

from lib.config import FEATURES_BUCKET
from lib.gcs import delete_file, storage_size

logger = logging.getLogger(__name__)


def save_features(
    domain_id: str,
    feature_id: str,
    gdf: gpd.GeoDataFrame,
    attributions: list[dict] | None = None,
) -> str:
    """Write a GeoDataFrame to GCS as a GeoParquet file.

    Using the gs:// prefix allows GeoPandas (via fsspec/gcsfs) to stream
    the upload directly to the bucket, bypassing the local container filesystem.

    Format choices: zstd compression (pyarrow default level) and row
    groups of 1000 (each row group becomes one ``partition_index`` for
    the streaming endpoint).

    ``attributions`` is stored JSON-encoded in the file's key-value metadata
    under ``attributions``, next to the GeoParquet ``geo`` key.
    """
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"

    # Same table GeoDataFrame.to_parquet builds; it exposes no hook for
    # extra file metadata.
    table = _geopandas_to_arrow(gdf)
    metadata = {
        **(table.schema.metadata or {}),
        b"attributions": json.dumps(attributions or []).encode(),
    }
    pq.write_table(
        table.replace_schema_metadata(metadata),
        path,
        compression="zstd",
        row_group_size=1000,
    )
    logger.info(f"Saved feature Parquet to {path}")

    return path


def feature_size(domain_id: str, feature_id: str) -> int:
    """Total GCS bytes of a feature's GeoParquet artifact — its footprint (#342)."""
    return storage_size(f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet")


def delete_features(domain_id: str, feature_id: str) -> None:
    """Delete a feature's Parquet file from GCS."""
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    try:
        delete_file(path)
        logger.info(f"Deleted feature Parquet at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete feature Parquet at {path}: {e}")
