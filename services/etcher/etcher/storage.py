"""Storage utilities for Features."""

import logging

import geopandas as gpd

from lib.config import FEATURES_BUCKET
from lib.gcs import delete_file

logger = logging.getLogger(__name__)


def save_features(domain_id: str, feature_id: str, gdf: gpd.GeoDataFrame) -> str:
    """Write a GeoDataFrame to GCS as a GeoParquet file.

    Using the gs:// prefix allows GeoPandas (via fsspec/gcsfs) to stream
    the upload directly to the bucket, bypassing the local container filesystem.

    Format choices: zstd compression (pyarrow default level) and row
    groups of 1000 (each row group becomes one ``partition_index`` for
    the streaming endpoint).
    """
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"

    gdf.to_parquet(
        path,
        compression="zstd",
        row_group_size=1000,
    )
    logger.info(f"Saved feature Parquet to {path}")

    return path


def delete_features(domain_id: str, feature_id: str) -> None:
    """Delete a feature's Parquet file from GCS."""
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    try:
        delete_file(path)
        logger.info(f"Deleted feature Parquet at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete feature Parquet at {path}: {e}")
