"""Storage utilities for Features."""

import logging

import geopandas as gpd

from lib.config import FEATURES_BUCKET
from lib.gcs import delete_file

logger = logging.getLogger(__name__)


def load_geojson(domain_id: str, feature_id: str) -> gpd.GeoDataFrame:
    """Load a feature's GeoJSON data from GCS as a GeoDataFrame."""
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
    logger.info(f"Loading feature GeoJSON from {path}")
    return gpd.read_file(path)


def save_geojson(domain_id: str, feature_id: str, gdf: gpd.GeoDataFrame) -> str:
    """Write a GeoDataFrame to GCS as a GeoJSON file.

    Using the gs:// prefix allows GeoPandas (via fsspec/gcsfs) to stream
    the upload directly to the bucket, bypassing the local container filesystem.
    """
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"

    # Write directly to GCS
    gdf.to_file(path, driver="GeoJSON")
    logger.info(f"Saved feature GeoJSON to {path}")

    return path


def delete_geojson(domain_id: str, feature_id: str) -> None:
    """Delete a feature's GeoJSON file from GCS."""
    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
    try:
        delete_file(path)
        logger.info(f"Deleted feature GeoJSON at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete feature GeoJSON at {path}: {e}")
