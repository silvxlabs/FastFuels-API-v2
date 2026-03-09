"""Storage utilities for Standgen."""

import logging

import dask.dataframe as dd
import pandas as pd
import xarray as xr

from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET, TABLES_BUCKET
from lib.gcs import delete_directory
from lib.zarr_utils import load_zarr

logger = logging.getLogger(__name__)


def load_pim_grid(grid_id: str) -> xr.Dataset:
    """Load a PIM grid's Zarr data from GCS."""
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    return load_zarr(path)


def load_tree_table(version: str) -> pd.DataFrame:
    """Load a TreeMap tree table from GCS as a pandas DataFrame."""
    path = f"gs://{TABLES_BUCKET}/TreeMap{version}_tree_table.parquet"
    logger.info(f"Loading tree table from {path}")
    return pd.read_parquet(path)


def load_inventory_parquet(inventory_id: str) -> dd.DataFrame:
    """Load an inventory's Parquet data from GCS as a dask DataFrame."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    logger.info(f"Loading inventory parquet from {path}")
    return dd.read_parquet(path)


def load_chm_grid(grid_id: str) -> xr.Dataset:
    """Load a CHM grid's Zarr data from GCS."""
    path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    return load_zarr(path)


def save_parquet(inventory_id: str, ddf: dd.DataFrame) -> str:
    """Write a dask DataFrame to GCS as partitioned Parquet.

    Each partition writes as a separate part-XXXX.parquet file.
    The full DataFrame is never materialized in memory.
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    ddf.to_parquet(path)
    logger.info(f"Saved inventory data to {path}")
    return path


def delete_parquet(inventory_id: str) -> None:
    """Delete inventory Parquet data from GCS."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    try:
        delete_directory(path)
        logger.info(f"Deleted inventory data at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete inventory data at {path}: {e}")
