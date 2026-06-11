"""Storage utilities for Standgen."""

import logging

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET, TABLES_BUCKET
from lib.errors import ProcessingError
from lib.gcs import delete_directory, exists, get_gcsfs_client
from lib.zarr_utils import load_zarr

logger = logging.getLogger(__name__)


def load_grid(grid_id: str) -> xr.Dataset:
    """Load a grid's Zarr data from GCS."""
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


def _write_parquet(ddf: dd.DataFrame, path: str) -> None:
    """Write ``ddf`` as partitioned Parquet with an aggregated ``_metadata`` file.

    ``write_index=False`` keeps dask from materializing the meaningless
    RangeIndex as a synthetic ``__null_dask_index__`` column in the file
    schema, which leaked into the API's data/metadata ``columns`` (#335).
    """
    ddf.to_parquet(path, write_metadata_file=True, write_index=False)


def save_parquet(inventory_id: str, ddf: dd.DataFrame) -> str:
    """Write a dask DataFrame to GCS as partitioned Parquet.

    Each partition writes as a separate part-XXXX.parquet file.
    The full DataFrame is never materialized in memory.
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    _write_parquet(ddf, path)
    logger.info(f"Saved inventory data to {path}")
    return path


def save_parquet_replace(inventory_id: str, ddf: dd.DataFrame) -> str:
    """Replace an inventory's Parquet data in place with ``ddf``.

    ``ddf`` is typically derived by reading the inventory's *own* current
    Parquet (an in-place modification). Writing straight back to the live path
    would corrupt the data: ``dd.read_parquet`` is lazy, so the read executes
    during this write and would race the overwrite. Instead write to a staging
    prefix, then swap it into place once the write (and its read) have
    completed:

    1. write ``ddf`` to ``{id}__rev`` (reads the live ``{id}`` here),
    2. delete the live ``{id}`` directory,
    3. server-side copy ``{id}__rev`` -> ``{id}`` (same bucket, no egress),
    4. delete the staging ``{id}__rev`` directory.
    """
    live_rel = f"{INVENTORIES_BUCKET}/{inventory_id}"
    staging_rel = f"{INVENTORIES_BUCKET}/{inventory_id}__rev"
    staging_uri = f"gs://{staging_rel}"

    # Clear any staging dir left behind by a previously failed attempt.
    if exists(staging_uri):
        delete_directory(staging_uri)

    _write_parquet(ddf, staging_uri)

    fs = get_gcsfs_client()

    # dask wrote staging through its own gcsfs instance, and the exists() probe
    # above cached this client's listing of the (then-absent) staging dir. Drop
    # the stale cache so the copy and the count check below list staging fresh.
    fs.invalidate_cache()
    staged_files = fs.find(staging_rel)

    delete_directory(f"gs://{live_rel}")
    # Recursive copy keeps fsspec's default on_error="ignore". gcsfs's
    # expand_path includes a synthetic directory-marker entry for the staging
    # prefix that 404s on rewrite; "ignore" skips it (only FileNotFoundError is
    # swallowed, so transient errors still raise). That same leniency would also
    # skip a genuinely missing part file, so verify completeness by file count
    # before deleting staging — never mark an incomplete dataset completed.
    fs.copy(staging_rel, live_rel, recursive=True)
    fs.invalidate_cache()
    live_files = fs.find(live_rel)
    if len(live_files) != len(staged_files):
        raise ProcessingError(
            code="INVENTORY_REWRITE_INCOMPLETE",
            message=(
                f"In-place rewrite of inventory {inventory_id} copied "
                f"{len(live_files)} of {len(staged_files)} files; staging "
                f"({staging_rel}) is preserved for recovery."
            ),
            suggestion="Retry the modification.",
        )
    delete_directory(staging_uri)

    logger.info(f"Replaced inventory data at gs://{live_rel}")
    return f"gs://{live_rel}"


def count_inventory_rows(inventory_id: str) -> int | None:
    """Total row count from an inventory Parquet's ``_metadata`` footer.

    Reads only the Parquet footer (no data scan, no dask compute), so this is a
    cheap alternative to materializing the DataFrame just to count rows. The
    ``_metadata`` file written by :func:`save_parquet` aggregates the row-group
    metadata across every part file, so ``num_rows`` is the total across all
    partitions.

    Returns None if the footer is missing, so a successful write is never turned
    into a job failure at the logging step.
    """
    path = f"{INVENTORIES_BUCKET}/{inventory_id}/_metadata"
    fs = get_gcsfs_client()
    try:
        with fs.open(path, "rb") as f:
            return pq.read_metadata(f).num_rows
    except FileNotFoundError:
        logger.warning(
            f"No Parquet _metadata for inventory {inventory_id}; skipping row count"
        )
        return None


def delete_parquet(inventory_id: str) -> None:
    """Delete inventory Parquet data from GCS."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    try:
        delete_directory(path)
        logger.info(f"Deleted inventory data at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete inventory data at {path}: {e}")
