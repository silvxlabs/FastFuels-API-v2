"""Storage utilities for Standgen."""

import io
import logging

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET, TABLES_BUCKET
from lib.gcs import delete_directory, get_gcsfs_client
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


def write_changed_partitions(inventory_id: str, transform) -> str:
    """Apply ``transform`` to each Parquet partition; overwrite only those that change.

    ``transform`` is a per-partition ``DataFrame -> DataFrame`` callable (e.g.
    ``lambda df: apply_modifications(df, mods)``). Every partition is read in one
    concurrent ``cat`` batch; ``transform`` is applied to each; only the partitions
    whose content actually changes are written back, in one concurrent ``pipe``
    batch, under their existing file names. A modification scoped to part of the
    domain therefore rewrites only the partitions it overlaps.

    The transform is per-partition and must not change the schema (modifications
    only transform values or drop rows). ``_metadata`` is intentionally left
    untouched: both the API (``pd.read_parquet`` per file) and dask
    (``dd.read_parquet`` re-reads each file footer) read the part files directly,
    so a stale ``_metadata`` never corrupts a read — it only affects the row
    counts the ``/data`` metadata endpoint reports after a row-removing mod.
    """
    fs = get_gcsfs_client()
    base = f"{INVENTORIES_BUCKET}/{inventory_id}"
    n = _apply_changed_partitions(fs, base, transform)
    logger.info(f"Rewrote {n} changed partition(s) for inventory {inventory_id}")
    return f"gs://{base}"


def _apply_changed_partitions(fs, base: str, transform) -> int:
    """Core of :func:`write_changed_partitions`, parameterized on the filesystem
    so it can be tested against local files. Returns the number of partitions
    rewritten."""
    part_paths = sorted(p for p in fs.find(base) if p.endswith(".parquet"))
    blobs = fs.cat(part_paths)  # one concurrent read of all partitions

    changed: dict[str, bytes] = {}
    for path in part_paths:
        old_df = pd.read_parquet(io.BytesIO(blobs[path]))
        # Copy so an in-place transform (apply_action mutates) leaves old_df
        # pristine for the change comparison.
        new_df = transform(old_df.copy())
        if not new_df.equals(old_df):
            buf = io.BytesIO()
            new_df.to_parquet(buf, index=False)
            changed[path] = buf.getvalue()

    if changed:
        fs.pipe(changed)  # one concurrent write of the changed partitions
    return len(changed)


def write_full_partitions(
    inventory_id: str, df: pd.DataFrame, rows_per_partition: int = 100_000
) -> str:
    """Replace an inventory's Parquet with a materialized DataFrame.

    Used for global (basal-area) treatments, whose result is a whole-stand
    reduction that re-partitions the data — so a per-partition diff does not
    apply. ``df`` must already be in memory (the caller computed it), so writing
    back over the live path does not race a lazy read. Because the partition set
    changes, the aggregated ``_metadata`` is removed (its file list would be
    wrong and dask trusts it for the file set) along with any stale part files
    from a larger previous partitioning; readers then list the directory fresh.
    """
    fs = get_gcsfs_client()
    base = f"{INVENTORIES_BUCKET}/{inventory_id}"
    n = _write_full(fs, base, df, rows_per_partition)
    logger.info(f"Wrote {n} partition(s) for inventory {inventory_id}")
    return f"gs://{base}"


def _write_full(fs, base: str, df: pd.DataFrame, rows_per_partition: int) -> int:
    """Core of :func:`write_full_partitions`, parameterized on the filesystem so
    it can be tested against local files. Returns the partition count written."""
    n = max(1, -(-len(df) // rows_per_partition))  # ceil division
    new: dict[str, bytes] = {}
    for k in range(n):
        chunk = df.iloc[k * rows_per_partition : (k + 1) * rows_per_partition]
        buf = io.BytesIO()
        chunk.to_parquet(buf, index=False)
        new[f"{base}/part.{k}.parquet"] = buf.getvalue()

    fs.pipe(new)  # one concurrent write of all partitions
    fs.invalidate_cache()
    obsolete = [
        p
        for p in fs.find(base)
        if (p.endswith(".parquet") and p not in new)
        or p.rsplit("/", 1)[-1] in ("_metadata", "_common_metadata")
    ]
    if obsolete:
        fs.rm(obsolete)
    return n


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
