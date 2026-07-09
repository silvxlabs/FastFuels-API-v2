"""Storage utilities for Standgen."""

import logging

import dask
import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET, TABLES_BUCKET
from lib.errors import ProcessingError
from lib.gcs import delete_directory, exists, get_gcsfs_client, storage_size
from lib.zarr_utils import load_zarr
from standgen.summarize import _build_column_stats_graph

logger = logging.getLogger(__name__)


FLOAT_STATS = {"min", "max", "mean", "std"}


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


def _compute_write_and_stats(
    write_delayed, stats_graph: dict[str, dict]
) -> dict[str, dict]:
    """Flatten lazy reductions and a deferred parquet write into a single
    dask.compute call, then reassemble the stats dict.

    Handles both continuous columns (which use a single ``.agg()`` lazy Series)
    and categorical columns (which use individual lazy scalars).

    Args:
        write_delayed: Deferred parquet write from ``ddf.to_parquet(..., compute=False)``.
        stats_graph: Dict of lazy reductions from ``_build_column_stats_graph``.

    Returns:
        Dict keyed by column key with computed summary stats.
    """
    # Collect all lazy objects alongside their (col_key, stat_key) address
    flat_lazy = []
    flat_keys = []
    for k in stats_graph:
        for s, val in stats_graph[k].items():
            if s == "type":
                continue
            flat_lazy.append(val)
            flat_keys.append((k, s))

    results = dask.compute(write_delayed, *flat_lazy)
    computed_values = results[1:]

    stats = {}
    for i, (k, s) in enumerate(flat_keys):
        if k not in stats:
            stats[k] = {"type": stats_graph[k]["type"]}
        val = computed_values[i]
        if s in FLOAT_STATS:
            stats[k][s] = float(val)
        else:
            stats[k][s] = int(val)

    for k, v in stats.items():
        if v["type"] == "continuous" and v["count"] == 0:
            v["min"] = v["max"] = v["mean"] = v["std"] = None

    return stats


def _write_parquet(ddf: dd.DataFrame, path: str) -> None:
    """Write ``ddf`` as partitioned Parquet with an aggregated ``_metadata`` file.

    ``write_index=False`` keeps dask from materializing the meaningless
    RangeIndex as a synthetic ``__null_dask_index__`` column in the file
    schema, which leaked into the API's data/metadata ``columns`` (#335).

    Returns a deferred write for fusing with stats reductions in a single
    ``dask.compute`` call.
    """
    return ddf.to_parquet(
        path, write_metadata_file=True, write_index=False, compute=False
    )


def save_parquet_with_summary(
    inventory_id: str,
    ddf: dd.DataFrame,
    columns: list[dict],
) -> tuple[str, dict[str, dict]]:
    """Write partitions to GCS and compute per-column summaries in one pass.

    Flattens all lazy scalar reductions from ``_build_column_stats_graph``
    alongside the deferred parquet write into a single ``dask.compute`` call.
    Because the scalars and the write share the same dask expression graph,
    each partition is materialized exactly once.
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    write_delayed = _write_parquet(ddf, path)
    stats_graph = _build_column_stats_graph(ddf, columns)
    stats = _compute_write_and_stats(write_delayed, stats_graph)

    logger.info(f"Saved inventory data with summaries to {path}")
    return path, stats


def save_parquet_replace_with_summary(
    inventory_id: str,
    ddf: dd.DataFrame,
    columns: list[dict],
) -> tuple[str, dict[str, dict]]:
    """Replace an inventory's Parquet data in place with ``ddf`` and compute
    per-column summaries in one pass.

    ``ddf`` is typically derived by reading the inventory's *own* current
    Parquet (an in-place modification). Writing straight back to the live path
    would corrupt the data: ``dd.read_parquet`` is lazy, so the read executes
    during this write and would race the overwrite. Instead, the stats
    reductions and the write are fused into a single dask.compute call against
    a staging prefix, then swapped into place once complete:

    1. fuse write + stats reductions to ``{id}__rev`` in a single ``dask.compute``
       call by flattening lazy scalars alongside the deferred write,
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

    write_delayed = _write_parquet(ddf, staging_uri)
    stats_graph = _build_column_stats_graph(ddf, columns)
    stats = _compute_write_and_stats(write_delayed, stats_graph)

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
    return f"gs://{live_rel}", stats


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


def inventory_size(inventory_id: str) -> int:
    """Total GCS bytes of an inventory's Parquet dataset — its absolute footprint.

    Sums every object under the inventory's prefix (all part files plus the
    ``_metadata``/``_common_metadata`` footers). Because it measures the live
    prefix, an in-place replace (:func:`save_parquet_replace`) reads as the new
    footprint, never an accumulation (#342).
    """
    return storage_size(f"gs://{INVENTORIES_BUCKET}/{inventory_id}")


def delete_parquet(inventory_id: str) -> None:
    """Delete inventory Parquet data from GCS."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    try:
        delete_directory(path)
        logger.info(f"Deleted inventory data at {path}")
    except Exception as e:
        logger.warning(f"Failed to delete inventory data at {path}: {e}")
