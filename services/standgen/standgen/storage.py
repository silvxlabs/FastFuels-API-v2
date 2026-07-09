"""Storage utilities for Standgen."""

import logging
import math

import dask
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET, TABLES_BUCKET
from lib.errors import ProcessingError
from lib.gcs import delete_directory, exists, get_gcsfs_client, storage_size
from lib.zarr_utils import load_zarr
from standgen.summarize import _build_column_stats_graph, _build_tree_forestry_graph

logger = logging.getLogger(__name__)


FLOAT_STATS = {"min", "max", "mean", "std"}


def _finite_or_none(value) -> float | None:
    """Coerce a reduction result to float, mapping NaN/inf to None.

    Continuous reductions can be non-finite: an all-null column yields NaN
    min/max/mean/std, and a column with a single non-null value yields NaN
    sample std (ddof=1). NaN/inf are not JSON-serializable — the API serves
    inventories with a JSONResponse that uses ``allow_nan=False`` — so they
    must not reach Firestore.
    """
    value = float(value)
    return value if math.isfinite(value) else None


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


def _fused_compute(
    write_delayed,
    stats_graph: dict[str, dict],
    forestry_delayed=None,
) -> tuple[dict[str, dict], dict | None]:
    """Flatten lazy reductions and a deferred parquet write into a single
    dask.compute call, then reassemble the stats dict.

    Handles both continuous columns (which use a single ``.agg()`` lazy Series)
    and categorical columns (which use individual lazy scalars).

    Args:
        write_delayed: Deferred parquet write from ``ddf.to_parquet(..., compute=False)``.
        stats_graph: Dict of lazy reductions from ``_build_column_stats_graph``.
        forestry_delayed: Optional dask.delayed from ``_build_tree_forestry_graph``.

    Returns:
        Tuple of (stats dict, forestry metrics dict or None).
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

    to_compute = [write_delayed, *flat_lazy]
    if forestry_delayed is not None:
        to_compute.append(forestry_delayed)

    results = dask.compute(*to_compute)
    computed_values = results[1 : 1 + len(flat_lazy)]
    forestry_metrics = results[-1] if forestry_delayed is not None else None

    stats = {}
    for i, (k, s) in enumerate(flat_keys):
        if k not in stats:
            stats[k] = {"type": stats_graph[k]["type"]}
        val = computed_values[i]
        if s in FLOAT_STATS:
            stats[k][s] = _finite_or_none(val)
        else:
            stats[k][s] = int(val)

    return stats, forestry_metrics


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


def _build_delayed_graph(
    ddf: dd.DataFrame,
    path: str,
    columns: list[dict],
    inventory_type: str,
    domain_gdf: gpd.GeoDataFrame | None,
    top_species_groups: int,
):
    write_delayed = _write_parquet(ddf, path)
    stats_graph = _build_column_stats_graph(ddf, columns)
    forestry_delayed = (
        _build_tree_forestry_graph(ddf, domain_gdf, top_species_groups)
        if inventory_type == "tree"
        and domain_gdf is not None
        and "dbh" in ddf.columns
        and "fia_species_code" in ddf.columns
        else None
    )
    return _fused_compute(write_delayed, stats_graph, forestry_delayed)


def save_parquet_with_summary(
    inventory_id: str,
    ddf: dd.DataFrame,
    columns: list[dict],
    inventory_type: str = "tree",
    domain_gdf: gpd.GeoDataFrame | None = None,
    top_species_groups: int = 5,
) -> tuple[str, dict[str, dict], dict | None]:
    """Write partitions to GCS and compute per-column summaries in one pass.

    Flattens all lazy scalar reductions from ``_build_column_stats_graph``
    alongside the deferred parquet write into a single ``dask.compute`` call.
    Because the scalars and the write share the same dask expression graph,
    each partition is materialized exactly once.

    For tree inventories, also computes stand-level forestry metrics in the same
    fused pass via _build_tree_forestry_graph. Requires ``domain_gdf`` for
    per-area metrics; if absent, forestry metrics return ``None``.

    Args:
        inventory_id: Inventory document ID.
        ddf: Lazy dask DataFrame to write and summarize.
        columns: List of column metadata dicts with 'key' and 'type' fields.
        inventory_type: Inventory type string.
        domain_gdf: Domain geometry for per-area forestry metric computation.
        top_species_groups: Number of top FIA species groups to include in
            dominant species list. Defaults to 5.

    Returns:
        Three-tuple of (GCS path, per-column stats dict, forestry metrics dict or None).
    """
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    stats, forestry_metrics = _build_delayed_graph(
        ddf, path, columns, inventory_type, domain_gdf, top_species_groups
    )

    logger.info(f"Saved inventory data with summaries to {path}")
    return path, stats, forestry_metrics


def save_parquet_replace_with_summary(
    inventory_id: str,
    ddf: dd.DataFrame,
    columns: list[dict],
    inventory_type: str = "tree",
    domain_gdf: gpd.GeoDataFrame | None = None,
    top_species_groups: int = 5,
) -> tuple[str, dict[str, dict], dict | None]:
    """Replace an inventory's Parquet data in place with ``ddf`` and compute
    per-column summaries in one pass.

    ``ddf`` is typically derived by reading the inventory's *own* current
    Parquet (an in-place modification). Writing straight back to the live path
    would corrupt the data: ``dd.read_parquet`` is lazy, so the read executes
    during this write and would race the overwrite. Instead, the stats
    reductions, forestry metrics (for tree inventories), and the write are fused
    into a single ``dask.compute`` call against a staging prefix, then swapped
    into place once complete:

    1. fuse write + stats reductions + forestry metrics to ``{id}__rev`` in a
       single ``dask.compute`` call by flattening lazy scalars alongside the
       deferred write,
    2. delete the live ``{id}`` directory,
    3. server-side copy ``{id}__rev`` -> ``{id}`` (same bucket, no egress),
    4. delete the staging ``{id}__rev`` directory.

    Requires ``domain_gdf`` for per-area forestry metrics; if absent, forestry
    metrics return ``None``.

    Args:
        inventory_id: Inventory document ID.
        ddf: Lazy dask DataFrame to write and summarize.
        columns: List of column metadata dicts with 'key' and 'type' fields.
        inventory_type: Inventory type string.
        domain_gdf: Domain geometry for per-area forestry metric computation.
        top_species_groups: Number of top FIA species groups to include in
            dominant species list. Defaults to 5.

    Returns:
        Three-tuple of (GCS path, per-column stats dict, forestry metrics dict or None).
    """
    live_rel = f"{INVENTORIES_BUCKET}/{inventory_id}"
    staging_rel = f"{INVENTORIES_BUCKET}/{inventory_id}__rev"
    staging_uri = f"gs://{staging_rel}"

    # Clear any staging dir left behind by a previously failed attempt.
    if exists(staging_uri):
        delete_directory(staging_uri)

    stats, forestry_metrics = _build_delayed_graph(
        ddf, staging_uri, columns, inventory_type, domain_gdf, top_species_groups
    )

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
    return f"gs://{live_rel}", stats, forestry_metrics


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
