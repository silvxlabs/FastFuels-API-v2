"""
PIM (Plot Imputation Map) expansion handler.

Generates tree inventories by expanding FIA plot data across a domain
using a spatial point process on a PIM grid (TreeMap).
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pint
import xarray as xr
from fastfuels_core.trees import TreeSample

from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document
from standgen.columns import (
    BASE_COLUMNS,
    DROP_COLUMNS,
    RENAME_MAP,
)
from standgen.modifications import (
    _has_spatial_condition,
    apply_modifications,
    resolve_spatial_conditions,
)
from standgen.storage import load_grid, load_tree_table, save_parquet_with_summary
from standgen.treatments import apply_treatments

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Product-to-required-band mapping for PIM inventory expansion
PIM_PLOT_ID_BANDS = {
    "treemap": "tm_id",
}

# Column names vary by TreeMap version
TREEMAP_COLUMNS = {
    "2022": {"tree_id": "TM_ID", "plot_id": "TM_ID", "plt_cn": "PLT_CN"},
    "2020": {"tree_id": "TM_ID", "plot_id": "TM_ID", "plt_cn": "PLT_CN"},
    "2016": {"tree_id": "tm_id", "plot_id": "tm_id", "plt_cn": "CN"},
    "2014": {"tree_id": "tl_id", "plot_id": "tl_id", "plt_cn": "CN"},
}

# Columns needed from the tree table (before version-specific renaming)
TREE_TABLE_COLUMNS = ["SPCD", "STATUSCD", "DIA", "HT", "CR", "TPA_UNADJ"]


def handle_pim(
    inventory: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process a PIM expansion inventory request.

    Args:
        inventory: Full inventory document from Firestore
        source: Source dict with pim-specific fields
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference' key and 'columns' key with per-column
        summary statistics populated.
    """
    inventory_id = inventory["id"]
    source_pim_grid_id = source["source_pim_grid_id"]
    seed = source["seed"]
    point_process = source["point_process"]

    # Load source PIM grid document from Firestore
    try:
        _, grid_snapshot = get_document(GRIDS_COLLECTION, source_pim_grid_id)
        grid_doc = grid_snapshot.to_dict()
    except DocumentNotFoundError:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Source PIM grid '{source_pim_grid_id}' not found.",
            suggestion="Ensure the source PIM grid exists and has been processed.",
        )

    grid_source = grid_doc.get("source", {})
    product = grid_source.get("product", "treemap")
    version = grid_source.get("version", "2022")

    # Resolve required plot ID band for this product
    plot_id_band = PIM_PLOT_ID_BANDS.get(product)
    if plot_id_band is None:
        raise ProcessingError(
            code="UNSUPPORTED_PRODUCT",
            message=f"PIM product '{product}' is not supported for inventory expansion.",
            suggestion="Supported products: treemap",
        )

    # Load PIM grid data
    progress("Loading PIM grid data...", 10)
    grid_ds = load_grid(source_pim_grid_id)

    # Validate required band exists in grid
    if plot_id_band not in grid_ds.data_vars:
        available = list(grid_ds.data_vars)
        raise ProcessingError(
            code="MISSING_BAND",
            message=(
                f"Source PIM grid is missing the required '{plot_id_band}' band. "
                f"Available bands: {available}"
            ),
            suggestion=(
                f"Create a PIM grid that includes the '{plot_id_band}' band. "
                f"This band is required for {product} inventory expansion."
            ),
        )

    # Convert raster to plots GeoDataFrame
    plots = raster_to_plots_gdf(grid_ds, plot_id_band)
    logger.info(
        f"Extracted {len(plots)} plot pixels from PIM grid",
        extra={"inventory_id": inventory_id},
    )

    # Load and prepare tree table
    progress("Loading tree table...", 20)
    tree_table = load_tree_table(version)
    unique_plot_ids = plots["PLOT_ID"].unique()
    tree_df = filter_and_convert_tree_table(tree_table, unique_plot_ids, version)
    logger.info(
        f"Filtered tree table to {len(tree_df)} trees from {len(unique_plot_ids)} plots",
        extra={"inventory_id": inventory_id},
    )

    # Create TreeSample
    tree_sample = TreeSample(tree_df)

    # Run point process expansion (lazy)
    progress("Running point process...", 30)
    ddf = tree_sample.expand_to_roi(
        point_process,
        domain_gdf,
        lazy=True,
        plots=plots,
        seed=seed,
        chunk_size=1000,
    )

    # Rename columns: fastfuels-core → v2 schema
    progress("Transforming columns...", 75)
    rename_cols = {k: v for k, v in RENAME_MAP.items() if k in ddf.columns}
    ddf = ddf.rename(columns=rename_cols)

    # Drop internal columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in ddf.columns]
    if cols_to_drop:
        ddf = ddf.drop(columns=cols_to_drop)

    # Select final column set
    ddf = ddf[BASE_COLUMNS]

    # Apply modifications if present. Resolve spatial-condition geometries once
    # here (off the per-partition path) when any are present.
    modifications = inventory.get("modifications", [])
    if modifications:
        progress("Applying modifications...", 77)
        if _has_spatial_condition(modifications):
            modifications = resolve_spatial_conditions(
                modifications, inventory["domain_id"], domain_gdf.crs
            )
        ddf = ddf.map_partitions(apply_modifications, modifications)

    # Apply treatments if present. apply_treatments keeps diameter treatments
    # lazy (row-local, streamed per-partition) and materializes only the treated
    # region for basal-area treatments (a global stand reduction, bounded by an
    # area limit). It returns a lazy dask DataFrame, so the partitioned write
    # below is unchanged.
    treatments = inventory.get("treatments", [])
    if treatments:
        progress("Applying treatments...", 78)
        if _has_spatial_condition(treatments):
            treatments = resolve_spatial_conditions(
                treatments, inventory["domain_id"], domain_gdf.crs
            )
        ddf = apply_treatments(ddf, treatments, domain_gdf, seed=seed)

    # Write Parquet to GCS (lazy — each partition writes separately)
    progress("Writing inventory data...", 80)
    _, stats = save_parquet_with_summary(inventory_id, ddf, inventory["columns"])

    # Compute georeference from domain
    progress("Computing georeference...", 95)
    georeference = compute_georeference(domain_gdf)

    progress("Complete", 100)

    return {
        "georeference": georeference,
        "columns": [
            {**col, "summary": stats.get(col["key"])} for col in inventory["columns"]
        ],
    }


def raster_to_plots_gdf(dataset: xr.Dataset, plot_id_band: str) -> gpd.GeoDataFrame:
    """Convert an xarray Dataset's plot ID variable to a plots GeoDataFrame.

    Includes ALL raster cells (no-plot cells map to 0) so that cells
    without trees act as zero-density anchors during density interpolation.
    Without these anchors, the interpolation produces positive density
    everywhere and trees get placed in areas that should be empty.

    Args:
        dataset: xarray Dataset with spatial coordinates
        plot_id_band: Name of the data variable containing plot IDs

    Returns:
        GeoDataFrame with PLOT_ID column and Point geometry at pixel centers
    """
    da = dataset[plot_id_band]
    nodata = da.rio.nodata
    df = da.to_dataframe(name="PLOT_ID").reset_index()
    # No-plot cells must read as 0 (the zero-density anchors described above).
    # Grids load raw (mask_and_scale=False), so the nodata sentinel is preserved
    # rather than turned into NaN — map it (and any NaN) to 0 explicitly.
    df["PLOT_ID"] = df["PLOT_ID"].where(df["PLOT_ID"] != nodata).fillna(0).astype(int)
    return gpd.GeoDataFrame(
        {"PLOT_ID": df["PLOT_ID"].values},
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=str(da.rio.crs),
    )


def filter_and_convert_tree_table(
    tree_table: pd.DataFrame,
    plot_ids: np.ndarray,
    version: str,
) -> pd.DataFrame:
    """Filter tree table to relevant plots and convert units to metric.

    Args:
        tree_table: Full TreeMap tree table
        plot_ids: Array of plot IDs present in the grid
        version: TreeMap version year

    Returns:
        DataFrame ready for TreeSample construction with columns:
        TREE_ID, PLOT_ID, SPCD, STATUSCD, DIA (cm), HT (m), CR (fraction), TPA (trees/m²)
    """
    col_map = TREEMAP_COLUMNS.get(version)
    if col_map is None:
        raise ProcessingError(
            code="UNSUPPORTED_VERSION",
            message=f"TreeMap version '{version}' is not supported.",
            suggestion="Supported versions: 2014, 2016, 2020, 2022",
        )

    tree_id_col = col_map["tree_id"]
    plot_id_col = col_map["plot_id"]

    # Select columns (version-specific ID + standard columns)
    select_cols = [tree_id_col, plot_id_col] + TREE_TABLE_COLUMNS
    # Deduplicate in case tree_id and plot_id are the same column
    select_cols = list(dict.fromkeys(select_cols))
    available = [c for c in select_cols if c in tree_table.columns]
    df = tree_table[available].copy()

    # Rename version-specific columns to standard names
    if tree_id_col != "TREE_ID" and tree_id_col in df.columns:
        df = df.rename(columns={tree_id_col: "TREE_ID"})
    if plot_id_col != "PLOT_ID":
        # plot_id_col might be same as tree_id_col (already renamed)
        src = "TREE_ID" if tree_id_col == plot_id_col else plot_id_col
        df["PLOT_ID"] = df[src]

    # Filter to plots present in the grid
    df = df[df["PLOT_ID"].isin(plot_ids)]

    # Unit conversions using pint
    # DIA: inches → cm
    df["DIA"] = Q_(df["DIA"].values, "inch").to("cm").magnitude

    # HT: feet → meters
    df["HT"] = Q_(df["HT"].values, "ft").to("m").magnitude

    # CR: percent → fraction (NaN for dead trees → fill with 0)
    df["CR"] = df["CR"].fillna(0).values / 100.0

    # TPA_UNADJ: trees/acre → trees/m²
    df["TPA"] = Q_(df["TPA_UNADJ"].values, "1/acre").to("1/m**2").magnitude
    if "TPA_UNADJ" in df.columns:
        df = df.drop(columns=["TPA_UNADJ"])

    return df


def compute_georeference(domain_gdf: gpd.GeoDataFrame) -> dict:
    """Compute georeference from domain GeoDataFrame (already in projected UTM CRS)."""
    bounds = domain_gdf.total_bounds  # [minx, miny, maxx, maxy]
    return {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in bounds],
    }
