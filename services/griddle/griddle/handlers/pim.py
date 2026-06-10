"""
PIM (Plot Imputation Map) source handlers.

Pure functions that fetch PIM data for a domain extent.
All handlers return xr.Dataset where each variable name is a band name.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from lib.alignment import RESAMPLING_METHOD_MAP, resolve_alignment_destination
from lib.config import RASTERS_BUCKET, TABLES_BUCKET
from lib.raster import RasterConnection, cog_env

# Column names vary by TreeMap version
TREEMAP_COLUMNS = {
    "2022": ("TM_ID", "PLT_CN"),
    "2020": ("TM_ID", "PLT_CN"),
    "2016": ("tm_id", "CN"),
    "2014": ("tl_id", "CN"),
}


def fetch_treemap(
    roi: gpd.GeoDataFrame,
    version: str,
    bands: list[str],
    progress: Callable[[str, int | None], None],
    extent_buffer_cells: int = 0,
    alignment: dict | None = None,
    target_grid_doc: dict | None = None,
) -> xr.Dataset:
    """Fetch TreeMap plot imputation raster data.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: TreeMap version year
        bands: List of band names to produce ("tm_id", "plt_cn")
        progress: Progress callback
        extent_buffer_cells: Result-grid cells of buffer around the ROI
        alignment: Alignment specification dict (see ``GridAlignmentSpecification``).
            Defaults to ``{"target": "domain"}`` when omitted. The TM_ID raster
            is reprojected to the alignment destination in the single
            ``rio.reproject`` performed inside ``extract_window``.
        target_grid_doc: Loaded grid document used as the alignment target
            when ``alignment["target"] == "grid"``. Required in that case.

    Returns:
        Dataset with one named variable per requested band. The PLT_CN band
        is derived from the aligned TM_ID values, so both bands share the
        same transform/shape.
    """
    alignment = alignment or {"target": "domain"}

    url = f"gs://{RASTERS_BUCKET}/TreeMap{version}.tif"
    with cog_env():
        raster = RasterConnection(url, connection_type="rioxarray", cache=True)
        method_name = alignment.get("method") or "nearest"
        dest = resolve_alignment_destination(
            alignment,
            roi,
            target_grid_doc,
            raster.target_native_resolution(roi)[0],
            extent_buffer_cells=extent_buffer_cells,
        )
        data = raster.extract_window(
            roi=roi,
            interpolation_padding_cells=extent_buffer_cells,
            resampling=RESAMPLING_METHOD_MAP[method_name],
            destination_resolution=alignment.get("resolution")
            if alignment["target"] == "native"
            else None,
            **dest,
        )
    tm_id_da = data.squeeze("band", drop=True)

    variables = {}

    if "tm_id" in bands:
        variables["tm_id"] = tm_id_da

    if "plt_cn" in bands:
        progress("Loading tree table for PLT_CN mapping...", 40)

        tm_col, cn_col = TREEMAP_COLUMNS.get(version, ("TM_ID", "PLT_CN"))
        table_url = f"gs://{TABLES_BUCKET}/TreeMap{version}_tree_table.parquet"
        df = pd.read_parquet(table_url, columns=[tm_col, cn_col])

        # Build unique TM_ID -> PLT_CN mapping
        mapping_df = df[[tm_col, cn_col]].drop_duplicates(subset=[tm_col])
        tm_ids = mapping_df[tm_col].values
        plt_cns = mapping_df[cn_col].values

        # Vectorized lookup via index array
        nodata_value = np.iinfo(np.int64).max
        max_tm_id = int(tm_ids.max())
        lookup = np.full(max_tm_id + 1, nodata_value, dtype=np.int64)
        # lookup = np.zeros(max_tm_id + 1, dtype=np.int64)
        lookup[tm_ids] = plt_cns

        # Clip raster values to valid range for lookup
        raw = tm_id_da.values.astype(np.int64)
        raw_clipped = np.clip(raw, 0, max_tm_id)
        plt_cn_values = lookup[raw_clipped]

        # Mask out of range values
        plt_cn_values[raw < 1] = nodata_value
        plt_cn_values[raw > max_tm_id] = nodata_value

        plt_cn_da = xr.DataArray(
            plt_cn_values,
            dims=tm_id_da.dims,
            coords=tm_id_da.coords,
        )
        plt_cn_da = plt_cn_da.rio.write_crs(tm_id_da.rio.crs)
        plt_cn_da = plt_cn_da.rio.write_transform(tm_id_da.rio.transform())
        plt_cn_da = plt_cn_da.rio.write_nodata(nodata_value)
        variables["plt_cn"] = plt_cn_da

    if not variables:
        raise ValueError("No bands requested — at least one band is required.")

    # Build dataset with CRS and transform
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(tm_id_da.rio.crs)
    ds = ds.rio.write_transform(tm_id_da.rio.transform())
    return ds
