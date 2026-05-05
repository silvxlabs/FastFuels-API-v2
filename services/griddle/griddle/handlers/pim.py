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
) -> xr.Dataset:
    """Fetch TreeMap plot imputation raster data.

    Args:
        roi: GeoDataFrame defining the region of interest
        version: TreeMap version year
        bands: List of band names to produce ("tm_id", "plt_cn")
        progress: Progress callback

    Returns:
        Dataset with one named variable per requested band
    """
    # Fetch the TreeMap COG raster (TM_ID pixel values)
    url = f"gs://{RASTERS_BUCKET}/TreeMap{version}.tif"
    with cog_env():
        raster = RasterConnection(url, connection_type="rioxarray", cache=True)
        data = raster.extract_window(
            roi=roi,
            interpolation_padding_cells=8,
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
        max_tm_id = int(tm_ids.max())
        lookup = np.zeros(max_tm_id + 1, dtype=np.int64)
        lookup[tm_ids] = plt_cns

        # Clip raster values to valid range for lookup
        raw = tm_id_da.values.astype(np.int64)
        raw_clipped = np.clip(raw, 0, max_tm_id)
        plt_cn_values = lookup[raw_clipped]

        # Zero out any values that were outside the mapping range
        plt_cn_values[raw < 0] = 0
        plt_cn_values[raw > max_tm_id] = 0

        plt_cn_da = xr.DataArray(
            plt_cn_values,
            dims=tm_id_da.dims,
            coords=tm_id_da.coords,
        )
        plt_cn_da = plt_cn_da.rio.write_crs(tm_id_da.rio.crs)
        plt_cn_da = plt_cn_da.rio.write_transform(tm_id_da.rio.transform())
        variables["plt_cn"] = plt_cn_da

    if not variables:
        raise ValueError("No bands requested — at least one band is required.")

    # Build dataset with CRS and transform
    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(tm_id_da.rio.crs)
    ds = ds.rio.write_transform(tm_id_da.rio.transform())
    return ds
