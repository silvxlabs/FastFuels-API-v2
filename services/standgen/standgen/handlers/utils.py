# standgen/algorithms/lmf.py (or wherever you keep your algorithms)

import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from scipy.ndimage import find_objects, label, maximum_filter
from shapely.geometry import Point


def find_treetops_lmf(
    chm_da: xr.DataArray, min_height: float, footprint_size: int
) -> gpd.GeoDataFrame:
    """Finds treetops from a CHM using a Local Maximum Filter (LMF).

    Args:
        chm_da (xr.DataArray): The Canopy Height Model data array.
        min_height (float): Minimum height threshold in CHM units (meters).
        footprint_size (int): Diameter of the circular footprint in pixels.
            Must be an odd integer.

    Returns:
        gpd.GeoDataFrame: Detected treetops with 'height' and 'geometry'.
    """
    if footprint_size % 2 == 0:
        raise ValueError("footprint_size must be an odd integer.")

    # Extract raw numpy array, transform, and crs directly from xarray
    chm = chm_da.values
    # Note: rioxarray must be imported/available for the .rio accessor
    transform = chm_da.rio.transform()
    crs = chm_da.rio.crs

    # Generate a circular footprint for the filter
    y, x = np.ogrid[
        -footprint_size // 2 : footprint_size // 2 + 1,
        -footprint_size // 2 : footprint_size // 2 + 1,
    ]
    footprint = x * x + y * y <= (footprint_size // 2) ** 2

    # Apply the maximum filter
    chm_max_filtered = maximum_filter(chm, footprint=footprint)

    # Identify peaks
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)

    # Find connected components of local maxima
    labeled_maxima, num_labels = label(local_maxima_mask)

    if num_labels == 0:
        return gpd.GeoDataFrame(columns=["height", "geometry"], crs=crs)

    # Find center of bounding box for each region
    object_slices = find_objects(labeled_maxima)
    rows, cols = [], []
    for slc in object_slices:
        row_center = (slc[0].start + slc[0].stop) // 2
        col_center = (slc[1].start + slc[1].stop) // 2
        rows.append(row_center)
        cols.append(col_center)

        # Get height and coordinates
        heights = chm[rows, cols]
        xs, ys = rio.transform.xy(transform, rows, cols)  # Safely get pixel centers

    # Create GeoDataFrame
    geometry = [Point(x, y) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame({"height": heights, "geometry": geometry}, crs=crs)
