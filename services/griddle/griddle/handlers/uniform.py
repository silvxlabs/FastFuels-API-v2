"""
Uniform (constant-value) grid handler.

Creates grids where every cell is filled with a user-specified constant value.
Returns xr.Dataset where each variable name is a band key.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_bounds


def create_uniform_grid(
    domain_gdf: gpd.GeoDataFrame,
    bands: list[dict],
    resolution: float,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Create a uniform grid for the given domain.

    Args:
        domain_gdf: GeoDataFrame defining the region of interest. Must be
            in a projected CRS (domains are always projected at creation).
        bands: List of band dicts from the source, each with "quantity" and
            "value" keys. The quantity's key (from UNIFORM_QUANTITY_DEFS)
            is used as the Dataset variable name.
        resolution: Grid cell size in meters.
        progress: Progress callback (message, percent).

    Returns:
        Dataset with one named variable per band, each filled with the
        constant value. CRS and transform are set via rioxarray.
    """
    progress("Preparing uniform grid...", 10)

    crs = domain_gdf.crs
    minx, miny, maxx, maxy = domain_gdf.total_bounds

    # Compute grid dimensions
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))

    # Build affine transform (north-up convention)
    transform = from_bounds(
        minx, miny, minx + width * resolution, miny + height * resolution, width, height
    )

    # Map quantity values to their band keys
    # The quantity enum value IS the key (e.g., "fuel_moisture.1hr")
    progress("Generating uniform grid...", 40)

    y_coords = (
        np.arange(height) * (-resolution)
        + (miny + height * resolution)
        - resolution / 2
    )
    x_coords = np.arange(width) * resolution + minx + resolution / 2

    variables = {}
    for band in bands:
        key = band["quantity"]
        value = band["value"]

        if isinstance(value, int):
            data = np.full((height, width), value, dtype=np.int32)
        else:
            data = np.full((height, width), value, dtype=np.float32)

        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        )
        variables[key] = da

    progress("Building dataset...", 80)

    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)

    return ds
