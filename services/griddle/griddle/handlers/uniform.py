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

from lib.alignment import lattice_from_bounds


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
        bands: List of band dicts from the source, each with "key" and
            "value" keys. The band's key (e.g., "fuel_moisture.1hr") is
            used as the Dataset variable name.
        resolution: Grid cell size in meters.
        progress: Progress callback (message, percent).

    Returns:
        Dataset with one named variable per band, each filled with the
        constant value. CRS and transform are set via rioxarray.
    """
    progress("Preparing uniform grid...", 10)

    crs = domain_gdf.crs
    minx, miny, _, _ = domain_gdf.total_bounds

    # Compute the domain-anchored lattice (transform + shape) shared with
    # every external-source fetcher.
    transform, (height, width) = lattice_from_bounds(
        tuple(domain_gdf.total_bounds), resolution
    )

    # Each band's "key" IS the storage key (e.g., "fuel_moisture.1hr")
    progress("Generating uniform grid...", 40)

    y_coords = (
        np.arange(height) * (-resolution)
        + (miny + height * resolution)
        - resolution / 2
    )
    x_coords = np.arange(width) * resolution + minx + resolution / 2

    variables = {}
    for band in bands:
        key = band["key"]
        value = band["value"]

        if isinstance(value, int):
            data = np.full((height, width), value, dtype=np.int32)
            nodata_value = np.iinfo(np.int32).max
        else:
            data = np.full((height, width), value, dtype=np.float32)
            nodata_value = np.nan

        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_nodata(nodata_value)
        variables[key] = da

    progress("Building dataset...", 80)

    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)

    return ds
