"""Grid-alignment math.

Pure functions used by both the API (validation) and Griddle (handler
runtime) to compute output lattices for raster reprojection.
"""

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.transform import from_bounds


def lattice_from_bounds(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> tuple[Affine, tuple[int, int]]:
    """Return (transform, (height, width)) for a north-up grid anchored at
    the lower-left of `bounds`, cell size `resolution`, covering `bounds`
    via ceil()."""
    minx, miny, maxx, maxy = bounds
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    transform = from_bounds(
        minx,
        miny,
        minx + width * resolution,
        miny + height * resolution,
        width,
        height,
    )
    return transform, (height, width)


def target_grid_bounds(georef: dict) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) for the lattice described by a
    persisted georeference dict (``transform``: 6-tuple, ``shape``:
    (height, width))."""
    a, _, c, _, e, f = georef["transform"]
    h, w = georef["shape"]
    return (c, f + h * e, c + w * a, f)


def resolve_alignment_destination(
    alignment: dict,
    domain_gdf: gpd.GeoDataFrame,
    target_grid_doc: dict | None,
    source_native_resolution: float,
) -> dict:
    """Map a persisted alignment dict to ``rio.reproject``-style destination
    kwargs.

    Returns a dict that may include ``destination_crs``,
    ``destination_transform``, ``destination_shape``. An empty dict means
    "no override; let the caller use its default reprojection path."
    """
    target = alignment["target"]

    if target == "domain":
        resolution = alignment.get("resolution") or source_native_resolution
        transform, shape = lattice_from_bounds(
            tuple(domain_gdf.total_bounds), resolution
        )
        return {
            "destination_crs": domain_gdf.crs,
            "destination_transform": transform,
            "destination_shape": shape,
        }

    if target == "grid":
        if target_grid_doc is None:
            raise ValueError(
                "alignment.target='grid' requires the target grid document"
            )
        georef = target_grid_doc["georeference"]
        if alignment.get("resolution") is None:
            return {
                "destination_crs": georef["crs"],
                "destination_transform": Affine(*georef["transform"]),
                "destination_shape": tuple(georef["shape"]),
            }
        transform, shape = lattice_from_bounds(
            target_grid_bounds(georef), alignment["resolution"]
        )
        return {
            "destination_crs": georef["crs"],
            "destination_transform": transform,
            "destination_shape": shape,
        }

    if target == "native":
        if alignment.get("resolution") is None:
            return {}
        return {"destination_crs": domain_gdf.crs}

    raise ValueError(f"unknown alignment target: {target}")
