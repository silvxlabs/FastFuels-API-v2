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
    ``(height, width)`` for 2D rasters or ``(z, height, width)`` for 3D
    voxel grids — only the trailing two dims are read)."""
    a, _, c, _, e, f = georef["transform"]
    h, w = georef["shape"][-2:]
    return (c, f + h * e, c + w * a, f)


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float]:
    """Symmetrically grow ``bounds`` by ``padding`` meters on every side."""
    minx, miny, maxx, maxy = bounds
    return (minx - padding, miny - padding, maxx + padding, maxy + padding)


def resolve_alignment_destination(
    alignment: dict,
    domain_gdf: gpd.GeoDataFrame,
    target_grid_doc: dict | None,
    source_native_resolution: float,
    extent_buffer_cells: int = 0,
) -> dict:
    """Map a persisted alignment dict to ``rio.reproject``-style destination
    kwargs.

    Returns a dict that may include ``destination_crs``,
    ``destination_transform``, ``destination_shape``. An empty dict means
    "no override; let the caller use its default reprojection path."

    ``extent_buffer_cells`` extends the destination lattice by N output
    cells on every side for ``target='domain'`` and ``target='grid'`` —
    the destination-override branch in ``extract_window`` skips the
    trailing ROI-clip step, so the buffer must be baked into the
    destination spec itself. Origin shifts by exactly
    ``extent_buffer_cells * resolution`` meters, so buffered output cells
    still nest cleanly with the same-anchor unbuffered lattice. For
    ``target='native'`` the buffer is applied at the clip step inside
    ``extract_window``, so the helper passes through unchanged.
    """
    target = alignment["target"]

    if target == "domain":
        resolution = alignment.get("resolution") or source_native_resolution
        bounds = _expand_bounds(
            tuple(domain_gdf.total_bounds), extent_buffer_cells * resolution
        )
        transform, shape = lattice_from_bounds(bounds, resolution)
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
        resolution = alignment.get("resolution")
        if resolution is None:
            # Cell size matches the target grid's transform.
            cell_size = abs(Affine(*georef["transform"]).a)
            bounds = _expand_bounds(
                target_grid_bounds(georef), extent_buffer_cells * cell_size
            )
            transform, shape = lattice_from_bounds(bounds, cell_size)
            return {
                "destination_crs": georef["crs"],
                "destination_transform": transform,
                "destination_shape": shape,
            }
        bounds = _expand_bounds(
            target_grid_bounds(georef), extent_buffer_cells * resolution
        )
        transform, shape = lattice_from_bounds(bounds, resolution)
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
