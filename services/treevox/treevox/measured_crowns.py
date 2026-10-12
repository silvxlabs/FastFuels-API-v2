"""Measured crown footprints for tree voxelization (#617).

Handler-side module — never imported by spawned workers.

A crown Feature holds one polygon per segmented crown, keyed by `tree_id` and
traced along the edges of the CHM grid it was built from. Rasterizing the
polygons back onto that CHM lattice recovers each crown's cells exactly; the
CHM gives each cell's top. Workers receive one `Footprint` per tree and build
the crown with fastfuels-core (`treevox.voxelize.build_measured_biomass`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers `.rio` accessor
import xarray as xr
from affine import Affine
from rasterio.features import rasterize

from lib.config import FEATURES_BUCKET, GRIDS_BUCKET
from lib.zarr_utils import load_zarr
from treevox.errors import ProcessingError
from treevox.firestore_io import load_feature
from treevox.voxelize import Footprint, stem_in_crown

logger = logging.getLogger(__name__)

TILE_CELLS = 2048  # CHM cells a side per tile read by `load_footprints`


@dataclass
class MeasuredFootprints:
    """Footprints of the inventory's trees that have a crown in the Feature.

    `footprints` is keyed by `tree_id`. `max_top` is the highest CHM top among
    them (m), or None when there are none. `ignored_crowns` counts the
    Feature's crowns whose tree is not in the inventory. `stem_outside` holds
    the trees whose matched crown does not contain their stem; their crowns
    are left out of `footprints`.
    """

    footprints: dict[int, Footprint]
    max_top: float | None
    ignored_crowns: int
    stem_outside: frozenset[int] = frozenset()


def load_footprints(feature_id: str, trees: pd.DataFrame) -> MeasuredFootprints:
    """Read a crown Feature and its CHM, and cut out each matched footprint.

    A crown matches a tree by `tree_id`. Crowns whose tree is not in `trees`
    (e.g. removed by a treatment) are ignored. A matched crown that does not
    contain its tree's stem, from `trees`' `x` and `y`, came from another
    inventory with the same `tree_id`s and is set aside.
    """
    tree_ids = trees["tree_id"]
    feature = load_feature(feature_id)
    chm_grid_id = feature["source"]["source_chm_grid_id"]

    crowns = gpd.read_parquet(
        f"gs://{FEATURES_BUCKET}/{feature['domain_id']}/{feature_id}.parquet",
        columns=["tree_id", "geometry"],
    )
    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty]
    matched = crowns[crowns["tree_id"].isin(tree_ids)].drop_duplicates("tree_id")
    ignored = int((~crowns["tree_id"].isin(tree_ids)).sum())
    if matched.empty:
        return MeasuredFootprints(footprints={}, max_top=None, ignored_crowns=ignored)

    chm = load_chm(chm_grid_id)
    footprints: dict[int, Footprint] = {}
    for tile in _tiles(matched, chm):
        values, transform = read_chm_window(chm, tile.total_bounds)
        footprints.update(
            footprints_from_crowns(
                tile["tree_id"].to_numpy(), tile.geometry.to_numpy(), values, transform
            )
        )
    stems = dict(
        zip(
            trees["tree_id"].astype("int64"),
            zip(trees["x"].astype("float64"), trees["y"].astype("float64")),
        )
    )
    stem_outside = frozenset(
        tree_id
        for tree_id, footprint in footprints.items()
        if footprint.mask.any() and not stem_in_crown(*stems[tree_id], footprint)
    )
    footprints = {k: f for k, f in footprints.items() if k not in stem_outside}
    tops = [np.nanmax(f.top) for f in footprints.values() if f.mask.any()]
    return MeasuredFootprints(
        footprints=footprints,
        max_top=float(max(tops)) if tops else None,
        ignored_crowns=ignored,
        stem_outside=stem_outside,
    )


def _tiles(crowns: gpd.GeoDataFrame, chm: xr.DataArray) -> list[gpd.GeoDataFrame]:
    """Group crowns into tiles of about `TILE_CELLS` CHM cells a side.

    A crown belongs to the tile holding its bounding box's north-west corner,
    so each tile's window is at most one crown wider than the tile. Reading
    the CHM one tile at a time keeps the parent's memory bounded by the tile
    rather than by the domain.
    """
    transform = chm.rio.transform()
    size = TILE_CELLS * transform.a
    bounds = crowns.bounds
    keys = pd.DataFrame(
        {
            "col": np.floor((bounds["minx"] - transform.c) / size),
            "row": np.floor((transform.f - bounds["maxy"]) / size),
        },
        index=crowns.index,
    )
    return [crowns.loc[index] for index in keys.groupby(["row", "col"]).groups.values()]


def load_chm(grid_id: str) -> xr.DataArray:
    """A grid's `chm` band, lazily, with no-data cells set to NaN."""
    ds = load_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}")
    if "chm" not in ds.data_vars:
        raise ProcessingError(
            code="MISSING_BAND",
            message=f"CHM grid {grid_id} has no 'chm' band.",
        )
    chm = ds["chm"]
    nodata = chm.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        chm = chm.where(chm != nodata)
    return chm


def read_chm_window(chm: xr.DataArray, bounds: np.ndarray) -> tuple[np.ndarray, Affine]:
    """Read the CHM cells covering `bounds` (minx, miny, maxx, maxy), plus one.

    Returns the float64 heights and the window's affine transform. The CHM
    must be north-up with square cells.
    """
    transform = chm.rio.transform()
    a, b, c, d, e, f = transform[:6]
    if b != 0 or d != 0 or a <= 0 or e >= 0 or not math.isclose(a, -e):
        raise ProcessingError(
            code="INVALID_CHM_GRID",
            message="The crown Feature's CHM grid is not north-up with square cells.",
        )
    nrows, ncols = chm.shape
    minx, miny, maxx, maxy = bounds
    col0 = max(0, math.floor((minx - c) / a) - 1)
    col1 = min(ncols, math.ceil((maxx - c) / a) + 1)
    row0 = max(0, math.floor((f - maxy) / -e) - 1)
    row1 = min(nrows, math.ceil((f - miny) / -e) + 1)
    row_dim, col_dim = chm.dims
    window = chm.isel({row_dim: slice(row0, row1), col_dim: slice(col0, col1)})
    values = np.asarray(window.values, dtype=np.float64)
    return values, transform * Affine.translation(col0, row0)


def footprints_from_crowns(
    tree_ids: np.ndarray,
    geometries: np.ndarray,
    chm: np.ndarray,
    transform: Affine,
) -> dict[int, Footprint]:
    """Rasterize crown polygons onto the CHM lattice and cut out each crown.

    The polygons follow CHM cell edges, so the cells whose centers they
    contain are exactly the crown's cells. Crowns never overlap. Cells with
    no CHM value are left out of the footprint. Returns one `Footprint` per
    tree, keyed by `tree_id`; a crown with no cells gets an empty footprint.
    """
    nrows, ncols = chm.shape
    labels = rasterize(
        zip(geometries, range(1, len(geometries) + 1)),
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    flat = labels.ravel()
    cells = np.flatnonzero(flat)
    order = np.argsort(flat[cells], kind="stable")
    cells = cells[order]
    starts = np.searchsorted(flat[cells], np.arange(1, len(geometries) + 2))
    rows, cols = np.divmod(cells, ncols)
    tops = chm.ravel()[cells]

    resolution = float(transform.a)
    footprints: dict[int, Footprint] = {}
    for k, tree_id in enumerate(tree_ids):
        r, c, t = (arr[starts[k] : starts[k + 1]] for arr in (rows, cols, tops))
        valid = np.isfinite(t)
        r, c, t = r[valid], c[valid], t[valid]
        if r.size == 0:
            footprints[int(tree_id)] = Footprint(
                mask=np.zeros((1, 1), dtype=bool),
                top=np.full((1, 1), np.nan),
                origin=(transform.c, transform.f),
                resolution=resolution,
            )
            continue
        r0, c0 = int(r.min()), int(c.min())
        shape = (int(r.max()) - r0 + 1, int(c.max()) - c0 + 1)
        mask = np.zeros(shape, dtype=bool)
        top = np.full(shape, np.nan)
        mask[r - r0, c - c0] = True
        top[r - r0, c - c0] = t
        x_west, y_north = transform * (c0, r0)
        footprints[int(tree_id)] = Footprint(
            mask=mask,
            top=top,
            origin=(float(x_west), float(y_north)),
            resolution=resolution,
        )
    return footprints
