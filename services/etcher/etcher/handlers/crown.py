"""
Crown feature handler for CHM segmentation.

Grows one crown per inventory tree on a CHM grid with core's Dalponte & Coomes
(2016) segmentation, traces each crown's cells into a polygon, and saves the
polygons keyed by the tree's ``tree_id`` as GeoParquet.
"""

import logging
import math

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import shapely
import xarray as xr
from affine import Affine
from fastfuels_core.itd.crown_segmentation import dalponte2016
from rasterio.features import shapes
from shapely.geometry import Point, shape

from etcher.errors import ProcessingError
from etcher.handlers.road import compute_georeference
from etcher.storage import save_features
from lib.config import GRIDS_BUCKET, INVENTORIES_BUCKET
from lib.zarr_utils import load_zarr

logger = logging.getLogger(__name__)

# Side length (cells) of each block segmented and traced on its own.
CHUNK_SIZE = 2048

SEED_COLUMNS = ["tree_id", "x", "y", "height"]


def handle_chm(
    feature: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process a CHM crown feature request.

    Args:
        feature: Full feature document from Firestore
        source: Source dict with the CHM, inventory and segmentation settings
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference' key
    """
    feature_id = feature["id"]
    domain_id = feature["domain_id"]
    settings = source["crown_segmentation"]

    progress("Loading tree inventory...", 10)
    trees = load_trees(source["source_inventory_id"])

    progress("Loading CHM...", 20)
    chm = load_chm(source["source_chm_grid_id"]).chunk(CHUNK_SIZE)

    progress("Placing seeds...", 30)
    seeds = resolve_seeds(trees, chm)
    logger.info(
        f"{len(seeds)} of {len(trees)} trees seed a crown",
        extra={"feature_id": feature_id},
    )

    progress("Segmenting crowns...", 40)
    labels = segment(chm, seeds, settings)

    progress("Tracing crown outlines...", 75)
    geometries = trace_crowns(
        labels,
        seeds["row"].to_numpy(),
        seeds["col"].to_numpy(),
        chm.rio.transform(),
        CHUNK_SIZE,
        margin_cells(settings["max_crown_radius"], chm.rio.transform()),
    )
    del labels
    # Seeds are ordered by tree_id, so the rows are too.
    crowns = gpd.GeoDataFrame(
        {"tree_id": seeds["tree_id"].to_numpy(dtype=np.int32)},
        geometry=geometries,
        crs=domain_gdf.crs,
    )

    progress("Saving features to storage...", 90)
    save_features(domain_id, feature_id, crowns)

    progress("Complete", 100)
    return {"georeference": compute_georeference(domain_gdf)}


def load_trees(inventory_id: str) -> pd.DataFrame:
    """Read every tree's ``tree_id``, position and height."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    return pd.read_parquet(path, columns=SEED_COLUMNS)


def load_chm(grid_id: str) -> xr.DataArray:
    """Load a grid's ``chm`` band with no-data cells set to NaN."""
    ds = load_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}")
    if "chm" not in ds.data_vars:
        raise ProcessingError(
            code="MISSING_BAND",
            message="Source grid is missing the required 'chm' band.",
        )
    chm = ds["chm"]
    nodata = chm.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        chm = chm.where(chm != nodata)
    return chm


def resolve_seeds(trees: pd.DataFrame, chm: xr.DataArray) -> pd.DataFrame:
    """Place each tree in its CHM cell and keep the trees that can grow a crown.

    Drops trees outside the CHM or on a no-data cell. Where several trees share
    a cell, the tallest keeps it, with ties going to the lower ``tree_id``.

    Returns the kept trees with ``row``, ``col`` and the cell-center ``x``,
    ``y``, ordered by ``tree_id``.
    """
    transform = chm.rio.transform()
    nrows, ncols = chm.shape
    x = trees["x"].to_numpy(dtype=np.float64)
    y = trees["y"].to_numpy(dtype=np.float64)
    inv = ~transform
    with np.errstate(invalid="ignore"):
        col = np.floor(inv.a * x + inv.b * y + inv.c)
        row = np.floor(inv.d * x + inv.e * y + inv.f)
    inside = (
        np.isfinite(row) & np.isfinite(col)
        & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
    )  # fmt: skip
    seeds = trees.loc[inside, ["tree_id", "height"]].copy()
    seeds["row"] = row[inside].astype(np.int64)
    seeds["col"] = col[inside].astype(np.int64)

    seeds = seeds.sort_values(
        ["height", "tree_id"], ascending=[False, True], na_position="last"
    )
    seeds = seeds.drop_duplicates(["row", "col"], keep="first")

    row_dim, col_dim = chm.dims
    values = chm.isel(
        {
            row_dim: xr.DataArray(seeds["row"].to_numpy(), dims="seed"),
            col_dim: xr.DataArray(seeds["col"].to_numpy(), dims="seed"),
        }
    ).values
    seeds = seeds[np.isfinite(values)]

    seeds = seeds.sort_values("tree_id", ignore_index=True)
    centers = transform @ (seeds["col"].to_numpy() + 0.5, seeds["row"].to_numpy() + 0.5)
    seeds["x"], seeds["y"] = centers
    return seeds


def segment(chm: xr.DataArray, seeds: pd.DataFrame, settings: dict) -> np.ndarray:
    """Label crowns on the CHM; label ``k`` is ``seeds`` row ``k - 1``."""
    try:
        labels = dalponte2016(
            chm,
            seeds[["x", "y"]],
            min_height=settings["min_height"],
            max_height=settings["max_height"],
            min_relative_height=settings["min_relative_height"],
            min_relative_crown_height=settings["min_relative_crown_height"],
            max_crown_radius=settings["max_crown_radius"],
        )
    except ValueError as e:
        raise ProcessingError(code="INVALID_SEGMENTATION_PARAMS", message=str(e))
    if not isinstance(labels.data, da.Array):
        return np.asarray(labels.values)
    # Write blocks straight into one array rather than concatenating copies.
    out = np.empty(labels.shape, dtype=np.int32)
    da.store(labels.data, out, lock=False)
    return out


def margin_cells(max_crown_radius: float, transform: Affine) -> int:
    """Cells between a seed and the farthest cell its crown can reach."""
    a, b, _, d, e, _ = transform[:6]
    cell_size = min(math.hypot(a, d), math.hypot(b, e))
    return math.ceil(max_crown_radius / cell_size)


def trace_crowns(
    labels: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    transform: Affine,
    chunk_size: int,
    margin: int,
) -> list:
    """Trace each crown's cells into a polygon along cell edges.

    Each crown is traced in the chunk containing its seed, from that chunk plus
    ``margin`` cells, which holds the whole crown. A crown in several 4-connected
    pieces (possible only where chunked segmentation disagrees across a chunk
    boundary) keeps the piece holding its seed.

    Polygons are traced in whole-cell (col, row) coordinates and then mapped
    through ``transform``, so vertices do not depend on the chunk layout.

    Returns one polygon per seed, in seed order.
    """
    a, b, c, d, e, f = transform[:6]

    def to_crs(cr: np.ndarray) -> np.ndarray:
        col, row = cr[:, 0], cr[:, 1]
        return np.column_stack((a * col + b * row + c, d * col + e * row + f))

    nrows, ncols = labels.shape
    n = len(seed_rows)
    geometries = np.full(n, None, dtype=object)
    for r0 in range(0, nrows, chunk_size):
        for c0 in range(0, ncols, chunk_size):
            r1, c1 = min(r0 + chunk_size, nrows), min(c0 + chunk_size, ncols)
            in_chunk = (
                (seed_rows >= r0) & (seed_rows < r1)
                & (seed_cols >= c0) & (seed_cols < c1)
            )  # fmt: skip
            if not in_chunk.any():
                continue
            keep = np.zeros(n + 1, dtype=bool)
            keep[np.flatnonzero(in_chunk) + 1] = True

            wr0, wc0 = max(0, r0 - margin), max(0, c0 - margin)
            wr1, wc1 = min(nrows, r1 + margin), min(ncols, c1 + margin)
            window = labels[wr0:wr1, wc0:wc1]
            traced = {}
            for geom, value in shapes(
                window,
                mask=keep[window],
                connectivity=4,
                transform=Affine.translation(wc0, wr0),
            ):
                i = int(value) - 1
                polygon = shape(geom)
                if i not in traced or polygon.contains(
                    Point(seed_cols[i] + 0.5, seed_rows[i] + 0.5)
                ):
                    traced[i] = polygon
            index = np.fromiter(traced, dtype=np.int64, count=len(traced))
            pixel = np.array(list(traced.values()), dtype=object)
            geometries[index] = shapely.transform(pixel, to_crs)
    return list(geometries)
