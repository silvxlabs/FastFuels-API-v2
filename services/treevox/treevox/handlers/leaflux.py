"""
services/treevox/treevox/handlers/leaflux.py

LeafLux irradiance handler. Routed to by dispatch on
(operation="irradiance", input="grid", entity="solar").

Chunks the domain in the original (x, y) frame. Each chunk is read with an
up-sun halo sized to the canopy height and solar zenith so it casts correct
shadows into its core; the halo is discarded on write. Sequential by design:
one chunk in flight at a time to stay within the service memory budget, not to
parallelize.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rioxarray  # noqa: F401 — registers `.rio` accessor
import xarray as xr
from leaflux import (
    Environment,
    LeafArea,
    SolarPosition,
    Terrain,
    attenuate_all,
)
from rasterio.enums import Resampling

from treevox import storage
from treevox.errors import ProcessingError

logger = logging.getLogger(__name__)

LEAF_AREA_DENSITY_KEY = "leaf_area_density"
ELEVATION_KEY = "elevation"
CANOPY_BAND = "irradiance.canopy.relative"
SURFACE_BAND = "irradiance.surface.relative"

MAX_ZENITH_DEG = 85.0
# Memory-bound window side: max (core + 2*halo) cells leaflux processes at once.
# ~1600 => ~2 GB peak at nz~40, fill~0.15, with terrain. Measure fill on a real
# stand before trusting this.
WINDOW_TARGET_CELLS = 1600
MAX_TERRAIN_NAN_FRACTION = 0.05
AIR_FILL = np.float32("nan")


@dataclass
class LeafluxResult:
    gcs_path: str
    georeference: dict
    chunk_shape: list[int]


@dataclass
class SourceGeometry:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    hr: float
    vr: float
    crs: str
    z_origin: float
    transform: list[float]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.z), len(self.y), len(self.x))


@dataclass
class ChunkPlan:
    nz: int
    ny: int
    nx: int
    hr: float
    vr: float
    core: int
    halo: int
    locations: list[tuple[int, int]]


def _open_source(source_grid_id: str) -> xr.Dataset:
    path = storage.gcs_path(source_grid_id)
    return xr.open_zarr(path, consolidated=True, decode_coords="all")  # lazy


def _read_geometry(src: xr.Dataset) -> SourceGeometry:
    return SourceGeometry(
        x=src.x.values,
        y=src.y.values,
        z=src.z.values,
        hr=float(src.attrs["transform"][0]),
        vr=float(src.attrs["z_resolution"]),
        crs=str(src.rio.crs),
        z_origin=float(src.attrs["z_origin"]),
        transform=list(src.attrs["transform"]),
    )


# Creates Leaflux SolarPosition from datetime, lat, and lon
def _solar_position(source: dict) -> SolarPosition:
    dt = source["date_time"]
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return SolarPosition(dt, source["latitude"], source["longitude"])


def _plan_chunks(geom: SourceGeometry, zenith_rad: float) -> ChunkPlan:
    nz, ny, nx = geom.shape
    halo = int(math.ceil((nz * geom.vr * math.tan(zenith_rad)) / geom.hr))
    # Symmetric halo: window = core + 2*halo must fit the budget.
    # TODO: could do directional (up-sun only) halo needs just core + halo -> fewer chunks
    # Keeping it simple for now
    core = max(min(WINDOW_TARGET_CELLS - 2 * halo, nx, ny), 1)
    n_rows = math.ceil(ny / core)
    n_cols = math.ceil(nx / core)
    locations = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    logger.info(
        f"Irradiance layout: {ny}x{nx} (y, x); core={core}, halo={halo}; "
        f"{n_rows}x{n_cols} = {len(locations)} chunks"
    )
    return ChunkPlan(nz, ny, nx, geom.hr, geom.vr, core, halo, locations)


def _window_slices(
    loc: tuple[int, int], plan: ChunkPlan
) -> tuple[slice, slice, slice, slice]:
    """Return (core_y, core_x, read_y, read_x): output core and the halo-expanded
    read window, all in global (y, x) indices."""
    r, c = loc
    y0, x0 = r * plan.core, c * plan.core
    y1, x1 = min(y0 + plan.core, plan.ny), min(x0 + plan.core, plan.nx)
    ry0, rx0 = max(y0 - plan.halo, 0), max(x0 - plan.halo, 0)
    ry1, rx1 = min(y1 + plan.halo, plan.ny), min(x1 + plan.halo, plan.nx)
    return slice(y0, y1), slice(x0, x1), slice(ry0, ry1), slice(rx0, rx1)


# Create a leaflux LeafArea array from the LAD band of the grid
def _read_leaf_area(src: xr.Dataset, read_y: slice, read_x: slice) -> LeafArea:
    lad_zyx = src[LEAF_AREA_DENSITY_KEY].isel(y=read_y, x=read_x).values  # Get LAD
    lad_yxz = np.transpose(
        np.asarray(lad_zyx, dtype=np.float32), (1, 2, 0)
    )  # Rearrange
    la = LeafArea.from_uniformgrid(lad_yxz)
    occupied, total = len(la.leaf_area), lad_yxz.size
    logger.info(f"fill: {occupied}/{total} = {occupied / total:.4f}")
    return la


def _load_dem(
    terrain_grid_id: str, src: xr.Dataset, z_origin: float, vr: float
) -> np.ndarray:
    dem = xr.open_zarr(
        storage.gcs_path(terrain_grid_id), consolidated=True, decode_coords="all"
    )
    target = src[LEAF_AREA_DENSITY_KEY].isel(z=0)
    logger.info(
        f"Terrain align: DEM {tuple(dem[ELEVATION_KEY].shape)} {dem.rio.crs} "
        f"-> source {tuple(target.shape)} {src.rio.crs}"
    )

    # TODO: Is this ok? Resample DEM bilinearly to get rid of step artifacts from
    # 10m DEMs
    aligned = (
        dem[ELEVATION_KEY]
        .rio.write_nodata(np.nan)
        .rio.reproject_match(target, resampling=Resampling.bilinear)
    )
    values = np.asarray(aligned.values, dtype=np.float32)

    nan_count = int(np.isnan(values).sum())
    covered = 1.0 - nan_count / values.size
    valid = values[~np.isnan(values)]
    elev_range = f"{valid.min():.1f}/{valid.max():.1f}" if valid.size else "n/a"
    logger.info(
        f"Terrain coverage: {covered * 100:.2f}% "
        f"({nan_count}/{values.size} cells missing); elev min/max {elev_range}"
    )

    if nan_count / values.size > MAX_TERRAIN_NAN_FRACTION:
        raise ProcessingError(
            code="TERRAIN_COVERAGE_INSUFFICIENT",
            message=(
                f"Terrain grid covers only {covered * 100:.1f}% of the source "
                f"domain ({nan_count} of {values.size} cells missing). Recreate "
                "the DEM over the full domain (optionally with an extent buffer)."
            ),
        )

    if nan_count:
        logger.info(
            f"Terrain: filling {nan_count} NaN cells (interpolate_na, nearest)."
        )
        values = np.asarray(
            aligned.rio.interpolate_na(method="nearest").values, dtype=np.float32
        )

    # Move DEM to be relative to LAD grid's z_origin
    values = (values - z_origin) / vr
    logger.info(
        f"Terrain re-zeroed to LAD datum: relative cell min/max "
        f"{np.min(values):.1f}/{np.max(values):.1f}"
    )
    return values  # (ny, nx), north -> south, LAD-relative cell units


def _terrain_offsets(
    xy: np.ndarray, terrain_win: np.ndarray, win_ny: int
) -> np.ndarray:
    # terrain height (cell units) beneath each (x, y) canopy point.
    # y is south->north (LeafArea); terrain_win row is north->south -> flip.
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, terrain_win.shape[1] - 1)
    yi = np.clip(
        win_ny - 1 - np.round(xy[:, 1]).astype(int), 0, terrain_win.shape[0] - 1
    )
    return terrain_win[yi, xi]


# Return window of DEM for chunked processing
# If there is no DEM, a flat plane is substituted
def _terrain_window(dem: np.ndarray | None, read_y: slice, read_x: slice) -> np.ndarray:
    win = (read_y.stop - read_y.start, read_x.stop - read_x.start)
    if dem is None:
        return np.zeros(win, dtype=np.float32)  # flat plane beneath the canopy
    return np.asarray(dem[read_y, read_x], dtype=np.float32)


def _canopy_core_array(
    canopy_stack: np.ndarray,
    plan: ChunkPlan,
    core_y: slice,
    core_x: slice,
    read_y: slice,
    read_x: slice,
) -> np.ndarray:
    # canopy_stack rows: (x, y, z, irr) in window-local coords, y south -> north.
    win_ny = read_y.stop - read_y.start
    x = np.round(canopy_stack[:, 0]).astype(int)
    y = win_ny - 1 - np.round(canopy_stack[:, 1]).astype(int)  # -> north -> south
    z = np.round(canopy_stack[:, 2]).astype(int)
    gy = read_y.start + y
    gx = read_x.start + x
    keep = (
        (gy >= core_y.start)
        & (gy < core_y.stop)
        & (gx >= core_x.start)
        & (gx < core_x.stop)
    )
    core = np.full(
        (plan.nz, core_y.stop - core_y.start, core_x.stop - core_x.start),
        AIR_FILL,
        dtype=np.float32,
    )
    core[z[keep], gy[keep] - core_y.start, gx[keep] - core_x.start] = canopy_stack[
        keep, 3
    ]
    return core


# TODO: currently, surface irradiance is stored at z=0 in a 3D grid, the rest is nans
# This could change, but I wanted to stick with the same format as the canopy for now
def _surface_core_3d(
    terrain_irr: np.ndarray,
    plan: ChunkPlan,
    core_y: slice,
    core_x: slice,
    read_y: slice,
    read_x: slice,
) -> np.ndarray:
    # terrain_irr is (win_ny, win_nx), y north -> south (already placed by leaflux).
    oy = core_y.start - read_y.start
    ox = core_x.start - read_x.start
    cny = core_y.stop - core_y.start
    cnx = core_x.stop - core_x.start
    surf = terrain_irr[oy : oy + cny, ox : ox + cnx].astype(np.float32)
    core = np.full((plan.nz, cny, cnx), AIR_FILL, dtype=np.float32)
    core[0] = surf  # 2D surface stored at z=0
    return core


def _init_output(
    out_path: str, geom: SourceGeometry, plan: ChunkPlan, requested: set[str]
) -> None:
    storage.init_store(
        out_path,
        x_coords=geom.x,
        y_coords=geom.y,
        z_coords=geom.z,
        hr=geom.hr,
        vr=geom.vr,
        crs=geom.crs,
        z_origin=geom.z_origin,
        requested_keys=sorted(requested),
        chunk_shape=(plan.nz, plan.core, plan.core),
    )


def _build_result(
    out_path: str, geom: SourceGeometry, plan: ChunkPlan
) -> LeafluxResult:
    georeference = {
        "crs": geom.crs,
        "transform": geom.transform,
        "shape": list(geom.shape),
        "z_resolution": geom.vr,
        "z_origin": geom.z_origin,
    }
    return LeafluxResult(out_path, georeference, [plan.nz, plan.core, plan.core])


def _process_chunk(
    loc: tuple[int, int],
    plan: ChunkPlan,
    src: xr.Dataset,
    out_path: str,
    dem: np.ndarray | None,
    sol: SolarPosition,
    extn: float,
    requested: set[str],
) -> None:
    core_y, core_x, read_y, read_x = _window_slices(loc, plan)
    win_ny = read_y.stop - read_y.start

    leaf_area = _read_leaf_area(src, read_y, read_x)
    needs_terrain = SURFACE_BAND in requested
    terrain_win = _terrain_window(dem, read_y, read_x) if needs_terrain else None

    if terrain_win is not None:
        # Move LAD onto DEM so that they are correctly relative to each other
        leaf_area.leaf_area[:, 2] += _terrain_offsets(
            leaf_area.leaf_area[:, :2], terrain_win, win_ny
        )

    terrain = Terrain(terrain_win) if terrain_win is not None else None
    env = Environment(
        leaf_area=leaf_area, terrain=terrain, voxel_dim=(plan.hr, plan.hr, plan.vr)
    )
    logger.info(f"voxel dim: {env.voxel_dim}")

    irr = attenuate_all(env, sol, extn=extn)

    # # ***TEMP***
    # plot_irradiance(
    #     irr=irr, terrain_coords=terrain, show_solar_vector=True,
    #     show_sensors=False, show_axes=True, show_canopy=False,
    # )

    data_vars: dict = {}
    if CANOPY_BAND in requested:
        canopy = irr.canopy_irradiance
        if terrain_win is not None:
            # Move the LAD back to flat, where it was originally, for consistency in the output
            canopy = canopy.copy()
            canopy[:, 2] -= _terrain_offsets(canopy[:, :2], terrain_win, win_ny)
        data_vars[CANOPY_BAND] = (
            ("z", "y", "x"),
            _canopy_core_array(canopy, plan, core_y, core_x, read_y, read_x),
        )
    if SURFACE_BAND in requested:
        data_vars[SURFACE_BAND] = (
            ("z", "y", "x"),
            _surface_core_3d(
                irr.terrain_irradiance, plan, core_y, core_x, read_y, read_x
            ),
        )
    storage.write_union(out_path, xr.Dataset(data_vars), core_y, core_x)


def _write_zero_result(
    src: xr.Dataset,
    geom: SourceGeometry,
    out_path: str,
    requested: set[str],
) -> LeafluxResult:
    # Sun at/below the horizon: relative irradiance is 0 wherever it exists.
    plan = _plan_chunks(geom, 0.0)  # halo 0 — no shadows to cast
    _init_output(out_path, geom, plan, requested)
    for loc in plan.locations:
        core_y, core_x, read_y, read_x = _window_slices(loc, plan)
        data_vars: dict = {}
        if CANOPY_BAND in requested:
            leaf_area = _read_leaf_area(src, read_y, read_x)
            n = len(leaf_area.leaf_area)
            zero_stack = np.column_stack(
                (leaf_area.leaf_area[:, :3], np.zeros(n, dtype=np.float32))
            )
            data_vars[CANOPY_BAND] = (
                ("z", "y", "x"),
                _canopy_core_array(zero_stack, plan, core_y, core_x, read_y, read_x),
            )
        if SURFACE_BAND in requested:
            cny, cnx = core_y.stop - core_y.start, core_x.stop - core_x.start
            core = np.full((plan.nz, cny, cnx), AIR_FILL, dtype=np.float32)
            core[0] = 0.0
            data_vars[SURFACE_BAND] = (("z", "y", "x"), core)
        storage.write_union(out_path, xr.Dataset(data_vars), core_y, core_x)  # Write
    return _build_result(out_path, geom, plan)


def run_leaflux(
    grid: dict,
    domain_gdf,  # geometry inherited from the source grid; here for parity/future use
    progress: Callable[[str, int | None], None],
) -> LeafluxResult:
    grid_id = grid["id"]
    source = grid["source"]
    requested = {b["key"] for b in grid["bands"]}

    progress("Opening source grid...", 5)
    src = _open_source(source["source_grid_id"])
    geom = _read_geometry(src)
    out_path = storage.gcs_path(grid_id)

    # TODO: confirm this is acceptable
    # Solar zenith angles greater than 85 degs are rejected, as at this point
    # irradiance is close to none and I want to avoid halo blow-up
    # SolarPosition already rejects agles that are below horizon, so this is
    # a small change to this.
    try:
        sol = _solar_position(source)
        too_low = math.degrees(sol.zenith) > MAX_ZENITH_DEG
    except ValueError:
        # SolarPosition rejects sun below the horizon (elevation < 0).
        too_low = True

    # If this is the case, all 0s are written
    if too_low:
        progress("Sun below threshold; writing zero irradiance...", 20)
        return _write_zero_result(src, geom, out_path, requested)

    # If no DEM is supplied, a flat plane will be substituted per chunk
    dem = None
    if SURFACE_BAND in requested and source.get("source_terrain_grid_id"):
        progress("Loading terrain...", 15)
        dem = _load_dem(source["source_terrain_grid_id"], src, geom.z_origin, geom.vr)

    plan = _plan_chunks(geom, sol.zenith)
    progress("Initializing output store...", 20)
    _init_output(out_path, geom, plan, requested)

    total = len(plan.locations)
    for i, loc in enumerate(plan.locations):
        # Processes chunk and writes
        _process_chunk(
            loc,
            plan,
            src,
            out_path,
            dem,
            sol,
            source["extinction_coefficient"],
            requested,
        )
        progress(f"Irradiance chunk {i + 1}/{total}...", 20 + int(75 * (i + 1) / total))

    progress("Finalizing...", 98)
    # Return result following convention set by voxelize
    return _build_result(out_path, geom, plan)
