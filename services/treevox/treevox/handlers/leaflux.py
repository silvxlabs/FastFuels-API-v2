"""
services/treevox/treevox/handlers/leaflux.py

LeafLux irradiance handler. Routed to by dispatch on
(operation="irradiance", input="grid", entity="solar").

The domain is split into a grid of core tiles. Each tile is read with an up-sun
halo so shadows fall correctly into its core; leaflux runs on that padded window,
the halo is trimmed off, and the core region is written straight to the output
zarr. A spawn Pool computes tiles in parallel and each worker writes its own
region (disjoint, chunk-aligned -> safe concurrent writes), returning nothing —
so no tile flows back through the parent and peak memory is WRITE_WORKERS * one
padded tile, bounded in domain. The source LAD and terrain grids are already
co-aligned on the domain lattice.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import dask.array as da
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401 — registers `.rio` accessor
import xarray as xr
from leaflux import Environment, LeafArea, SolarPosition, Terrain, attenuate_all

from treevox import storage

logger = logging.getLogger(__name__)

LEAF_AREA_DENSITY_KEY = "leaf_area_density"
ELEVATION_KEY = "elevation"
CANOPY_BAND = "irradiance.canopy.relative"
SURFACE_BAND = "irradiance.surface.relative"

MAX_ZENITH_DEG = 85.0  # past this the sun is treated as down (near-dark)
WINDOW_TARGET_CELLS = 1600  # core + 2*halo cap; ~2 GB peak at nz~40, fill~0.15
AIR_FILL = np.float32("nan")

# One tile is computed and written per worker, so peak memory is roughly
# WRITE_WORKERS * one padded tile (WINDOW_TARGET_CELLS sizes a tile to ~2 GB).
# Size to the Cloud Run instance memory (~= instance_GB / 2). Default is a single
# worker: bounded like the original one-tile target, but without the dask
# processes-scheduler parent-gather that made memory grow with domain area. Raise
# it on a larger instance to parallelize.
WRITE_WORKERS = int(os.getenv("LEAFLUX_WRITE_WORKERS", "1"))


class LeafluxResult:
    def __init__(self, gcs_path: str, georeference: dict, chunk_shape: list[int]):
        self.gcs_path = gcs_path
        self.georeference = georeference
        self.chunk_shape = chunk_shape


def _open(grid_id: str) -> xr.Dataset:
    return xr.open_zarr(
        storage.gcs_path(grid_id), consolidated=True, decode_coords="all"
    )


def _domain_centroid_lat_lon(domain_gdf) -> tuple[float, float]:
    """Return (lat, lon) of the domain centroid in EPSG:4326."""
    centroid = domain_gdf.geometry.union_all().centroid
    point = gpd.GeoSeries([centroid], crs=domain_gdf.crs).to_crs("EPSG:4326").iloc[0]
    return float(point.y), float(point.x)


def _sun(source: dict, domain_gdf) -> tuple[SolarPosition | None, bool]:
    """SolarPosition and a night flag; None sun when it is below the horizon."""
    dt = source["date_time"]
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    try:
        latitude, longitude = _domain_centroid_lat_lon(domain_gdf=domain_gdf)
        sol = SolarPosition(dt, latitude, longitude)
        return sol, math.degrees(sol.zenith) > MAX_ZENITH_DEG
    except ValueError:  # SolarPosition rejects a sun below the horizon
        return None, True


# --- Tile geometry ---------------------------------------------------------


@dataclass(frozen=True)
class _Tile:
    """One output tile: a core region [r0:r1, c0:c1] plus the padded window
    [pr0:pr1, pc0:pc1] (core + up-sun halo, clipped to the grid) that is read and
    fed to leaflux. All indices are global cell indices into the (z, y, x) grid."""

    r0: int
    r1: int
    c0: int
    c1: int
    pr0: int
    pr1: int
    pc0: int
    pc1: int

    @property
    def pad_zyx(self) -> tuple:
        """Index into a (z, y, x) grid for the padded read (all z)."""
        return (slice(None), slice(self.pr0, self.pr1), slice(self.pc0, self.pc1))

    @property
    def pad_yx(self) -> tuple:
        """Index into a (y, x) grid for the padded read (e.g. terrain)."""
        return (slice(self.pr0, self.pr1), slice(self.pc0, self.pc1))

    @property
    def core_in_pad(self) -> tuple:
        """Index that trims a padded (z, y, x) result back to the core region."""
        return (
            slice(None),
            slice(self.r0 - self.pr0, self.r0 - self.pr0 + (self.r1 - self.r0)),
            slice(self.c0 - self.pc0, self.c0 - self.pc0 + (self.c1 - self.c0)),
        )

    def region(self, nz: int) -> dict:
        """The output-zarr region this tile's core is written to."""
        return {
            "z": slice(0, nz),
            "y": slice(self.r0, self.r1),
            "x": slice(self.c0, self.c1),
        }

    def origin(self, ny: int) -> tuple[float, float]:
        """Global leaflux (x, y) origin for the padded window.

        LeafLux floors rotated coordinates onto a 1x1 lattice, so a tile only
        reproduces a whole-domain run when its lattice is anchored to the global
        frame. leaflux x = column and y runs south->north (ny - 1 - row), so the
        padded window's north-west corner (pc0, pr0) with height (pr1 - pr0) maps
        to origin (pc0, ny - pr1)."""
        return (float(self.pc0), float(ny - self.pr1))


def _tile_at(row: int, col: int, core: int, halo: int, ny: int, nx: int) -> _Tile:
    r0, r1 = row * core, min(row * core + core, ny)
    c0, c1 = col * core, min(col * core + core, nx)
    return _Tile(
        r0=r0,
        r1=r1,
        c0=c0,
        c1=c1,
        pr0=max(0, r0 - halo),
        pr1=min(ny, r1 + halo),
        pc0=max(0, c0 - halo),
        pc1=min(nx, c1 + halo),
    )


def _plan_tiles(ny: int, nx: int, core: int, halo: int) -> list[_Tile]:
    """Every core tile covering the (ny, nx) grid; cores are disjoint and their
    union is the whole grid."""
    return [
        _tile_at(row, col, core, halo, ny, nx)
        for row in range(math.ceil(ny / core))
        for col in range(math.ceil(nx / core))
    ]


# --- Per-tile irradiance ---------------------------------------------------


def _leaf_area(lad: np.ndarray) -> LeafArea:
    """LeafArea from a (z, y, x) array (from_uniformgrid wants (y, x, z))."""
    return LeafArea.from_uniformgrid(np.transpose(lad.astype(np.float32), (1, 2, 0)))


def _scatter(stack: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Scatter leaflux's (x, y, z, irr) point cloud (y south->north) into a dense
    (z, y, x) grid; air cells stay AIR_FILL."""
    _, ny, _ = shape
    x = np.round(stack[:, 0]).astype(int)
    y = ny - 1 - np.round(stack[:, 1]).astype(int)  # south->north -> north->south
    z = np.round(stack[:, 2]).astype(int)
    grid = np.full(shape, AIR_FILL, dtype=np.float32)
    grid[z, y, x] = stack[:, 3]
    return grid


def _terrain_lift(leaf_area: LeafArea, terrain: np.ndarray) -> None:
    """Lift the canopy onto the terrain (cell units) so shadows cast correctly.
    Leaf y is south->north; terrain rows are north->south."""
    xy = leaf_area.leaf_area[:, :2]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, terrain.shape[1] - 1)
    yi = np.clip(
        terrain.shape[0] - 1 - np.round(xy[:, 1]).astype(int), 0, terrain.shape[0] - 1
    )
    leaf_area.leaf_area[:, 2] += terrain[yi, xi]


def _canopy_irradiance(lad, origin, sol, extn, voxel, night) -> np.ndarray:
    """Relative canopy irradiance for a padded (z, y, x) LAD tile, scattered back
    onto the tile grid. No leaflux run when it is night or the tile is empty
    (leaflux can't reduce an empty leaf stack)."""
    if night or not np.any(lad > 0):
        fill = np.float32(0.0) if night else AIR_FILL  # 0 where canopy at night
        return np.where(lad > 0, fill, AIR_FILL).astype(np.float32)
    env = Environment(leaf_area=_leaf_area(lad), voxel_dim=voxel)
    irr = attenuate_all(env, sol, extn=extn, origin=origin)
    return _scatter(irr.canopy_irradiance, lad.shape)


def _surface_irradiance(lad, terrain, origin, sol, extn, voxel, night) -> np.ndarray:
    """Relative ground irradiance for a padded tile, placed at z=0 of a (z, y, x)
    grid (all other cells AIR_FILL). `terrain` is a (y, x) surface in cell units;
    the canopy is lifted onto it so shadows cast from the right height."""
    grid = np.full(lad.shape, AIR_FILL, dtype=np.float32)
    if night:
        grid[0] = 0.0
        return grid
    leaf_area = _leaf_area(lad)
    _terrain_lift(leaf_area, terrain)
    env = Environment(leaf_area=leaf_area, terrain=Terrain(terrain), voxel_dim=voxel)
    irr = attenuate_all(env, sol, extn=extn, origin=origin)
    grid[0] = irr.terrain_irradiance
    return grid


# --- Per-tile parallel writer ----------------------------------------------


@dataclass(frozen=True)
class _WriteJob:
    """Everything a worker needs to fill tiles, all picklable (no open datasets —
    those are opened once per worker in `_open_worker_sources`)."""

    source_grid_id: str
    dem_id: str | None
    out_path: str
    nz: int
    ny: int
    nx: int
    core: int
    halo: int
    z_origin: float
    vr: float
    voxel: tuple[float, float, float]
    extn: float
    sol: SolarPosition | None
    night: bool
    want_canopy: bool
    want_surface: bool


# Per-worker source handles, opened once by the Pool initializer (or in-process
# for the single-worker path) and reused across every tile that worker fills.
_WORKER: dict = {}


def _open_worker_sources(job: _WriteJob) -> None:
    _WORKER["job"] = job
    _WORKER["lad_ds"] = _open(job.source_grid_id)
    _WORKER["dem_ds"] = _open(job.dem_id) if job.dem_id else None


def _read_terrain(tile: _Tile, shape: tuple[int, int]) -> np.ndarray:
    """The tile's padded terrain in cell units, or a flat plane at the LAD datum
    when there is no terrain grid."""
    job: _WriteJob = _WORKER["job"]
    dem_ds = _WORKER["dem_ds"]
    if dem_ds is None:
        return np.zeros(shape, dtype=np.float32)
    elev = dem_ds[ELEVATION_KEY][tile.pad_yx].values.astype(np.float32)
    return (elev - job.z_origin) / job.vr


def _write_tile(tile: _Tile) -> None:
    job: _WriteJob = _WORKER["job"]
    lad = _WORKER["lad_ds"][LEAF_AREA_DENSITY_KEY][tile.pad_zyx].values.astype(
        np.float32
    )
    origin = tile.origin(job.ny)

    data: dict = {}
    if job.want_canopy:
        canopy = _canopy_irradiance(
            lad, origin, job.sol, job.extn, job.voxel, job.night
        )
        data[CANOPY_BAND] = (("z", "y", "x"), canopy[tile.core_in_pad])
    if job.want_surface:
        terrain = _read_terrain(tile, lad.shape[1:])
        surface = _surface_irradiance(
            lad, terrain, origin, job.sol, job.extn, job.voxel, job.night
        )
        data[SURFACE_BAND] = (("z", "y", "x"), surface[tile.core_in_pad])

    xr.Dataset(data).to_zarr(
        job.out_path, region=tile.region(job.nz), safe_chunks=False
    )


def _write_tiles(
    tiles: list[_Tile], job: _WriteJob, progress: Callable[[str, int | None], None]
) -> None:
    n = len(tiles)
    workers = max(1, min(WRITE_WORKERS, n))
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        progress(f"Computing irradiance ({done}/{n} tiles)...", 25 + int(70 * done / n))

    if workers == 1:
        _open_worker_sources(job)
        for tile in tiles:
            _write_tile(tile)
            _tick()
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            workers, initializer=_open_worker_sources, initargs=(job,)
        ) as pool:
            for _ in pool.imap_unordered(_write_tile, tiles, chunksize=1):
                _tick()


# --- Orchestration ---------------------------------------------------------


def _init_output(
    out_path: str, lad_ds: xr.Dataset, bands: list[str], core: int
) -> None:
    """Write the output-zarr skeleton: metadata, coords, CRS and per-band attrs,
    but no data (compute=False). Chunked at the tile size so each per-tile region
    write lands in its own chunk (disjoint tiles -> safe concurrent writes). The
    workers fill the data regions afterwards."""
    nz, ny, nx = lad_ds.sizes["z"], lad_ds.sizes["y"], lad_ds.sizes["x"]
    skeleton = xr.Dataset(
        {
            b: (
                ("z", "y", "x"),
                da.full((nz, ny, nx), np.nan, chunks=(nz, core, core), dtype="float32"),
            )
            for b in bands
        },
        coords={"z": lad_ds.z, "y": lad_ds.y, "x": lad_ds.x},
    ).rio.write_crs(str(lad_ds.rio.crs))
    for b in bands:
        skeleton[b].attrs["grid_mapping"] = "spatial_ref"
    skeleton.attrs["transform"] = list(lad_ds.attrs["transform"])
    skeleton.attrs["z_origin"] = float(lad_ds.attrs["z_origin"])
    skeleton.attrs["z_resolution"] = float(lad_ds.attrs["z_resolution"])
    skeleton.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
        compute=False,
        encoding={b: {"fill_value": float("nan")} for b in bands},
    )


def run_leaflux(
    grid: dict,
    domain_gdf: gpd.GeoDataFrame,  # inherited from the source grid; here for dispatch parity
    progress: Callable[[str, int | None], None],
) -> LeafluxResult:
    source = grid["source"]
    requested = [b["key"] for b in grid["bands"]]
    out_path = storage.gcs_path(grid["id"])

    progress("Opening source grids...", 10)
    lad_ds = _open(source["source_grid_id"])
    nz, ny, nx = lad_ds.sizes["z"], lad_ds.sizes["y"], lad_ds.sizes["x"]
    hr = float(lad_ds.attrs["transform"][0])
    vr = float(lad_ds.attrs["z_resolution"])
    z_origin = float(lad_ds.attrs["z_origin"])

    sol, night = _sun(source, domain_gdf)
    halo = 0 if night else math.ceil((nz * vr * math.tan(sol.zenith)) / hr)
    core = max(min(WINDOW_TARGET_CELLS - 2 * halo, nx, ny), 1)
    bands = [b for b in (CANOPY_BAND, SURFACE_BAND) if b in requested]

    progress("Writing irradiance grid...", 20)
    _init_output(out_path, lad_ds, bands, core)

    job = _WriteJob(
        source_grid_id=source["source_grid_id"],
        dem_id=source.get("source_terrain_grid_id")
        if SURFACE_BAND in requested
        else None,
        out_path=out_path,
        nz=nz,
        ny=ny,
        nx=nx,
        core=core,
        halo=halo,
        z_origin=z_origin,
        vr=vr,
        voxel=(hr, hr, vr),
        extn=source["extinction_coefficient"],
        sol=sol,
        night=night,
        want_canopy=CANOPY_BAND in requested,
        want_surface=SURFACE_BAND in requested,
    )
    progress("Computing irradiance...", 25)
    _write_tiles(_plan_tiles(ny, nx, core, halo), job, progress)

    progress("Finalizing...", 98)
    return LeafluxResult(
        out_path,
        {
            "crs": str(lad_ds.rio.crs),
            "transform": list(lad_ds.attrs["transform"]),
            "shape": [nz, ny, nx],
            "z_resolution": vr,
            "z_origin": z_origin,
        },
        [nz, core, core],
    )
