"""Integration test for the LeafLux irradiance handler.

The handler tiles the domain with `da.map_overlap`, casts shadows into each
tile from an up-sun halo, and trims the halo back off. The correctness claim is
that this chunked-with-halo computation reproduces a single, un-chunked leaflux
run over the whole domain.

This test verifies exactly that. It:

  1. Builds a REAL 3D leaf-area-density grid by voxelizing the Blue Mountain PIM
     inventory (via ``treevox_runner``).
  2. Synthesizes a terrain grid aligned to that grid's lattice (the handler
     assumes the terrain grid shares the source lattice, so an aligned grid is
     the valid input; relief is kept small so terrain shadows stay within the
     canopy-sized halo).
  3. Runs the handler with the tile size shrunk so the domain spans many tiles.
  4. Runs leaflux ONCE over the whole domain, independently of the handler, as
     the reference.
  5. Asserts the two output grids are equal, cell for cell.
"""

from __future__ import annotations

import math
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
from leaflux import Environment, LeafArea, SolarPosition, Terrain, attenuate_all
from treevox.handlers import leaflux as handler

from lib.config import GRIDS_BUCKET
from lib.gcs.blobs import delete_directory, exists
from lib.zarr_utils import save_zarr

pytestmark = pytest.mark.integration

LAD_KEY = "leaf_area_density"
CANOPY_BAND = "irradiance.canopy.relative"
SURFACE_BAND = "irradiance.surface.relative"

# Blue Mountain (western Montana). Only needs to yield a daytime sun; the same
# SolarPosition drives both the handler and the reference, so exact coords do
# not affect the equality check.
LATITUDE = 46.9
LONGITUDE = -114.0
DATE_TIME = datetime(2023, 6, 21, 19, 0, 0)  # ~solar noon UTC in summer
EXTINCTION = 0.5


def _reference_scatter(stack: np.ndarray, nz: int, ny: int, nx: int) -> np.ndarray:
    """Independent scatter of leaflux's (x, y, z, irr) point cloud (y south->
    north) into a dense (z, y, x) grid — written without the handler helpers."""
    x = np.round(stack[:, 0]).astype(int)
    y = ny - 1 - np.round(stack[:, 1]).astype(int)
    z = np.round(stack[:, 2]).astype(int)
    grid = np.full((nz, ny, nx), np.nan, dtype=np.float32)
    grid[z, y, x] = stack[:, 3]
    return grid


def _reference_leaflux(
    lad: np.ndarray,
    elevation: np.ndarray,
    z_origin: float,
    vr: float,
    hr: float,
    sol: SolarPosition,
) -> dict[str, np.ndarray]:
    """Run leaflux once over the whole domain — the ground truth.

    Mirrors the handler's modeling choices (canopy ignores terrain; surface is
    computed with terrain and stored at z=0) but shares none of its code or
    chunking.
    """
    nz, ny, nx = lad.shape
    yxz = np.transpose(lad.astype(np.float32), (1, 2, 0))

    # Canopy: no terrain.
    canopy_env = Environment(
        leaf_area=LeafArea.from_uniformgrid(yxz), voxel_dim=(hr, hr, vr)
    )
    canopy_irr = attenuate_all(canopy_env, sol, extn=EXTINCTION)
    canopy = _reference_scatter(canopy_irr.canopy_irradiance, nz, ny, nx)

    # Surface: with terrain (cell units), canopy lifted onto it.
    terrain = (elevation.astype(np.float32) - z_origin) / vr
    la = LeafArea.from_uniformgrid(yxz)
    xy = la.leaf_area[:, :2]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, nx - 1)
    yi = np.clip(ny - 1 - np.round(xy[:, 1]).astype(int), 0, ny - 1)
    la.leaf_area[:, 2] += terrain[yi, xi]
    surf_env = Environment(
        leaf_area=la, terrain=Terrain(terrain), voxel_dim=(hr, hr, vr)
    )
    surf_irr = attenuate_all(surf_env, sol, extn=EXTINCTION)
    surface = np.full((nz, ny, nx), np.nan, dtype=np.float32)
    surface[0] = surf_irr.terrain_irradiance
    return {CANOPY_BAND: canopy, SURFACE_BAND: surface}


def _assert_equal(
    name: str, got: np.ndarray, ref: np.ndarray, atol: float = 1e-4
) -> None:
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    got_nan, ref_nan = np.isnan(got), np.isnan(ref)
    assert np.array_equal(got_nan, ref_nan), (
        f"{name}: populated-cell masks differ "
        f"(handler {(~got_nan).sum()} cells, reference {(~ref_nan).sum()})"
    )
    diff = np.abs(got[~got_nan] - ref[~ref_nan])
    assert np.allclose(got[~got_nan], ref[~ref_nan], atol=atol, rtol=0.0), (
        f"{name}: max abs diff {diff.max():.2e} exceeds atol {atol}"
    )


def test_handler_matches_unchunked_leaflux(treevox_runner, monkeypatch):
    # 1. Real voxelized leaf-area-density grid.
    source = treevox_runner(
        static_inventory="static-test-blue-mtn-pim-inventory",
        bands=[LAD_KEY],
    )
    lad_ds = source.ds
    lad_grid_id = source.grid_id
    nz, ny, nx = lad_ds.sizes["z"], lad_ds.sizes["y"], lad_ds.sizes["x"]
    hr = float(lad_ds.attrs["transform"][0])
    vr = float(lad_ds.attrs["z_resolution"])
    z_origin = float(lad_ds.attrs["z_origin"])
    crs = str(lad_ds.rio.crs)
    lad_np = lad_ds[LAD_KEY].values.astype(np.float32)

    dem_id = f"test-{uuid4().hex}"
    leaflux_id = f"test-{uuid4().hex}"
    dem_path = f"gs://{GRIDS_BUCKET}/{dem_id}"
    out_path = f"gs://{GRIDS_BUCKET}/{leaflux_id}"

    try:
        # 2. Terrain grid aligned to the LAD lattice: a gentle tilted, wavy
        #    surface a few metres above the datum (relief << nz*vr so terrain
        #    shadows never exceed the canopy-sized halo).
        xs, ys = lad_ds.x.values, lad_ds.y.values
        gx = (xs - xs.min())[None, :]
        gy = (ys - ys.min())[:, None]
        span = max(gx.max(), 1.0)
        relief = 4.0 * (gx / span) + 1.5 * np.sin(gy / 40.0)
        elevation = (z_origin + relief).astype(np.float32)
        elevation = np.broadcast_to(elevation, (ny, nx)).copy()

        dem = xr.Dataset(
            {"elevation": (("y", "x"), elevation)}, coords={"y": ys, "x": xs}
        ).rio.write_crs(crs)  # sets the spatial_ref/grid_mapping linkage itself
        save_zarr(dem_path, dem, chunk_shape=(ny, nx))

        # 3. Shrink the tile so the domain spans many tiles (real halo/tiling).
        sol = SolarPosition(DATE_TIME, LATITUDE, LONGITUDE)
        assert math.degrees(sol.zenith) < handler.MAX_ZENITH_DEG, "sun must be up"
        halo = math.ceil((nz * vr * math.tan(sol.zenith)) / hr)
        core_target = max(8, min(ny, nx) // 3)
        monkeypatch.setattr(handler, "WINDOW_TARGET_CELLS", core_target + 2 * halo)
        # Exercise the per-tile parallel writer with several workers writing their
        # own regions concurrently (not the single-worker in-process fallback).
        monkeypatch.setattr(handler, "WRITE_WORKERS", 4)
        core = max(min(core_target, nx, ny), 1)
        n_tiles = math.ceil(ny / core) * math.ceil(nx / core)
        assert n_tiles > 1, f"test must span multiple tiles, got {n_tiles}"

        grid = {
            "id": leaflux_id,
            "source": {
                "operation": "irradiance",
                "input": "grid",
                "entity": "solar",
                "source_grid_id": lad_grid_id,
                "source_terrain_grid_id": dem_id,
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "date_time": DATE_TIME,
                "extinction_coefficient": EXTINCTION,
            },
            "bands": [{"key": CANOPY_BAND}, {"key": SURFACE_BAND}],
        }

        result = handler.run_leaflux(grid, None, lambda *a, **k: None)
        assert result.georeference["shape"] == [nz, ny, nx]

        out = xr.open_zarr(out_path, consolidated=True, decode_coords="all")

        # 4 + 5. Independent single-shot reference, then compare cell for cell.
        ref = _reference_leaflux(lad_np, elevation, z_origin, vr, hr, sol)
        _assert_equal(CANOPY_BAND, out[CANOPY_BAND].values, ref[CANOPY_BAND])
        _assert_equal(SURFACE_BAND, out[SURFACE_BAND].values, ref[SURFACE_BAND])

        # The output must carry the source lattice for downstream consumers.
        assert str(out.rio.crs) == crs
        np.testing.assert_array_equal(out.x.values, xs)
        np.testing.assert_array_equal(out.y.values, ys)
        out.close()
    finally:
        for path in (dem_path, out_path):
            if exists(path):
                delete_directory(path)
