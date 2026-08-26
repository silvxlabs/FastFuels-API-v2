"""Unit tests for treevox.handlers.leaflux.

The integration test (tests/integration/test_leaflux.py) is the ground truth: it
runs the whole handler against an independent whole-domain leaflux run on the
static fixture. These unit tests cover the atomic pieces underneath it — tile
geometry, the coordinate/orientation conventions, the domain->sun resolution,
and the night/empty short-circuits — plus one fast, GCS-free bridge test that
the per-tile assembly reproduces a whole-domain run.
"""

from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from leaflux import SolarPosition
from shapely.geometry import box
from treevox.handlers import leaflux as handler
from treevox.handlers.leaflux import (
    CANOPY_BAND,
    SURFACE_BAND,
    _canopy_irradiance,
    _domain_centroid_lat_lon,
    _leaf_area,
    _plan_tiles,
    _scatter,
    _sun,
    _surface_irradiance,
    _terrain_lift,
    _tile_at,
)


def _boom(*args, **kwargs):
    raise AssertionError("leaflux model must not run on the night/empty path")


def _domain_at(lat, lon, half=0.001):
    """A one-row domain GeoDataFrame whose centroid is (lat, lon) in EPSG:4326."""
    return gpd.GeoDataFrame(
        geometry=[box(lon - half, lat - half, lon + half, lat + half)],
        crs="EPSG:4326",
    )


# --- Tile geometry ---


def test_tile_interior_has_full_halo():
    t = _tile_at(row=1, col=1, core=32, halo=8, ny=100, nx=100)
    assert (t.r0, t.r1, t.c0, t.c1) == (32, 64, 32, 64)  # core
    assert (t.pr0, t.pr1, t.pc0, t.pc1) == (24, 72, 24, 72)  # padded by the halo
    # trimming the padded window recovers the core (halo peeled off each side)
    assert t.core_in_pad == (slice(None), slice(8, 40), slice(8, 40))


def test_tile_edges_clip_to_grid():
    tl = _tile_at(0, 0, core=32, halo=8, ny=100, nx=100)  # no halo above/left
    assert (tl.pr0, tl.pc0) == (0, 0)
    assert tl.core_in_pad == (slice(None), slice(0, 32), slice(0, 32))

    br = _tile_at(3, 3, core=32, halo=8, ny=100, nx=100)  # partial, clips at 100
    assert (br.r0, br.r1, br.c0, br.c1) == (96, 100, 96, 100)
    assert (br.pr1, br.pc1) == (100, 100)
    assert br.core_in_pad == (slice(None), slice(8, 12), slice(8, 12))


def test_plan_tiles_cover_domain_disjointly():
    ny, nx, core, halo = 70, 90, 32, 8
    tiles = _plan_tiles(ny, nx, core, halo)
    assert len(tiles) == math.ceil(ny / core) * math.ceil(nx / core)
    cover = np.zeros((ny, nx), dtype=int)
    for t in tiles:
        cover[t.r0 : t.r1, t.c0 : t.c1] += 1
    assert (cover == 1).all()  # every cell covered exactly once


def test_core_in_pad_selects_the_core_cells():
    # label a padded window by global column index; trimming must yield exactly
    # the core's global columns.
    t = _tile_at(1, 1, core=4, halo=2, ny=20, nx=20)
    cols = np.broadcast_to(np.arange(t.pc0, t.pc1), (1, t.pr1 - t.pr0, t.pc1 - t.pc0))
    trimmed = cols[t.core_in_pad]
    assert trimmed.shape[2] == t.c1 - t.c0
    np.testing.assert_array_equal(trimmed[0, 0], np.arange(t.c0, t.c1))


@pytest.mark.parametrize(
    "row,col,expected",
    [(0, 0, (0.0, 60.0)), (1, 1, (24.0, 28.0)), (2, 2, (56.0, 0.0))],
)
def test_origin_anchors_padded_window_to_global_frame(row, col, expected):
    t = _tile_at(row, col, core=32, halo=8, ny=100, nx=100)
    assert t.origin(100) == expected


def test_single_tile_origin_is_whole_domain_zero():
    # a tile that covers the whole grid must have origin (0, 0), matching leaflux's
    # default -> a one-tile run == a no-tiling run.
    t = _tile_at(0, 0, core=50, halo=8, ny=50, nx=50)
    assert (t.pr0, t.pr1, t.pc0, t.pc1) == (0, 50, 0, 50)
    assert t.origin(50) == (0.0, 0.0)


# --- Coordinate / orientation conventions ---


def test_scatter_places_points_with_row_flip():
    shape = (2, 3, 4)  # (z, y, x)
    stack = np.array([[1, 0, 0, 0.5], [3, 2, 1, 0.25]], dtype=np.float32)
    grid = _scatter(stack, shape)
    # point x=1, y(south->north)=0 -> north-south row = ny-1-0 = 2
    assert grid[0, 2, 1] == np.float32(0.5)
    # point x=3, y=2 -> row = 3-1-2 = 0, z=1
    assert grid[1, 0, 3] == np.float32(0.25)
    assert np.isnan(grid).sum() == 2 * 3 * 4 - 2  # everything else is air


def test_leaf_area_transposes_zyx_to_yxz(monkeypatch):
    captured = {}

    class FakeLeafArea:
        @staticmethod
        def from_uniformgrid(arr):
            captured["shape"] = arr.shape
            return "leaf-area"

    monkeypatch.setattr(handler, "LeafArea", FakeLeafArea)
    out = _leaf_area(np.zeros((3, 4, 5), dtype=np.float32))  # (z, y, x)
    assert out == "leaf-area"
    assert captured["shape"] == (4, 5, 3)  # (y, x, z)


def test_terrain_lift_shifts_z_by_terrain_under_leaf():
    la = SimpleNamespace(
        leaf_area=np.array([[1.0, 0.0, 5.0, 0.2]], dtype=np.float32)  # x=1, y=0, z=5
    )
    terrain = np.arange(6, dtype=np.float32).reshape(2, 3)  # H=2, W=3
    _terrain_lift(la, terrain)
    # leaf y=0 (south) -> terrain row H-1-0 = 1; x=1 -> terrain[1, 1] = 4
    assert la.leaf_area[0, 2] == np.float32(9.0)


# --- Domain -> sun resolution ---


def test_domain_centroid_lat_lon_returns_center():
    lat, lon = _domain_centroid_lat_lon(_domain_at(46.9, -114.0))
    assert lat == pytest.approx(46.9)
    assert lon == pytest.approx(-114.0)


def test_domain_centroid_lat_lon_reprojects_from_projected_crs():
    # A domain stored in a projected CRS must still resolve to the right lat/lon.
    projected = _domain_at(46.9, -114.0).to_crs("EPSG:32611")
    lat, lon = _domain_centroid_lat_lon(projected)
    assert lat == pytest.approx(46.9, abs=1e-4)
    assert lon == pytest.approx(-114.0, abs=1e-4)


def test_sun_daytime_returns_position_not_night():
    sol, night = _sun(
        {"date_time": datetime(2023, 6, 21, 19, 0, 0)},
        _domain_at(46.9, -114.0),
    )
    assert sol is not None
    assert night is False
    assert math.degrees(sol.zenith) < handler.MAX_ZENITH_DEG


def test_sun_below_horizon_is_night():
    _, night = _sun(
        {"date_time": datetime(2023, 6, 21, 9, 0, 0)},  # ~1-2am local at lon -114
        _domain_at(46.9, -114.0),
    )
    assert night is True


def test_sun_location_comes_from_domain():
    # Same instant, two domains: the sun must track the domain centroid, not any
    # coordinate on the source dict.
    source = {"date_time": datetime(2023, 6, 21, 19, 0, 0)}
    montana, _ = _sun(source, _domain_at(46.9, -114.0))
    florida, _ = _sun(source, _domain_at(25.8, -80.2))
    assert montana is not None and florida is not None
    assert abs(montana.zenith - florida.zenith) > 1e-3


# --- Night / empty short-circuits (must not run the model) ---


def test_canopy_night_zeros_canopy_without_model(monkeypatch):
    monkeypatch.setattr(handler, "attenuate_all", _boom)
    lad = np.zeros((2, 3, 3), dtype=np.float32)
    lad[0, 1, 1] = 0.6
    out = _canopy_irradiance(lad, (0.0, 0.0), None, 0.5, (1, 1, 1), night=True)
    assert out.shape == lad.shape
    assert out[0, 1, 1] == np.float32(0.0)  # canopy cell reads 0 at night
    assert np.isnan(out[0, 0, 0])  # air stays air


def test_canopy_empty_returns_all_air_without_model(monkeypatch):
    monkeypatch.setattr(handler, "attenuate_all", _boom)
    out = _canopy_irradiance(
        np.zeros((2, 3, 3), dtype=np.float32),
        (0.0, 0.0),
        None,
        0.5,
        (1, 1, 1),
        night=False,
    )
    assert np.isnan(out).all()


def test_surface_night_zeros_ground_without_model(monkeypatch):
    monkeypatch.setattr(handler, "attenuate_all", _boom)
    lad = np.zeros((2, 3, 3), dtype=np.float32)
    lad[0, 1, 1] = 0.6
    out = _surface_irradiance(
        lad,
        np.zeros((3, 3), dtype=np.float32),
        (0.0, 0.0),
        None,
        0.5,
        (1, 1, 1),
        night=True,
    )
    assert out.shape == (3, 3)  # 2-D ground plane
    assert (out == 0.0).all()  # ground reads 0 at night


# --- Bridge: tiled assembly == whole-domain run (fast, no GCS) ---


def test_tiled_assembly_matches_whole_domain():
    nz, ny, nx = 8, 24, 24
    lad = np.zeros((nz, ny, nx), dtype=np.float32)
    lad[4, 6:9, 6:9] = 0.6  # occluders straddling tile seams
    lad[5, 15:18, 14:17] = 0.5
    lad[3, 10:12, 2:4] = 0.4
    terrain = np.zeros((ny, nx), dtype=np.float32)
    sol = SolarPosition(datetime(2023, 6, 21, 22, 0, 0), 46.9, -114.0)
    voxel, extn = (1.0, 1.0, 1.0), 0.5
    halo = math.ceil(nz * math.tan(sol.zenith) / voxel[0])
    core = 8
    tiles = _plan_tiles(ny, nx, core, halo)
    assert len(tiles) > 1  # non-vacuous: real tiling
    assert halo < ny  # a padded window is not the whole grid

    canopy = np.full((nz, ny, nx), np.nan, dtype=np.float32)
    surface = np.full((ny, nx), np.nan, dtype=np.float32)
    for t in tiles:
        origin = t.origin(ny)
        canopy[:, t.r0 : t.r1, t.c0 : t.c1] = _canopy_irradiance(
            lad[t.pad_zyx], origin, sol, extn, voxel, night=False
        )[t.core_in_pad]
        surface[t.r0 : t.r1, t.c0 : t.c1] = _surface_irradiance(
            lad[t.pad_zyx], terrain[t.pad_yx], origin, sol, extn, voxel, night=False
        )[t.core_in_pad_yx]

    ref_canopy = _canopy_irradiance(lad, (0.0, 0.0), sol, extn, voxel, night=False)
    ref_surface = _surface_irradiance(
        lad, terrain, (0.0, 0.0), sol, extn, voxel, night=False
    )

    for name, got, ref in [
        ("canopy", canopy, ref_canopy),
        ("surface", surface, ref_surface),
    ]:
        got_nan, ref_nan = np.isnan(got), np.isnan(ref)
        assert np.array_equal(got_nan, ref_nan), f"{name}: populated masks differ"
        np.testing.assert_allclose(
            got[~got_nan], ref[~ref_nan], atol=1e-6, err_msg=name
        )


# --- run_leaflux orchestration wiring (mocked I/O, no GCS) ---


def test_run_leaflux_wires_job_and_tiles(monkeypatch):
    fake_lad = SimpleNamespace(
        sizes={"z": 6, "y": 40, "x": 40},
        attrs={
            "transform": [2.0, 0, 0, 0, -2.0, 0],
            "z_resolution": 1.0,
            "z_origin": 100.0,
        },
        z=np.arange(6),
        y=np.arange(40),
        x=np.arange(40),
        rio=SimpleNamespace(crs="EPSG:32611"),
    )
    monkeypatch.setattr(handler, "_open", lambda gid: fake_lad)
    monkeypatch.setattr(handler, "_init_output", lambda *a, **k: None)
    monkeypatch.setattr(handler, "WINDOW_TARGET_CELLS", 24)  # shrink to force >1 tile
    captured = {}

    def fake_write(tiles, job, progress):
        captured["tiles"], captured["job"] = tiles, job

    monkeypatch.setattr(handler, "_write_tiles", fake_write)

    grid = {
        "id": "test-out",
        "bands": [{"key": CANOPY_BAND}, {"key": SURFACE_BAND}],
        "source": {
            "source_lad_grid_id": "src-lad",
            "source_terrain_grid_id": "src-dem",
            "date_time": datetime(2023, 6, 21, 19, 0, 0),
            "extinction_coefficient": 0.5,
        },
    }
    result = handler.run_leaflux(grid, _domain_at(46.9, -114.0), lambda *a, **k: None)

    job = captured["job"]
    assert job.want_canopy and job.want_surface
    assert job.dem_id == "src-dem"
    assert job.voxel == (2.0, 2.0, 1.0)
    assert (job.nz, job.ny, job.nx) == (6, 40, 40)

    cover = np.zeros((40, 40), dtype=int)
    for t in captured["tiles"]:
        cover[t.r0 : t.r1, t.c0 : t.c1] += 1
    assert (cover == 1).all()  # planned tiles tile the domain

    # Canopy requested -> 3D grid with z metadata.
    assert job.is_3d is True
    assert result.georeference["shape"] == [6, 40, 40]
    assert "z_resolution" in result.georeference
    assert "z_origin" in result.georeference
    assert result.chunk_shape[0] == 6


def test_run_leaflux_surface_only_is_2d(monkeypatch):
    """A surface-only request yields a 2D (y, x) grid, not a 3D grid with NaN
    padding layers."""
    fake_lad = SimpleNamespace(
        sizes={"z": 6, "y": 30, "x": 30},
        attrs={
            "transform": [2.0, 0, 0, 0, -2.0, 0],
            "z_resolution": 1.0,
            "z_origin": 100.0,
        },
        z=np.arange(6),
        y=np.arange(30),
        x=np.arange(30),
        rio=SimpleNamespace(crs="EPSG:32611"),
    )
    monkeypatch.setattr(handler, "_open", lambda gid: fake_lad)
    monkeypatch.setattr(handler, "_init_output", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(
        handler, "_write_tiles", lambda tiles, job, progress: captured.update(job=job)
    )

    grid = {
        "id": "test-out",
        "bands": [{"key": SURFACE_BAND}],
        "source": {
            "source_lad_grid_id": "src-lad",
            "source_terrain_grid_id": "src-dem",
            "date_time": datetime(2023, 6, 21, 19, 0, 0),
            "extinction_coefficient": 0.5,
        },
    }
    result = handler.run_leaflux(grid, _domain_at(46.9, -114.0), lambda *a, **k: None)

    job = captured["job"]
    assert job.want_surface and not job.want_canopy
    assert job.is_3d is False
    assert job.dem_id == "src-dem"
    # 2D geometry: (y, x) only, no z axis or z metadata.
    assert result.georeference["shape"] == [30, 30]
    assert "z_resolution" not in result.georeference
    assert "z_origin" not in result.georeference
    assert len(result.chunk_shape) == 2


def test_run_leaflux_no_surface_band_skips_terrain(monkeypatch):
    fake_lad = SimpleNamespace(
        sizes={"z": 6, "y": 20, "x": 20},
        attrs={
            "transform": [2.0, 0, 0, 0, -2.0, 0],
            "z_resolution": 1.0,
            "z_origin": 100.0,
        },
        z=np.arange(6),
        y=np.arange(20),
        x=np.arange(20),
        rio=SimpleNamespace(crs="EPSG:32611"),
    )
    monkeypatch.setattr(handler, "_open", lambda gid: fake_lad)
    monkeypatch.setattr(handler, "_init_output", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(
        handler, "_write_tiles", lambda tiles, job, progress: captured.update(job=job)
    )

    grid = {
        "id": "test-out",
        "bands": [{"key": CANOPY_BAND}],
        "source": {
            "source_lad_grid_id": "src-lad",
            "source_terrain_grid_id": "src-dem",  # present but not requested
            "date_time": datetime(2023, 6, 21, 19, 0, 0),
            "extinction_coefficient": 0.5,
        },
    }
    handler.run_leaflux(grid, _domain_at(46.9, -114.0), lambda *a, **k: None)
    job = captured["job"]
    assert job.want_canopy and not job.want_surface
    assert job.dem_id is None  # no terrain opened when surface isn't requested
