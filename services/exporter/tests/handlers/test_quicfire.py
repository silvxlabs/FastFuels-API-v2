"""
Unit tests for the QUIC-Fire export handler.

These tests build synthetic xarray Datasets in memory, patch the GCS-loading
and Firestore lookups, capture the zip the handler builds, and verify the
output file contents byte-for-byte (Fortran record dtype/shape, surface +
canopy stitching math, Y-flip, NaN handling, optional file presence).

End-to-end tests against real GCS live in tests/integration/test_quicfire.py.
"""

import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from exporter.handlers.quicfire import export_quicfire
from rasterio.transform import from_bounds
from scipy.io import FortranFile

_NX = 4
_NY = 3
_NZ = 5
_DZ = 1.0
_TRANSFORM = list(
    from_bounds(500000, 5200000, 500000 + _NX * 30, 5200000 + _NY * 30, _NX, _NY)
)


def noop_progress(message: str, percent: int | None = None):
    pass


def _make_3d_dataset(bands: dict[str, np.ndarray]) -> xr.Dataset:
    """Build a tree-voxel-shaped Dataset with (z, y, x) dims."""
    ds = xr.Dataset(
        data_vars={name: (("z", "y", "x"), arr) for name, arr in bands.items()},
        coords={
            "z": np.arange(_NZ, dtype=np.float64),
            "y": np.arange(_NY, dtype=np.float64),
            "x": np.arange(_NX, dtype=np.float64),
        },
    )
    ds = ds.rio.write_crs("EPSG:32611")
    return ds


def _make_2d_dataset(bands: dict[str, np.ndarray]) -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={name: (("y", "x"), arr) for name, arr in bands.items()},
        coords={
            "y": np.arange(_NY, dtype=np.float64),
            "x": np.arange(_NX, dtype=np.float64),
        },
    )
    ds = ds.rio.write_crs("EPSG:32611")
    return ds


def _fake_domain_doc() -> dict:
    """Minimal valid domain doc with one Polygon feature."""
    return {
        "id": "test-domain",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"role": "domain"},
                "geometry": {
                    "type": "Polygon",
                    # JSON-stringified per the Firestore convention.
                    "coordinates": json.dumps(
                        [
                            [
                                [-114.0, 46.8],
                                [-113.99, 46.8],
                                [-113.99, 46.81],
                                [-114.0, 46.81],
                                [-114.0, 46.8],
                            ]
                        ]
                    ),
                },
            }
        ],
    }


def _build_source(
    *,
    canopy_grid_id: str = "tree",
    surface_grid_id: str = "uniform",
    topo_grid_id: str | None = None,
    canopy_savr_grid_id: str | None = None,
    surface_savr_grid_id: str | None = None,
) -> dict:
    source: dict = {
        "name": "quicfire",
        "domain_id": "test-domain",
        "canopy_bulk_density": {
            "grid_id": canopy_grid_id,
            "band": "bulk_density.foliage.live",
        },
        "canopy_moisture": {"grid_id": canopy_grid_id, "band": "fuel_moisture.live"},
        "surface_fuel_load": {"grid_id": surface_grid_id, "band": "fuel_load.1hr"},
        "surface_fuel_depth": {"grid_id": surface_grid_id, "band": "fuel_depth"},
        "surface_moisture": {"grid_id": surface_grid_id, "band": "fuel_moisture.1hr"},
        "rhof_merge": "sum",
        "moist_merge": "weighted_avg",
        "savr_merge": "weighted_avg",
        "resolved": {
            "domain": {"crs": "EPSG:32611"},
            "fire_grid": {
                "nx": _NX,
                "ny": _NY,
                "nz": _NZ,
                "transform": _TRANSFORM,
                "z_origin": 0.0,
                "z_resolution": _DZ,
                "crs": "EPSG:32611",
            },
            "roles": {},
        },
    }
    if topo_grid_id:
        source["topography"] = {"grid_id": topo_grid_id, "band": "elevation"}
    if canopy_savr_grid_id and surface_savr_grid_id:
        source["canopy_savr"] = {"grid_id": canopy_savr_grid_id, "band": "savr.foliage"}
        source["surface_savr"] = {"grid_id": surface_savr_grid_id, "band": "savr.1hr"}
    return source


@pytest.fixture
def captured_zip(tmp_path) -> Iterator[dict]:
    """Patch GCS upload + domain Firestore read; capture the staged zip locally.

    Yields a mutable dict whose ``zip_path`` key the handler populates with
    the local path of the zip it would have uploaded. Tests inspect the zip
    directly after calling ``export_quicfire``.
    """
    captured = {"zip_path": None, "gcs_path": None}

    def fake_upload(zip_path, export):
        # Copy out of tmp before the handler's TemporaryDirectory cleanup.
        copied = tmp_path / "captured.zip"
        copied.write_bytes(Path(zip_path).read_bytes())
        captured["zip_path"] = str(copied)
        captured["gcs_path"] = f"gs://exports/{export['id']}/captured.zip"
        return captured["gcs_path"]

    class _FakeSnapshot:
        def to_dict(self):
            return _fake_domain_doc()

    def fake_get_document(_collection, _doc_id, **_kwargs):
        return None, _FakeSnapshot()

    with (
        patch("exporter.handlers.quicfire._upload_zip", side_effect=fake_upload),
        patch("exporter.handlers.quicfire.get_document", side_effect=fake_get_document),
    ):
        yield captured


@pytest.fixture
def patch_load_grid():
    """Return a context manager that patches load_grid_zarr with a dict-of-Datasets."""

    def _patch(grids: dict[str, xr.Dataset]):
        return patch(
            "exporter.handlers.quicfire.load_grid_zarr",
            side_effect=lambda gid: grids[gid],
        )

    return _patch


def _read_3d(zip_path: str, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            data = f.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with FortranFile(tmp_path, "r") as ff:
            arr = ff.read_record(dtype=np.float32)
    finally:
        os.unlink(tmp_path)
    return arr.reshape(_NZ, _NY, _NX)


def _read_2d(zip_path: str, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            data = f.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with FortranFile(tmp_path, "r") as ff:
            arr = ff.read_record(dtype=np.float32)
    finally:
        os.unlink(tmp_path)
    return arr.reshape(_NY, _NX)


class TestExportQuicfireMinimal:
    def test_minimal_zip_contents(self, captured_zip, patch_load_grid):
        canopy_rhof = np.full((_NZ, _NY, _NX), 0.2, dtype=np.float32)
        canopy_rhof[0] = 0.0  # no canopy at the bottom slab
        canopy_moist = np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32)
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": canopy_moist,
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            gcs_path = export_quicfire(
                {"id": "exp-1", "name": ""},
                _build_source(),
                noop_progress,
            )

        assert gcs_path == captured_zip["gcs_path"]
        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            assert set(zf.namelist()) == {
                "treesrhof.dat",
                "treesmoist.dat",
                "treesfueldepth.dat",
                "metadata.json",
                "domain.geojson",
            }

    def test_treesrhof_surface_at_k0_canopy_above(self, captured_zip, patch_load_grid):
        canopy_rhof = np.full((_NZ, _NY, _NX), 0.2, dtype=np.float32)
        canopy_rhof[0] = 0.0
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-2", "name": ""}, _build_source(), noop_progress)

        rhof = _read_3d(captured_zip["zip_path"], "treesrhof.dat")
        # Surface contribution at k=0: load/dz = 0.5/1.0 = 0.5.
        assert np.allclose(rhof[0], 0.5)
        # Canopy at k>0 unchanged (Y-flip is a permutation, all values still 0.2).
        assert np.allclose(rhof[1:], 0.2)

    def test_treesmoist_mass_weighted(self, captured_zip, patch_load_grid):
        # Canopy bulk density at k=0 = 0.3, surface_load/dz = 0.7 → total = 1.0.
        # canopy_moist = 0.5 (50%), surf_moist = 0.1 (10%)
        # weighted_avg = (0.3*0.5 + 0.7*0.1) / 1.0 = 0.22
        canopy_rhof = np.full((_NZ, _NY, _NX), 0.3, dtype=np.float32)
        canopy_moist = np.full((_NZ, _NY, _NX), 50.0, dtype=np.float32)
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": canopy_moist,
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.7, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 10.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-3", "name": ""}, _build_source(), noop_progress)

        moist = _read_3d(captured_zip["zip_path"], "treesmoist.dat")
        # eps in the divide makes the result 0.22 - tiny correction; tolerance accommodates.
        assert np.allclose(moist[0], 0.22, atol=1e-5)
        # Above k=0, moist = canopy_moist / 100 = 0.5
        assert np.allclose(moist[1:], 0.5)

    def test_treesfueldepth_surface_only_at_k0(self, captured_zip, patch_load_grid):
        canopy_rhof = np.zeros((_NZ, _NY, _NX), dtype=np.float32)
        canopy_rhof[2] = 0.4
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.13, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-4", "name": ""}, _build_source(), noop_progress)

        depth = _read_3d(captured_zip["zip_path"], "treesfueldepth.dat")
        assert np.allclose(depth[0], 0.13)
        assert np.allclose(depth[1:], 0.0)

    def test_dtype_is_float32(self, captured_zip, patch_load_grid):
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float64
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float64),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float64),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float64),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float64),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-5", "name": ""}, _build_source(), noop_progress)

        rhof = _read_3d(captured_zip["zip_path"], "treesrhof.dat")
        assert rhof.dtype == np.float32

    def test_nan_replaced_with_zero(self, captured_zip, patch_load_grid):
        canopy_rhof = np.full((_NZ, _NY, _NX), 0.2, dtype=np.float32)
        canopy_rhof[0, 0, 0] = (
            np.nan
        )  # NaN at k=0 corner; gets stitched then NaN-cleaned
        canopy_rhof[3, 1, 1] = np.nan  # NaN at k>0
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-6", "name": ""}, _build_source(), noop_progress)

        rhof = _read_3d(captured_zip["zip_path"], "treesrhof.dat")
        assert not np.any(np.isnan(rhof))


class TestYFlip:
    def test_y_flip_applied_to_3d(self, captured_zip, patch_load_grid):
        # Mark each y row with a unique value so Y-flip is detectable.
        canopy_rhof = np.zeros((_NZ, _NY, _NX), dtype=np.float32)
        canopy_rhof[2, :, :] = (
            np.arange(_NY).reshape(_NY, 1) + 1
        )  # row j gets value j+1 at k=2

        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire({"id": "exp-7", "name": ""}, _build_source(), noop_progress)

        rhof = _read_3d(captured_zip["zip_path"], "treesrhof.dat")
        # Original row j had value j+1; after Y-flip, row j holds (NY-j) at k=2.
        for j in range(_NY):
            assert np.allclose(rhof[2, j, :], _NY - j)

    def test_y_flip_applied_to_topo(self, captured_zip, patch_load_grid):
        elevation = (
            np.arange(_NY).reshape(_NY, 1).repeat(_NX, axis=1).astype(np.float32) + 1
        )
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float32
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )
        topo_ds = _make_2d_dataset({"elevation": elevation})

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds, "topo": topo_ds}):
            export_quicfire(
                {"id": "exp-8", "name": ""},
                _build_source(topo_grid_id="topo"),
                noop_progress,
            )

        topo = _read_2d(captured_zip["zip_path"], "topo.dat")
        for j in range(_NY):
            assert np.allclose(topo[j, :], _NY - j)


class TestOptionalLayers:
    def test_topography_present_when_role_set(self, captured_zip, patch_load_grid):
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float32
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )
        topo_ds = _make_2d_dataset(
            {"elevation": np.full((_NY, _NX), 1500.0, dtype=np.float32)}
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds, "topo": topo_ds}):
            export_quicfire(
                {"id": "exp-9", "name": ""},
                _build_source(topo_grid_id="topo"),
                noop_progress,
            )

        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            assert "topo.dat" in zf.namelist()
        topo = _read_2d(captured_zip["zip_path"], "topo.dat")
        assert np.allclose(topo, 1500.0)

    def test_treesss_present_when_savr_pair_set(self, captured_zip, patch_load_grid):
        canopy_rhof = np.full((_NZ, _NY, _NX), 0.5, dtype=np.float32)
        canopy_rhof[0] = 0.0  # surface-only at bottom slab
        canopy_savr = np.full((_NZ, _NY, _NX), 1000.0, dtype=np.float32)
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": canopy_rhof,
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
                "savr.foliage": canopy_savr,
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.full((_NY, _NX), 0.5, dtype=np.float32),
                "fuel_depth": np.full((_NY, _NX), 0.1, dtype=np.float32),
                "fuel_moisture.1hr": np.full((_NY, _NX), 6.0, dtype=np.float32),
                "savr.1hr": np.full((_NY, _NX), 2000.0, dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire(
                {"id": "exp-10", "name": ""},
                _build_source(
                    canopy_savr_grid_id="tree", surface_savr_grid_id="uniform"
                ),
                noop_progress,
            )

        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            assert "treesss.dat" in zf.namelist()

        ss = _read_3d(captured_zip["zip_path"], "treesss.dat")
        # Surface-only k=0: SAVR = 2000, particle size = 2/2000 = 0.001 m
        assert np.allclose(ss[0], 0.001)
        # Canopy above k=0: SAVR = 1000, particle size = 2/1000 = 0.002 m
        assert np.allclose(ss[1:], 0.002)

    def test_treesss_absent_when_savr_not_set(self, captured_zip, patch_load_grid):
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float32
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire(
                {"id": "exp-11", "name": ""}, _build_source(), noop_progress
            )

        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            assert "treesss.dat" not in zf.namelist()


class TestMetadata:
    def test_metadata_contents(self, captured_zip, patch_load_grid):
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float32
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire(
                {"id": "exp-meta", "name": "blue mountain run"},
                _build_source(),
                noop_progress,
            )

        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            with zf.open("metadata.json") as f:
                meta = json.loads(f.read())

        assert meta["format"] == "quicfire"
        assert meta["export_id"] == "exp-meta"
        assert meta["export_name"] == "blue mountain run"
        fg = meta["fire_grid"]
        assert fg["nx"] == _NX
        assert fg["ny"] == _NY
        assert fg["nz"] == _NZ
        assert fg["dz"] == _DZ
        assert fg["crs"] == "EPSG:32611"
        assert "completed_on" in meta
        assert meta["source"]["name"] == "quicfire"


class TestDomainGeojson:
    def test_domain_geojson_parsed_and_written(self, captured_zip, patch_load_grid):
        tree_ds = _make_3d_dataset(
            {
                "bulk_density.foliage.live": np.zeros(
                    (_NZ, _NY, _NX), dtype=np.float32
                ),
                "fuel_moisture.live": np.full((_NZ, _NY, _NX), 100.0, dtype=np.float32),
            }
        )
        surf_ds = _make_2d_dataset(
            {
                "fuel_load.1hr": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_depth": np.zeros((_NY, _NX), dtype=np.float32),
                "fuel_moisture.1hr": np.zeros((_NY, _NX), dtype=np.float32),
            }
        )

        with patch_load_grid({"tree": tree_ds, "uniform": surf_ds}):
            export_quicfire(
                {"id": "exp-dom", "name": ""}, _build_source(), noop_progress
            )

        with zipfile.ZipFile(captured_zip["zip_path"]) as zf:
            with zf.open("domain.geojson") as f:
                gj = json.loads(f.read())

        assert gj["type"] == "FeatureCollection"
        feature = gj["features"][0]
        # Coordinates were JSON-stringified in the fake doc; the handler must parse them.
        coords = feature["geometry"]["coordinates"]
        assert isinstance(coords, list)
        assert isinstance(coords[0], list)  # outer ring
        assert isinstance(coords[0][0], list)  # first vertex
        assert len(coords[0]) == 5  # closed polygon
