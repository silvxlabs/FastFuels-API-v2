"""
Unit tests for the landscape export handler.

These tests build synthetic xarray Datasets in memory, patch the GCS-loading
layer, capture the GeoTIFF the handler builds, and verify the output with
rasterio (band order, int16 scaled encodings, nodata handling, cropping,
georeference, band metadata).

The handler consumes a pre-validated ``source`` dict, so fixtures mirror what
the API's ``LandscapeExportSource`` persists.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from exporter.errors import ProcessingError
from exporter.handlers.landscape import export_landscape
from rasterio.transform import from_bounds

_NX = 4
_NY = 3
_DX = 30.0
_WEST = 500000.0
_NORTH = 5200000.0 + _NY * _DX
_CRS = "EPSG:32611"
_TRANSFORM = list(
    from_bounds(_WEST, _NORTH - _NY * _DX, _WEST + _NX * _DX, _NORTH, _NX, _NY)
)

_BAND_ORDER = [
    "elevation",
    "slope",
    "aspect",
    "fuel_model",
    "canopy_cover",
    "canopy_height",
    "canopy_base_height",
    "canopy_bulk_density",
]


def noop_progress(message: str, percent: int | None = None):
    pass


def _make_2d_dataset(
    bands: dict[str, np.ndarray],
    *,
    west: float = _WEST,
    north: float = _NORTH,
) -> xr.Dataset:
    sample = next(iter(bands.values()))
    ny, nx = sample.shape
    ds = xr.Dataset(
        data_vars={name: (("y", "x"), arr) for name, arr in bands.items()},
        coords={
            "y": north - _DX * (np.arange(ny) + 0.5),
            "x": west + _DX * (np.arange(nx) + 0.5),
        },
    )
    return ds.rio.write_crs(_CRS)


def _role_values() -> dict[str, np.ndarray]:
    """Distinct constant fields per role so band order is verifiable."""
    return {
        "elevation": np.full((_NY, _NX), 1500.0),
        "slope": np.full((_NY, _NX), 20.0),
        "aspect": np.full((_NY, _NX), 180.0),
        "fbfm": np.full((_NY, _NX), 122.0),
        "cc": np.full((_NY, _NX), 45.0),
        "chm": np.full((_NY, _NX), 18.3),
        "cbh": np.full((_NY, _NX), 2.6),
        "cbd": np.full((_NY, _NX), 0.12),
    }


def _build_source() -> dict:
    return {
        "name": "landscape",
        "domain_id": "test-domain",
        "fire_behavior_fuel_model": "fbfm40",
        "elevation": {"grid_id": "topo", "band": "elevation"},
        "slope": {"grid_id": "topo", "band": "slope"},
        "aspect": {"grid_id": "topo", "band": "aspect"},
        "fuel_model": {"grid_id": "fbfm", "band": "fbfm"},
        "canopy_cover": {"grid_id": "canopy", "band": "cc"},
        "canopy_height": {"grid_id": "canopy", "band": "chm"},
        "canopy_base_height": {"grid_id": "canopy", "band": "cbh"},
        "canopy_bulk_density": {"grid_id": "canopy", "band": "cbd"},
        "resolved": {
            "landscape_grid": {
                "nx": _NX,
                "ny": _NY,
                "dx": _DX,
                "dy": _DX,
                "transform": _TRANSFORM,
                "crs": _CRS,
            },
        },
    }


def _build_grids(values: dict[str, np.ndarray] | None = None) -> dict[str, xr.Dataset]:
    v = values or _role_values()
    return {
        "topo": _make_2d_dataset(
            {"elevation": v["elevation"], "slope": v["slope"], "aspect": v["aspect"]}
        ),
        "fbfm": _make_2d_dataset({"fbfm": v["fbfm"]}),
        "canopy": _make_2d_dataset(
            {"cc": v["cc"], "chm": v["chm"], "cbh": v["cbh"], "cbd": v["cbd"]}
        ),
    }


@pytest.fixture
def captured_tif(tmp_path):
    """Patch the GCS upload; capture the staged GeoTIFF locally."""
    captured = {"tif_path": None, "gcs_path": None}

    def fake_upload(tif_path, export):
        copied = tmp_path / "captured.tif"
        copied.write_bytes(Path(tif_path).read_bytes())
        captured["tif_path"] = str(copied)
        captured["gcs_path"] = f"gs://exports/{export['id']}/captured.tif"
        return captured["gcs_path"]

    with patch("exporter.handlers.landscape._upload_tif", side_effect=fake_upload):
        yield captured


@pytest.fixture
def patch_load_grid():
    """Return a context manager that patches load_grid_zarr with a dict-of-Datasets."""

    def _patch(grids: dict[str, xr.Dataset]):
        return patch(
            "exporter.handlers.landscape.load_grid_zarr",
            side_effect=lambda gid: grids[gid],
        )

    return _patch


class TestExportLandscape:
    def test_profile_and_georeference(self, captured_tif, patch_load_grid):
        with patch_load_grid(_build_grids()):
            gcs_path = export_landscape(
                {"id": "e1", "name": "test"}, _build_source(), noop_progress
            )
        assert gcs_path == captured_tif["gcs_path"]
        with rasterio.open(captured_tif["tif_path"]) as src:
            assert src.count == 8
            assert all(dt == "int16" for dt in src.dtypes)
            assert src.nodatavals == (-9999,) * 8
            assert src.width == _NX
            assert src.height == _NY
            assert src.crs.to_epsg() == 32611
            assert list(src.transform)[:6] == pytest.approx(_TRANSFORM[:6])

    def test_band_order_and_scaled_encodings(self, captured_tif, patch_load_grid):
        with patch_load_grid(_build_grids()):
            export_landscape({"id": "e1", "name": ""}, _build_source(), noop_progress)
        with rasterio.open(captured_tif["tif_path"]) as src:
            data = src.read()
        # Unscaled bands round to the nearest integer.
        assert (data[0] == 1500).all()  # elevation, m
        assert (data[1] == 20).all()  # slope, deg
        assert (data[2] == 180).all()  # aspect, deg
        assert (data[3] == 122).all()  # fuel model code
        assert (data[4] == 45).all()  # canopy cover, %
        # LANDFIRE scaled encodings.
        assert (data[5] == 183).all()  # canopy height, m * 10
        assert (data[6] == 26).all()  # canopy base height, m * 10
        assert (data[7] == 12).all()  # canopy bulk density, kg/m**3 * 100

    def test_band_descriptions_and_tags(self, captured_tif, patch_load_grid):
        with patch_load_grid(_build_grids()):
            export_landscape({"id": "e1", "name": ""}, _build_source(), noop_progress)
        with rasterio.open(captured_tif["tif_path"]) as src:
            assert src.descriptions == (
                "Elevation",
                "Slope",
                "Aspect",
                "Fuel Model",
                "Canopy Cover",
                "Canopy Height",
                "Canopy Base Height",
                "Canopy Bulk Density",
            )
            assert src.tags(1)["BandName"] == "Elevation"
            assert src.tags(1)["Units"] == "meters"
            assert src.tags(4)["Units"] == (
                "Scott and Burgan Fire Behavior Fuel Models"
            )
            assert src.tags(6)["Units"] == "meters * 10"
            assert src.tags(8)["Units"] == "kg/m^3 * 100"

    def test_fbfm13_declaration_reflected_in_tags(self, captured_tif, patch_load_grid):
        source = _build_source()
        source["fire_behavior_fuel_model"] = "fbfm13"
        with patch_load_grid(_build_grids()):
            export_landscape({"id": "e1", "name": ""}, source, noop_progress)
        with rasterio.open(captured_tif["tif_path"]) as src:
            assert src.tags(4)["Units"] == "Anderson Fire Behavior Fuel Models"

    def test_nan_becomes_nodata(self, captured_tif, patch_load_grid):
        values = _role_values()
        values["chm"] = values["chm"].copy()
        values["chm"][1, 2] = np.nan
        with patch_load_grid(_build_grids(values)):
            export_landscape({"id": "e1", "name": ""}, _build_source(), noop_progress)
        with rasterio.open(captured_tif["tif_path"]) as src:
            ch = src.read(6)
        assert ch[1, 2] == -9999
        assert (ch != -9999).sum() == _NY * _NX - 1

    def test_oversized_role_grid_cropped(self, captured_tif, patch_load_grid):
        # Role grids extend one cell west and north of the landscape; the
        # handler must crop by integer slicing to the landscape window.
        values = _role_values()
        ny, nx = _NY + 1, _NX + 1
        big = {k: np.full((ny, nx), np.nan) for k in values}
        for k, v in values.items():
            big[k][1:, 1:] = v  # landscape window holds the real values
        grids = {
            "topo": _make_2d_dataset(
                {
                    "elevation": big["elevation"],
                    "slope": big["slope"],
                    "aspect": big["aspect"],
                },
                west=_WEST - _DX,
                north=_NORTH + _DX,
            ),
            "fbfm": _make_2d_dataset(
                {"fbfm": big["fbfm"]}, west=_WEST - _DX, north=_NORTH + _DX
            ),
            "canopy": _make_2d_dataset(
                {
                    "cc": big["cc"],
                    "chm": big["chm"],
                    "cbh": big["cbh"],
                    "cbd": big["cbd"],
                },
                west=_WEST - _DX,
                north=_NORTH + _DX,
            ),
        }
        with patch_load_grid(grids):
            export_landscape({"id": "e1", "name": ""}, _build_source(), noop_progress)
        with rasterio.open(captured_tif["tif_path"]) as src:
            data = src.read()
        assert data.shape == (8, _NY, _NX)
        assert (data != -9999).all()  # no NaN border cells leaked in
        assert (data[0] == 1500).all()

    def test_missing_band_raises_processing_error(self, patch_load_grid):
        grids = _build_grids()
        grids["canopy"] = grids["canopy"].drop_vars("cbd")
        with patch_load_grid(grids):
            with pytest.raises(ProcessingError) as exc:
                export_landscape(
                    {"id": "e1", "name": ""}, _build_source(), noop_progress
                )
        assert exc.value.code == "BAND_NOT_FOUND"

    def test_grid_load_failure_raises_processing_error(self):
        def boom(_gid):
            raise RuntimeError("zarr gone")

        with patch("exporter.handlers.landscape.load_grid_zarr", side_effect=boom):
            with pytest.raises(ProcessingError) as exc:
                export_landscape(
                    {"id": "e1", "name": ""}, _build_source(), noop_progress
                )
        assert exc.value.code == "GRID_LOAD_ERROR"
