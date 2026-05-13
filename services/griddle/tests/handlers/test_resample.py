"""
Tests for the resample handler.

Tests cover:
- Single-variable upsampling and downsampling
- Multi-variable resampling with variable preservation
- Per-variable resampling method overrides
- Spatial metadata preservation (CRS, coordinates)
- Error handling for missing/empty source grids
- Progress callback behavior
- Zarr round-trip with Dataset.rio.to_raster()

All tests use ``alignment={"target": "native", "resolution": ...}`` so the
output preserves the source pixel anchor; the resolution change is what's
under test, not the lattice anchor (covered by alignment-specific tests).
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.resample import resample_grid
from shapely.geometry import box

from lib.errors import ProcessingError
from lib.zarr_utils import load_zarr, save_zarr


def _make_mock_source_ds(
    variables: dict[str, np.ndarray],
    y_coords=None,
    x_coords=None,
    crs="EPSG:32611",
    resolution=29.945,
):
    """Create a mock Dataset mimicking real GCS format.

    Uses real spatial metadata: CRS EPSG:32611, ~30m resolution.
    Each variable is a named 2D (y, x) array — no "band" dimension.

    Args:
        variables: Mapping of variable name to 2D numpy array (y, x)
        y_coords: explicit y coordinates (auto-generated if None)
        x_coords: explicit x coordinates (auto-generated if None)
        crs: coordinate reference system string
        resolution: pixel size for auto-generated coordinates
    """
    first = next(iter(variables.values()))
    ny, nx = first.shape

    if y_coords is None:
        y_coords = 5190907.682 - np.arange(ny, dtype=np.float64) * resolution
    if x_coords is None:
        x_coords = 719959.803 + np.arange(nx, dtype=np.float64) * resolution

    data_vars = {}
    for name, arr in variables.items():
        da = xr.DataArray(
            data=arr.astype(np.float32),
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        data_vars[name] = da

    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform()
    return ds


def _domain_gdf(crs="EPSG:32611"):
    """Domain covering the synthetic source-data extent (Blue Mountain area)."""
    return gpd.GeoDataFrame(
        {
            "geometry": [
                box(719959.803, 5190907.682 - 300.0, 719959.803 + 300.0, 5190907.682)
            ]
        },
        crs=crs,
    )


def _native_alignment(resolution: float) -> dict:
    return {"target": "native", "resolution": resolution}


def _band_types(*names: str, kind: str = "continuous") -> dict[str, str]:
    return {n: kind for n in names}


class TestResampleGrid:
    """Tests for the resample_grid function."""

    @patch("griddle.handlers.resample.load_zarr")
    def test_single_variable_upsampling(self, mock_load_zarr):
        """30m -> 10m upsampling produces ~3x output dimensions."""
        data = np.random.rand(10, 10)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        da = result["elevation"]
        assert da.dims == ("y", "x")
        # Output should be ~3x source in each dim (±1 pixel for alignment)
        assert da.shape[0] >= 28 and da.shape[0] <= 32
        assert da.shape[1] >= 28 and da.shape[1] <= 32
        # Resolution should match target
        res = result.rio.resolution()
        assert abs(res[0]) == pytest.approx(10.0, rel=0.01)
        assert abs(res[1]) == pytest.approx(10.0, rel=0.01)

    @patch("griddle.handlers.resample.load_zarr")
    def test_single_variable_downsampling(self, mock_load_zarr):
        """10m -> 30m downsampling produces ~1/3 output dimensions."""
        data = np.random.rand(30, 30)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=10.0
        )
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(30.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        da = result["elevation"]
        assert da.dims == ("y", "x")
        # Output should be ~1/3 source in each dim
        assert da.shape[0] >= 9 and da.shape[0] <= 11
        assert da.shape[1] >= 9 and da.shape[1] <= 11
        res = result.rio.resolution()
        assert abs(res[0]) == pytest.approx(30.0, rel=0.01)
        assert abs(res[1]) == pytest.approx(30.0, rel=0.01)

    @patch("griddle.handlers.resample.load_zarr")
    def test_multi_variable_preserves_all_variables(self, mock_load_zarr):
        """Multi-variable resample preserves all variables with correct names."""
        variables = {
            "fbfm": np.random.rand(10, 10),
            "canopy_cover": np.random.rand(10, 10),
            "canopy_height": np.random.rand(10, 10),
        }
        mock_load_zarr.return_value = _make_mock_source_ds(variables, resolution=30.0)
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("fbfm", "canopy_cover", "canopy_height"),
            progress=progress,
        )

        assert set(result.data_vars) == {"fbfm", "canopy_cover", "canopy_height"}
        for var_name in result.data_vars:
            assert result[var_name].dims == ("y", "x")

    @patch("griddle.handlers.resample.load_zarr")
    def test_method_override_per_variable(self, mock_load_zarr):
        """Per-variable overrides apply the correct method: nearest preserves
        exact source values while bilinear produces interpolated values."""
        # 4x4 grid with distinct integer values per variable
        categorical = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        continuous = np.array(
            [
                [0.0, 10.0, 20.0, 30.0],
                [0.0, 10.0, 20.0, 30.0],
                [0.0, 10.0, 20.0, 30.0],
                [0.0, 10.0, 20.0, 30.0],
            ]
        )
        variables = {
            "fbfm": categorical,
            "canopy_cover": continuous,
        }
        mock_load_zarr.return_value = _make_mock_source_ds(variables, resolution=30.0)
        progress = MagicMock()

        # alignment.method=bilinear is the global default; method_overrides flips
        # fbfm to nearest. This exercises both overrides and continuous fallback.
        result = resample_grid(
            source_grid_id="test-grid",
            alignment={"target": "native", "resolution": 10.0, "method": "bilinear"},
            method_overrides={"fbfm": "nearest"},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("fbfm", "canopy_cover"),
            progress=progress,
        )

        assert set(result.data_vars) == {"fbfm", "canopy_cover"}

        # Nearest variable (fbfm): all output values must be exact source values
        fbfm_vals = result["fbfm"].values.astype(float)
        source_vals = {1.0, 2.0, 3.0, 4.0}
        unique_output = set(np.unique(fbfm_vals[~np.isnan(fbfm_vals)]))
        assert unique_output.issubset(source_vals), (
            f"Nearest interpolation should only produce source values {source_vals}, "
            f"got {unique_output}"
        )

        # Bilinear variable (canopy_cover): should have interpolated (non-source) values
        cc_vals = result["canopy_cover"].values.astype(float)
        valid_cc = cc_vals[~np.isnan(cc_vals)]
        source_cc = {0.0, 10.0, 20.0, 30.0}
        non_source = [v for v in valid_cc if v not in source_cc]
        assert len(non_source) > 0, (
            "Bilinear interpolation should produce intermediate values "
            "between source values, but all output values are exact source values"
        )

    @patch("griddle.handlers.resample.load_zarr")
    def test_spatial_metadata_preserved(self, mock_load_zarr):
        """Output CRS matches source CRS."""
        data = np.random.rand(10, 10)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, crs="EPSG:32611", resolution=30.0
        )
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        assert result.rio.crs is not None
        assert result.rio.crs.to_epsg() == 32611

    @patch("griddle.handlers.resample.load_zarr")
    def test_coordinates_are_correct(self, mock_load_zarr):
        """Output coordinates are spaced at target resolution."""
        data = np.random.rand(10, 10)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        # y and x coordinate spacing should match target resolution
        y_diff = np.abs(np.diff(result.coords["y"].values))
        x_diff = np.abs(np.diff(result.coords["x"].values))
        np.testing.assert_allclose(y_diff, 10.0, rtol=0.01)
        np.testing.assert_allclose(x_diff, 10.0, rtol=0.01)

    @patch("griddle.handlers.resample.load_zarr")
    def test_source_grid_not_found_raises(self, mock_load_zarr):
        """Missing source grid raises ProcessingError."""
        mock_load_zarr.side_effect = FileNotFoundError("not found")
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            resample_grid(
                source_grid_id="missing-grid",
                alignment=_native_alignment(10.0),
                method_overrides={},
                domain_gdf=_domain_gdf(),
                target_grid_doc=None,
                band_types=_band_types("elevation"),
                progress=progress,
            )

        assert exc_info.value.code == "SOURCE_GRID_NOT_FOUND"

    @patch("griddle.handlers.resample.load_zarr")
    def test_empty_dataset_raises(self, mock_load_zarr):
        """Dataset with no data vars raises ProcessingError."""
        mock_load_zarr.return_value = xr.Dataset()
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            resample_grid(
                source_grid_id="test-grid",
                alignment=_native_alignment(10.0),
                method_overrides={},
                domain_gdf=_domain_gdf(),
                target_grid_doc=None,
                band_types={},
                progress=progress,
            )

        assert exc_info.value.code == "SOURCE_GRID_READ_ERROR"

    @patch("griddle.handlers.resample.load_zarr")
    def test_progress_callbacks(self, mock_load_zarr):
        """Handler calls progress at expected stages."""
        data = np.random.rand(6, 6)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        progress = MagicMock()

        resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        assert progress.call_count >= 3
        messages = [c[0][0] for c in progress.call_args_list]
        assert any("Loading" in m for m in messages)
        assert any("Resampling" in m or "Resample" in m for m in messages)

    @patch("griddle.handlers.resample.load_zarr")
    def test_empty_overrides_uses_default(self, mock_load_zarr):
        """Empty method_overrides uses default method for all variables."""
        variables = {
            "fbfm": np.random.rand(6, 6),
            "canopy_cover": np.random.rand(6, 6),
        }
        mock_load_zarr.return_value = _make_mock_source_ds(variables, resolution=30.0)
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment={"target": "native", "resolution": 10.0, "method": "bilinear"},
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("fbfm", "canopy_cover"),
            progress=progress,
        )

        assert set(result.data_vars) == {"fbfm", "canopy_cover"}
        for var_name in result.data_vars:
            assert result[var_name].dims == ("y", "x")


class TestResampleAlignment:
    """Coverage for the three alignment targets supported by resample."""

    @patch("griddle.handlers.resample.load_zarr")
    def test_domain_target_anchors_at_domain_origin(self, mock_load_zarr):
        """target='domain' lands the output at the domain's lower-left."""
        data = np.random.rand(20, 20)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        domain = _domain_gdf()
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment={"target": "domain", "resolution": 10.0},
            method_overrides={},
            domain_gdf=domain,
            target_grid_doc=None,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        transform = result["elevation"].rio.transform()
        domain_minx = domain.total_bounds[0]
        domain_maxy = domain.total_bounds[3]
        # Anchor: transform.c == domain.minx, transform.f == domain.maxy.
        assert transform.c == pytest.approx(domain_minx)
        assert transform.f == pytest.approx(domain_maxy)
        assert abs(transform.a) == pytest.approx(10.0)

    @patch("griddle.handlers.resample.load_zarr")
    def test_grid_target_exact_match(self, mock_load_zarr):
        """target='grid' with no resolution yields exact transform/shape."""
        data = np.random.rand(20, 20)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        target_grid_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (5.0, 0.0, 720100.0, 0.0, -5.0, 5190800.0),
                "shape": (40, 40),
            }
        }
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment={"target": "grid", "grid_id": "x"},
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=target_grid_doc,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        transform = result["elevation"].rio.transform()
        assert transform.c == pytest.approx(720100.0)
        assert transform.f == pytest.approx(5190800.0)
        assert abs(transform.a) == pytest.approx(5.0)
        assert result["elevation"].shape == (40, 40)

    @patch("griddle.handlers.resample.load_zarr")
    def test_grid_target_with_new_resolution(self, mock_load_zarr):
        """target='grid' with explicit resolution preserves origin, recomputes shape."""
        data = np.random.rand(20, 20)
        mock_load_zarr.return_value = _make_mock_source_ds(
            {"elevation": data}, resolution=30.0
        )
        # Target grid: 30m cells, 10x10, anchored at (720100, 5190200) lower-left.
        target_grid_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (30.0, 0.0, 720100.0, 0.0, -30.0, 5190500.0),
                "shape": (10, 10),
            }
        }
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment={"target": "grid", "grid_id": "x", "resolution": 10.0},
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=target_grid_doc,
            band_types=_band_types("elevation"),
            progress=progress,
        )

        transform = result["elevation"].rio.transform()
        # Anchor is preserved (target's lower-left); cell size is 10m.
        assert transform.c == pytest.approx(720100.0)
        assert transform.f == pytest.approx(5190500.0)
        assert abs(transform.a) == pytest.approx(10.0)
        # Shape: target bounds 300x300 / 10m = 30x30
        assert result["elevation"].shape == (30, 30)


class TestResampleZarrRoundTrip:
    """Verify resample output survives a zarr save/load/to_raster cycle.

    This catches the class of bug where variables are dropped or
    spatial metadata is lost through the storage boundary.
    """

    @patch("griddle.handlers.resample.load_zarr")
    def test_round_trip_preserves_all_variables(self, mock_load_zarr, tmp_path):
        """All source variables survive resample → zarr round-trip."""
        variables = {
            "fuel_load.1hr": np.ones((10, 10), dtype=np.float32),
            "fuel_depth": np.full((10, 10), 0.5, dtype=np.float32),
            "savr.1hr": np.full((10, 10), 7218.0, dtype=np.float32),
        }
        mock_load_zarr.return_value = _make_mock_source_ds(variables, resolution=30.0)
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("fuel_load.1hr", "fuel_depth", "savr.1hr"),
            progress=progress,
        )

        save_zarr(str(tmp_path / "resample.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "resample.zarr"))

        assert set(loaded.data_vars) == {"fuel_load.1hr", "fuel_depth", "savr.1hr"}
        for var_name in loaded.data_vars:
            assert loaded[var_name].dims == ("y", "x")

    @patch("griddle.handlers.resample.load_zarr")
    def test_round_trip_to_raster_succeeds(self, mock_load_zarr, tmp_path):
        """Dataset.rio.to_raster() works after resample → zarr round-trip.

        This is the exact operation the exporter performs.
        """
        variables = {
            "fuel_load.1hr": np.ones((10, 10), dtype=np.float32),
            "fuel_depth": np.full((10, 10), 0.5, dtype=np.float32),
            "savr.1hr": np.full((10, 10), 7218.0, dtype=np.float32),
        }
        mock_load_zarr.return_value = _make_mock_source_ds(variables, resolution=30.0)
        progress = MagicMock()

        result = resample_grid(
            source_grid_id="test-grid",
            alignment=_native_alignment(10.0),
            method_overrides={},
            domain_gdf=_domain_gdf(),
            target_grid_doc=None,
            band_types=_band_types("fuel_load.1hr", "fuel_depth", "savr.1hr"),
            progress=progress,
        )

        save_zarr(str(tmp_path / "resample.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "resample.zarr"))

        # Full Dataset to_raster — the exact operation the exporter performs
        out_path = str(tmp_path / "multiband.tif")
        loaded.rio.to_raster(out_path)
        assert (tmp_path / "multiband.tif").exists()
