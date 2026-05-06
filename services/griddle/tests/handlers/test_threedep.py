"""
Tests for 3DEP handler.

Pure unit tests — all external dependencies are mocked. No network access.
Integration tests live in tests/integration/test_threedep.py.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.errors import ProcessingError
from griddle.handlers.threedep import (
    _compute_slope_aspect,
    _fetch_and_mosaic_tiles,
    _validate_dem_has_data,
    fetch_topography,
)
from rasterio.transform import from_bounds
from shapely.geometry import box
from xarray import DataArray

# Helpers


def _make_mock_raster(
    values: np.ndarray, crs: str = "EPSG:32611", resolution: float = 1.0
):
    """Create a mock RasterConnection that returns a DataArray.

    Follows the pattern from test_chm.py: builds a real xr.DataArray with
    CRS/transform, wraps in a MagicMock with .extract_window.return_value
    and .raster_resolution.
    """
    height, width = values.shape
    transform = from_bounds(
        300000,
        4100000,
        300000 + width * resolution,
        4100000 + height * resolution,
        width,
        height,
    )

    da = xr.DataArray(
        values[np.newaxis, :, :],
        dims=["band", "y", "x"],
        coords={
            "band": [1],
            "y": np.linspace(4100000 + height * resolution, 4100000, height),
            "x": np.linspace(300000, 300000 + width * resolution, width),
        },
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    mock_raster = MagicMock()
    mock_raster.raster_resolution = resolution
    mock_raster.extract_window.return_value = da
    return mock_raster


def _make_mock_dem(shape: tuple = (50, 50), value: float = 1000.0) -> DataArray:
    """Create a small mock DEM DataArray with spatial metadata."""
    h, w = shape
    data = np.full(shape, value, dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": np.linspace(4100000 + h, 4100000, h),
            "x": np.linspace(300000, 300000 + w, w),
        },
    )
    da = da.rio.write_crs("EPSG:32611")
    da = da.rio.write_transform(
        from_bounds(300000, 4100000, 300000 + w, 4100000 + h, w, h)
    )
    return da


def _make_roi() -> gpd.GeoDataFrame:
    """Small synthetic ROI in EPSG:32611."""
    geom = box(300010, 4100010, 300090, 4100090)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32611")


# Fixtures


@pytest.fixture
def flat_dem() -> xr.DataArray:
    """Flat DEM at 1000m elevation."""
    data = np.full((100, 100), 1000.0, dtype=np.float64)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": np.arange(100, 0, -1, dtype=np.float64),
            "x": np.arange(100, dtype=np.float64),
        },
    )
    da = da.rio.write_crs("EPSG:32611")
    da = da.rio.write_transform()
    return da


@pytest.fixture
def tilted_dem() -> xr.DataArray:
    """DEM tilted northward at 45 degrees (1m rise per 1m run)."""
    y = np.arange(100, 0, -1, dtype=np.float64)
    x = np.arange(100, dtype=np.float64)
    yy, _ = np.meshgrid(y, x, indexing="ij")
    da = xr.DataArray(yy * 1.0, dims=["y", "x"], coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:32611")
    da = da.rio.write_transform()
    return da


@pytest.fixture
def east_tilted_dem() -> xr.DataArray:
    """DEM tilted eastward at 45 degrees (1m rise per 1m run)."""
    y = np.arange(100, 0, -1, dtype=np.float64)
    x = np.arange(100, dtype=np.float64)
    _, xx = np.meshgrid(y, x, indexing="ij")
    da = xr.DataArray(xx * 1.0, dims=["y", "x"], coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:32611")
    da = da.rio.write_transform()
    return da


# Slope/Aspect computation


class TestComputeSlopeAspect:
    """Tests for slope/aspect computation on synthetic DEMs."""

    def test_flat_slope_near_zero(self, flat_dem):
        slope, _ = _compute_slope_aspect(flat_dem, cell_size=1.0)
        assert np.allclose(slope.values, 0.0, atol=1e-6)

    def test_tilted_slope_45_degrees(self, tilted_dem):
        slope, _ = _compute_slope_aspect(tilted_dem, cell_size=1.0)
        interior = slope.values[10:-10, 10:-10]
        assert np.allclose(interior, 45.0, atol=0.5)

    def test_north_rising_aspect_180(self, tilted_dem):
        """North-rising slope descends toward south (180 degrees)."""
        _, aspect = _compute_slope_aspect(tilted_dem, cell_size=1.0)
        interior = aspect.values[10:-10, 10:-10]
        assert np.allclose(interior, 180.0, atol=1.0)

    def test_east_rising_aspect_270(self, east_tilted_dem):
        """East-rising slope descends toward west (270 degrees)."""
        _, aspect = _compute_slope_aspect(east_tilted_dem, cell_size=1.0)
        interior = aspect.values[10:-10, 10:-10]
        assert np.allclose(interior, 270.0, atol=1.0)

    def test_slope_non_negative(self, flat_dem):
        slope, _ = _compute_slope_aspect(flat_dem, cell_size=1.0)
        assert np.all(slope.values >= 0)

    def test_slope_at_most_90(self, tilted_dem):
        slope, _ = _compute_slope_aspect(tilted_dem, cell_size=1.0)
        assert np.all(slope.values <= 90)

    def test_aspect_in_range(self, tilted_dem):
        _, aspect = _compute_slope_aspect(tilted_dem, cell_size=1.0)
        assert np.all(aspect.values >= 0)
        assert np.all(aspect.values < 360)

    def test_preserves_shape_and_type(self, flat_dem):
        slope, aspect = _compute_slope_aspect(flat_dem, cell_size=1.0)
        assert slope.shape == flat_dem.shape
        assert aspect.shape == flat_dem.shape
        assert isinstance(slope, DataArray)
        assert isinstance(aspect, DataArray)


# fetch_topography pipeline (all external calls mocked)


class TestFetchTopography:
    """Mock-based tests for the full fetch_topography pipeline."""

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_elevation_only_returns_dataset(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, metadata = fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())

        assert isinstance(ds, xr.Dataset)
        assert "elevation" in ds.data_vars
        assert ds.rio.crs is not None

    @patch(
        "griddle.handlers.threedep._clip_to_roi",
        side_effect=lambda da, roi, extent_buffer_cells=0: da,
    )
    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_slope_aspect_computed(self, mock_discover, mock_fetch, _mock_clip):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, _ = fetch_topography(_make_roi(), 10, ["slope", "aspect"], MagicMock())

        assert "slope" in ds.data_vars
        assert "aspect" in ds.data_vars

    @patch(
        "griddle.handlers.threedep._clip_to_roi",
        side_effect=lambda da, roi, extent_buffer_cells=0: da,
    )
    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_all_bands(self, mock_discover, mock_fetch, _mock_clip):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, _ = fetch_topography(
            _make_roi(), 10, ["elevation", "slope", "aspect"], MagicMock()
        )

        assert set(ds.data_vars) == {"elevation", "slope", "aspect"}

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_elevation_only_passes_buffer_unchanged(self, mock_discover, mock_fetch):
        """Without derivatives, the user's extent_buffer_cells reaches the fetch verbatim."""
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        fetch_topography(
            _make_roi(), 10, ["elevation"], MagicMock(), extent_buffer_cells=4
        )

        assert mock_fetch.call_args[0][4] == 4

    @patch(
        "griddle.handlers.threedep._clip_to_roi",
        side_effect=lambda da, roi, extent_buffer_cells=0: da,
    )
    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_derivatives_fetch_extra_cells_internally(
        self, mock_discover, mock_fetch, mock_clip
    ):
        """With derivatives, the fetch buffer is extent_buffer_cells + gradient overhead.

        The user's extent_buffer_cells is honored for the *output* extent — the
        extra DEM cells are clipped away by _clip_to_roi.
        """
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        from griddle.handlers.threedep import _DERIVATIVE_GRADIENT_OVERHEAD_CELLS

        fetch_topography(
            _make_roi(), 10, ["slope", "aspect"], MagicMock(), extent_buffer_cells=4
        )

        # _fetch_and_mosaic_tiles is called with extent_buffer_cells + overhead
        assert mock_fetch.call_args[0][4] == 4 + _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        # Each clip call uses extent_buffer_cells=4 (the user-requested output buffer)
        for call in mock_clip.call_args_list:
            assert call.args[2] == 4 or call.kwargs.get("extent_buffer_cells") == 4

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_returns_tile_metadata(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        _, metadata = fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())

        assert metadata["tile_count"] == 1
        assert "native_crs" in metadata

    def test_invalid_resolution_raises(self):
        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 5, ["elevation"], MagicMock())
        assert exc_info.value.code == "INVALID_RESOLUTION"

    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_no_tiles_raises(self, mock_discover):
        mock_discover.return_value = []
        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())
        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_progress_called(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        progress = MagicMock()
        fetch_topography(_make_roi(), 10, ["elevation"], progress)

        assert progress.call_count >= 3

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_all_nodata_raises_coverage_error(self, mock_discover, mock_fetch):
        """Tiles found but all nodata should raise COVERAGE_ERROR."""
        mock_discover.return_value = ["https://example.com/tile.tif"]
        nodata_dem = _make_mock_dem(value=-999999.0)
        nodata_dem = nodata_dem.rio.write_nodata(-999999.0)
        mock_fetch.return_value = nodata_dem

        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 1, ["elevation"], MagicMock())
        assert exc_info.value.code == "COVERAGE_ERROR"


# _fetch_and_mosaic_tiles (mocked RasterConnection)


class TestFetchAndMosaicTiles:
    """Mock-based tests for tile fetching and mosaicking."""

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_single_tile_returns_data(self, mock_rc_class):
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        result = _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            extent_buffer_cells=0,
            progress=MagicMock(),
        )

        assert isinstance(result, DataArray)
        assert result.shape == (50, 50)
        call_kwargs = mock_rc_class.return_value.extract_window.call_args[1]
        assert "projection_padding_meters" not in call_kwargs
        assert call_kwargs["interpolation_padding_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 1, 12])
    @patch("griddle.handlers.threedep.RasterConnection")
    def test_extent_buffer_cells_threaded_through(self, mock_rc_class, buffer):
        """Caller-supplied extent_buffer_cells reaches extract_window unchanged."""
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            extent_buffer_cells=buffer,
            progress=MagicMock(),
        )

        call_kwargs = mock_rc_class.return_value.extract_window.call_args[1]
        assert call_kwargs["interpolation_padding_cells"] == buffer

    @patch("griddle.handlers.threedep.merge_arrays")
    @patch("griddle.handlers.threedep.RasterConnection")
    def test_multi_tile_merges(self, mock_rc_class, mock_merge):
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)
        mock_merge.return_value = DataArray(np.full((50, 50), 1234.0))

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/a.tif", "https://example.com/b.tif"],
            resolution=10,
            extent_buffer_cells=0,
            progress=MagicMock(),
        )

        mock_merge.assert_called_once()


# DEM data validation


class TestValidateDemHasData:
    """Tests for nodata detection on fetched DEMs."""

    def test_valid_dem_passes(self):
        dem = _make_mock_dem(value=1000.0)
        _validate_dem_has_data(dem, resolution=10)  # should not raise

    def test_all_nodata_raises(self):
        dem = _make_mock_dem(value=-999999.0)
        dem = dem.rio.write_nodata(-999999.0)
        with pytest.raises(ProcessingError) as exc_info:
            _validate_dem_has_data(dem, resolution=1)
        assert exc_info.value.code == "COVERAGE_ERROR"

    def test_all_nan_raises(self):
        dem = _make_mock_dem(value=np.nan)
        with pytest.raises(ProcessingError) as exc_info:
            _validate_dem_has_data(dem, resolution=10)
        assert exc_info.value.code == "COVERAGE_ERROR"

    def test_partial_data_passes(self):
        """DEM with some valid pixels should pass."""
        data = np.full((50, 50), -999999.0, dtype=np.float32)
        data[25, 25] = 1000.0  # one valid pixel
        da = xr.DataArray(data, dims=["y", "x"])
        da = da.rio.write_crs("EPSG:32611")
        da = da.rio.write_nodata(-999999.0)
        _validate_dem_has_data(da, resolution=10)  # should not raise
