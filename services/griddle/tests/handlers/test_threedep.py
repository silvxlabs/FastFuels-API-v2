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
from griddle.handlers.threedep import (
    _DERIVATIVE_GRADIENT_OVERHEAD_CELLS,
    _compute_slope_aspect,
    _fetch_and_mosaic_tiles,
    _validate_dem_has_data,
    fetch_topography,
)
from rasterio.transform import from_bounds
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box
from xarray import DataArray

from lib.errors import ProcessingError
from lib.threedep import TnmSearchError

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
    mock_raster.raster_x_resolution = resolution
    mock_raster.target_native_resolution.return_value = (resolution, resolution)
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

    @pytest.mark.parametrize("sentinel", [-999999.0, np.nan])
    def test_nodata_dem_cells_masked(self, tilted_dem, sentinel):
        """Cells whose gradient stencil touches nodata are NaN; the rest are exact."""
        dem = tilted_dem.copy(data=tilted_dem.values.copy())
        dem.values[40:42, 50:52] = sentinel
        dem = dem.rio.write_nodata(sentinel)

        slope, aspect = _compute_slope_aspect(dem, cell_size=1.0)

        bad = np.zeros(dem.shape, dtype=bool)
        bad[39:43, 50:52] = True
        bad[40:42, 49:53] = True
        assert np.isnan(slope.values[bad]).all()
        assert np.isnan(aspect.values[bad]).all()
        assert np.allclose(slope.values[~bad], 45.0, atol=1e-5)
        assert np.allclose(aspect.values[~bad], 180.0, atol=1e-5)
        assert np.isnan(slope.rio.nodata)
        assert np.isnan(aspect.rio.nodata)

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
        mock_fetch.return_value = (_make_mock_dem(), ["https://example.com/tile.tif"])

        ds, metadata = fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())

        assert isinstance(ds, xr.Dataset)
        assert "elevation" in ds.data_vars
        assert ds.rio.crs is not None

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_slope_aspect_computed(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        # Mosaic carries the gradient overhead; output strips it back down.
        mock_fetch.return_value = (
            _make_mock_dem(shape=(50 + 2 * _DERIVATIVE_GRADIENT_OVERHEAD_CELLS,) * 2),
            ["https://example.com/tile.tif"],
        )

        ds, _ = fetch_topography(_make_roi(), 10, ["slope", "aspect"], MagicMock())

        assert "slope" in ds.data_vars
        assert "aspect" in ds.data_vars

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_all_bands(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = (
            _make_mock_dem(shape=(50 + 2 * _DERIVATIVE_GRADIENT_OVERHEAD_CELLS,) * 2),
            ["https://example.com/tile.tif"],
        )

        ds, _ = fetch_topography(
            _make_roi(), 10, ["elevation", "slope", "aspect"], MagicMock()
        )

        assert set(ds.data_vars) == {"elevation", "slope", "aspect"}

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_elevation_only_passes_buffer_unchanged(self, mock_discover, mock_fetch):
        """Without derivatives, the user's extent_buffer_cells reaches the fetch verbatim."""
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = (_make_mock_dem(), ["https://example.com/tile.tif"])

        fetch_topography(
            _make_roi(), 10, ["elevation"], MagicMock(), extent_buffer_cells=4
        )

        assert mock_fetch.call_args[0][4] == 4

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_derivatives_fetch_extra_cells_internally(self, mock_discover, mock_fetch):
        """With derivatives, the fetch buffer is extent_buffer_cells + gradient overhead.

        The user's extent_buffer_cells is honored for the output extent — the
        extra cells are stripped by _trim_derivative_overhead after the gradient.
        """
        mock_discover.return_value = ["https://example.com/tile.tif"]
        # Output should be 50x50 after stripping overhead from each side.
        out_size = 50
        in_size = out_size + 2 * _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        mock_fetch.return_value = (
            _make_mock_dem(shape=(in_size, in_size)),
            ["https://example.com/tile.tif"],
        )

        ds, _ = fetch_topography(
            _make_roi(), 10, ["slope", "aspect"], MagicMock(), extent_buffer_cells=4
        )

        # _fetch_and_mosaic_tiles is called with extent_buffer_cells + overhead
        assert mock_fetch.call_args[0][4] == 4 + _DERIVATIVE_GRADIENT_OVERHEAD_CELLS
        # Output bands are stripped back down to the user-requested extent size.
        assert ds["slope"].shape == (out_size, out_size)
        assert ds["aspect"].shape == (out_size, out_size)

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_tiles_arc_second")
    def test_returns_tile_metadata(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = (_make_mock_dem(), ["https://example.com/tile.tif"])

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
        mock_fetch.return_value = (_make_mock_dem(), ["https://example.com/tile.tif"])

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
        mock_fetch.return_value = (nodata_dem, ["https://example.com/tile.tif"])

        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 1, ["elevation"], MagicMock())
        assert exc_info.value.code == "COVERAGE_ERROR"


# Tile discovery per source resolution

PROJECT_URLS = [
    "https://example.com/1m/Projects/A/TIFF/USGS_1M_11_a.tif",
    "https://example.com/1m/Projects/A/TIFF/USGS_1M_12_b.tif",
]


class TestTileDiscovery:
    """Which discovery path each source resolution takes."""

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_s1m_tiles")
    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_2m_reads_lidar_projects(self, mock_projects, mock_s1m, mock_fetch):
        mock_projects.return_value = (PROJECT_URLS, ["20210616"])
        mock_fetch.return_value = (_make_mock_dem(), PROJECT_URLS)

        _, metadata = fetch_topography(_make_roi(), 2, ["elevation"], MagicMock())

        mock_s1m.assert_not_called()
        assert mock_fetch.call_args[0][1] == PROJECT_URLS
        assert mock_fetch.call_args[0][2] == 2
        assert metadata["tile_source"] == "projects"
        assert metadata["tiles"] == PROJECT_URLS
        assert metadata["tile_count"] == 2
        assert metadata["native_crs"] is None
        assert metadata["acquisition_dates"] == ["20210616"]

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_s1m_tiles")
    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_1m_prefers_s1m(self, mock_projects, mock_s1m, mock_fetch):
        s1m_urls = ["https://example.com/S1M/s1m.tif"]
        mock_s1m.return_value = (s1m_urls, ["20230515"])
        mock_fetch.return_value = (_make_mock_dem(), s1m_urls)

        _, metadata = fetch_topography(_make_roi(), 1, ["elevation"], MagicMock())

        mock_projects.assert_not_called()
        assert metadata["tile_source"] == "s1m"
        assert metadata["native_crs"] == "EPSG:6350"

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep.discover_s1m_tiles")
    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_1m_falls_back_to_projects(self, mock_projects, mock_s1m, mock_fetch):
        mock_s1m.return_value = ([], [])
        mock_projects.return_value = (PROJECT_URLS, ["20210616"])
        mock_fetch.return_value = (_make_mock_dem(), PROJECT_URLS)

        _, metadata = fetch_topography(_make_roi(), 1, ["elevation"], MagicMock())

        assert mock_fetch.call_args[0][1] == PROJECT_URLS
        # Full resolution: the source resolution passed through is still 1.
        assert mock_fetch.call_args[0][2] == 1
        assert metadata["tile_source"] == "projects"

    @patch("griddle.handlers.threedep.discover_s1m_tiles")
    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_1m_no_tiles_anywhere_raises_coverage_error(self, mock_projects, mock_s1m):
        mock_s1m.return_value = ([], [])
        mock_projects.return_value = ([], [])

        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 1, ["elevation"], MagicMock())
        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_project_search_retried_with_timeout(self, mock_projects):
        mock_projects.return_value = ([], [])

        with pytest.raises(ProcessingError):
            fetch_topography(_make_roi(), 2, ["elevation"], MagicMock())

        kwargs = mock_projects.call_args.kwargs
        assert kwargs["attempts"] > 1
        assert kwargs["timeout"] > 0

    @patch("griddle.handlers.threedep.discover_lidar_project_tiles")
    def test_project_search_failure_is_tile_source_unavailable(self, mock_projects):
        mock_projects.side_effect = TnmSearchError("HTTP 504")

        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 2, ["elevation"], MagicMock())
        assert exc_info.value.code == "TILE_SOURCE_UNAVAILABLE"


# _fetch_and_mosaic_tiles (mocked RasterConnection)


class TestFetchAndMosaicTiles:
    """Mock-based tests for tile fetching and mosaicking."""

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_single_tile_returns_data(self, mock_rc_class):
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        result, used_urls = _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "domain"},
            target_grid_doc=None,
        )

        assert isinstance(result, DataArray)
        assert result.shape == (50, 50)
        assert used_urls == ["https://example.com/tile.tif"]
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
            alignment={"target": "domain"},
            target_grid_doc=None,
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
            alignment={"target": "domain"},
            target_grid_doc=None,
        )

        mock_merge.assert_called_once()

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_alignment_threaded_to_extract_window(self, mock_rc_class):
        """Each tile reprojects directly onto the alignment lattice — single
        reproject, no second alignment pass downstream.

        Domain target with explicit resolution: extract_window receives
        destination_transform/destination_shape derived from the ROI bounds,
        not the source's native cell size.
        """
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "domain", "resolution": 4.0},
            target_grid_doc=None,
        )

        call_kwargs = mock_rc_class.return_value.extract_window.call_args[1]
        # ROI is 80m x 80m at 4m -> 20 x 20 destination cells.
        assert call_kwargs["destination_shape"] == (20, 20)
        assert abs(call_kwargs["destination_transform"].a) == pytest.approx(4.0)
        assert call_kwargs["destination_crs"] == _make_roi().crs

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_native_alignment_passes_no_destination_lattice(self, mock_rc_class):
        """target='native' without resolution: no destination override, the
        default extract_window path runs (single reproject to ROI CRS)."""
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "native"},
            target_grid_doc=None,
        )

        call_kwargs = mock_rc_class.return_value.extract_window.call_args[1]
        assert "destination_transform" not in call_kwargs
        assert "destination_shape" not in call_kwargs
        assert call_kwargs.get("destination_crs") is None

    @pytest.mark.parametrize(
        "resolution, open_options",
        [(1, {}), (2, {"OVERVIEW_LEVEL": 0}), (10, {}), (30, {})],
    )
    @patch("griddle.handlers.threedep.RasterConnection")
    def test_2m_opens_first_overview(self, mock_rc_class, resolution, open_options):
        """2 opens each tile's first (2 m) overview; others open full resolution."""
        mock_rc_class.return_value = _make_mock_raster(np.full((50, 50), 1.0))

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=resolution,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "domain"},
            target_grid_doc=None,
        )

        _, kwargs = mock_rc_class.call_args
        assert {k: v for k, v in kwargs.items() if k.isupper()} == open_options

    @patch("griddle.handlers.threedep.merge_arrays")
    @patch("griddle.handlers.threedep.RasterConnection")
    def test_one_destination_lattice_for_all_tiles(self, mock_rc_class, mock_merge):
        """Tiles in different UTM zones report slightly different native cell
        sizes in the domain CRS; every tile still lands on the first's lattice."""
        values = np.full((50, 50), 1.0, dtype=np.float32)
        zone_11 = _make_mock_raster(values, resolution=1.9991)
        zone_12 = _make_mock_raster(values, resolution=1.9992)
        mock_rc_class.side_effect = [zone_11, zone_12]
        mock_merge.return_value = DataArray(values)

        _fetch_and_mosaic_tiles(
            _make_roi(),
            PROJECT_URLS,
            resolution=2,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "domain"},
            target_grid_doc=None,
        )

        first = zone_11.extract_window.call_args.kwargs
        second = zone_12.extract_window.call_args.kwargs
        assert first["destination_transform"] == second["destination_transform"]
        assert first["destination_shape"] == second["destination_shape"]
        assert abs(first["destination_transform"].a) == pytest.approx(1.9991)

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_tile_outside_domain_skipped(self, mock_rc_class):
        """A tile whose EPSG:4326 bbox matched but which misses the domain."""
        outside = _make_mock_raster(np.full((50, 50), 1.0))
        outside.extract_window.side_effect = NoDataInBounds("No data found in bounds.")
        inside = _make_mock_raster(np.full((50, 50), 1234.0))
        mock_rc_class.side_effect = [outside, inside]

        result, used_urls = _fetch_and_mosaic_tiles(
            _make_roi(),
            PROJECT_URLS,
            resolution=2,
            extent_buffer_cells=0,
            progress=MagicMock(),
            alignment={"target": "domain"},
            target_grid_doc=None,
        )

        assert used_urls == PROJECT_URLS[1:]
        assert float(result.max()) == 1234.0

    @patch("griddle.handlers.threedep.RasterConnection")
    def test_no_tile_reaches_domain_raises_coverage_error(self, mock_rc_class):
        outside = _make_mock_raster(np.full((50, 50), 1.0))
        outside.extract_window.side_effect = NoDataInBounds("No data found in bounds.")
        mock_rc_class.return_value = outside

        with pytest.raises(ProcessingError) as exc_info:
            _fetch_and_mosaic_tiles(
                _make_roi(),
                PROJECT_URLS,
                resolution=2,
                extent_buffer_cells=0,
                progress=MagicMock(),
                alignment={"target": "domain"},
                target_grid_doc=None,
            )
        assert exc_info.value.code == "COVERAGE_ERROR"

    def test_merge_keeps_first_tile_where_both_valid(self):
        """Overlap order is the discovery order: the first valid value wins
        and later tiles only fill the first's nodata."""
        nodata = -999999.0
        first = _make_mock_dem(value=10.0).rio.write_nodata(nodata)
        first.values[:, :25] = nodata
        second = _make_mock_dem(value=20.0).rio.write_nodata(nodata)

        with patch("griddle.handlers.threedep.RasterConnection") as mock_rc_class:
            mock_rc_class.side_effect = [
                _raster_returning(first),
                _raster_returning(second),
            ]
            result, _ = _fetch_and_mosaic_tiles(
                _make_roi(),
                PROJECT_URLS,
                resolution=2,
                extent_buffer_cells=0,
                progress=MagicMock(),
                alignment={"target": "native"},
                target_grid_doc=None,
            )

        assert np.all(result.values[:, :25] == 20.0)
        assert np.all(result.values[:, 25:] == 10.0)


def _raster_returning(da: DataArray) -> MagicMock:
    raster = MagicMock()
    raster.target_native_resolution.return_value = (1.0, 1.0)
    raster.extract_window.return_value = da.expand_dims(band=[1])
    return raster


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
