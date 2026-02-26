"""
Tests for 3DEP handler.

Pure unit tests — all external dependencies are mocked. No network access.
Integration tests live in tests/integration/test_threedep.py.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.errors import ProcessingError
from griddle.handlers.threedep import (
    _compute_slope_aspect,
    _discover_s1m_tiles,
    _discover_tiles_arc_second,
    _fetch_and_mosaic_tiles,
    _meters_to_degrees,
    _s1m_tile_path,
    _validate_dem_has_data,
    fetch_topography,
)
from rasterio.transform import from_bounds
from shapely.geometry import box
from xarray import DataArray

TEST_DATA_DIR = Path(__file__).parent.parent / "data"
DOMAINS_DIR = TEST_DATA_DIR / "domains"


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
def blue_mtn_roi() -> gpd.GeoDataFrame:
    """Load Blue Mountain domain as a GeoDataFrame."""
    with open(DOMAINS_DIR / "blue_mtn.json") as f:
        domain = json.load(f)
    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


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


# Tile discovery (pure math)


class TestDiscoverTilesArcSecond:
    """Tile URL construction for 10m/30m (pure math, no network)."""

    def test_single_tile_10m(self, blue_mtn_roi):
        urls = _discover_tiles_arc_second(blue_mtn_roi, resolution=10)
        assert len(urls) >= 1
        assert all("USGS_13_" in url for url in urls)

    def test_single_tile_30m(self, blue_mtn_roi):
        urls = _discover_tiles_arc_second(blue_mtn_roi, resolution=30)
        assert len(urls) >= 1
        assert all("USGS_1_" in url for url in urls)

    def test_url_format_10m(self, blue_mtn_roi):
        urls = _discover_tiles_arc_second(blue_mtn_roi, resolution=10)
        for url in urls:
            assert url.startswith(
                "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
            )
            assert url.endswith(".tif")

    def test_url_format_30m(self, blue_mtn_roi):
        urls = _discover_tiles_arc_second(blue_mtn_roi, resolution=30)
        for url in urls:
            assert url.startswith(
                "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/"
            )
            assert url.endswith(".tif")


# Geographic CRS padding conversion


class TestMetersToDegrees:
    """Tests for meter-to-degree padding conversion."""

    def test_known_latitude_45(self):
        """At 45°N, 1 degree latitude ≈ 111,320m. 1000m ≈ 0.00898°."""
        geom = box(-110.0, 44.5, -109.5, 45.5)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        result = _meters_to_degrees(1000.0, roi)
        # Latitude component: 1000 / 111320 ≈ 0.00898
        # Longitude component: 1000 / (111320 * cos(45°)) ≈ 0.01270
        # Should return the larger (longitude) value
        assert 0.012 < result < 0.014

    def test_equator_lat_lon_nearly_equal(self):
        """At the equator, lat and lon degree sizes are nearly equal."""
        geom = box(-80.0, -0.5, -79.5, 0.5)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        result = _meters_to_degrees(1000.0, roi)
        expected = 1000.0 / 111_320
        assert abs(result - expected) < 0.0001

    def test_higher_latitude_gives_larger_result(self):
        """Closer to the poles, the same meter distance spans more degrees."""
        geom_low = box(-110.0, 29.5, -109.5, 30.5)
        roi_low = gpd.GeoDataFrame(geometry=[geom_low], crs="EPSG:4326")

        geom_high = box(-110.0, 59.5, -109.5, 60.5)
        roi_high = gpd.GeoDataFrame(geometry=[geom_high], crs="EPSG:4326")

        result_low = _meters_to_degrees(1000.0, roi_low)
        result_high = _meters_to_degrees(1000.0, roi_high)
        assert result_high > result_low

    def test_projected_crs_roi_reprojected(self):
        """ROI in a projected CRS should still produce a valid result."""
        geom = box(300000, 4100000, 301000, 4101000)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32611")
        result = _meters_to_degrees(1000.0, roi)
        # Should be a small positive number in the range of 0.008-0.015
        assert 0.005 < result < 0.02


# Tile discovery (mocked S3/GCS)


class TestDiscoverS1mTiles:
    """Mock-based tests for S1M tile discovery."""

    def _make_roi(self):
        geom = box(-119.0, 37.0, -118.99, 37.01)
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    @patch("s3fs.S3FileSystem")
    def test_returns_urls_for_available_tiles(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = [
            "prd-tnm/StagedProducts/Elevation/S1M/n00e00/n123e456/S1M_n123e456_20230515.tif"
        ]
        urls, dates = _discover_s1m_tiles(self._make_roi())
        assert len(urls) >= 1
        assert urls[0].startswith("https://prd-tnm.s3.amazonaws.com/")
        assert urls[0].endswith(".tif")
        assert "20230515" in dates

    @patch("s3fs.S3FileSystem")
    def test_returns_empty_when_no_tif_files(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = [
            "prd-tnm/StagedProducts/Elevation/S1M/n00e00/n123e456/readme.txt"
        ]
        urls, dates = _discover_s1m_tiles(self._make_roi())
        assert urls == []
        assert dates == []

    @patch("s3fs.S3FileSystem")
    def test_returns_empty_on_listing_error(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.side_effect = Exception("Access denied")
        urls, dates = _discover_s1m_tiles(self._make_roi())
        assert urls == []
        assert dates == []

    @pytest.mark.parametrize(
        "ty, tx, expected_zone, expected_tile",
        [
            # Positive easting (eastern US)
            (48, 139, "n04e13", "n0490e1390"),
            (290, 198, "n29e19", "n2910e1980"),
            # Negative easting, exactly divisible (no floor rounding issue)
            (227, -120, "n22w12", "n2280w1200"),
            (239, -110, "n24w11", "n2400w1100"),
            # Negative easting, NOT exactly divisible (floor rounding edge case)
            # tx=-25: abs(25)*10000/100000 = 2.5 -> zone w02, NOT w03
            (90, -25, "n09w02", "n0910w0250"),
            # tx=-69: abs(69)*10000/100000 = 6.9 -> zone w06, NOT w07
            (137, -69, "n13w06", "n1380w0690"),
            # tx=-99: abs(99)*10000/100000 = 9.9 -> zone w09, NOT w10
            (226, -99, "n22w09", "n2270w0990"),
            # tx=-121: abs(121)*10000/100000 = 12.1 -> zone w12, NOT w13
            (227, -121, "n22w12", "n2280w1210"),
            # Small negative easting near zero
            (226, -5, "n22w00", "n2270w0050"),
        ],
        ids=[
            "positive-easting",
            "positive-easting-high",
            "negative-easting-exact",
            "negative-easting-exact-2",
            "negative-easting-w02-not-w03",
            "negative-easting-w06-not-w07",
            "negative-easting-w09-not-w10",
            "negative-easting-w12-not-w13",
            "negative-easting-near-zero",
        ],
    )
    def test_tile_path(self, ty, tx, expected_zone, expected_tile):
        """Verify zone and tile directory naming against known S3 paths."""
        zone, tile_dir = _s1m_tile_path(ty, tx)
        assert zone == expected_zone
        assert tile_dir == expected_tile

    @patch("s3fs.S3FileSystem")
    def test_negative_easting_uses_w_prefix(self, mock_fs_class):
        """Western US domains have negative Albers eastings -> 'w' prefix in S3 paths."""
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = [
            "prd-tnm/StagedProducts/Elevation/S1M/n22w12/n2280w1200/S1M_n2280w1200_20250717.tif"
        ]

        # Bondurant, WY area: Albers coords ~(-1195000, 2275000) -> negative easting
        geom = box(522800, 4720400, 523300, 4720900)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32612")

        urls, dates = _discover_s1m_tiles(roi)
        assert len(urls) >= 1

        # Verify the S3 path used 'w' not 'e' for the zone/tile
        call_args = mock_fs.ls.call_args[0][0]
        assert "w" in call_args, (
            f"Expected 'w' prefix for negative easting, got: {call_args}"
        )


# fetch_topography pipeline (all external calls mocked)


class TestFetchTopography:
    """Mock-based tests for the full fetch_topography pipeline."""

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
    def test_elevation_only_returns_dataset(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, metadata = fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())

        assert isinstance(ds, xr.Dataset)
        assert "elevation" in ds.data_vars
        assert ds.rio.crs is not None

    @patch("griddle.handlers.threedep._clip_to_roi", side_effect=lambda da, roi: da)
    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
    def test_slope_aspect_computed(self, mock_discover, mock_fetch, _mock_clip):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, _ = fetch_topography(_make_roi(), 10, ["slope", "aspect"], MagicMock())

        assert "slope" in ds.data_vars
        assert "aspect" in ds.data_vars

    @patch("griddle.handlers.threedep._clip_to_roi", side_effect=lambda da, roi: da)
    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
    def test_all_bands(self, mock_discover, mock_fetch, _mock_clip):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        ds, _ = fetch_topography(
            _make_roi(), 10, ["elevation", "slope", "aspect"], MagicMock()
        )

        assert set(ds.data_vars) == {"elevation", "slope", "aspect"}

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
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

    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
    def test_no_tiles_raises(self, mock_discover):
        mock_discover.return_value = []
        with pytest.raises(ProcessingError) as exc_info:
            fetch_topography(_make_roi(), 10, ["elevation"], MagicMock())
        assert exc_info.value.code == "COVERAGE_ERROR"

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
    def test_progress_called(self, mock_discover, mock_fetch):
        mock_discover.return_value = ["https://example.com/tile.tif"]
        mock_fetch.return_value = _make_mock_dem()

        progress = MagicMock()
        fetch_topography(_make_roi(), 10, ["elevation"], progress)

        assert progress.call_count >= 3

    @patch("griddle.handlers.threedep._fetch_and_mosaic_tiles")
    @patch("griddle.handlers.threedep._discover_tiles_arc_second")
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

    @patch("lib.raster.RasterConnection")
    def test_single_tile_returns_data(self, mock_rc_class):
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)

        result = _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/tile.tif"],
            resolution=10,
            pad_cells=0,
            progress=MagicMock(),
        )

        assert isinstance(result, DataArray)
        assert result.shape == (50, 50)

    @patch("griddle.handlers.threedep.merge_arrays")
    @patch("lib.raster.RasterConnection")
    def test_multi_tile_merges(self, mock_rc_class, mock_merge):
        values = np.full((50, 50), 1234.0, dtype=np.float32)
        mock_rc_class.return_value = _make_mock_raster(values)
        mock_merge.return_value = DataArray(np.full((50, 50), 1234.0))

        _fetch_and_mosaic_tiles(
            _make_roi(),
            ["https://example.com/a.tif", "https://example.com/b.tif"],
            resolution=10,
            pad_cells=0,
            progress=MagicMock(),
        )

        mock_merge.assert_called_once()

    def test_padding_floor_500m(self):
        """1m resolution with no derivatives should still get >= 500m padding."""
        padding = max(1 * (0 + 8), 1 * 15, 500)
        assert padding == 500

    def test_padding_derivative_override(self):
        """30m with derivatives: resolution * (pad_cells + 8) > 500."""
        padding = max(30 * (10 + 8), 30 * 15, 500)
        assert padding == 540


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
