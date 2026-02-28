"""
Unit tests for lib.threedep tile discovery utilities.

Tests pure-math functions (s1m_tile_path, discover_tiles_arc_second) that
don't require network access. S1M discovery (S3 I/O) is tested via mocks.

Run with: uv run --extra threedep pytest tests/test_threedep.py -v
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import box

from lib.threedep import (
    S3_BASE,
    discover_s1m_tiles,
    discover_tiles_arc_second,
    s1m_tile_path,
)


class TestS1mTilePath:
    """Tests for s1m_tile_path index-to-path conversion."""

    def test_positive_easting(self):
        """Positive easting should use 'e' prefix."""
        zone, tile_dir = s1m_tile_path(ty=100, tx=50)
        assert "e" in tile_dir
        assert "w" not in tile_dir

    def test_negative_easting(self):
        """Negative easting should use 'w' prefix."""
        zone, tile_dir = s1m_tile_path(ty=100, tx=-50)
        assert "w" in tile_dir

    def test_zone_grouping(self):
        """Zone should group by 100km blocks."""
        zone, _ = s1m_tile_path(ty=0, tx=0)
        # ty=0 → top_n=1 → zone_n = floor(1*10000/100000) = 0
        # tx=0 → zone_e = floor(0*10000/100000) = 0
        assert zone == "n00e00"

    def test_tile_dir_uses_top_left_corner(self):
        """Tile dir northing should be (ty + 1) * 10."""
        _, tile_dir = s1m_tile_path(ty=5, tx=3)
        # Northing: (5+1)*10 = 60 → n0060
        # Easting: 3*10 = 30 → e0030
        assert tile_dir == "n0060e0030"

    def test_known_values(self):
        """Test a specific known tile path."""
        zone, tile_dir = s1m_tile_path(ty=400, tx=-200)
        # top_n = 401, zone_n = floor(401*10000/100000) = floor(40.1) = 40
        # tx=-200, zone_e = floor(200*10000/100000) = floor(20.0) = 20
        assert zone == "n40w20"
        # n_label: (401)*10 = 4010 → n4010
        # e_label: abs(-200)*10 = 2000 → w2000
        assert tile_dir == "n4010w2000"


class TestDiscoverTilesArcSecond:
    """Tests for 10m/30m tile URL construction (pure math, no I/O)."""

    def test_single_tile_10m(self):
        """Small ROI within one 1x1 degree cell should return one tile."""
        geom = box(-110.5, 44.5, -110.4, 44.6)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        urls = discover_tiles_arc_second(roi, 10)
        assert len(urls) == 1
        assert "USGS_13_n45w111.tif" in urls[0]

    def test_single_tile_30m(self):
        """30m uses product code '1' instead of '13'."""
        geom = box(-110.5, 44.5, -110.4, 44.6)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        urls = discover_tiles_arc_second(roi, 30)
        assert len(urls) == 1
        assert "/1/TIFF/current/" in urls[0]
        assert "USGS_1_n45w111.tif" in urls[0]

    def test_multi_tile_spanning_degree_boundary(self):
        """ROI spanning a degree boundary should return multiple tiles."""
        geom = box(-111.1, 44.9, -110.9, 45.1)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        urls = discover_tiles_arc_second(roi, 10)
        assert len(urls) == 4  # spans 2 lat x 2 lon

    def test_urls_use_s3_base(self):
        """All URLs should start with the S3 base."""
        geom = box(-105.5, 40.5, -105.4, 40.6)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        urls = discover_tiles_arc_second(roi, 10)
        for url in urls:
            assert url.startswith(S3_BASE)

    def test_projected_crs_roi(self):
        """ROI in a projected CRS should be reprojected to EPSG:4326."""
        geom = box(300000, 4100000, 301000, 4101000)
        roi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32611")
        urls = discover_tiles_arc_second(roi, 10)
        assert len(urls) >= 1
        for url in urls:
            assert url.endswith(".tif")


class TestDiscoverS1mTiles:
    """Mock-based tests for S1M tile discovery (avoids real S3 I/O)."""

    def _make_roi(self):
        geom = box(-119.0, 37.0, -118.99, 37.01)
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    @patch("s3fs.S3FileSystem")
    def test_returns_urls_and_dates(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = [
            "prd-tnm/StagedProducts/Elevation/S1M/n00e00/tile/S1M_tile_20230515.tif"
        ]
        urls, dates = discover_s1m_tiles(self._make_roi())
        assert len(urls) >= 1
        assert urls[0].startswith("https://prd-tnm.s3.amazonaws.com/")
        assert "20230515" in dates

    @patch("s3fs.S3FileSystem")
    def test_returns_empty_when_no_tifs(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = ["prd-tnm/readme.txt"]
        urls, dates = discover_s1m_tiles(self._make_roi())
        assert urls == []
        assert dates == []

    @patch("s3fs.S3FileSystem")
    def test_returns_empty_on_s3_error(self, mock_fs_class):
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.side_effect = FileNotFoundError("no such key")
        urls, dates = discover_s1m_tiles(self._make_roi())
        assert urls == []
        assert dates == []

    @patch("s3fs.S3FileSystem")
    def test_deduplicates_acquisition_dates(self, mock_fs_class):
        """Duplicate acquisition dates across tiles should be deduplicated."""
        mock_fs = MagicMock()
        mock_fs_class.return_value = mock_fs
        mock_fs.ls.return_value = [
            "prd-tnm/StagedProducts/Elevation/S1M/n00e00/tile/S1M_tile_20230515.tif"
        ]
        urls, dates = discover_s1m_tiles(self._make_roi())
        assert len(urls) >= 1
        assert dates == ["20230515"]
