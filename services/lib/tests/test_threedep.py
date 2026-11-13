"""
Unit tests for lib.threedep tile discovery utilities.

Tests pure-math functions (s1m_tile_path, discover_tiles_arc_second) that
don't require network access. S1M discovery (S3 I/O) is tested via mocks, and
lidar project discovery against recorded TNM Access responses.

Run with: uv run --extra threedep pytest tests/test_threedep.py -v
"""

import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
import requests
from shapely.geometry import box

from lib.testing import SHARED_TEST_DATA_DIR
from lib.threedep import (
    S3_BASE,
    TNM_1M_DATASET,
    TnmSearchError,
    discover_lidar_project_tiles,
    discover_s1m_tiles,
    discover_tiles_arc_second,
    s1m_tile_path,
)

TNM_RESPONSES_DIR = SHARED_TEST_DATA_DIR / "threedep"
PROJECTS_BASE = f"{S3_BASE}/1m/Projects"


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


def _tnm_response(body: dict | None = None, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    return response


def _recorded(name: str) -> dict:
    with open(TNM_RESPONSES_DIR / name) as f:
        return json.load(f)


class TestDiscoverLidarProjectTiles:
    """Lidar project tile discovery against recorded TNM Access responses.

    Recorded 2026-11-13 from the CSKT area near Ronan, MT, where S1M has no
    tiles: the bbox ``-114.20,47.55,-114.17,47.57`` (one tile in each of UTM
    zones 11 and 12) and ``-114.40,47.63,-114.39,47.65`` (three flights).
    """

    def _roi(self, minx, miny, maxx, maxy):
        return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326)

    @patch("lib.threedep.requests.get")
    def test_ronan_tiles_from_both_utm_zones(self, mock_get):
        mock_get.return_value = _tnm_response(_recorded("tnm_1m_cskt_ronan.json"))

        urls, dates = discover_lidar_project_tiles(
            self._roi(-114.20, 47.55, -114.17, 47.57)
        )

        project = f"{PROJECTS_BASE}/MT_NRCS_MT_2018_D18/TIFF"
        assert urls == [
            f"{project}/USGS_1M_11_x71y528_MT_NRCS_MT_2018_D18.tif",
            f"{project}/USGS_1M_12_x26y528_MT_NRCS_MT_2018_D18.tif",
        ]
        assert dates == ["20210616"]

    @patch("lib.threedep.requests.get")
    def test_query_parameters(self, mock_get):
        mock_get.return_value = _tnm_response(_recorded("tnm_1m_cskt_ronan.json"))

        discover_lidar_project_tiles(
            self._roi(-114.20, 47.55, -114.17, 47.57), timeout=7.0
        )

        params = mock_get.call_args.kwargs["params"]
        assert params["datasets"] == TNM_1M_DATASET
        assert params["prodFormats"] == "GeoTIFF"
        assert params["offset"] == 0
        bbox = [float(v) for v in params["bbox"].split(",")]
        assert bbox == pytest.approx([-114.20, 47.55, -114.17, 47.57])
        assert mock_get.call_args.kwargs["timeout"] == 7.0

    @patch("lib.threedep.requests.get")
    def test_overlapping_flights_newest_first(self, mock_get):
        """Newest publication date first; ties broken by URL."""
        mock_get.return_value = _tnm_response(
            _recorded("tnm_1m_cskt_three_flights.json")
        )

        urls, dates = discover_lidar_project_tiles(
            self._roi(-114.40, 47.63, -114.39, 47.65)
        )

        names = [url.rsplit("/", 1)[-1] for url in urls]
        assert names == [
            "USGS_1M_11_x69y528_MT_Statewide_Phase3_2021_B21.tif",
            "USGS_1M_11_x69y529_MT_Statewide_Phase3_2021_B21.tif",
            "USGS_1M_11_x69y528_MT_Statewide_Phase2_2020_B20.tif",
            "USGS_1M_11_x69y528_MT_NRCS_MT_2018_D18.tif",
            "USGS_1M_11_x69y529_MT_NRCS_MT_2018_D18.tif",
        ]
        assert dates == ["20210616", "20240407", "20241126"]

    @patch("lib.threedep.requests.get")
    def test_order_independent_of_response_order(self, mock_get):
        body = _recorded("tnm_1m_cskt_three_flights.json")
        mock_get.return_value = _tnm_response(body)
        roi = self._roi(-114.40, 47.63, -114.39, 47.65)
        expected, _ = discover_lidar_project_tiles(roi)

        body["items"].reverse()
        mock_get.return_value = _tnm_response(body)
        assert discover_lidar_project_tiles(roi)[0] == expected

    @patch("lib.threedep.requests.get")
    def test_pages_until_total(self, mock_get):
        body = _recorded("tnm_1m_cskt_three_flights.json")
        items = body["items"]
        mock_get.side_effect = [
            _tnm_response({**body, "items": items[:3]}),
            _tnm_response({**body, "items": items[3:]}),
        ]

        urls, _ = discover_lidar_project_tiles(
            self._roi(-114.40, 47.63, -114.39, 47.65)
        )

        assert len(urls) == 5
        offsets = [call.kwargs["params"]["offset"] for call in mock_get.call_args_list]
        assert offsets == [0, 3]

    @patch("lib.threedep.requests.get")
    def test_no_tiles(self, mock_get):
        mock_get.return_value = _tnm_response(
            {"total": 0, "items": [], "errors": [], "messages": []}
        )
        urls, dates = discover_lidar_project_tiles(
            self._roi(-125.0, 40.0, -124.9, 40.1)
        )
        assert urls == []
        assert dates == []

    @patch("lib.threedep.time.sleep")
    @patch("lib.threedep.requests.get")
    def test_retries_timeouts_and_5xx(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.Timeout("read timed out"),
            _tnm_response(status_code=504),
            _tnm_response(_recorded("tnm_1m_cskt_ronan.json")),
        ]

        urls, _ = discover_lidar_project_tiles(
            self._roi(-114.20, 47.55, -114.17, 47.57), attempts=3
        )

        assert len(urls) == 2
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("lib.threedep.time.sleep")
    @patch("lib.threedep.requests.get")
    def test_raises_after_attempts_exhausted(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.Timeout("read timed out")

        with pytest.raises(TnmSearchError):
            discover_lidar_project_tiles(
                self._roi(-114.20, 47.55, -114.17, 47.57), attempts=2
            )
        assert mock_get.call_count == 2

    @patch("lib.threedep.requests.get")
    def test_single_attempt_by_default(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("refused")

        with pytest.raises(TnmSearchError):
            discover_lidar_project_tiles(self._roi(-114.20, 47.55, -114.17, 47.57))
        assert mock_get.call_count == 1

    @patch("lib.threedep.requests.get")
    def test_4xx_not_retried(self, mock_get):
        mock_get.return_value = _tnm_response(status_code=400)

        with pytest.raises(TnmSearchError):
            discover_lidar_project_tiles(
                self._roi(-114.20, 47.55, -114.17, 47.57), attempts=3
            )
        assert mock_get.call_count == 1

    @patch("lib.threedep.requests.get")
    def test_reported_errors_raise(self, mock_get):
        mock_get.return_value = _tnm_response(
            {"total": 0, "items": [], "errors": ["bad bbox"], "messages": []}
        )

        with pytest.raises(TnmSearchError):
            discover_lidar_project_tiles(self._roi(-114.20, 47.55, -114.17, 47.57))
