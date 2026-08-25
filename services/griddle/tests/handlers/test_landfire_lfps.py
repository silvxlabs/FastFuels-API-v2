"""
Unit tests for LFPS job orchestration in griddle.handlers.landfire_lfps.

All against mocked lib.landfire calls -- no live LFPS calls, and
time.sleep is mocked so tests don't actually wait.
"""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from affine import Affine
from griddle.handlers.landfire_lfps import (
    _LFPS_AOI_PADDING_METERS,
    _LFPS_NATIVE_RESOLUTION,
    _lfps_aoi_bbox,
    _lfps_layer_name,
    fetch_lfps,
)
from rasterio.warp import transform_bounds

from lib.errors import ProcessingError
from lib.landfire import LfpsJob, LfpsJobFailedError, LfpsProduct
from lib.raster import REPROJECTION_GUARD_CELLS
from lib.testing import SHARED_TEST_DOMAINS_DIR


def _seasonal_product(layer_name: str = "LF2025_FBFM40_SP26") -> LfpsProduct:
    """A live seasonal catalog entry, for patching resolve_seasonal_product."""
    return LfpsProduct(
        layer_name=layer_name,
        product_name="Seasonal FBFM40",
        theme="Seasonal Fuels",
        acronym="FBFM40",
        version="LF2025",
        conus=True,
        geo_areas="All",
        season="SP",
        season_year=2026,
    )


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    with open(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json") as f:
        domain = json.load(f)
    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


def _make_zip(*names: str) -> bytes:
    """Build an in-memory zip containing the given (empty) file names."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"")
    return buf.getvalue()


class TestLfpsAoiBbox:
    """Unit tests for _lfps_aoi_bbox."""

    def test_native_buffer_widens_the_aoi(self, roi):
        no_buffer = _lfps_aoi_bbox(roi, {"target": "native"}, None, 0)
        buffered = _lfps_aoi_bbox(roi, {"target": "native"}, None, 10)

        w0, s0, e0, n0 = (float(x) for x in no_buffer.split())
        w1, s1, e1, n1 = (float(x) for x in buffered.split())
        assert w1 < w0
        assert s1 < s0
        assert e1 > e0
        assert n1 > n0

    def test_native_explicit_resolution_used_for_buffer(self, roi):
        result = _lfps_aoi_bbox(roi, {"target": "native", "resolution": 100.0}, None, 5)

        minx, miny, maxx, maxy = roi.total_bounds
        buf = 5 * 100.0
        pad = (
            REPROJECTION_GUARD_CELLS * _LFPS_NATIVE_RESOLUTION
            + _LFPS_AOI_PADDING_METERS
        )
        expected = transform_bounds(
            roi.crs,
            "EPSG:4326",
            minx - buf - pad,
            miny - buf - pad,
            maxx + buf + pad,
            maxy + buf + pad,
        )
        actual = tuple(float(x) for x in result.split())
        assert actual == pytest.approx(expected)

    def test_native_zero_buffer_matches_unbuffered_bounds(self, roi):
        result = _lfps_aoi_bbox(roi, {"target": "native"}, None, 0)

        minx, miny, maxx, maxy = roi.total_bounds
        pad = (
            REPROJECTION_GUARD_CELLS * _LFPS_NATIVE_RESOLUTION
            + _LFPS_AOI_PADDING_METERS
        )
        expected = transform_bounds(
            roi.crs,
            "EPSG:4326",
            minx - pad,
            miny - pad,
            maxx + pad,
            maxy + pad,
        )
        actual = tuple(float(x) for x in result.split())
        assert actual == pytest.approx(expected)

    @patch("griddle.handlers.landfire_lfps.resolve_alignment_destination")
    def test_domain_grid_uses_lattice_bounds_not_roi(self, mock_resolve, roi):
        mock_resolve.return_value = {
            "destination_transform": Affine(30.0, 0.0, 100000.0, 0.0, -30.0, 200300.0),
            "destination_shape": (10, 10),
            "destination_crs": "EPSG:32611",
        }

        result = _lfps_aoi_bbox(roi, {"target": "grid"}, {"id": "some-grid"}, 0)

        w, _, e, _ = (float(x) for x in result.split())
        roi_w, _, roi_e, _ = roi.to_crs("EPSG:4326").total_bounds
        assert e < roi_w  # entirely west of the RO

    @patch("griddle.handlers.landfire_lfps.resolve_alignment_destination")
    def test_domain_grid_buffer_not_applied_twice(self, mock_resolve, roi):
        mock_resolve.return_value = {
            "destination_transform": Affine(30.0, 0.0, 100000.0, 0.0, -30.0, 200300.0),
            "destination_shape": (10, 10),
            "destination_crs": "EPSG:32611",
        }

        no_buffer = _lfps_aoi_bbox(roi, {"target": "grid"}, {"id": "some-grid"}, 0)
        with_buffer = _lfps_aoi_bbox(roi, {"target": "grid"}, {"id": "some-grid"}, 10)

        assert no_buffer == with_buffer


class TestLfpsLayerName:
    """Unit tests for _lfps_layer_name."""

    def test_annual_layer_name(self):
        assert _lfps_layer_name("fbfm40", "2025") == "LF2025_FBFM40"

    @patch("griddle.handlers.landfire_lfps.resolve_seasonal_product")
    def test_seasonal_layer_name(self, mock_resolve):
        """The seasonal layer name comes from the live catalog entry, not
        from assuming the season year is version + 1."""
        mock_resolve.return_value = _seasonal_product("LF2025_FBFM40_SP26")
        assert _lfps_layer_name("fbfm40", "2025", "SP") == "LF2025_FBFM40_SP26"
        mock_resolve.assert_called_once_with("FBFM40", "2025", "SP")

    @patch("griddle.handlers.landfire_lfps.resolve_seasonal_product")
    def test_seasonal_not_live_raises(self, mock_resolve):
        mock_resolve.return_value = None
        with pytest.raises(ProcessingError) as exc_info:
            _lfps_layer_name("fbfm40", "2025", "FA")
        assert exc_info.value.code == "SEASONAL_NOT_AVAILABLE"

    def test_seasonal_requires_fbfm40(self):
        """LANDFIRE's Seasonal Fuels product only publishes FBFM40 -- any
        other product with a season set should fail clearly rather than
        silently building a nonexistent layer name."""
        with pytest.raises(ProcessingError) as exc_info:
            _lfps_layer_name("fccs", "2025", "SP")
        assert exc_info.value.code == "SEASONAL_NOT_SUPPORTED"

    def test_unknown_season_raises(self):
        with pytest.raises(ProcessingError) as exc_info:
            _lfps_layer_name("fbfm40", "2025", "WINTER")
        assert exc_info.value.code == "INVALID_SEASON"


class TestFetchLfps:
    """Unit tests for the fetch_lfps context manager."""

    @patch("griddle.handlers.landfire_lfps.resolve_seasonal_product")
    @patch("griddle.handlers.landfire_lfps.time.sleep")
    @patch("griddle.handlers.landfire_lfps.download")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_succeeds_and_yields_the_tif_path(
        self, mock_submit, mock_poll, mock_download, mock_sleep, mock_resolve, roi
    ):
        mock_resolve.return_value = _seasonal_product("LF2025_FBFM40_SP26")
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")
        mock_poll.return_value = LfpsJob(
            job_id="job-1", status="Succeeded", output_file="https://.../job-1.zip"
        )
        mock_download.return_value = _make_zip("result.tfw", "result.tif")
        progress = MagicMock()

        with fetch_lfps(
            roi, "fbfm40", "2025", {"target": "domain"}, None, 0, progress, "SP"
        ) as path:
            assert Path(path).name == "result.tif"
            assert Path(path).exists()

        assert not Path(path).exists()  # cleaned up once the `with` block exits

        layers = mock_submit.call_args[0][0]
        assert layers == ["LF2025_FBFM40_SP26"]
        mock_poll.assert_called_once_with("job-1")
        mock_download.assert_called_once()

    @patch("griddle.handlers.landfire_lfps.time.sleep")
    @patch("griddle.handlers.landfire_lfps.download")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_keeps_polling_until_succeeded(
        self, mock_submit, mock_poll, mock_download, mock_sleep, roi
    ):
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")
        mock_poll.side_effect = [
            LfpsJob(job_id="job-1", status="Executing"),
            LfpsJob(job_id="job-1", status="Executing"),
            LfpsJob(
                job_id="job-1", status="Succeeded", output_file="https://x/job-1.zip"
            ),
        ]
        mock_download.return_value = _make_zip("result.tif")

        with fetch_lfps(
            roi, "fbfm40", "2025", {"target": "domain"}, None, 0, MagicMock()
        ) as path:
            assert Path(path).name == "result.tif"

        assert mock_poll.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("griddle.handlers.landfire_lfps.time.sleep")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_failed_status_raises_with_lfps_message(
        self, mock_submit, mock_poll, mock_sleep, roi
    ):
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")
        mock_poll.side_effect = LfpsJobFailedError("ERROR: bad AOI")

        with pytest.raises(ProcessingError) as exc_info:
            with fetch_lfps(
                roi, "fbfm40", "2025", {"target": "domain"}, None, 0, MagicMock()
            ):
                pass

        assert exc_info.value.code == "LFPS_JOB_FAILED"
        assert "bad AOI" in exc_info.value.message

    @patch("griddle.handlers.landfire_lfps._LFPS_JOB_TIMEOUT_SECONDS", -1)
    @patch("griddle.handlers.landfire_lfps.time.sleep")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_timeout_raises_without_polling(
        self, mock_submit, mock_poll, mock_sleep, roi
    ):
        """A negative timeout means the deadline has already passed by the
        time the loop's first check runs -- deterministic, no flaky timing."""
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")

        with pytest.raises(ProcessingError) as exc_info:
            with fetch_lfps(
                roi, "fbfm40", "2025", {"target": "domain"}, None, 0, MagicMock()
            ):
                pass

        assert exc_info.value.code == "LFPS_TIMEOUT"
        mock_poll.assert_not_called()

    @patch("griddle.handlers.landfire_lfps.time.sleep")
    @patch("griddle.handlers.landfire_lfps.download")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_no_tif_in_zip_raises(
        self, mock_submit, mock_poll, mock_download, mock_sleep, roi
    ):
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")
        mock_poll.return_value = LfpsJob(
            job_id="job-1", status="Succeeded", output_file="https://x/job-1.zip"
        )
        mock_download.return_value = _make_zip("readme.txt")

        with pytest.raises(ProcessingError) as exc_info:
            with fetch_lfps(
                roi, "fbfm40", "2025", {"target": "domain"}, None, 0, MagicMock()
            ):
                pass

        assert exc_info.value.code == "LFPS_OUTPUT_INVALID"
