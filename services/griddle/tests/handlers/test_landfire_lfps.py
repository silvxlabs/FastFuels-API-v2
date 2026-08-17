"""
Unit tests for the seasonal-FBFM40 LFPS orchestration in
griddle.handlers.landfire_lfps.

All against mocked lib.landfire calls, mocked Firestore writes, and a
mocked task re-enqueue -- no live LFPS or Cloud Tasks calls. Covers the
submit/poll/timeout/failure state machine; the fetch/unzip/align step
itself is covered separately -- see tests/handlers/test_landfire.py for
the shared raster pipeline, and tests/integration/test_landfire_lfps.py
for a real LFPS submit/poll/download/fetch pass.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from griddle.handlers.landfire_lfps import (
    _LFPS_JOB_TIMEOUT_SECONDS,
    process_lfps_fbfm40,
)

from lib.errors import ProcessingDeferred, ProcessingError
from lib.landfire import LfpsJob, LfpsJobFailedError
from lib.testing import SHARED_TEST_DOMAINS_DIR


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    """Real domain geometry -- only needed so `_lfps_aoi_bbox` has valid
    bounds/CRS to work with; its content doesn't matter since `submit_job`
    is mocked."""
    with open(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json") as f:
        domain = json.load(f)
    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


def _grid(lfps_job=None, grid_id="test-grid-id"):
    return {"id": grid_id, "lfps_job": lfps_job}


def _source(season="SP", **overrides):
    source = {"product": "fbfm40", "version": "2025", "season": season}
    source.update(overrides)
    return source


class TestProcessLfpsFbfm40NoJob:
    """No lfps_job on the grid doc yet -- submits a new job."""

    @patch("griddle.handlers.landfire_lfps.tasks.enqueue_delayed_task")
    @patch("griddle.handlers.landfire_lfps.update_document")
    @patch("griddle.handlers.landfire_lfps.submit_job")
    def test_submits_writes_state_and_defers(
        self, mock_submit, mock_update, mock_enqueue, roi
    ):
        mock_submit.return_value = LfpsJob(job_id="job-1", status="Pending")
        progress = MagicMock()

        with pytest.raises(ProcessingDeferred):
            process_lfps_fbfm40(_grid(), roi, _source(), None, progress)

        layers, _aoi = mock_submit.call_args[0]
        assert layers == ["LF2025_FBFM40_SP26"]

        mock_update.assert_called_once()
        _collection, grid_id, data = mock_update.call_args[0]
        assert grid_id == "test-grid-id"
        assert data["lfps_job"]["job_id"] == "job-1"
        assert "submitted_at" in data["lfps_job"]

        mock_enqueue.assert_called_once_with("test-grid-id", 10)


class TestProcessLfpsFbfm40InProgress:
    """lfps_job already present, LFPS says it's still running."""

    @patch("griddle.handlers.landfire_lfps.tasks.enqueue_delayed_task")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    def test_under_timeout_reenqueues(self, mock_poll, mock_enqueue, roi):
        mock_poll.return_value = LfpsJob(job_id="job-1", status="Executing")
        grid = _grid({"job_id": "job-1", "submitted_at": datetime.now(UTC)})
        progress = MagicMock()

        with pytest.raises(ProcessingDeferred):
            process_lfps_fbfm40(grid, roi, _source(), None, progress)

        mock_poll.assert_called_once_with("job-1")
        mock_enqueue.assert_called_once_with("test-grid-id", 10)

    @patch("griddle.handlers.landfire_lfps.tasks.enqueue_delayed_task")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    def test_over_timeout_fails_without_reenqueue(self, mock_poll, mock_enqueue, roi):
        mock_poll.return_value = LfpsJob(job_id="job-1", status="Executing")
        stale = datetime.now(UTC) - timedelta(seconds=_LFPS_JOB_TIMEOUT_SECONDS + 1)
        grid = _grid({"job_id": "job-1", "submitted_at": stale})
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            process_lfps_fbfm40(grid, roi, _source(), None, progress)

        assert exc_info.value.code == "LFPS_TIMEOUT"
        mock_enqueue.assert_not_called()


class TestProcessLfpsFbfm40Succeeded:
    """LFPS reports the job done -- downloads and hands off to the fetch."""

    @patch("griddle.handlers.landfire_lfps._fetch_downloaded_fbfm40")
    @patch("griddle.handlers.landfire_lfps.download")
    @patch("griddle.handlers.landfire_lfps.poll_status")
    def test_downloads_and_continues(self, mock_poll, mock_download, mock_fetch, roi):
        job = LfpsJob(
            job_id="job-1", status="Succeeded", output_file="https://.../job-1.zip"
        )
        mock_poll.return_value = job
        mock_download.return_value = b"PK\x03\x04zip-bytes"
        mock_result = MagicMock()
        mock_fetch.return_value = mock_result
        grid = _grid({"job_id": "job-1", "submitted_at": datetime.now(UTC)})
        progress = MagicMock()

        result = process_lfps_fbfm40(grid, roi, _source(), None, progress)

        mock_download.assert_called_once_with(job)
        mock_fetch.assert_called_once()
        assert result is mock_result


class TestProcessLfpsFbfm40Failed:
    """LFPS reports the job failed -- surfaces its own message."""

    @patch("griddle.handlers.landfire_lfps.poll_status")
    def test_failed_status_raises_with_lfps_message(self, mock_poll, roi):
        mock_poll.side_effect = LfpsJobFailedError("ERROR: bad AOI")
        grid = _grid({"job_id": "job-1", "submitted_at": datetime.now(UTC)})
        progress = MagicMock()

        with pytest.raises(ProcessingError) as exc_info:
            process_lfps_fbfm40(grid, roi, _source(), None, progress)

        assert exc_info.value.code == "LFPS_JOB_FAILED"
        assert "bad AOI" in exc_info.value.message
