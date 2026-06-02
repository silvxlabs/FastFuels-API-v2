"""
Tests for Griddle main module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from griddle.main import _band_nodata, process_grid_request

from lib.config import GRIDS_COLLECTION


def _ds_with_band(name, data, nodata):
    """Build a single-variable Dataset with optional nodata for _band_nodata."""
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": [1.0, 0.0], "x": [0.0, 1.0]})
    da = da.rio.write_crs("EPSG:5070").rio.write_transform(Affine(10, 0, 0, 0, -10, 0))
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return xr.Dataset({name: da})


class TestBandNodata:
    """_band_nodata reads a band's nodata for propagation onto the resource."""

    def test_integer_sentinel_returned_as_int(self):
        ds = _ds_with_band(
            "fbfm", np.array([[101, 102], [103, 32767]], np.int16), 32767
        )
        result = _band_nodata(ds, "fbfm")
        assert result == 32767
        assert isinstance(result, int)

    def test_nan_nodata_returns_none(self):
        ds = _ds_with_band(
            "elev", np.array([[1.0, 2.0], [3.0, np.nan]], np.float32), np.float32("nan")
        )
        assert _band_nodata(ds, "elev") is None

    def test_no_nodata_returns_none(self):
        ds = _ds_with_band("cc", np.array([[1.0, 2.0], [3.0, 4.0]], np.float32), None)
        assert _band_nodata(ds, "cc") is None

    def test_dotted_layerset_key_resolves_to_variable(self):
        ds = _ds_with_band("loading", np.array([[1, 32767], [3, 4]], np.int16), 32767)
        assert _band_nodata(ds, "loading.litter") == 32767

    def test_missing_variable_returns_none(self):
        ds = _ds_with_band("fbfm", np.array([[1, 2], [3, 4]], np.int16), 32767)
        assert _band_nodata(ds, "nonexistent") is None


class MockRequest:
    """Mock Flask request object."""

    def __init__(self, json_data=None, headers=None):
        self._json = json_data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


class TestProcessGridRequest:
    """Tests for process_grid_request HTTP handler."""

    def test_missing_grid_id_returns_400(self):
        """Return 400 when grid_id is missing."""
        request = MockRequest(json_data={})

        response, status_code = process_grid_request(request)

        assert status_code == 400
        assert "id" in response.lower()

    def test_empty_body_returns_400(self):
        """Return 400 when request body is empty."""
        request = MockRequest(json_data=None)

        response, status_code = process_grid_request(request)

        assert status_code == 400

    @patch("griddle.main.update_status")
    def test_retry_marks_job_failed(self, mock_update_status):
        """On retry (retry_count > 0), mark job as failed and return 200."""
        request = MockRequest(
            json_data={"id": "test-grid-id"},
            headers={"X-CloudTasks-TaskRetryCount": "1"},
        )

        response, status_code = process_grid_request(request)

        assert status_code == 200
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        assert call_args[0][0] == "test-grid-id"
        assert call_args[0][1] == "failed"
        assert "UNEXPECTED_FAILURE" in call_args[1]["error"]["code"]

    @patch("griddle.main.load_grid")
    def test_grid_not_found_returns_200(self, mock_load_grid):
        """Return 200 when grid is not found (already deleted)."""
        from lib.firestore import DocumentNotFoundError

        mock_load_grid.side_effect = DocumentNotFoundError(
            GRIDS_COLLECTION, "missing-id"
        )
        request = MockRequest(json_data={"id": "missing-id"})

        response, status_code = process_grid_request(request)

        assert status_code == 200

    @patch("griddle.main.update_document")
    @patch("griddle.main.save_zarr")
    @patch("griddle.main.dispatch_handler")
    @patch("griddle.main._load_domain")
    @patch("griddle.main.update_status")
    @patch("griddle.main.update_progress")
    @patch("griddle.main.load_grid")
    def test_successful_processing(
        self,
        mock_load_grid,
        mock_update_progress,
        mock_update_status,
        mock_load_domain,
        mock_dispatch,
        mock_save_zarr,
        mock_update_document,
    ):
        """Successful processing returns 200 and updates status to complete."""
        # Setup mocks
        mock_load_grid.return_value = {
            "id": "test-grid-id",
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }
        mock_load_domain.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.rio.crs = "EPSG:32610"
        mock_result.rio.transform.return_value = [1, 0, 0, 0, -1, 0]
        mock_result.shape = (100, 100)
        mock_dispatch.return_value = mock_result

        request = MockRequest(json_data={"id": "test-grid-id"})

        response, status_code = process_grid_request(request)

        assert status_code == 200
        # Check that status was updated to running and then complete
        assert mock_update_status.call_count == 2
        first_call = mock_update_status.call_args_list[0]
        second_call = mock_update_status.call_args_list[1]
        assert first_call[0] == ("test-grid-id", "running")
        assert second_call[0][1] == "completed"

    @patch("griddle.main.update_document")
    @patch("griddle.main.save_zarr")
    @patch("griddle.main.dispatch_handler")
    @patch("griddle.main._load_domain")
    @patch("griddle.main.update_status")
    @patch("griddle.main.update_progress")
    @patch("griddle.main.load_grid")
    def test_chunk_shape_passed_to_save_zarr(
        self,
        mock_load_grid,
        mock_update_progress,
        mock_update_status,
        mock_load_domain,
        mock_dispatch,
        mock_save_zarr,
        mock_update_document,
    ):
        """chunks.shape from grid doc is passed to save_zarr."""
        mock_load_grid.return_value = {
            "id": "test-grid-id",
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
            "chunks": {"shape": [256, 256], "count": None, "count_by_axis": None},
        }
        mock_load_domain.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.rio.crs = "EPSG:32610"
        mock_result.rio.transform.return_value = [1, 0, 0, 0, -1, 0]
        mock_result.rio.height = 100
        mock_result.rio.width = 100
        mock_result.shape = (100, 100)
        mock_dispatch.return_value = mock_result

        request = MockRequest(json_data={"id": "test-grid-id"})
        process_grid_request(request)

        mock_save_zarr.assert_called_once_with(
            "test-grid-id", mock_result, chunk_shape=(256, 256)
        )

        completion_call = next(
            c for c in mock_update_status.call_args_list if c.args[1] == "completed"
        )
        assert completion_call.kwargs["chunks"] == {
            "shape": [256, 256],
            "count": 1,
            "count_by_axis": {"y": 1, "x": 1},
        }

    @patch("griddle.main.update_document")
    @patch("griddle.main.save_zarr")
    @patch("griddle.main.dispatch_handler")
    @patch("griddle.main._load_domain")
    @patch("griddle.main.update_status")
    @patch("griddle.main.update_progress")
    @patch("griddle.main.load_grid")
    def test_chunk_shape_defaults_when_missing(
        self,
        mock_load_grid,
        mock_update_progress,
        mock_update_status,
        mock_load_domain,
        mock_dispatch,
        mock_save_zarr,
        mock_update_document,
    ):
        """chunk shape defaults to (512, 512) for grids without chunks set."""
        mock_load_grid.return_value = {
            "id": "test-grid-id",
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }
        mock_load_domain.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.rio.crs = "EPSG:32610"
        mock_result.rio.transform.return_value = [1, 0, 0, 0, -1, 0]
        mock_result.rio.height = 100
        mock_result.rio.width = 100
        mock_result.shape = (100, 100)
        mock_dispatch.return_value = mock_result

        request = MockRequest(json_data={"id": "test-grid-id"})
        process_grid_request(request)

        mock_save_zarr.assert_called_once_with(
            "test-grid-id", mock_result, chunk_shape=(512, 512)
        )

    @patch("griddle.main._load_domain")
    @patch("griddle.main.dispatch_handler")
    @patch("griddle.main.update_status")
    @patch("griddle.main.load_grid")
    def test_unexpected_error_returns_500(
        self, mock_load_grid, mock_update_status, mock_dispatch, mock_load_domain
    ):
        """Unexpected error returns 500 to trigger Cloud Tasks retry."""
        mock_load_grid.return_value = {
            "id": "test-grid-id",
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }
        mock_load_domain.return_value = MagicMock()
        mock_dispatch.side_effect = RuntimeError("Unexpected error")

        request = MockRequest(json_data={"id": "test-grid-id"})

        response, status_code = process_grid_request(request)

        assert status_code == 500

    @patch("griddle.main._load_domain")
    @patch("griddle.main.dispatch_handler")
    @patch("griddle.main.update_status")
    @patch("griddle.main.load_grid")
    def test_processing_error_returns_200(
        self, mock_load_grid, mock_update_status, mock_dispatch, mock_load_domain
    ):
        """ProcessingError returns 200 (error recorded, no retry needed)."""
        from lib.errors import ProcessingError

        mock_load_grid.return_value = {
            "id": "test-grid-id",
            "source": {"name": "landfire", "product": "fbfm40"},
            "domain_id": "test-domain-id",
        }
        mock_load_domain.return_value = MagicMock()
        mock_dispatch.side_effect = ProcessingError(
            code="TEST_ERROR",
            message="Test error message",
        )

        request = MockRequest(json_data={"id": "test-grid-id"})

        response, status_code = process_grid_request(request)

        assert status_code == 200
        # Check that status was updated to failed
        last_call = mock_update_status.call_args_list[-1]
        assert last_call[0][1] == "failed"
