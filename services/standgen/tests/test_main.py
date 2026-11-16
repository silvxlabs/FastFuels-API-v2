"""Tests for standgen main orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from lib.errors import CancelledException, ProcessingError


class MockRequest:
    def __init__(self, data, headers=None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


@pytest.fixture
def mock_inventory():
    return {
        "id": "test-inventory-123",
        "domain_id": "test-domain-456",
        "source": {"name": "pim", "source_pim_grid_id": "grid-789", "seed": 42},
        "status": "pending",
    }


@patch("standgen.main.inventory_size")
@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_happy_path(
    mock_load,
    mock_status,
    mock_dispatch,
    mock_load_domain,
    mock_inventory_size,
    mock_inventory,
):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.return_value = {
        "georeference": {"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
        "columns": [
            {
                "key": "x",
                "type": "continuous",
                "unit": "m",
                "summary": {
                    "type": "continuous",
                    "count": 100,
                    "null_count": 0,
                    "min": 0.0,
                    "max": 1000.0,
                    "mean": 500.0,
                    "std": 100.0,
                },
            }
        ],
        "forestry_metrics": {
            "type": "tree",
            "tree_count": 100,
            "basal_area_per_area": 120.5,
            "tree_density": 200.0,
            "quadratic_mean_diameter": 9.3,
            "dominant_species_groups": [],
        },
    }
    mock_inventory_size.return_value = 4096

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    assert response == "OK"
    mock_status.assert_any_call("test-inventory-123", "running")
    # The inventory's Parquet footprint is recorded on completion (#342).
    mock_inventory_size.assert_called_once_with("test-inventory-123")
    mock_status.assert_any_call(
        "test-inventory-123",
        "completed",
        georeference={"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
        columns=[
            {
                "key": "x",
                "type": "continuous",
                "unit": "m",
                "summary": {
                    "type": "continuous",
                    "count": 100,
                    "null_count": 0,
                    "min": 0.0,
                    "max": 1000.0,
                    "mean": 500.0,
                    "std": 100.0,
                },
            }
        ],
        forestry_metrics={
            "type": "tree",
            "tree_count": 100,
            "basal_area_per_area": 120.5,
            "tree_density": 200.0,
            "quadratic_mean_diameter": 9.3,
            "dominant_species_groups": [],
        },
        size_bytes=4096,
        extra=None,
    )


def test_missing_id():
    from standgen.main import process_inventory_request

    request = MockRequest({})
    response, status_code = process_inventory_request(request)
    assert status_code == 400


def test_missing_body():
    from standgen.main import process_inventory_request

    request = MockRequest(None)
    response, status_code = process_inventory_request(request)
    assert status_code == 400


@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_retry_of_running_inventory_marks_it_failed(mock_load, mock_status):
    """A retry against a "running" inventory means the previous attempt died."""
    from standgen.main import process_inventory_request

    mock_load.return_value = {"status": "running", "domain_id": "d"}
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    mock_status.assert_called_once()
    call_args = mock_status.call_args
    assert call_args[0][1] == "failed"
    assert call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"


@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_retry_of_pending_inventory_is_processed(mock_load, mock_status):
    """A retry against a "pending" inventory never ran (Cloud Run rejected the
    first dispatch before any container started, #664) and must be processed."""
    from standgen.main import process_inventory_request

    mock_load.return_value = {"status": "pending", "domain_id": "d"}
    # Cancel at the "running" transition so the test stops before the handler
    # pipeline, which is covered elsewhere.
    mock_status.side_effect = CancelledException("stop here")
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    mock_status.assert_called_once_with("test-123", "running")


@pytest.mark.parametrize("terminal", ["completed", "failed"])
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_retry_of_finished_inventory_is_left_alone(mock_load, mock_status, terminal):
    """A retry against a terminal inventory means the previous attempt finished
    but its response was lost; the recorded outcome must survive."""
    from standgen.main import process_inventory_request

    mock_load.return_value = {"status": terminal, "domain_id": "d"}
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    mock_status.assert_not_called()


@patch("standgen.main.load_inventory")
def test_inventory_not_found(mock_load):
    from standgen.main import process_inventory_request

    from lib.firestore import DocumentNotFoundError

    mock_load.side_effect = DocumentNotFoundError("not found")

    request = MockRequest({"id": "test-123"})
    response, status_code = process_inventory_request(request)
    assert status_code == 200


@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_processing_error(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_inventory
):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = ProcessingError(
        code="MISSING_BAND", message="Missing tm_id band"
    )

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    # Should have called update_status with "failed"
    failed_calls = [c for c in mock_status.call_args_list if c[0][1] == "failed"]
    assert len(failed_calls) == 1


@patch("standgen.main.delete_parquet")
@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_cancelled_during_processing(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_delete, mock_inventory
):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = CancelledException("cancelled")

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    mock_delete.assert_called_once_with("test-inventory-123")


@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_unexpected_error_records_failure(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_inventory
):
    """An unexpected error is recorded as failed immediately, not left to a
    retry that may not exist (#664)."""
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = RuntimeError("unexpected")

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    assert mock_status.call_args[0] == ("test-inventory-123", "failed")
    assert mock_status.call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"


@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_missing_source_returns_200_not_retried(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_inventory
):
    """A deleted input surfaces as FileNotFoundError (zarr GroupNotFoundError /

    GCS 404) — a terminal not-found, not a transient fault. Mark failed with
    SOURCE_NOT_FOUND and return 200 so the job is not retried.
    """
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = FileNotFoundError(
        "No group found in store 'gs://bucket/test-grid' at path ''"
    )

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    failed_calls = [c for c in mock_status.call_args_list if c[0][1] == "failed"]
    assert len(failed_calls) == 1
    assert failed_calls[-1][1]["error"]["code"] == "SOURCE_NOT_FOUND"


@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_cancelled_before_processing(mock_load, mock_status, mock_inventory):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_status.side_effect = CancelledException("cancelled")

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
