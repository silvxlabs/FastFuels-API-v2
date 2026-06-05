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


@patch("standgen.main._load_domain")
@patch("standgen.main.dispatch_handler")
@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_happy_path(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_inventory
):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.return_value = {
        "georeference": {"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
    }

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    assert response == "OK"
    mock_status.assert_any_call("test-inventory-123", "running")
    mock_status.assert_any_call(
        "test-inventory-123",
        "completed",
        georeference={"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
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
def test_retry_marks_failed(mock_load, mock_status):
    from standgen.main import process_inventory_request

    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_inventory_request(request)

    assert status_code == 200
    mock_status.assert_called_once()
    call_args = mock_status.call_args
    assert call_args[0][1] == "failed"


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
def test_unexpected_error_returns_500(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_inventory
):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = RuntimeError("unexpected")

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 500


@patch("standgen.main.update_status")
@patch("standgen.main.load_inventory")
def test_cancelled_before_processing(mock_load, mock_status, mock_inventory):
    from standgen.main import process_inventory_request

    mock_load.return_value = mock_inventory
    mock_status.side_effect = CancelledException("cancelled")

    request = MockRequest({"id": "test-inventory-123"})
    response, status_code = process_inventory_request(request)

    assert status_code == 200
