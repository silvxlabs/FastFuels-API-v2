"""Tests for features main orchestrator."""

from unittest.mock import MagicMock, patch

import pytest
from etcher.errors import CancelledException, ProcessingError


class MockRequest:
    def __init__(self, data, headers=None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


@pytest.fixture
def mock_feature():
    return {
        "id": "test-feature-123",
        "domain_id": "test-domain-456",
        "type": "road",
        "source": {"product": "osm"},
        "status": "pending",
    }


@patch("etcher.main.feature_size")
@patch("etcher.main._load_domain")
@patch("etcher.main.dispatch_handler")
@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_happy_path(
    mock_load,
    mock_status,
    mock_dispatch,
    mock_load_domain,
    mock_feature_size,
    mock_feature,
):
    from etcher.main import process_feature_request

    mock_load.return_value = mock_feature
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.return_value = {
        "georeference": {"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
    }
    mock_feature_size.return_value = 4096

    request = MockRequest({"id": "test-feature-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 200
    assert response == "OK"
    mock_status.assert_any_call("test-feature-123", "running")
    # The feature's GeoParquet footprint is recorded on completion (#342).
    mock_feature_size.assert_called_once_with("test-domain-456", "test-feature-123")
    mock_status.assert_any_call(
        "test-feature-123",
        "completed",
        georeference={"crs": "EPSG:32610", "bounds": [0, 0, 1000, 1000]},
        size_bytes=4096,
    )


def test_missing_id():
    from etcher.main import process_feature_request

    request = MockRequest({})
    response, status_code = process_feature_request(request)
    assert status_code == 400


def test_missing_body():
    from etcher.main import process_feature_request

    request = MockRequest(None)
    response, status_code = process_feature_request(request)
    assert status_code == 400


@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_retry_of_running_feature_marks_it_failed(mock_load, mock_status):
    """A retry against a "running" feature means the previous attempt died."""
    from etcher.main import process_feature_request

    mock_load.return_value = {"status": "running", "domain_id": "d"}
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_feature_request(request)

    assert status_code == 200
    mock_status.assert_called_once()
    call_args = mock_status.call_args
    assert call_args[0][1] == "failed"
    assert call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"


@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_retry_of_pending_feature_is_processed(mock_load, mock_status):
    """A retry against a "pending" feature never ran (Cloud Run rejected the
    first dispatch before any container started, #664) and must be processed."""
    from etcher.main import process_feature_request

    mock_load.return_value = {"status": "pending", "domain_id": "d"}
    # Cancel at the "running" transition so the test stops before the handler
    # pipeline, which is covered elsewhere.
    mock_status.side_effect = CancelledException("stop here")
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_feature_request(request)

    assert status_code == 200
    mock_status.assert_called_once_with("test-123", "running")


@pytest.mark.parametrize("terminal", ["completed", "failed"])
@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_retry_of_finished_feature_is_left_alone(mock_load, mock_status, terminal):
    """A retry against a terminal feature means the previous attempt finished
    but its response was lost; the recorded outcome must survive."""
    from etcher.main import process_feature_request

    mock_load.return_value = {"status": terminal, "domain_id": "d"}
    request = MockRequest(
        {"id": "test-123"},
        headers={"X-CloudTasks-TaskRetryCount": "1"},
    )
    response, status_code = process_feature_request(request)

    assert status_code == 200
    mock_status.assert_not_called()


@patch("etcher.main.load_feature")
def test_feature_not_found(mock_load):
    from etcher.main import process_feature_request

    from lib.firestore import DocumentNotFoundError

    mock_load.side_effect = DocumentNotFoundError("not found")

    request = MockRequest({"id": "test-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 200


@patch("etcher.main._load_domain")
@patch("etcher.main.dispatch_handler")
@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_processing_error(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_feature
):
    from etcher.main import process_feature_request

    mock_load.return_value = mock_feature
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = ProcessingError(
        code="UNKNOWN_PRODUCT", message="Unknown product 'osm' for water features."
    )

    request = MockRequest({"id": "test-feature-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 200
    # Should have called update_status with "failed"
    failed_calls = [c for c in mock_status.call_args_list if c[0][1] == "failed"]
    assert len(failed_calls) == 1


@patch("etcher.main.delete_features")
@patch("etcher.main._load_domain")
@patch("etcher.main.dispatch_handler")
@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_cancelled_during_processing(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_delete, mock_feature
):
    from etcher.main import process_feature_request

    mock_load.return_value = mock_feature
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = CancelledException("cancelled")

    request = MockRequest({"id": "test-feature-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 200
    # Must assert with both domain_id and feature_id
    mock_delete.assert_called_once_with("test-domain-456", "test-feature-123")


@patch("etcher.main._load_domain")
@patch("etcher.main.dispatch_handler")
@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_unexpected_error_returns_500(
    mock_load, mock_status, mock_dispatch, mock_load_domain, mock_feature
):
    from etcher.main import process_feature_request

    mock_load.return_value = mock_feature
    mock_load_domain.return_value = MagicMock()
    mock_dispatch.side_effect = RuntimeError("unexpected")

    request = MockRequest({"id": "test-feature-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 500


@patch("etcher.main.update_status")
@patch("etcher.main.load_feature")
def test_cancelled_before_processing(mock_load, mock_status, mock_feature):
    from etcher.main import process_feature_request

    mock_load.return_value = mock_feature
    mock_status.side_effect = CancelledException("cancelled")

    request = MockRequest({"id": "test-feature-123"})
    response, status_code = process_feature_request(request)

    assert status_code == 200
