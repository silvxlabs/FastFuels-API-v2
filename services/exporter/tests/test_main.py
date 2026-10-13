"""Unit tests for exporter.main — the Cloud Tasks entry point's retry arms.

Firestore helpers are mocked at the ``exporter.main`` namespace; the handler
pipeline itself is covered by tests/handlers.
"""

from unittest.mock import patch

import pytest
from exporter.errors import CancelledException
from exporter.main import process_export_request

from lib.firestore import DocumentNotFoundError


class MockRequest:
    def __init__(self, data, headers=None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


def test_missing_id_returns_400():
    assert process_export_request(MockRequest({}))[1] == 400


@patch("exporter.main.load_export", side_effect=DocumentNotFoundError("gone"))
def test_deleted_export_is_acknowledged(_mock_load):
    assert process_export_request(MockRequest({"id": "e1"}))[1] == 200


@patch("exporter.main.update_status")
@patch("exporter.main.load_export")
def test_retry_of_running_export_marks_it_failed(mock_load, mock_status):
    """A retry against a "running" export means the previous attempt died."""
    mock_load.return_value = {"status": "running", "domain_id": "d"}
    response, code = process_export_request(
        MockRequest({"id": "e1"}, {"X-CloudTasks-TaskRetryCount": "1"})
    )
    assert code == 200
    mock_status.assert_called_once()
    assert mock_status.call_args[0][:2] == ("e1", "failed")
    assert mock_status.call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"


@patch("exporter.main.update_status")
@patch("exporter.main.load_export")
def test_retry_of_pending_export_is_processed(mock_load, mock_status):
    """A retry against a "pending" export never ran (Cloud Run rejected the
    first dispatch before any container started, #664) and must be processed."""
    mock_load.return_value = {"status": "pending", "domain_id": "d"}
    # Cancel at the "running" transition so the test stops before the handler
    # pipeline, which is covered elsewhere.
    mock_status.side_effect = CancelledException("stop here")
    response, code = process_export_request(
        MockRequest({"id": "e1"}, {"X-CloudTasks-TaskRetryCount": "1"})
    )
    assert code == 200
    mock_status.assert_called_once_with("e1", "running")


@pytest.mark.parametrize("terminal", ["completed", "failed"])
@patch("exporter.main.update_status")
@patch("exporter.main.load_export")
def test_retry_of_finished_export_is_left_alone(mock_load, mock_status, terminal):
    """A retry against a terminal export means the previous attempt finished
    but its response was lost; the recorded outcome must survive."""
    mock_load.return_value = {"status": terminal, "domain_id": "d"}
    response, code = process_export_request(
        MockRequest({"id": "e1"}, {"X-CloudTasks-TaskRetryCount": "1"})
    )
    assert code == 200
    mock_status.assert_not_called()


@patch("exporter.main.dispatch_handler", side_effect=RuntimeError("boom"))
@patch("exporter.main.update_status")
@patch("exporter.main.load_export")
def test_unexpected_error_records_failure(mock_load, mock_status, _mock_dispatch):
    """An unexpected error is recorded as failed immediately, not left to a
    retry that may not exist (#664)."""
    mock_load.return_value = {"status": "pending", "domain_id": "d"}
    response, code = process_export_request(MockRequest({"id": "e1"}))
    assert code == 200
    assert mock_status.call_args[0] == ("e1", "failed")
    assert mock_status.call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"
