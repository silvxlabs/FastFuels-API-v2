"""
Unit tests for lakitu.main, the Cloud Tasks entry point.

Everything below the entry point is mocked at the ``lakitu.main`` namespace, so
these exercise the request handling, status transitions, and the exception
ladder without touching Firestore, GCS, or the network.
"""

from unittest.mock import patch

import pytest
from lakitu.main import process_point_cloud_request

from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError


class MockRequest:
    def __init__(self, data, headers=None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


@pytest.fixture
def point_cloud():
    return {
        "id": "pc-1",
        "domain_id": "domain-1",
        "type": "als",
        "status": "pending",
        "source": {"name": "3dep"},
    }


@pytest.fixture
def handler_result():
    return {
        "size_bytes": 4096,
        "georeference": {
            "crs": "EPSG:32612",
            "bounds": [0, 0, 0, 1, 1, 1],
        },
        "summary": {"point_count": 10, "point_classes": [2], "density": 1.0},
        "source_extra": {"datasets": ["ACQ"], "coverage_fraction": 1.0},
    }


class TestRequestHandling:
    """Tests for the request envelope."""

    def test_missing_id_returns_400(self):
        assert process_point_cloud_request(MockRequest({}))[1] == 400

    def test_missing_body_returns_400(self):
        assert process_point_cloud_request(MockRequest(None))[1] == 400

    @patch("lakitu.main.delete_cloud")
    @patch("lakitu.main.update_status")
    def test_retry_marks_failed_without_reprocessing(self, mock_status, mock_delete):
        """A retry means the previous attempt died mid-flight.

        Re-running could burn another long fetch, so the resource is failed and
        the task acknowledged.
        """
        response, code = process_point_cloud_request(
            MockRequest({"id": "pc-1"}, {"X-CloudTasks-TaskRetryCount": "1"})
        )
        assert code == 200
        assert mock_status.call_args[0][1] == "failed"
        assert mock_status.call_args[1]["error"]["code"] == "UNEXPECTED_FAILURE"
        mock_delete.assert_called_once_with("pc-1")

    @patch("lakitu.main.load_point_cloud", side_effect=DocumentNotFoundError("gone"))
    def test_deleted_point_cloud_is_acknowledged(self, _mock_load):
        assert process_point_cloud_request(MockRequest({"id": "pc-1"}))[1] == 200


class TestSuccess:
    """Tests for the completion path."""

    @patch("lakitu.main.update_progress")
    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_happy_path_records_every_derived_field(
        self,
        mock_load,
        mock_status,
        _mock_domain,
        mock_dispatch,
        _mock_progress,
        point_cloud,
        handler_result,
    ):
        mock_load.return_value = point_cloud
        mock_dispatch.return_value = handler_result

        response, code = process_point_cloud_request(MockRequest({"id": "pc-1"}))
        assert code == 200

        statuses = [call[0][1] for call in mock_status.call_args_list]
        assert statuses == ["running", "completed"]

        completion = mock_status.call_args_list[-1][1]
        assert completion["georeference"] == handler_result["georeference"]
        assert completion["summary"] == handler_result["summary"]
        assert completion["source_extra"] == handler_result["source_extra"]
        # Storage quota aggregates size_bytes; without it the cloud is free.
        assert completion["size_bytes"] == 4096

    @patch("lakitu.main.update_progress")
    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_checksum_is_never_written(
        self,
        mock_load,
        mock_status,
        _mock_domain,
        mock_dispatch,
        _mock_progress,
        point_cloud,
        handler_result,
    ):
        """The creation-time checksum must survive processing.

        Derived resources compare against it to detect staleness, so rewriting
        it here would make every derivative look stale.
        """
        mock_load.return_value = point_cloud
        mock_dispatch.return_value = handler_result

        process_point_cloud_request(MockRequest({"id": "pc-1"}))
        for call in mock_status.call_args_list:
            assert "checksum" not in call[1]


class TestFailurePaths:
    """Tests for the exception ladder."""

    @patch("lakitu.main.delete_cloud")
    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_processing_error_is_terminal(
        self,
        mock_load,
        mock_status,
        _mock_domain,
        mock_dispatch,
        mock_delete,
        point_cloud,
    ):
        """A handled failure records itself and acknowledges the task."""
        mock_load.return_value = point_cloud
        mock_dispatch.side_effect = ProcessingError(
            code="COVERAGE_ERROR", message="no coverage"
        )

        response, code = process_point_cloud_request(MockRequest({"id": "pc-1"}))
        assert code == 200
        assert mock_status.call_args[0][1] == "failed"
        assert mock_status.call_args[1]["error"]["code"] == "COVERAGE_ERROR"
        mock_delete.assert_called_once_with("pc-1")

    @patch("lakitu.main.delete_cloud")
    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_cancellation_cleans_up_and_acknowledges(
        self,
        mock_load,
        mock_status,
        _mock_domain,
        mock_dispatch,
        mock_delete,
        point_cloud,
    ):
        mock_load.return_value = point_cloud
        mock_dispatch.side_effect = CancelledException("deleted")

        response, code = process_point_cloud_request(MockRequest({"id": "pc-1"}))
        assert code == 200
        mock_delete.assert_called_once_with("pc-1")

    @patch("lakitu.main.update_status", side_effect=CancelledException("deleted"))
    @patch("lakitu.main.load_point_cloud")
    def test_cancelled_before_processing(self, mock_load, _mock_status, point_cloud):
        mock_load.return_value = point_cloud
        assert process_point_cloud_request(MockRequest({"id": "pc-1"}))[1] == 200

    @patch("lakitu.main.delete_cloud")
    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_missing_input_is_terminal_not_retried(
        self,
        mock_load,
        mock_status,
        _mock_domain,
        mock_dispatch,
        _mock_delete,
        point_cloud,
    ):
        """A deleted input never reappears, so retrying only adds noise."""
        mock_load.return_value = point_cloud
        mock_dispatch.side_effect = FileNotFoundError("gone")

        response, code = process_point_cloud_request(MockRequest({"id": "pc-1"}))
        assert code == 200
        assert mock_status.call_args[1]["error"]["code"] == "SOURCE_NOT_FOUND"

    @patch("lakitu.main.dispatch_handler")
    @patch("lakitu.main._load_domain")
    @patch("lakitu.main.update_status")
    @patch("lakitu.main.load_point_cloud")
    def test_unexpected_error_returns_500(
        self, mock_load, _mock_status, _mock_domain, mock_dispatch, point_cloud
    ):
        """An unexpected fault is left for the retry to make terminal."""
        mock_load.return_value = point_cloud
        mock_dispatch.side_effect = RuntimeError("boom")

        response, code = process_point_cloud_request(MockRequest({"id": "pc-1"}))
        assert code == 500
