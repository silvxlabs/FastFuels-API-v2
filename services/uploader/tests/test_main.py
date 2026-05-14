"""
Unit tests for uploader/main.py

Tests the process_upload entry point in isolation: error handling,
idempotency, path validation, and unknown resource type routing.
All GCP I/O (Firestore, dispatch) is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
from cloudevents.http import CloudEvent
from uploader.main import process_upload

from lib.errors import ProcessingError


def _make_event(bucket: str, name: str) -> CloudEvent:
    return CloudEvent(
        attributes={
            "type": "google.cloud.storage.object.v1.finalized",
            "source": "test",
        },
        data={"bucket": bucket, "name": name},
    )


def _make_doc(status: str = "pending") -> dict:
    return {"id": "inv-123", "domain_id": "dom-123", "status": status, "source": {}}


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_processing_error_writes_failed_status(mock_get, mock_update, mock_dispatch):
    """ProcessingError from handler → status=failed + error dict in Firestore."""
    mock_get.return_value = (None, MagicMock(to_dict=lambda: _make_doc("pending")))
    mock_dispatch.side_effect = ProcessingError(
        code="SCHEMA_VALIDATION_ERROR",
        message="height column missing",
    )

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    update_data = mock_update.call_args_list[-1][0][2]
    assert update_data["status"] == "failed"
    assert update_data["error"]["code"] == "SCHEMA_VALIDATION_ERROR"
    assert update_data["error"]["message"] == "height column missing"


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_already_completed_skips_dispatch(mock_get, mock_update, mock_dispatch):
    """Resource already completed → dispatch never called."""
    mock_get.return_value = (None, MagicMock(to_dict=lambda: _make_doc("completed")))

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    mock_dispatch.assert_not_called()
    mock_update.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_already_failed_skips_dispatch(mock_get, mock_update, mock_dispatch):
    """Resource already failed → dispatch never called."""
    mock_get.return_value = (None, MagicMock(to_dict=lambda: _make_doc("failed")))

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    mock_dispatch.assert_not_called()
    mock_update.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_malformed_path_returns_early(mock_get, mock_update, mock_dispatch):
    """Object path with wrong number of segments → no Firestore access."""
    event = _make_event("uploads-bucket", "bad-path.csv")
    process_upload(event)

    mock_get.assert_not_called()
    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_unknown_resource_type_returns_early(mock_get, mock_update, mock_dispatch):
    """Object path with unrecognised resource type → no Firestore access."""
    event = _make_event("uploads-bucket", "widgets/widget-123/file.csv")
    process_upload(event)

    mock_get.assert_not_called()
    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document")
@patch("uploader.main.get_document")
def test_unexpected_exception_reraises(mock_get, mock_update, mock_dispatch):
    """Unexpected (non-ProcessingError) exception propagates for Eventarc retry."""
    mock_get.return_value = (None, MagicMock(to_dict=lambda: _make_doc("pending")))
    mock_dispatch.side_effect = RuntimeError("disk full")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    with pytest.raises(RuntimeError, match="disk full"):
        process_upload(event)
