"""
Unit tests for uploader/main.py

Tests the process_upload entry point in isolation: error handling,
idempotency, path validation, and unknown resource type routing.
All GCP I/O (Firestore, dispatch) is mocked. The claim transaction itself is
exercised against live Firestore in tests/integration/test_claim.py.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from cloudevents.http import CloudEvent
from uploader.main import CLAIM_LEASE, is_claimable, process_upload, update_resource

from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError
from tests.integration.staging import staged_object_name


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
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_processing_error_writes_failed_status(mock_claim, mock_update, mock_dispatch):
    """ProcessingError from handler → status=failed + error dict in Firestore."""
    mock_claim.return_value = (True, _make_doc("pending"))
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
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_already_completed_skips_dispatch(mock_claim, mock_update, mock_dispatch):
    """Resource already completed → dispatch never called."""
    mock_claim.return_value = (False, _make_doc("completed"))

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    mock_dispatch.assert_not_called()
    mock_update.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_already_failed_skips_dispatch(mock_claim, mock_update, mock_dispatch):
    """Resource already failed → dispatch never called."""
    mock_claim.return_value = (False, _make_doc("failed"))

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    mock_dispatch.assert_not_called()
    mock_update.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_claimed_by_another_run_raises_for_retry(
    mock_claim, mock_update, mock_dispatch
):
    """Claim held by another run → no dispatch, and raise so the event is retried (#709)."""
    doc = _make_doc("running") | {"claimed_on": datetime.now(UTC)}
    mock_claim.return_value = (False, doc)

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    with pytest.raises(RuntimeError, match="claimed by another run"):
        process_upload(event)

    mock_dispatch.assert_not_called()
    mock_update.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.claim_resource")
def test_missing_doc_returns_without_dispatch(mock_claim, mock_dispatch):
    mock_claim.side_effect = DocumentNotFoundError("gone")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    assert process_upload(event) is None

    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.claim_resource")
def test_cancelled_during_claim_returns_without_dispatch(mock_claim, mock_dispatch):
    mock_claim.side_effect = CancelledException("Resource inv-123 was cancelled")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    assert process_upload(event) is None

    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.claim_resource")
def test_claimed_doc_is_passed_to_handler(mock_claim, mock_dispatch):
    doc = _make_doc("pending")
    mock_claim.return_value = (True, doc)

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    process_upload(event)

    mock_dispatch.assert_called_once_with(
        "inventories",
        "inv-123",
        "uploads-bucket",
        "inventories/inv-123/upload.csv",
        doc,
    )


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_malformed_path_returns_early(mock_claim, mock_update, mock_dispatch):
    """Object path with wrong number of segments → no Firestore access."""
    event = _make_event("uploads-bucket", "bad-path.csv")
    process_upload(event)

    mock_claim.assert_not_called()
    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_unknown_resource_type_returns_early(mock_claim, mock_update, mock_dispatch):
    """Object path with unrecognised resource type → no Firestore access."""
    event = _make_event("uploads-bucket", "widgets/widget-123/file.csv")
    process_upload(event)

    mock_claim.assert_not_called()
    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_staged_test_upload_is_ignored(mock_claim, mock_update, mock_dispatch):
    """A name from the integration suite's staging helper is inert here (#349).

    The integration tests stage their uploads in the live, Eventarc-triggered
    UPLOADS_BUCKET and then call handlers directly. Staging under a resource
    type this dispatcher does not own is what stops the deployed service from
    picking those objects up and racing the in-process handler over the same
    object and document, so it has to keep returning before any Firestore
    access.
    """
    name = staged_object_name("test-abc123", "upload.nc")
    process_upload(_make_event("uploads-bucket", name))

    mock_claim.assert_not_called()
    mock_dispatch.assert_not_called()


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_unexpected_exception_reraises(mock_claim, mock_update, mock_dispatch):
    """Unexpected (non-ProcessingError) exception propagates for Eventarc retry."""
    mock_claim.return_value = (True, _make_doc("pending"))
    mock_dispatch.side_effect = RuntimeError("disk full")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    with pytest.raises(RuntimeError, match="disk full"):
        process_upload(event)


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_cancelled_during_processing_is_swallowed(
    mock_claim, mock_update, mock_dispatch
):
    """A mid-run delete (own-doc write-back → CancelledException) ends the job
    gracefully: no re-raise, so Eventarc does not retry a deleted resource
    (#441, #593). The completion write is what raises in production; here the
    guarded handler's CancelledException stands in for it."""
    mock_claim.return_value = (True, _make_doc("pending"))
    mock_dispatch.side_effect = CancelledException("resource deleted")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    result = process_upload(event)  # must not raise

    assert result is None
    assert not any(
        c[0][2].get("status") == "failed" for c in mock_update.call_args_list
    )


@patch("uploader.dispatch.dispatch_handler")
@patch("uploader.main.update_document_or_cancel")
@patch("uploader.main.claim_resource")
def test_missing_input_writes_source_not_found_without_retry(
    mock_claim, mock_update, mock_dispatch
):
    """A missing INPUT resource stays a terminal SOURCE_NOT_FOUND failure and is
    not re-raised — no #420 regression from the own-doc cancellation guard."""
    mock_claim.return_value = (True, _make_doc("pending"))
    mock_dispatch.side_effect = FileNotFoundError("gs://uploads/inventories/x/y 404")

    event = _make_event("uploads-bucket", "inventories/inv-123/upload.csv")
    result = process_upload(event)  # terminal, not re-raised

    assert result is None
    update_data = mock_update.call_args_list[-1][0][2]
    assert update_data["status"] == "failed"
    assert update_data["error"]["code"] == "SOURCE_NOT_FOUND"


class TestUpdateResource:
    """The guarded own-doc write-back helper (#593)."""

    @patch("uploader.main.update_document_or_cancel")
    def test_success_writes_through(self, mock_update):
        update_resource("inventories", "inv-1", {"status": "completed"})
        mock_update.assert_called_once_with(
            "inventories", "inv-1", {"status": "completed"}
        )

    @patch("uploader.main.update_document_or_cancel")
    def test_deleted_doc_raises_cancelled(self, mock_update):
        mock_update.side_effect = CancelledException("Resource inv-1 was cancelled")
        with pytest.raises(CancelledException):
            update_resource("inventories", "inv-1", {"status": "completed"})


class TestIsClaimable:
    """Which docs a delivery may claim (#709)."""

    NOW = datetime(2026, 10, 14, 18, 15, tzinfo=UTC)

    def test_pending(self):
        assert is_claimable({"status": "pending"}, self.NOW)

    def test_running_with_fresh_claim(self):
        doc = {"status": "running", "claimed_on": self.NOW - timedelta(seconds=25)}
        assert not is_claimable(doc, self.NOW)

    def test_running_at_lease_boundary(self):
        doc = {"status": "running", "claimed_on": self.NOW - CLAIM_LEASE}
        assert not is_claimable(doc, self.NOW)

    def test_running_with_stale_claim(self):
        doc = {
            "status": "running",
            "claimed_on": self.NOW - CLAIM_LEASE - timedelta(seconds=1),
        }
        assert is_claimable(doc, self.NOW)

    def test_running_without_claim(self):
        assert is_claimable({"status": "running"}, self.NOW)

    @pytest.mark.parametrize("status", ["completed", "failed"])
    def test_terminal(self, status):
        doc = {"status": status, "claimed_on": self.NOW - 2 * CLAIM_LEASE}
        assert not is_claimable(doc, self.NOW)
