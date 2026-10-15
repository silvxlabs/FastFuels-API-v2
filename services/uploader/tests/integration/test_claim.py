"""
Integration tests for the uploader's claim on a resource doc (#709).

Runs process_upload against live Firestore with dispatch patched out, so the
claim transaction is real but no handler runs and nothing touches GCS. The
event's object name is never written to UPLOADS_BUCKET.
"""

import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
from cloudevents.http import CloudEvent
from uploader.main import CLAIM_LEASE, process_upload

from lib.config import INVENTORIES_COLLECTION
from lib.firestore import (
    delete_document,
    get_document,
    set_document,
    update_document,
)


def _make_event(resource_id: str) -> CloudEvent:
    return CloudEvent(
        attributes={
            "type": "google.cloud.storage.object.v1.finalized",
            "source": "test",
        },
        data={"bucket": "uploads", "name": f"inventories/{resource_id}/upload.csv"},
    )


@pytest.fixture
def make_inventory():
    """Create a test inventory doc with the given fields; delete on teardown."""
    ids = []

    def _make(**fields) -> str:
        inventory_id = f"test-{uuid4().hex}"
        ids.append(inventory_id)
        now = datetime.now(UTC)
        doc = {
            "id": inventory_id,
            "owner_id": "test-owner",
            "type": "tree",
            "created_on": now,
            "modified_on": now,
            "source": {"name": "upload", "format": "csv"},
        } | fields
        set_document(INVENTORIES_COLLECTION, inventory_id, doc)
        return inventory_id

    yield _make

    for inventory_id in ids:
        delete_document(INVENTORIES_COLLECTION, inventory_id)


def _read(inventory_id: str) -> dict:
    _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
    return snapshot.to_dict()


def test_concurrent_deliveries_dispatch_once(make_inventory):
    """Five simultaneous deliveries of one event → exactly one handler run."""
    inventory_id = make_inventory(status="pending")
    event = _make_event(inventory_id)
    deliveries = 5
    barrier = threading.Barrier(deliveries)
    errors = []

    def deliver():
        barrier.wait()
        try:
            process_upload(event)
        except Exception as e:
            errors.append(e)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        threads = [threading.Thread(target=deliver) for _ in range(deliveries)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # The losers raise so Eventarc retries them; none of them dispatch.
    assert len(errors) == deliveries - 1
    assert all("claimed by another run" in str(e) for e in errors)
    assert mock_dispatch.call_count == 1
    doc = _read(inventory_id)
    assert doc["status"] == "running"
    assert doc["claimed_on"] is not None


def test_pending_is_claimed(make_inventory):
    inventory_id = make_inventory(status="pending")
    before = datetime.now(UTC)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        process_upload(_make_event(inventory_id))

    mock_dispatch.assert_called_once()
    doc = _read(inventory_id)
    assert doc["status"] == "running"
    assert doc["claimed_on"] >= before
    assert doc["modified_on"] == doc["claimed_on"]


def test_stale_claim_is_taken_over(make_inventory):
    """A claim older than the lease belongs to a dead run → retry proceeds."""
    stale = datetime.now(UTC) - CLAIM_LEASE - timedelta(seconds=60)
    inventory_id = make_inventory(status="running", claimed_on=stale)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        process_upload(_make_event(inventory_id))

    mock_dispatch.assert_called_once()
    assert _read(inventory_id)["claimed_on"] > stale


def test_fresh_claim_raises_for_retry(make_inventory):
    """A claim within the lease → duplicate doesn't dispatch, and raises to be retried."""
    fresh = datetime.now(UTC) - timedelta(seconds=30)
    inventory_id = make_inventory(status="running", claimed_on=fresh)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        with pytest.raises(RuntimeError, match="claimed by another run"):
            process_upload(_make_event(inventory_id))

    mock_dispatch.assert_not_called()
    doc = _read(inventory_id)
    assert doc["status"] == "running"
    assert doc["claimed_on"] == fresh


def test_retry_after_failed_run_is_not_dropped(make_inventory):
    """A run that fails leaves its claim; retries keep raising until the lease expires, then take over."""
    inventory_id = make_inventory(status="pending")
    event = _make_event(inventory_id)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        mock_dispatch.side_effect = RuntimeError("transient failure")
        with pytest.raises(RuntimeError, match="transient failure"):
            process_upload(event)

        # Retry inside the lease: raises again rather than acking and dropping the event.
        mock_dispatch.side_effect = None
        with pytest.raises(RuntimeError, match="claimed by another run"):
            process_upload(event)
        assert mock_dispatch.call_count == 1

        # Retry after the lease: the dead run's claim is taken over.
        stale = datetime.now(UTC) - CLAIM_LEASE - timedelta(seconds=60)
        update_document(INVENTORIES_COLLECTION, inventory_id, {"claimed_on": stale})
        process_upload(event)
        assert mock_dispatch.call_count == 2

    assert _read(inventory_id)["claimed_on"] > stale


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_terminal_is_skipped(make_inventory, status):
    stale = datetime.now(UTC) - 2 * CLAIM_LEASE
    inventory_id = make_inventory(status=status, claimed_on=stale)

    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        process_upload(_make_event(inventory_id))

    mock_dispatch.assert_not_called()
    doc = _read(inventory_id)
    assert doc["status"] == status
    assert doc["claimed_on"] == stale


def test_missing_doc_is_skipped():
    with patch("uploader.dispatch.dispatch_handler") as mock_dispatch:
        assert process_upload(_make_event(f"test-{uuid4().hex}")) is None

    mock_dispatch.assert_not_called()
