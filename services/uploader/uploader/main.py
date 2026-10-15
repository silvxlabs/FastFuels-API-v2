"""
Uploader main entry point.

Cloud Function triggered by Eventarc GCS object finalization events.
Extracts the resource type and ID from the object path, then dispatches
to the appropriate handler.

Object path convention: {resource_type}/{resource_id}/{filename}

Unlike HTTP-triggered services (griddle, standgen), this function returns
None on success. Raising an unhandled exception signals Eventarc to retry.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime, timedelta

import functions_framework
from cloudevents.http import CloudEvent
from google.api_core.exceptions import NotFound
from google.cloud import firestore

from lib.config import (
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_COLLECTION,
)
from lib.errors import CancelledException, ProcessingError
from lib.firestore import (
    DocumentNotFoundError,
    firestore_client,
    update_document_or_cancel,
)

_RESOURCE_COLLECTIONS = {
    "inventories": INVENTORIES_COLLECTION,
    "grids": GRIDS_COLLECTION,
    "pointclouds": POINT_CLOUDS_COLLECTION,
}

# Cloud Run hard-kills a request at its 600 s timeout, so a claim older than
# that plus a margin belongs to a dead run and may be taken over.
CLAIM_LEASE = timedelta(seconds=660)


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging."""

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("resource_type", "resource_id"):
            value = getattr(record, field, None)
            if value:
                log_entry[field] = value
        if record.exc_info:
            log_entry["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        print(json.dumps(log_entry), file=sys.stderr)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(StructuredLogHandler())


def update_status(
    collection: str,
    resource_id: str,
    status: str,
    error: dict | None = None,
) -> None:
    """Update resource status in Firestore.

    Raises:
        CancelledException: If the resource document has been deleted
    """
    data = {
        "status": status,
        "modified_on": datetime.now(UTC),
    }
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}
    if error is not None:
        data["error"] = error

    update_document_or_cancel(collection, resource_id, data)


def update_resource(collection: str, resource_id: str, data: dict) -> None:
    """Guarded write-back of result/metadata to a resource's own document.

    Handlers write their completion payload (status, georeference, summary, …)
    through this guard so a mid-run delete surfaces as CancelledException —
    handled as graceful cancellation — instead of escaping as an unhandled
    DocumentNotFoundError that makes Eventarc retry a permanently-missing
    resource for 24h (#441, #593).
    """
    update_document_or_cancel(collection, resource_id, data)


def is_claimable(doc: dict, now: datetime) -> bool:
    """Whether a delivery may claim this resource for processing.

    True for a pending doc, or a running doc whose claim has outlived
    CLAIM_LEASE. A running doc with no claim predates claiming and is
    treated as stale.
    """
    status = doc.get("status")
    if status == "pending":
        return True
    if status == "running":
        claimed_on = doc.get("claimed_on")
        return claimed_on is None or now - claimed_on > CLAIM_LEASE
    return False


@firestore.transactional
def _claim_in_transaction(
    transaction: firestore.Transaction, ref: firestore.DocumentReference
) -> tuple[bool, dict]:
    snapshot = ref.get(transaction=transaction)
    if not snapshot.exists:
        raise DocumentNotFoundError(f"Document not found: {ref.path}")
    doc = snapshot.to_dict()
    now = datetime.now(UTC)
    if not is_claimable(doc, now):
        return False, doc
    transaction.update(
        ref, {"status": "running", "modified_on": now, "claimed_on": now}
    )
    return True, doc


def claim_resource(collection: str, resource_id: str) -> tuple[bool, dict]:
    """Atomically claim a resource for processing and mark it running.

    Eventarc delivers at least once, and a duplicate delivery must not process
    the same upload concurrently: each run clears its output before writing
    (#709). Only the delivery that wins the claim proceeds.

    Returns:
        (claimed, doc): whether this delivery holds the claim, and the
        resource doc as read before claiming.

    Raises:
        DocumentNotFoundError: If the resource document does not exist.
        CancelledException: If the document was deleted during the claim.
    """
    ref = firestore_client.collection(collection).document(resource_id)
    try:
        return _claim_in_transaction(firestore_client.transaction(), ref)
    except NotFound:
        raise CancelledException(f"Resource {resource_id} was cancelled")


@functions_framework.cloud_event
def process_upload(cloud_event: CloudEvent) -> None:
    """Main entry point. Triggered by Eventarc GCS object finalization.

    Extracts bucket and object path from the cloud event, validates the path
    structure, claims the resource document, and dispatches to the appropriate
    handler.

    Returning normally (None) signals Eventarc the event was handled.
    Raising an unhandled exception signals Eventarc to retry.
    """
    from uploader.dispatch import dispatch_handler

    data = cloud_event.data
    bucket = data["bucket"]
    object_name = data["name"]

    # Validate path structure: {resource_type}/{resource_id}/{filename}
    parts = object_name.split("/")
    if len(parts) != 3:
        logger.error(f"Unexpected object path format: {object_name}")
        return

    resource_type, resource_id, _ = parts
    ids = {"resource_type": resource_type, "resource_id": resource_id}

    # Resolve Firestore collection — ignore objects we don't own
    collection = _RESOURCE_COLLECTIONS.get(resource_type)
    if collection is None:
        logger.warning(
            f"Ignoring object with unknown resource type: {object_name}",
            extra=ids,
        )
        return

    logger.info("Upload event received", extra=ids)

    try:
        claimed, doc = claim_resource(collection, resource_id)
    except DocumentNotFoundError:
        logger.info("Resource not found (already deleted?)", extra=ids)
        return
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return

    # Duplicate delivery: already terminal, or another run holds the claim
    if not claimed:
        status = doc.get("status")
        if status == "running":
            logger.info(
                f"Resource claimed by another run at {doc.get('claimed_on')}, skipping",
                extra=ids,
            )
        else:
            logger.info(f"Resource already {status}, skipping", extra=ids)
        return

    try:
        dispatch_handler(resource_type, resource_id, bucket, object_name, doc)

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)

    except ProcessingError as e:
        # Expected, handled terminal outcome — log at WARNING, not ERROR.
        logger.warning(f"Processing failed: {e.code} - {e.message}", extra=ids)
        try:
            update_status(collection, resource_id, "failed", error=e.to_dict())
        except CancelledException:
            pass

    except FileNotFoundError as e:
        # A referenced input was deleted while this job was queued or
        # running — a benign race (user deleted the resource, or test
        # teardown), not a system fault. Record a terminal failure and do
        # NOT re-raise: re-raising makes Eventarc retry a permanently-missing
        # object, wasting the attempt and amplifying log noise.
        logger.warning(f"Input not found (deleted during processing?): {e}", extra=ids)
        try:
            update_status(
                collection,
                resource_id,
                "failed",
                error={
                    "code": "SOURCE_NOT_FOUND",
                    "message": (
                        "A required input resource was not found. It may have "
                        "been deleted before processing completed."
                    ),
                },
            )
        except CancelledException:
            pass

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        raise  # Let Eventarc retry
