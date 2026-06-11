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
from datetime import UTC, datetime

import functions_framework
from cloudevents.http import CloudEvent

from lib.config import (
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_COLLECTION,
)
from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError, get_document, update_document

_RESOURCE_COLLECTIONS = {
    "inventories": INVENTORIES_COLLECTION,
    "grids": GRIDS_COLLECTION,
    "pointclouds": POINT_CLOUDS_COLLECTION,
}


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

    try:
        update_document(collection, resource_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Resource {resource_id} was cancelled")


@functions_framework.cloud_event
def process_upload(cloud_event: CloudEvent) -> None:
    """Main entry point. Triggered by Eventarc GCS object finalization.

    Extracts bucket and object path from the cloud event, validates the path
    structure, loads the resource document, checks idempotency, and dispatches
    to the appropriate handler.

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

    # Load resource document
    try:
        _, snapshot = get_document(collection, resource_id)
        doc = snapshot.to_dict()
    except DocumentNotFoundError:
        logger.info("Resource not found (already deleted?)", extra=ids)
        return

    # Idempotency: skip if already in a terminal state
    status = doc.get("status")
    if status in ("completed", "failed"):
        logger.info(f"Resource already {status}, skipping", extra=ids)
        return

    # Mark as running
    try:
        update_status(collection, resource_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return

    try:
        dispatch_handler(resource_type, resource_id, bucket, object_name, doc)

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)

    except ProcessingError as e:
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        try:
            update_status(collection, resource_id, "failed", error=e.to_dict())
        except CancelledException:
            pass

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        raise  # Let Eventarc retry
