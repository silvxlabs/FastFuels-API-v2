"""
exporter main entry point.

Cloud Function that processes export requests via Cloud Tasks HTTP trigger.
All infrastructure concerns (Firestore, GCS, progress) are handled here.
Handlers are pure functions that transform data and return file paths.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

import functions_framework
from flask import Request

from exporter.dispatch import dispatch_handler
from exporter.errors import CancelledException, ProcessingError
from exporter.storage import delete_export_files, generate_signed_download
from lib.config import EXPORTS_COLLECTION
from lib.firestore import DocumentNotFoundError, get_document, update_document
from lib.gcs import storage_size


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging."""

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("export_id", "domain_id", "traceback"):
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

UNEXPECTED_FAILURE_MESSAGE = (
    "Export failed unexpectedly. Please try again or contact the development team."
)

SOURCE_NOT_FOUND_MESSAGE = (
    "A required input resource was not found. It may have been deleted "
    "before processing completed."
)


def load_export(export_id: str) -> dict:
    """Load export document from Firestore."""
    _, snapshot = get_document(EXPORTS_COLLECTION, export_id)
    return snapshot.to_dict()


def update_progress(
    export_id: str,
    message: str,
    percent: int | None = None,
) -> None:
    """Update export progress information."""
    progress = {"message": message}
    if percent is not None:
        progress["percent"] = percent

    try:
        update_document(
            EXPORTS_COLLECTION,
            export_id,
            {
                "progress": progress,
                "modified_on": datetime.now(UTC),
            },
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Export {export_id} was cancelled")


def update_status(
    export_id: str,
    status: str,
    signed_url: str | None = None,
    size_bytes: int | None = None,
    error: dict | None = None,
) -> None:
    """Update export status.

    size_bytes is the GCS artifact footprint of the export in bytes, recorded
    on completion for per-owner storage quota accounting (#342).
    """
    data = {
        "status": status,
        "modified_on": datetime.now(UTC),
    }

    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}

    if signed_url is not None:
        data["signed_url"] = signed_url

    if size_bytes is not None:
        data["size_bytes"] = size_bytes

    if error is not None:
        data["error"] = error

    try:
        update_document(EXPORTS_COLLECTION, export_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Export {export_id} was cancelled")


def make_progress_callback(export_id: str):
    """Create a progress callback bound to an export_id."""

    def callback(message: str, percent: int | None = None):
        update_progress(export_id, message, percent)

    return callback


@functions_framework.http
def process_export_request(request: Request):
    """Main entry point for export processing.

    Triggered by Cloud Tasks HTTP request containing {"id": "..."}.
    """
    data = request.get_json(silent=True)
    export_id = data.get("id") if data else None

    if not export_id:
        logger.error("No id in request body")
        return "Missing id", 400

    # Check if this is a retry
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"export_id": export_id},
        )
        try:
            update_status(
                export_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200

    logger.info("Processing started", extra={"export_id": export_id})

    # Load export document
    try:
        export = load_export(export_id)
    except DocumentNotFoundError:
        logger.info(
            "Export not found (already deleted?)", extra={"export_id": export_id}
        )
        return "OK", 200

    domain_id = export.get("domain_id")
    ids = {"export_id": export_id, "domain_id": domain_id}

    # Update status to running
    try:
        update_status(export_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    try:
        progress_callback = make_progress_callback(export_id)
        gcs_path = dispatch_handler(export, progress_callback)

        expiration_days = export.get("expiration_days", 7)

        progress_callback("Generating signed URL...", 90)

        signed_url = generate_signed_download(gcs_path, expiration_days)
        update_status(
            export_id,
            "completed",
            signed_url=signed_url,
            size_bytes=storage_size(gcs_path),
        )

        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        delete_export_files(export_id)
        return "OK", 200

    except ProcessingError as e:
        # Expected, handled terminal outcome — log at WARNING, not ERROR.
        log_extra = {**ids}
        if e.traceback:
            log_extra["traceback"] = e.traceback
        logger.warning(f"Processing failed: {e.code} - {e.message}", extra=log_extra)
        try:
            update_status(export_id, "failed", error=e.to_dict())
        except CancelledException:
            delete_export_files(export_id)
        return "OK", 200

    except FileNotFoundError as e:
        # A referenced input (source grid zarr, ...) was deleted while this
        # export was queued or running — a benign race (user deleted the
        # resource, or test teardown), not a system fault. zarr's
        # GroupNotFoundError and the GCS 404 both subclass FileNotFoundError.
        # Record a terminal failure and return 200: the object will never
        # reappear, so a retry is wasted and only amplifies log noise.
        logger.warning(f"Input not found (deleted during processing?): {e}", extra=ids)
        try:
            update_status(
                export_id,
                "failed",
                error={"code": "SOURCE_NOT_FOUND", "message": SOURCE_NOT_FOUND_MESSAGE},
            )
        except CancelledException:
            delete_export_files(export_id)
        return "OK", 200

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        return "Internal error", 500


# Local development entry point
if __name__ == "__main__":
    import os

    class MockRequest:
        """Simple mock request for local testing."""

        def __init__(self, data: dict, headers: dict | None = None):
            self._json = data
            self.headers = headers or {}

        def get_json(self, silent: bool = False):
            return self._json

    export_id = os.environ.get("EXPORT_ID")
    if export_id:
        request = MockRequest({"id": export_id})
        response, status_code = process_export_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set EXPORT_ID environment variable for local testing")
