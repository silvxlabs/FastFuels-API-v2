"""
Griddle main entry point.

Cloud Function that processes grid requests via Cloud Tasks HTTP trigger.
All infrastructure concerns (Firestore, GCS, progress) are handled here.
Handlers are pure functions that just transform data.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

import functions_framework
from flask import Request

from griddle.dispatch import dispatch_handler
from griddle.errors import CancelledException, ProcessingError
from griddle.storage import delete_zarr, save_zarr
from lib.config import GRIDS_COLLECTION
from lib.firestore import DocumentNotFoundError, get_document, update_document


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging.

    Includes grid_id in log entries when set via `extra={'grid_id': ...}`.
    """

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("grid_id", "domain_id"):
            value = getattr(record, field, None)
            if value:
                log_entry[field] = value
        if record.exc_info:
            log_entry["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        print(json.dumps(log_entry), file=sys.stderr)


# Configure structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(StructuredLogHandler())


# Error message for unexpected failures (shown to users)
UNEXPECTED_FAILURE_MESSAGE = (
    "Job failed unexpectedly. Please try again or contact the development team."
)


def load_grid(grid_id: str) -> dict:
    """Load grid document from Firestore.

    Args:
        grid_id: Grid document ID

    Returns:
        Grid document as dict

    Raises:
        DocumentNotFoundError: If grid not found
    """
    _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    return snapshot.to_dict()


def update_progress(
    grid_id: str,
    message: str,
    percent: int | None = None,
) -> None:
    """Update grid progress information.

    Args:
        grid_id: Grid document ID
        message: Progress message
        percent: Optional completion percentage (0-100)

    Raises:
        CancelledException: If grid was deleted (user cancelled)
    """
    progress = {"message": message}
    if percent is not None:
        progress["percent"] = percent

    try:
        update_document(
            GRIDS_COLLECTION,
            grid_id,
            {
                "progress": progress,
                "modified_on": datetime.now(UTC),
            },
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def update_status(
    grid_id: str,
    status: str,
    georeference: dict | None = None,
    error: dict | None = None,
) -> None:
    """Update grid status.

    Args:
        grid_id: Grid document ID
        status: New status ("running", "completed", "failed")
        georeference: Optional georeference dict (for completed status)
        error: Optional error dict (for failed status)

    Raises:
        CancelledException: If grid was deleted (user cancelled)
    """
    data = {
        "status": status,
        "modified_on": datetime.now(UTC),
    }

    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}

    if georeference is not None:
        data["georeference"] = georeference

    if error is not None:
        data["error"] = error

    try:
        update_document(GRIDS_COLLECTION, grid_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def make_progress_callback(grid_id: str):
    """Create a progress callback bound to a grid_id.

    Args:
        grid_id: Grid document ID

    Returns:
        Callback function(message, percent)
    """

    def callback(message: str, percent: int | None = None):
        update_progress(grid_id, message, percent)

    return callback


@functions_framework.http
def process_grid_request(request: Request):
    """Main entry point for grid processing.

    Triggered by Cloud Tasks HTTP request containing {"id": "..."}.

    Cloud Tasks retry behavior:
    - max_attempts=2: First attempt processes normally, second attempt marks as failed
    - X-CloudTasks-TaskRetryCount header indicates retry count (0 on first attempt)

    Flow:
    1. Check if this is a retry (previous attempt crashed)
    2. Parse request to get grid_id
    3. Load grid document from Firestore
    4. Update status to "running"
    5. Dispatch to appropriate handler
    6. Save result to Zarr
    7. Update status to "completed" with georeference
    8. On error, update status to "failed" with error details

    Returns:
        Tuple of (response_body, status_code)
    """
    # Parse request body
    data = request.get_json(silent=True)
    grid_id = data.get("id") if data else None

    if not grid_id:
        logger.error("No id in request body")
        return "Missing id", 400

    # Check if this is a retry (previous attempt crashed without updating status)
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"grid_id": grid_id},
        )
        try:
            update_status(
                grid_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
        except (CancelledException, DocumentNotFoundError):
            pass  # Grid was deleted, nothing to update
        return "OK", 200  # Return 200 to prevent further retries

    logger.info("Processing started", extra={"grid_id": grid_id})

    # Load grid document
    try:
        grid = load_grid(grid_id)
    except DocumentNotFoundError:
        logger.info("Grid not found (already deleted?)", extra={"grid_id": grid_id})
        return "OK", 200  # Grid was deleted, nothing to do

    domain_id = grid.get("domain_id")
    ids = {"grid_id": grid_id, "domain_id": domain_id}

    # Update status to running
    try:
        update_status(grid_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    try:
        # Dispatch to handler
        progress_callback = make_progress_callback(grid_id)
        result = dispatch_handler(grid, progress_callback)

        # TODO: Apply modifications if present
        # if grid.get("modifications"):
        #     update_progress(grid_id, "Applying modifications...", 80)
        #     result = apply_modifications(result, grid["modifications"])

        # Save to Zarr
        update_progress(grid_id, "Saving...", 90)
        chunk_shape = tuple(grid.get("chunk_shape", (512, 512)))
        save_zarr(grid_id, result, chunk_shape=chunk_shape)

        # Update status to completed with georeference
        transform = result.rio.transform()
        update_status(
            grid_id,
            "completed",
            georeference={
                "crs": str(result.rio.crs),
                "transform": list(transform)[:6],
                "shape": [result.rio.height, result.rio.width],
            },
        )

        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        # Grid was deleted during processing - clean up
        logger.info("Cancelled during processing", extra=ids)
        delete_zarr(grid_id)
        return "OK", 200

    except ProcessingError as e:
        # Handler raised a structured error
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        try:
            update_status(grid_id, "failed", error=e.to_dict())
        except CancelledException:
            delete_zarr(grid_id)
        return "OK", 200  # Return 200 - error is recorded, no need to retry

    except Exception as e:
        # Unexpected error - let Cloud Tasks retry
        # On retry, the retry_count check above will mark it as failed
        logger.exception(f"Unexpected error: {e}", extra=ids)
        return "Internal error", 500  # Return 500 to trigger retry


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

    grid_id = os.environ.get("GRID_ID")
    if grid_id:
        request = MockRequest({"id": grid_id})
        response, status_code = process_grid_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set GRID_ID environment variable for local testing")
