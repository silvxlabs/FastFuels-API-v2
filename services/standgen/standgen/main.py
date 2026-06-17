"""
Standgen main entry point.

Cloud Function that processes inventory requests via Cloud Tasks HTTP trigger.
All infrastructure concerns (Firestore, GCS, progress) are handled here.
Handlers return results; main.py writes them to storage and updates Firestore.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

import functions_framework
from flask import Request

from lib.config import DOMAINS_COLLECTION, INVENTORIES_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError, get_document, update_document
from standgen.dispatch import dispatch_handler
from standgen.handlers.modifications import apply_in_place_modifications
from standgen.handlers.treatments import apply_in_place_treatments
from standgen.storage import delete_parquet


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging.

    Includes inventory_id and domain_id in log entries when set via extra.
    """

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("inventory_id", "domain_id"):
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

UNEXPECTED_FAILURE_MESSAGE = (
    "Job failed unexpectedly. Please try again or contact the development team."
)


def load_inventory(inventory_id: str) -> dict:
    """Load inventory document from Firestore."""
    _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
    return snapshot.to_dict()


def update_progress(inventory_id, message, percent=None):
    """Update inventory progress. Raises CancelledException if deleted."""
    progress = {"message": message}
    if percent is not None:
        progress["percent"] = percent
    try:
        update_document(
            INVENTORIES_COLLECTION,
            inventory_id,
            {"progress": progress, "modified_on": datetime.now(UTC)},
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Inventory {inventory_id} was cancelled")


def update_status(
    inventory_id, status, georeference=None, columns=None, error=None, extra=None
):
    """Update inventory status.

    Args:
        inventory_id: Inventory document ID.
        status: New status value ('pending', 'running', 'completed', 'failed').
        georeference: Spatial reference to write on completion.
        columns: Column list with populated summaries to write on completion.
        error: Error details to write on failure.
        extra: Additional fields merged into the update — used to clear
            ``pending_modifications`` or ``pending_treatments`` once an
            in-place handler completes.
    """
    data = {"status": status, "modified_on": datetime.now(UTC)}
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}
    if georeference is not None:
        data["georeference"] = georeference
    if columns is not None:
        data["columns"] = columns
    if error is not None:
        data["error"] = error
    if extra:
        data.update(extra)
    try:
        update_document(INVENTORIES_COLLECTION, inventory_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Inventory {inventory_id} was cancelled")


def make_progress_callback(inventory_id):
    def callback(message, percent=None):
        update_progress(inventory_id, message, percent)

    return callback


def _load_domain(domain_id: str):
    """Load domain from Firestore and parse into GeoDataFrame.

    Raises ProcessingError for domain-related failures.
    """
    try:
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        return parse_domain_gdf(snapshot.to_dict())
    except DocumentNotFoundError:
        raise ProcessingError(
            code="DOMAIN_NOT_FOUND",
            message=f"Domain {domain_id} not found.",
            suggestion="Ensure the domain exists before creating an inventory.",
        )
    except EmptyDomainError:
        raise ProcessingError(
            code="EMPTY_DOMAIN",
            message="Domain has no geometry.",
            suggestion="Create a domain with at least one polygon feature.",
        )
    except InvalidGeometryError as e:
        raise ProcessingError(
            code="INVALID_GEOMETRY",
            message=str(e),
            suggestion="Ensure the domain has valid GeoJSON geometry.",
        )


@functions_framework.http
def process_inventory_request(request: Request):
    """Main entry point. Triggered by Cloud Tasks HTTP with {"id": "..."}."""
    # Read inventory ID from request body
    data = request.get_json(silent=True)
    inventory_id = data.get("id") if data else None
    if not inventory_id:
        logger.error("No id in request body")
        return "Missing id", 400

    # Check if we've already tried to process this task before (Cloud Tasks sets X-CloudTasks-TaskRetryCount header)
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"inventory_id": inventory_id},
        )
        try:
            update_status(
                inventory_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200

    logger.info("Processing started", extra={"inventory_id": inventory_id})
    try:
        inventory = load_inventory(inventory_id)
    except DocumentNotFoundError:
        logger.info(
            "Inventory not found (already deleted?)",
            extra={"inventory_id": inventory_id},
        )
        return "OK", 200

    domain_id = inventory.get("domain_id")
    ids = {"inventory_id": inventory_id, "domain_id": domain_id}

    try:
        update_status(inventory_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    # This is where the main processing happens.
    # Handlers will call the progress callback to update progress and check for cancellation.
    try:
        domain_gdf = _load_domain(inventory["domain_id"])
        progress_callback = make_progress_callback(inventory_id)
        # An in-place modification or treatment queues only the new delta in
        # pending_modifications / pending_treatments; apply it to the inventory's
        # own data rather than re-deriving from the root source. The two are
        # mutually exclusive per request — each endpoint sets only its own queue.
        # The pending delta is merged into the cumulative ledger only here, in
        # the same update that flips status to completed, so the stored ledger
        # always equals the applied data: a failed run leaves the ledger as-is
        # and pending_* intact for a retry (#319).
        pending_modifications = inventory.get("pending_modifications") or []
        pending_treatments = inventory.get("pending_treatments") or []
        if pending_modifications:
            result = apply_in_place_modifications(
                inventory, domain_gdf, progress_callback
            )
            completion_extra = {
                "modifications": (inventory.get("modifications") or [])
                + pending_modifications,
                "pending_modifications": [],
            }
        elif pending_treatments:
            result = apply_in_place_treatments(inventory, domain_gdf, progress_callback)
            completion_extra = {
                "treatments": (inventory.get("treatments") or []) + pending_treatments,
                "pending_treatments": [],
            }
        else:
            result = dispatch_handler(inventory, domain_gdf, progress_callback)
            completion_extra = None
        update_status(
            inventory_id,
            "completed",
            georeference=result["georeference"],
            columns=result["columns"],
            extra=completion_extra,
        )
        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        delete_parquet(inventory_id)
        return "OK", 200

    except ProcessingError as e:
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        try:
            update_status(inventory_id, "failed", error=e.to_dict())
        except CancelledException:
            delete_parquet(inventory_id)
        return "OK", 200

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        return "Internal error", 500


if __name__ == "__main__":
    import os

    class MockRequest:
        def __init__(self, data, headers=None):
            self._json = data
            self.headers = headers or {}

        def get_json(self, silent=False):
            return self._json

    inventory_id = os.environ.get("INVENTORY_ID")
    if inventory_id:
        request = MockRequest({"id": inventory_id})
        response, status_code = process_inventory_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set INVENTORY_ID environment variable for local testing")
