"""
Griddle main entry point.

Cloud Function that processes grid requests via Cloud Tasks HTTP trigger.
All infrastructure concerns (Firestore, GCS, progress) are handled here.
Handlers are pure functions that just transform data.
"""

import json
import logging
import math
import sys
import traceback
from datetime import UTC, datetime

import functions_framework
from flask import Request

from griddle.dispatch import dispatch_handler
from griddle.modifications import apply_modifications
from griddle.storage import delete_zarr, save_zarr
from griddle.summarize import summarize_dataset
from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError, get_document, update_document
from lib.grids import compute_chunks_doc


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
    chunks: dict | None = None,
    error: dict | None = None,
) -> None:
    """Update grid status.

    Args:
        grid_id: Grid document ID
        status: New status ("running", "completed", "failed")
        georeference: Optional georeference dict (for completed status)
        chunks: Optional chunks layout dict (for completed status)
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

    if chunks is not None:
        data["chunks"] = chunks

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
            suggestion="Ensure the domain exists before creating a grid.",
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


def _band_nodata(result, key: str):
    """Resolve a band's nodata value from the produced Dataset for the resource.

    Reads the value the data actually carries (the source raster's nodata tag,
    or the fill a constructed band uses) — it is not invented. Layerset band
    keys are ``<var>.<band>`` while nodata lives on the variable, so fall back
    to the pre-dot variable name. Returns ``None`` when the band has no nodata,
    or when the sentinel is NaN (which has no JSON representation).
    """
    var = key if key in result.data_vars else key.rsplit(".", 1)[0]
    if var not in result.data_vars:
        return None
    nodata = result[var].rio.nodata
    if nodata is None:
        return None
    value = nodata.item() if hasattr(nodata, "item") else nodata
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


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
    6. Compute band summaries and write back to Firestore
    7. Save result to Zarr
    8. Update status to "completed" with georeference
    9. On error, update status to "failed" with error details

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
        # Load and parse domain
        domain_gdf = _load_domain(grid["domain_id"])

        # Dispatch to handler
        progress_callback = make_progress_callback(grid_id)
        result = dispatch_handler(grid, domain_gdf, progress_callback)

        # Write back enriched source metadata (e.g., 3DEP tile provenance)
        update_document(GRIDS_COLLECTION, grid_id, {"source": grid["source"]})

        # Compute band summaries, propagate nodata, and write back to Firestore
        summaries = summarize_dataset(result, grid["bands"])
        bands_with_summaries = [
            {**band, "summary": summaries.get(band["key"])} for band in grid["bands"]
        ]

        # Apply modifications if present
        if grid.get("modifications"):
            update_progress(grid_id, "Applying modifications...", 80)
            result = apply_modifications(
                result, grid["modifications"], grid["domain_id"]
            )

        # Propagate each band's nodata value from the produced data onto the
        # grid resource so consumers see it without opening the Zarr (NaN
        # floats surface as null). Read after modifications so it reflects the
        # final stored data.
        for band in bands_with_summaries:
            band["nodata"] = _band_nodata(result, band["key"])
        update_document(GRIDS_COLLECTION, grid_id, {"bands": bands_with_summaries})

        # Save to Zarr
        update_progress(grid_id, "Saving...", 90)
        chunk_shape = tuple((grid.get("chunks") or {}).get("shape") or (512, 512))
        save_zarr(grid_id, result, chunk_shape=chunk_shape)

        # Update status to completed with georeference and chunks layout
        transform = result.rio.transform()
        grid_shape = (result.rio.height, result.rio.width)
        update_status(
            grid_id,
            "completed",
            georeference={
                "crs": str(result.rio.crs),
                "transform": list(transform)[:6],
                "shape": list(grid_shape),
            },
            chunks=compute_chunks_doc(grid_shape, chunk_shape),
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
