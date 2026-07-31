"""
Lakitu main entry point.

Cloud Function that produces point clouds via Cloud Tasks HTTP trigger. All
infrastructure concerns (Firestore, GCS, progress) are handled here; handlers
fetch and assemble points and hand back a buffer.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

import functions_framework
from flask import Request

from lakitu.dispatch import dispatch_handler
from lakitu.storage import delete_cloud, write_cloud
from lib.config import DOMAINS_COLLECTION, POINT_CLOUDS_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.errors import CancelledException, ProcessingError
from lib.firestore import DocumentNotFoundError, get_document, update_document


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging.

    Includes point_cloud_id and domain_id in log entries when set via
    `extra={'point_cloud_id': ...}`.
    """

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("point_cloud_id", "domain_id"):
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

SOURCE_NOT_FOUND_MESSAGE = (
    "A required input resource was not found. It may have been deleted "
    "before processing completed."
)


def load_point_cloud(point_cloud_id: str) -> dict:
    """Load point cloud document from Firestore.

    Args:
        point_cloud_id: Point cloud document ID

    Returns:
        Point cloud document as dict

    Raises:
        DocumentNotFoundError: If point cloud not found
    """
    _, snapshot = get_document(POINT_CLOUDS_COLLECTION, point_cloud_id)
    return snapshot.to_dict()


def update_progress(
    point_cloud_id: str, message: str, percent: int | None = None
) -> None:
    """Update point cloud progress information.

    Args:
        point_cloud_id: Point cloud document ID
        message: Progress message
        percent: Optional completion percentage (0-100)

    Raises:
        CancelledException: If the point cloud was deleted (user cancelled)
    """
    progress = {"message": message}
    if percent is not None:
        progress["percent"] = percent

    try:
        update_document(
            POINT_CLOUDS_COLLECTION,
            point_cloud_id,
            {"progress": progress, "modified_on": datetime.now(UTC)},
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Point cloud {point_cloud_id} was cancelled")


def update_status(
    point_cloud_id: str,
    status: str,
    georeference: dict | None = None,
    summary: dict | None = None,
    size_bytes: int | None = None,
    source_extra: dict | None = None,
    error: dict | None = None,
) -> None:
    """Update point cloud status and any derived fields.

    ``checksum`` is deliberately never written here. The document update merges,
    so omitting it preserves the value assigned at creation — which is what
    downstream resources compare against to detect staleness.

    Args:
        point_cloud_id: Point cloud document ID
        status: New status value
        georeference: CRS and 3D bounds, on success
        summary: Point count, classes, and density, on success
        size_bytes: Stored size, on success. Storage quota aggregates this
            field, so a completed cloud without it is invisible to the quota.
        source_extra: Provenance resolved during processing, merged into the
            existing source block
        error: Structured error, on failure

    Raises:
        CancelledException: If the point cloud was deleted (user cancelled)
    """
    data: dict = {"status": status, "modified_on": datetime.now(UTC)}
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}

    if georeference is not None:
        data["georeference"] = georeference
    if summary is not None:
        data["summary"] = summary
    if size_bytes is not None:
        data["size_bytes"] = size_bytes
    if source_extra:
        # Dotted keys merge into the stored source map rather than replacing it,
        # so the discriminator and the request's own pin survive.
        for key, value in source_extra.items():
            data[f"source.{key}"] = value
    if error is not None:
        data["error"] = error

    try:
        update_document(POINT_CLOUDS_COLLECTION, point_cloud_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Point cloud {point_cloud_id} was cancelled")


def make_progress_callback(point_cloud_id: str):
    """Build a progress callback bound to a point cloud."""

    def callback(message: str, percent: int | None = None) -> None:
        update_progress(point_cloud_id, message, percent)

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
            suggestion="Ensure the domain exists before creating a point cloud.",
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
def process_point_cloud_request(request: Request):
    """Main entry point. Triggered by Cloud Tasks HTTP with {"id": "..."}."""
    data = request.get_json(silent=True)
    point_cloud_id = data.get("id") if data else None
    if not point_cloud_id:
        logger.error("No id in request body")
        return "Missing id", 400

    # Cloud Tasks sets X-CloudTasks-TaskRetryCount. A retry means the previous
    # attempt died without recording anything, so fail the resource rather than
    # spend another hour of fetching on it.
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"point_cloud_id": point_cloud_id},
        )
        try:
            update_status(
                point_cloud_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
        except (CancelledException, DocumentNotFoundError):
            pass
        delete_cloud(point_cloud_id)
        return "OK", 200  # Return 200 to prevent further retries

    logger.info("Processing started", extra={"point_cloud_id": point_cloud_id})
    try:
        point_cloud = load_point_cloud(point_cloud_id)
    except DocumentNotFoundError:
        logger.info(
            "Point cloud not found (already deleted?)",
            extra={"point_cloud_id": point_cloud_id},
        )
        return "OK", 200

    domain_id = point_cloud.get("domain_id")
    ids = {"point_cloud_id": point_cloud_id, "domain_id": domain_id}

    try:
        update_status(point_cloud_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    try:
        domain_gdf = _load_domain(point_cloud["domain_id"])
        progress_callback = make_progress_callback(point_cloud_id)

        result = dispatch_handler(point_cloud, domain_gdf, progress_callback)

        progress_callback("Storing point cloud", 95)
        size_bytes = write_cloud(point_cloud_id, result["buffer"])

        update_status(
            point_cloud_id,
            "completed",
            georeference=result["georeference"],
            summary=result["summary"],
            size_bytes=size_bytes,
            source_extra=result.get("source_extra"),
        )
        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        delete_cloud(point_cloud_id)
        return "OK", 200

    except ProcessingError as e:
        # Expected, handled terminal outcome — log at WARNING, not ERROR.
        logger.warning(f"Processing failed: {e.code} - {e.message}", extra=ids)
        delete_cloud(point_cloud_id)
        try:
            update_status(point_cloud_id, "failed", error=e.to_dict())
        except CancelledException:
            pass
        return "OK", 200

    except FileNotFoundError as e:
        # A referenced input was deleted while this job was queued or running —
        # a benign race, not a system fault. Record a terminal failure and
        # return 200: the object will never reappear, so a retry is wasted.
        logger.warning(f"Input not found (deleted during processing?): {e}", extra=ids)
        delete_cloud(point_cloud_id)
        try:
            update_status(
                point_cloud_id,
                "failed",
                error={
                    "code": "SOURCE_NOT_FOUND",
                    "message": SOURCE_NOT_FOUND_MESSAGE,
                },
            )
        except CancelledException:
            pass
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

    point_cloud_id = os.environ.get("POINT_CLOUD_ID")
    if point_cloud_id:
        request = MockRequest({"id": point_cloud_id})
        response, status_code = process_point_cloud_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set POINT_CLOUD_ID environment variable for local testing")
