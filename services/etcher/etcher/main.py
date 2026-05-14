"""
Features main entry point.

Cloud Function that processes feature generation requests via Cloud Tasks HTTP trigger.
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

from etcher.dispatch import dispatch_handler
from etcher.errors import CancelledException, ProcessingError
from etcher.storage import delete_geojson
from lib.config import DOMAINS_COLLECTION, FEATURES_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.firestore import DocumentNotFoundError, get_document, update_document


class StructuredLogHandler(logging.Handler):
    """Log handler that outputs JSON for Cloud Logging.

    Includes feature_id and domain_id in log entries when set via extra.
    """

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("feature_id", "domain_id"):
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


def load_feature(feature_id: str) -> dict:
    """Load feature document from Firestore."""
    _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
    return snapshot.to_dict()


def update_progress(feature_id, message, percent=None):
    """Update feature progress. Raises CancelledException if deleted."""
    progress = {"message": message}
    if percent is not None:
        progress["percent"] = percent
    try:
        update_document(
            FEATURES_COLLECTION,
            feature_id,
            {"progress": progress, "modified_on": datetime.now(UTC)},
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Feature {feature_id} was cancelled")


def update_status(feature_id, status, georeference=None, error=None):
    """Update feature status."""
    data = {"status": status, "modified_on": datetime.now(UTC)}
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}
    if georeference is not None:
        data["georeference"] = georeference
    if error is not None:
        data["error"] = error
    try:
        update_document(FEATURES_COLLECTION, feature_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Feature {feature_id} was cancelled")


def make_progress_callback(feature_id):
    def callback(message, percent=None):
        update_progress(feature_id, message, percent)

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
            suggestion="Ensure the domain exists before creating a feature.",
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
def process_feature_request(request: Request):
    """Main entry point. Triggered by Cloud Tasks HTTP with {"id": "..."}."""
    # Read feature ID from request body
    data = request.get_json(silent=True)
    feature_id = data.get("id") if data else None
    if not feature_id:
        logger.error("No id in request body")
        return "Missing id", 400

    # Check if we've already tried to process this task before (Cloud Tasks sets X-CloudTasks-TaskRetryCount header)
    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"feature_id": feature_id},
        )
        try:
            update_status(
                feature_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200

    logger.info("Processing started", extra={"feature_id": feature_id})
    try:
        feature = load_feature(feature_id)
    except DocumentNotFoundError:
        logger.info(
            "Feature not found (already deleted?)",
            extra={"feature_id": feature_id},
        )
        return "OK", 200

    domain_id = feature.get("domain_id")
    ids = {"feature_id": feature_id, "domain_id": domain_id}

    try:
        update_status(feature_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    # This is where the main processing happens.
    # Handlers will call the progress callback to update progress and check for cancellation.
    try:
        domain_gdf = _load_domain(domain_id)
        progress_callback = make_progress_callback(feature_id)
        result = dispatch_handler(feature, domain_gdf, progress_callback)
        update_status(
            feature_id,
            "completed",
            georeference=result.get("georeference"),
        )
        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        delete_geojson(domain_id, feature_id)
        return "OK", 200

    except ProcessingError as e:
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        try:
            update_status(feature_id, "failed", error=e.to_dict())
        except CancelledException:
            delete_geojson(domain_id, feature_id)
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

    feature_id = os.environ.get("FEATURE_ID")
    if feature_id:
        request = MockRequest({"id": feature_id})
        response, status_code = process_feature_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set FEATURE_ID environment variable for local testing")
