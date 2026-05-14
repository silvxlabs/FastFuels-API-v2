"""Cloud Function entry for treevox.

Thin HTTP layer only. All voxelization work lives in `treevox.orchestrator`;
Firestore reads/writes in `treevox.firestore_io`; inventory parquet I/O in
`treevox.inventory_io`; error types in `treevox.errors`.

Retry semantics (via `X-CloudTasks-TaskRetryCount`):
- First attempt processes normally.
- Any retry attempt marks the grid as failed and returns 200 so Cloud Tasks
  stops retrying.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback

import functions_framework
from flask import Request

from lib.config import GRIDS_COLLECTION
from lib.firestore import DocumentNotFoundError, update_document
from lib.grids import compute_chunks_doc
from treevox import storage
from treevox.errors import CancelledException, ProcessingError
from treevox.firestore_io import (
    load_domain,
    load_grid,
    make_progress_callback,
    update_status,
)
from treevox.orchestrator import dispatch_handler

# GRPC_VERBOSITY and GRPC_ENABLE_FORK_SUPPORT are set before this module
# imports via the Dockerfile (production) and services/treevox/conftest.py
# (tests). They silence gRPC's atfork warnings that would otherwise spam
# "fork_posix.cc" / "ev_poll_posix.cc" when the multiprocessing Pool spawns.


class StructuredLogHandler(logging.Handler):
    """JSON log handler for Cloud Logging; carries grid_id / domain_id via extra."""

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


# Attach the handler to the package logger so descendant modules
# (treevox.orchestrator, treevox.storage, …) inherit it without each
# needing its own handler.
_pkg_logger = logging.getLogger("treevox")
_pkg_logger.setLevel(logging.INFO)
if not any(isinstance(h, StructuredLogHandler) for h in _pkg_logger.handlers):
    _pkg_logger.addHandler(StructuredLogHandler())
_pkg_logger.propagate = False

logger = logging.getLogger(__name__)


UNEXPECTED_FAILURE_MESSAGE = (
    "Job failed unexpectedly. Please try again or contact the development team."
)


@functions_framework.http
def process_grid_request(request: Request):
    """Cloud Tasks HTTP trigger — expects `{"id": grid_id}`."""
    data = request.get_json(silent=True)
    grid_id = data.get("id") if data else None

    if not grid_id:
        logger.error("No id in request body")
        return "Missing id", 400

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
            storage.delete_zarr(storage.gcs_path(grid_id))
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200

    logger.info("Processing started", extra={"grid_id": grid_id})

    try:
        grid = load_grid(grid_id)
    except DocumentNotFoundError:
        logger.info("Grid not found (already deleted?)", extra={"grid_id": grid_id})
        return "OK", 200

    domain_id = grid.get("domain_id")
    ids = {"grid_id": grid_id, "domain_id": domain_id}

    try:
        update_status(grid_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    try:
        domain_gdf = load_domain(grid["domain_id"])
        progress_callback = make_progress_callback(grid_id)

        # This is where we run the actual process
        result = dispatch_handler(grid, domain_gdf, progress_callback)

        chunks_doc = compute_chunks_doc(
            result.georeference["shape"], result.chunk_shape
        )
        update_document(GRIDS_COLLECTION, grid_id, {"chunks": chunks_doc})
        update_status(grid_id, "completed", georeference=result.georeference)

        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        storage.delete_zarr(storage.gcs_path(grid_id))
        return "OK", 200

    except ProcessingError as e:
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        storage.delete_zarr(storage.gcs_path(grid_id))
        try:
            update_status(grid_id, "failed", error=e.to_dict())
        except CancelledException:
            pass
        return "OK", 200

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        storage.delete_zarr(storage.gcs_path(grid_id))
        return "Internal error", 500


class MockRequest:
    """Simple mock request for local testing via `GRID_ID=... uv run treevox/main.py`."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


if __name__ == "__main__":
    grid_id = os.environ.get("GRID_ID")
    if grid_id:
        request = MockRequest({"id": grid_id})
        response, status_code = process_grid_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set GRID_ID environment variable for local testing")
