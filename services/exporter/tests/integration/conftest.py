"""
Shared fixtures and helpers for exporter integration tests.

Supports two execution modes:
- local: Directly calls process_export_request() with a MockRequest
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion

The main fixture is ``exporter_runner``, which handles the full lifecycle:
Firestore setup -> exporter execution -> polling -> invariant checks -> cleanup.
Tests receive the final export document dict and write their own assertions.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import gcsfs
import pytest

from lib.config import (
    DEPLOYMENT_ENV,
    EXPORTS_BUCKET,
    EXPORTS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists

logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path(__file__).parent.parent / "data"
EXPORTS_DIR = TEST_DATA_DIR / "exports"

# Grid JSON templates live in griddle's test data (shared format)
GRIDS_DIR = Path(__file__).parents[3] / "griddle" / "tests" / "data" / "grids"


def load_json(path: Path) -> dict:
    """Load a JSON file from the test data directory."""
    with open(path) as f:
        return json.load(f)


class MockRequest:
    """Minimal Flask Request mock for calling process_export_request locally."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


def _run_exporter(export_id: str) -> None:
    """Execute exporter processing for an export.

    In local mode, directly calls process_export_request with a MockRequest.
    In deployed mode, enqueues via Cloud Tasks.
    """
    if DEPLOYMENT_ENV == "local":
        _run_local(export_id)
    else:
        _run_deployed(export_id)


def _run_local(export_id: str) -> None:
    """Call process_export_request directly with a MockRequest."""
    from exporter.main import process_export_request

    request = MockRequest(data={"id": export_id})
    response, status_code = process_export_request(request)

    if status_code != 200:
        pytest.fail(f"process_export_request returned {status_code}: {response}")


def _run_deployed(export_id: str) -> None:
    """Enqueue an export processing task via Cloud Tasks."""
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import run_v2, tasks_v2
    from google.cloud.tasks_v2 import HttpMethod

    from lib.config import EXPORTER_QUEUE, EXPORTER_SERVICE, GCP_PROJECT, GCP_REGION

    async def _enqueue():
        # Look up the Cloud Run service URL
        run_client = run_v2.ServicesAsyncClient()
        service_name = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{EXPORTER_SERVICE}"
        )
        svc = await run_client.get_service(name=service_name)
        url = svc.uri

        # Create the Cloud Task
        tasks_client = tasks_v2.CloudTasksAsyncClient()
        parent = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, EXPORTER_QUEUE)
        task = tasks_v2.Task(
            name=tasks_client.task_path(
                GCP_PROJECT, GCP_REGION, EXPORTER_QUEUE, export_id
            ),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": export_id}).encode(),
            },
        )
        try:
            await tasks_client.create_task(parent=parent, task=task)
        except AlreadyExists:
            pass

    asyncio.run(_enqueue())


def _poll_for_completion(export_id: str, timeout: int = 120) -> dict:
    """Poll Firestore until the export reaches a terminal status.

    Uses exponential backoff starting at 2s, maxing at 10s.
    Fails immediately if status becomes "failed".

    Args:
        export_id: Export document ID to poll.
        timeout: Maximum seconds to wait for completion.

    Returns:
        The final export document as a dict.
    """
    start = time.time()
    interval = 2.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Export {export_id} did not complete within {timeout}s")

        _, snapshot = get_document(EXPORTS_COLLECTION, export_id)
        export = snapshot.to_dict()
        status = export.get("status")
        progress = export.get("progress")

        logger.info(
            f"Export {export_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return export
        if status == "failed":
            error = export.get("error", {})
            pytest.fail(
                f"Export {export_id} failed: {error.get('code')} - {error.get('message')}"
            )

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)


@pytest.fixture
def source_grid(request):
    """Copy a static fixture zarr to a test-specific path.

    Used with ``@pytest.mark.parametrize("source_grid", [...], indirect=True)``
    to provide a completed source grid for export tests.

    The static zarr is copied to a unique test path, a Firestore document is
    created from the corresponding JSON template, and both are cleaned up on
    teardown.
    """
    static_name = request.param
    grid_id = f"test-{uuid4().hex}"

    # Copy static zarr to test-specific path
    fs = gcsfs.GCSFileSystem()
    src = f"{GRIDS_BUCKET}/{static_name}"
    dst = f"{GRIDS_BUCKET}/{grid_id}"
    fs.cp(src, dst, recursive=True)

    # Create Firestore doc from JSON template
    grid_data = load_json(GRIDS_DIR / f"{static_name}.json")
    grid_data["id"] = grid_id
    set_document(GRIDS_COLLECTION, grid_id, grid_data)

    yield grid_id

    # Cleanup
    gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(GRIDS_COLLECTION, grid_id)


@pytest.fixture
def exporter_runner():
    """Run the exporter for a source grid and return the final export document.

    Handles the full lifecycle: Firestore document creation, exporter
    execution, polling (deployed mode), and invariant checks. Cleans up
    all Firestore documents and GCS export files on teardown.

    Usage::

        def test_something(exporter_runner, source_grid):
            export = exporter_runner(source_grid, "geotiff.json")
            assert export["status"] == "completed"
    """
    export_ids = []

    def _run(
        grid_id: str,
        export_file: str,
        timeout: int = 120,
        source_overrides: dict | None = None,
    ) -> dict:
        # Load export template
        export_data = load_json(EXPORTS_DIR / export_file)
        export_id = f"test-{uuid4().hex}"
        export_data["id"] = export_id
        export_data["source"]["grid_ids"] = [grid_id]

        # Apply optional source overrides (e.g., band subset)
        if source_overrides:
            export_data["source"].update(source_overrides)

        # Create Firestore export document
        set_document(EXPORTS_COLLECTION, export_id, export_data)
        export_ids.append(export_id)

        # Run exporter
        _run_exporter(export_id)

        # Get final export state
        if DEPLOYMENT_ENV != "local":
            export = _poll_for_completion(export_id, timeout=timeout)
        else:
            _, snapshot = get_document(EXPORTS_COLLECTION, export_id)
            export = snapshot.to_dict()

        # Verify common invariants
        assert export["status"] == "completed", (
            f"Export did not complete: status={export['status']}, "
            f"error={export.get('error')}"
        )
        assert export["signed_url"] is not None, "signed_url should be set"

        return export

    yield _run

    # Teardown: delete GCS export files and Firestore documents
    for export_id in export_ids:
        gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(EXPORTS_COLLECTION, export_id)
