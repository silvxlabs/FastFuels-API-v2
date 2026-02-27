"""
Shared fixtures and helpers for standgen integration tests.

Supports two execution modes:
- local: Directly calls process_inventory_request() with a MockRequest
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion

The main fixture is ``standgen_runner``, which handles the full lifecycle:
Firestore setup -> standgen execution -> polling -> parquet verification -> cleanup.
Tests receive the completed inventory document and write their own assertions.
"""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import gcsfs
import pytest

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists
from lib.testing import (
    SHARED_TEST_DOMAINS_DIR,
    SHARED_TEST_GRIDS_DIR,
    SHARED_TEST_INVENTORIES_DIR,
)

logger = logging.getLogger(__name__)

INVENTORIES_DIR = SHARED_TEST_INVENTORIES_DIR
DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR
GRIDS_DIR = SHARED_TEST_GRIDS_DIR


def load_json(path: Path) -> dict:
    """Load a JSON file from the test data directory."""
    with open(path) as f:
        return json.load(f)


class MockRequest:
    """Minimal Flask Request mock for calling process_inventory_request locally."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


def _run_standgen(inventory_id: str) -> None:
    """Execute standgen processing for an inventory.

    In local mode, directly calls process_inventory_request with a MockRequest.
    In deployed mode, enqueues via Cloud Tasks.
    """
    if DEPLOYMENT_ENV == "local":
        _run_local(inventory_id)
    else:
        _run_deployed(inventory_id)


def _run_local(inventory_id: str) -> None:
    """Call process_inventory_request directly with a MockRequest."""
    from standgen.main import process_inventory_request

    request = MockRequest(data={"id": inventory_id})
    response, status_code = process_inventory_request(request)

    if status_code != 200:
        pytest.fail(f"process_inventory_request returned {status_code}: {response}")


def _run_deployed(inventory_id: str) -> None:
    """Enqueue an inventory processing task via Cloud Tasks."""
    import asyncio

    from google.api_core.exceptions import AlreadyExists
    from google.cloud import run_v2, tasks_v2
    from google.cloud.tasks_v2 import HttpMethod

    from lib.config import GCP_PROJECT, GCP_REGION, STANDGEN_QUEUE, STANDGEN_SERVICE

    async def _enqueue():
        run_client = run_v2.ServicesAsyncClient()
        service_name = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{STANDGEN_SERVICE}"
        )
        svc = await run_client.get_service(name=service_name)
        url = svc.uri

        tasks_client = tasks_v2.CloudTasksAsyncClient()
        parent = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, STANDGEN_QUEUE)
        task = tasks_v2.Task(
            name=tasks_client.task_path(
                GCP_PROJECT, GCP_REGION, STANDGEN_QUEUE, inventory_id
            ),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": inventory_id}).encode(),
            },
        )
        try:
            await tasks_client.create_task(parent=parent, task=task)
        except AlreadyExists:
            pass

    asyncio.run(_enqueue())


def _poll_for_completion(inventory_id: str, timeout: int = 300) -> dict:
    """Poll Firestore until the inventory reaches a terminal status.

    Uses exponential backoff starting at 2s, maxing at 10s.
    Fails immediately if status becomes "failed".
    """
    start = time.time()
    interval = 2.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Inventory {inventory_id} did not complete within {timeout}s")

        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()
        status = inventory.get("status")
        progress = inventory.get("progress")

        logger.info(
            f"Inventory {inventory_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return inventory
        if status == "failed":
            error = inventory.get("error", {})
            pytest.fail(
                f"Inventory {inventory_id} failed: "
                f"{error.get('code')} - {error.get('message')}"
            )

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)


def _stringify_coordinates(domain_data: dict) -> dict:
    """Stringify nested coordinate arrays for Firestore compatibility.

    Firestore doesn't support nested arrays, so coordinates must be stored
    as JSON strings. This mirrors what the API does on domain creation.
    """
    import copy

    data = copy.deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


@pytest.fixture(scope="module")
def module_pim_grid(request):
    """Module-scoped variant of source_pim_grid.

    Copies a static PIM grid to a test-specific GCS path and creates the
    Firestore document. Shared across all tests in a module for efficiency.

    Used with ``@pytest.mark.parametrize("module_pim_grid", [...], indirect=True)``
    or requested directly from module-scoped fixtures.
    """
    static_name = request.param
    grid_id = f"test-{uuid4().hex}"

    fs = gcsfs.GCSFileSystem()
    src = f"{GRIDS_BUCKET}/{static_name}"
    dst = f"{GRIDS_BUCKET}/{grid_id}"
    fs.cp(src, dst, recursive=True)

    grid_data = load_json(GRIDS_DIR / f"{static_name}.json")
    grid_data["id"] = grid_id
    set_document(GRIDS_COLLECTION, grid_id, grid_data)

    yield grid_id

    gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(GRIDS_COLLECTION, grid_id)


@pytest.fixture
def source_pim_grid(request):
    """Copy a static PIM grid fixture to a test-specific path.

    Used with ``@pytest.mark.parametrize("source_pim_grid", [...], indirect=True)``
    to provide a completed PIM grid for standgen.

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
def standgen_runner():
    """Run standgen for a (domain, inventory) pair and return the completed doc.

    Handles the full lifecycle: Firestore document creation, standgen
    execution, polling (deployed mode), and output verification. Cleans up
    all Firestore documents and GCS data on teardown.

    Usage::

        @pytest.mark.parametrize(
            "source_pim_grid", ["static-test-blue-mtn-pim-treemap"], indirect=True
        )
        def test_something(standgen_runner, source_pim_grid):
            inventory = standgen_runner(
                "blue_mtn.json", "pim_treemap.json",
                source_pim_grid_id=source_pim_grid,
            )
            assert inventory["georeference"] is not None
    """
    domain_ids = []
    inventory_ids = []

    def _run(
        domain_file: str,
        inventory_file: str,
        source_pim_grid_id: str,
        timeout: int = 300,
        source_overrides: dict | None = None,
    ) -> dict:
        # Create domain document
        domain_data = load_json(DOMAINS_DIR / domain_file)
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        domain_ids.append(domain_id)

        # Create inventory document
        inventory_data = load_json(INVENTORIES_DIR / inventory_file)
        inventory_data["domain_id"] = domain_id
        inventory_data["source"]["source_pim_grid_id"] = source_pim_grid_id
        if source_overrides:
            inventory_data["source"].update(source_overrides)
        inventory_id = f"test-{uuid4().hex}"
        inventory_data["id"] = inventory_id
        set_document(INVENTORIES_COLLECTION, inventory_id, inventory_data)
        inventory_ids.append(inventory_id)

        # Run standgen
        _run_standgen(inventory_id)

        # Get final inventory state
        if DEPLOYMENT_ENV != "local":
            inventory = _poll_for_completion(inventory_id, timeout=timeout)
        else:
            _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
            inventory = snapshot.to_dict()

        # Verify common invariants
        assert inventory["status"] == "completed", (
            f"Expected completed, got {inventory['status']}. "
            f"Error: {inventory.get('error')}"
        )
        assert inventory.get("georeference") is not None, (
            "georeference should be populated after processing"
        )

        return inventory

    yield _run

    # Teardown: delete GCS parquet data, delete Firestore documents
    for inventory_id in inventory_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, inventory_id)

    for domain_id in domain_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    """Cleanly shut down gcsfs sessions after all tests complete.

    Prevents RuntimeError during pytest atexit phase when fsspec IO thread
    tries to close aiohttp session bound to a different loop.
    """
    yield

    import fsspec.asyn as fasyn
    import gcsfs as _gcsfs

    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=5)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None

    _gcsfs.GCSFileSystem.clear_instance_cache()
