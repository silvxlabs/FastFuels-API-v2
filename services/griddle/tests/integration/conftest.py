"""
Shared fixtures and helpers for griddle integration tests.

Supports two execution modes:
- local: Directly calls process_grid_request() with a MockRequest
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion

The main fixture is ``griddle_runner``, which handles the full lifecycle:
Firestore setup -> griddle execution -> polling -> zarr open -> cleanup.
Tests receive a GriddleResult (ds, grid_id) and write their own assertions.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import gcsfs
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import (
    delete_directory,
    delete_file,
    exists,
    upload_file,
)
from lib.testing import (
    SHARED_TEST_DOMAINS_DIR,
    SHARED_TEST_FEATURES_DIR,
    SHARED_TEST_GRIDS_DIR,
)
from lib.zarr_utils import load_zarr


class GriddleResult(NamedTuple):
    ds: xr.Dataset
    grid_id: str


logger = logging.getLogger(__name__)

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR
GRIDS_DIR = SHARED_TEST_GRIDS_DIR
FEATURES_DIR = SHARED_TEST_FEATURES_DIR


def load_json(path: Path) -> dict:
    """Load a JSON file from the test data directory."""
    with open(path) as f:
        return json.load(f)


class MockRequest:
    """Minimal Flask Request mock for calling process_grid_request locally."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


def _run_griddle(grid_id: str) -> None:
    """Execute griddle processing for a grid.

    In local mode, directly calls process_grid_request with a MockRequest.
    In deployed mode, enqueues via Cloud Tasks.
    """
    if DEPLOYMENT_ENV == "local":
        _run_local(grid_id)
    else:
        _run_deployed(grid_id)


def _run_local(grid_id: str) -> None:
    """Call process_grid_request directly with a MockRequest."""
    from griddle.main import process_grid_request

    request = MockRequest(data={"id": grid_id})
    response, status_code = process_grid_request(request)

    if status_code != 200:
        pytest.fail(f"process_grid_request returned {status_code}: {response}")


def _run_deployed(grid_id: str) -> None:
    """Enqueue a grid processing task via Cloud Tasks."""
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import run_v2, tasks_v2
    from google.cloud.tasks_v2 import HttpMethod

    from lib.config import GCP_PROJECT, GCP_REGION, GRIDDLE_QUEUE, GRIDDLE_SERVICE

    async def _enqueue():
        # Look up the Cloud Run service URL
        run_client = run_v2.ServicesAsyncClient()
        service_name = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{GRIDDLE_SERVICE}"
        )
        svc = await run_client.get_service(name=service_name)
        url = svc.uri

        # Create the Cloud Task
        tasks_client = tasks_v2.CloudTasksAsyncClient()
        parent = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, GRIDDLE_QUEUE)
        task = tasks_v2.Task(
            name=tasks_client.task_path(
                GCP_PROJECT, GCP_REGION, GRIDDLE_QUEUE, grid_id
            ),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": grid_id}).encode(),
            },
        )
        try:
            await tasks_client.create_task(parent=parent, task=task)
        except AlreadyExists:
            pass

    asyncio.run(_enqueue())


def _poll_for_completion(grid_id: str, timeout: int = 300) -> dict:
    """Poll Firestore until the grid reaches a terminal status.

    Uses exponential backoff starting at 2s, maxing at 10s.
    Fails immediately if status becomes "failed".

    Args:
        grid_id: Grid document ID to poll.
        timeout: Maximum seconds to wait for completion.

    Returns:
        The final grid document as a dict.
    """
    start = time.time()
    interval = 2.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Grid {grid_id} did not complete within {timeout}s")

        _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
        grid = snapshot.to_dict()
        status = grid.get("status")
        progress = grid.get("progress")

        logger.info(
            f"Grid {grid_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return grid
        if status == "failed":
            error = grid.get("error", {})
            pytest.fail(
                f"Grid {grid_id} failed: {error.get('code')} - {error.get('message')}"
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


@pytest.fixture
def griddle_runner():
    """Run griddle for a (domain, grid) pair and return the output dataset.

    Handles the full lifecycle: Firestore document creation, griddle
    execution, polling (deployed mode), and zarr opening. Cleans up
    all Firestore documents and GCS data on teardown.

    Usage::

        def test_something(griddle_runner):
            result = griddle_runner("blue_mtn.json", "landfire_fbfm40.json")
            assert "fbfm" in result.ds.data_vars
    """
    domain_ids = []
    grid_ids = []
    feature_blobs: list[str] = []
    feature_doc_ids: list[str] = []
    datasets = []

    def _run(
        domain_file: str,
        grid_file: str,
        timeout: int = 300,
        source_overrides: dict | None = None,
        feature_file: str | None = None,
        feature_id: str | None = None,
        feature_doc: bool = False,
        feature_type: str = "road",
        modifications: list[dict] | None = None,
    ) -> GriddleResult:
        # Create domain document
        domain_data = load_json(DOMAINS_DIR / domain_file)
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        domain_ids.append(domain_id)

        # Optionally upload a feature to the FEATURES_BUCKET path the API
        # would write to (``{domain_id}/{feature_id}.parquet``). Used by
        # handlers that read a Feature from GCS — currently just layerset.
        # Auto-injects ``layerset_id`` into ``source_overrides`` when the
        # caller didn't already set one.
        #
        # Fixtures live on disk as GeoJSON (diffable text, per the testing
        # protocol). The production handler reads GeoParquet, so we
        # convert in-memory and upload the Parquet form to GCS — the
        # on-disk fixture format and production storage format are
        # deliberately decoupled.
        if feature_file is not None:
            if feature_id is None:
                feature_id = f"test-{uuid4().hex}"
            feature_gcs_path = (
                f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
            )
            gdf = gpd.read_file(FEATURES_DIR / feature_file)
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                gdf.to_parquet(tmp_path, compression="zstd", row_group_size=1000)
                upload_file(tmp_path, feature_gcs_path)
            finally:
                os.unlink(tmp_path)
            feature_blobs.append(feature_gcs_path)
            source_overrides = dict(source_overrides or {})
            source_overrides.setdefault("layerset_id", feature_id)

            # Modifications conditions with `source: "feature"` require a
            # completed Feature Firestore doc (domain_id + status check).
            # Layerset handlers ignore this — the layerset path reads the
            # parquet directly and never touches the FEATURES_COLLECTION.
            if feature_doc:
                set_document(
                    FEATURES_COLLECTION,
                    feature_id,
                    {
                        "id": feature_id,
                        "domain_id": domain_id,
                        "type": feature_type,
                        "status": "completed",
                        "source": {"product": "test"},
                    },
                )
                feature_doc_ids.append(feature_id)

        # Create grid document
        grid_data = load_json(GRIDS_DIR / grid_file)
        grid_data["domain_id"] = domain_id
        if source_overrides:
            grid_data["source"].update(source_overrides)
        if modifications is not None:
            grid_data["modifications"] = modifications
        grid_id = f"test-{uuid4().hex}"
        grid_data["id"] = grid_id
        set_document(GRIDS_COLLECTION, grid_id, grid_data)
        grid_ids.append(grid_id)

        # Run griddle
        _run_griddle(grid_id)

        # Get final grid state
        if DEPLOYMENT_ENV != "local":
            grid = _poll_for_completion(grid_id, timeout=timeout)
        else:
            _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
            grid = snapshot.to_dict()

        # Verify common invariants that should always hold
        assert grid["status"] == "completed"
        geo = grid["georeference"]
        assert geo is not None, "georeference should be populated after processing"
        assert geo["crs"] is not None
        assert len(geo["transform"]) == 6
        assert len(geo["shape"]) == 2
        assert all(s > 0 for s in geo["shape"])

        # Open zarr and return the dataset for test-specific assertions
        ds = load_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}")

        for var in ds.data_vars:
            if ds[var].dtype == np.float64:
                ds[var] = ds[var].astype(np.float32)
        datasets.append(ds)
        return GriddleResult(ds=ds, grid_id=grid_id)

    yield _run

    # Teardown: close datasets, delete GCS data, delete Firestore documents
    for ds in datasets:
        ds.close()

    for grid_id in grid_ids:
        gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(GRIDS_COLLECTION, grid_id)

    for feature_blob in feature_blobs:
        if exists(feature_blob):
            delete_file(feature_blob)

    for feature_doc_id in feature_doc_ids:
        delete_document(FEATURES_COLLECTION, feature_doc_id)

    for domain_id in domain_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def source_grid(request):
    """Copy a static fixture zarr to a test-specific path.

    Used with ``@pytest.mark.parametrize("source_grid", [...], indirect=True)``
    to provide a completed source grid for transform handlers (lookup, resample).

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


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    """Cleanly shut down gcsfs sessions after all tests complete.

    gcsfs registers a weakref finalizer that calls close_session() when
    instances are garbage collected. During Python's atexit phase, this
    finalizer tries to close the aiohttp session via fsspec's IO thread
    loop, but the session's internal Futures are bound to a different loop,
    producing a RuntimeError.

    Fix: stop the fsspec IO thread loop before clearing the instance cache.
    This forces the finalizer into the synchronous force_close path
    (connector._close()), which doesn't involve cross-loop Future issues.
    """
    yield

    import fsspec.asyn as fasyn
    import gcsfs

    # Stop the fsspec IO thread loop so gcsfs finalizers use the safe
    # synchronous cleanup path instead of async cross-loop calls.
    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=5)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None

    gcsfs.GCSFileSystem.clear_instance_cache()
