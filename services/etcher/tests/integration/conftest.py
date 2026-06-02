"""
Shared fixtures and helpers for feature integration tests.

Supports two execution modes:
- local: Directly calls process_feature_request() with a MockRequest
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion

The main fixture is ``features_runner``, which handles the full lifecycle:
Firestore setup -> feature execution -> polling -> Parquet verification -> cleanup.
"""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import pytest

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_file, exists
from lib.testing import (
    SHARED_TEST_DOMAINS_DIR,
    SHARED_TEST_FEATURES_DIR,
)

logger = logging.getLogger(__name__)

FEATURES_DIR = SHARED_TEST_FEATURES_DIR
DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


def load_json(path: Path) -> dict:
    """Load a JSON file from the test data directory."""
    with open(path) as f:
        return json.load(f)


class MockRequest:
    """Minimal Flask Request mock for calling process_feature_request locally."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


def _run_feature_job(feature_id: str) -> None:
    """Execute feature processing.

    In local mode, directly calls process_feature_request with a MockRequest.
    In deployed mode, enqueues via Cloud Tasks.
    """
    if DEPLOYMENT_ENV == "local":
        _run_local(feature_id)
    else:
        _run_deployed(feature_id)


def _run_local(feature_id: str) -> None:
    """Call process_feature_request directly with a MockRequest."""
    from etcher.main import process_feature_request

    request = MockRequest(data={"id": feature_id})
    response, status_code = process_feature_request(request)

    if status_code != 200:
        pytest.fail(f"process_feature_request returned {status_code}: {response}")


def _run_deployed(feature_id: str) -> None:
    """Enqueue a feature processing task via Cloud Tasks."""
    import asyncio

    from google.api_core.exceptions import AlreadyExists
    from google.cloud import run_v2, tasks_v2
    from google.cloud.tasks_v2 import HttpMethod

    from lib.config import FEATURES_QUEUE, FEATURES_SERVICE, GCP_PROJECT, GCP_REGION

    async def _enqueue():
        run_client = run_v2.ServicesAsyncClient()
        service_name = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{FEATURES_SERVICE}"
        )
        svc = await run_client.get_service(name=service_name)
        url = svc.uri

        tasks_client = tasks_v2.CloudTasksAsyncClient()
        parent = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, FEATURES_QUEUE)
        task = tasks_v2.Task(
            name=tasks_client.task_path(
                GCP_PROJECT, GCP_REGION, FEATURES_QUEUE, feature_id
            ),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": feature_id}).encode(),
            },
        )
        try:
            await tasks_client.create_task(parent=parent, task=task)
        except AlreadyExists:
            pass

    asyncio.run(_enqueue())


def _poll_for_completion(feature_id: str, timeout: int = 300) -> dict:
    """Poll Firestore until the feature reaches a terminal status."""
    start = time.time()
    interval = 2.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Feature {feature_id} did not complete within {timeout}s")

        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()
        status = feature.get("status")
        progress = feature.get("progress")

        logger.info(
            f"Feature {feature_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return feature
        if status == "failed":
            error = feature.get("error", {})
            pytest.fail(
                f"Feature {feature_id} failed: "
                f"{error.get('code')} - {error.get('message')}"
            )

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)


def _stringify_coordinates(domain_data: dict) -> dict:
    """Stringify nested coordinate arrays for Firestore compatibility."""
    import copy

    data = copy.deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


@pytest.fixture
def features_runner():
    """Run feature generation for a (domain, feature) pair.

    Handles the full lifecycle and cleans up Firestore/GCS on teardown.
    """
    domain_ids = []
    feature_ids = []

    def _run(
        domain_file: str,
        feature_file: str,
        timeout: int = 300,
    ) -> tuple[str, dict]:
        # Create domain document
        domain_data = load_json(DOMAINS_DIR / domain_file)
        domain_id = f"test-{uuid4().hex}"
        data = _stringify_coordinates(domain_data)
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        domain_ids.append(domain_id)

        # Create feature document
        feature_data = load_json(FEATURES_DIR / feature_file)
        feature_data["domain_id"] = domain_id
        feature_id = f"test-{uuid4().hex}"
        feature_data["id"] = feature_id
        set_document(FEATURES_COLLECTION, feature_id, feature_data)
        feature_ids.append((domain_id, feature_id))

        # Run feature job
        _run_feature_job(feature_id)

        # Get final state
        if DEPLOYMENT_ENV != "local":
            feature = _poll_for_completion(feature_id, timeout=timeout)
        else:
            _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
            feature = snapshot.to_dict()

        assert feature["status"] == "completed", (
            f"Expected completed, got {feature['status']}. "
            f"Error: {feature.get('error')}"
        )
        assert feature.get("georeference") is not None, (
            "georeference should be populated after processing"
        )

        return domain_id, feature

    yield _run

    # Teardown
    for domain_id, feature_id in feature_ids:
        gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
        if exists(gcs_path):
            delete_file(gcs_path)
        delete_document(FEATURES_COLLECTION, feature_id)

    for domain_id in domain_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    """Cleanly shut down gcsfs sessions after all tests complete."""
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
