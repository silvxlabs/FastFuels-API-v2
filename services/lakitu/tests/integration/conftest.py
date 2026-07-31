"""
Shared fixtures for lakitu integration tests.

These run the worker end to end against the live USGS 3DEP archive and real
Firestore/GCS. Domains are seeded per test with a `test-` prefixed id and torn
down afterwards.
"""

import json
import uuid
from copy import deepcopy

import pytest

from lib.config import DOMAINS_COLLECTION, POINT_CLOUDS_COLLECTION
from lib.firestore import delete_document, get_document, set_document
from lib.gcs import delete_directory, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR, load_json


def stringify_coordinates(domain_data: dict) -> dict:
    """JSON-encode nested coordinate arrays, as the API does on domain create.

    Firestore rejects nested arrays, so domains are stored with their
    coordinates as strings and `parse_domain_gdf` decodes them back.
    """
    data = deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if isinstance(coords, list):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


class MockRequest:
    """Stands in for the flask request Cloud Tasks would deliver."""

    def __init__(self, data, headers=None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent=False):
        return self._json


@pytest.fixture
def seeded_domain():
    """Seed a domain from a shared fixture file and clean it up afterwards."""
    created: list[str] = []

    def seed(filename: str) -> str:
        domain = load_json(SHARED_TEST_DOMAINS_DIR / filename)
        domain["id"] = f"test-{uuid.uuid4().hex}"
        set_document(DOMAINS_COLLECTION, domain["id"], stringify_coordinates(domain))
        created.append(domain["id"])
        return domain["id"]

    yield seed

    for domain_id in created:
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def seeded_point_cloud():
    """Seed a pending 3DEP point cloud, cleaning up its doc and artifacts."""
    created: list[str] = []

    def seed(domain_id: str, source: dict | None = None) -> str:
        point_cloud_id = f"test-{uuid.uuid4().hex}"
        set_document(
            POINT_CLOUDS_COLLECTION,
            point_cloud_id,
            {
                "id": point_cloud_id,
                "checksum": uuid.uuid4().hex,
                "domain_id": domain_id,
                "type": "als",
                "name": "Integration test cloud",
                "description": "",
                "status": "pending",
                "progress": None,
                "source": source or {"name": "3dep"},
                "georeference": None,
                "summary": None,
                "error": None,
                "tags": [],
                "owner_id": "test-owner",
            },
        )
        created.append(point_cloud_id)
        return point_cloud_id

    yield seed

    for point_cloud_id in created:
        from lakitu.storage import cloud_path

        directory = cloud_path(point_cloud_id).rsplit("/", 1)[0]
        if exists(directory):
            delete_directory(directory)
        delete_document(POINT_CLOUDS_COLLECTION, point_cloud_id)


@pytest.fixture
def read_point_cloud():
    """Read a point cloud document back from Firestore."""

    def read(point_cloud_id: str) -> dict:
        _, snapshot = get_document(POINT_CLOUDS_COLLECTION, point_cloud_id)
        return snapshot.to_dict()

    return read


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
