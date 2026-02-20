"""
Pytest configuration for API tests.

Provides shared fixtures:
- test_owner_id: Lazy lookup of owner from pre-seeded test key in Firestore
- cleanup_stale_test_data: Removes leftover test data (triggered via client)
- client: HTTP client for API requests (depends on cleanup)
- firestore_client: Firestore client for direct database operations
- domain_for_testing: A domain owned by test-owner for tests that need a domain
- domain_with_different_owner: A domain owned by different-owner for ownership tests
"""

import os
import threading

import gcsfs
import pytest
from api.auth import hash_api_key
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from httpx import Client

from lib.config import (
    APPLICATIONS_COLLECTION,
    DOMAINS_COLLECTION,
    EXPORTS_BUCKET,
    EXPORTS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    KEYS_COLLECTION,
)
from tests import fixtures
from tests.fixtures import make_application_data, make_domain_data

TEST_URL = os.getenv("TEST_API_URL", "http://127.0.0.1:8080")
TEST_API_KEY = os.environ.get("TEST_API_KEY", "")
COLLECTIONS = [
    DOMAINS_COLLECTION,
    GRIDS_COLLECTION,
    EXPORTS_COLLECTION,
    APPLICATIONS_COLLECTION,
    KEYS_COLLECTION,
]


def _delete_docs_by_owner(
    fs_client, collection: str, owner_id: str, skip_ids: set[str] | None = None
) -> list[str]:
    """Delete all docs in a collection matching owner_id. Returns deleted doc IDs."""
    query = fs_client.collection(collection).where(
        filter=FieldFilter("owner_id", "==", owner_id)
    )
    docs = [doc for doc in query.stream() if not skip_ids or doc.id not in skip_ids]
    if not docs:
        return []

    doc_ids = [doc.id for doc in docs]
    for i in range(0, len(docs), 500):
        batch = fs_client.batch()
        for doc in docs[i : i + 500]:
            batch.delete(doc.reference)
        batch.commit()
    return doc_ids


@pytest.fixture(scope="session")
def test_owner_id():
    """Resolve the test owner from the pre-seeded key in Firestore."""
    key_hash = hash_api_key(TEST_API_KEY)
    doc = firestore.Client().collection(KEYS_COLLECTION).document(key_hash).get()
    if not doc.exists:
        raise RuntimeError("TEST_API_KEY is not valid.")
    owner_id = doc.to_dict()["owner_id"]
    fixtures.DEFAULT_OWNER_ID = owner_id
    return owner_id


@pytest.fixture(scope="session")
def cleanup_stale_test_data(test_owner_id):
    """Remove leftover test data from previous runs before tests start."""
    key_hash = hash_api_key(TEST_API_KEY)
    owner_ids = [test_owner_id, "different-owner"]

    fs_client = firestore.Client()
    gcs = gcsfs.GCSFileSystem()

    # Collect grid and export IDs before deleting so we can clean up GCS
    grid_ids = []
    export_ids = []
    for owner_id in owner_ids:
        query = fs_client.collection(GRIDS_COLLECTION).where(
            filter=FieldFilter("owner_id", "==", owner_id)
        )
        grid_ids.extend(doc.id for doc in query.stream())

        query = fs_client.collection(EXPORTS_COLLECTION).where(
            filter=FieldFilter("owner_id", "==", owner_id)
        )
        export_ids.extend(doc.id for doc in query.stream())

    # Delete GCS data for stale grids
    if grid_ids:
        gcs_paths = [f"{GRIDS_BUCKET}/{gid}" for gid in grid_ids]
        try:
            gcs.rm(gcs_paths, recursive=True)
        except Exception as e:
            print(f"\nWarning: Failed to clean up grid GCS data: {e}")

    # Delete GCS data for stale exports
    if export_ids:
        gcs_paths = [f"{EXPORTS_BUCKET}/{eid}" for eid in export_ids]
        try:
            gcs.rm(gcs_paths, recursive=True)
        except Exception as e:
            print(f"\nWarning: Failed to clean up export GCS data: {e}")

    # Delete Firestore documents
    deleted_counts = {}
    for collection in COLLECTIONS:
        skip_ids = {key_hash} if collection == KEYS_COLLECTION else None
        count = 0
        for owner_id in owner_ids:
            count += len(
                _delete_docs_by_owner(fs_client, collection, owner_id, skip_ids)
            )
        if count:
            deleted_counts[collection] = count

    # Also clean keys by creator_id (application keys have owner_id = app ID)
    for owner_id in owner_ids:
        query = fs_client.collection(KEYS_COLLECTION).where(
            filter=FieldFilter("creator_id", "==", owner_id)
        )
        docs = [doc for doc in query.stream() if doc.id != key_hash]
        for i in range(0, len(docs), 500):
            batch = fs_client.batch()
            for doc in docs[i : i + 500]:
                batch.delete(doc.reference)
            batch.commit()

    if deleted_counts:
        print(f"\nCleaned up stale test data: {deleted_counts}")

    yield


@pytest.fixture(scope="session")
def client(cleanup_stale_test_data):
    """Session-scoped HTTP client for API tests."""
    headers = {"API-KEY": TEST_API_KEY}
    with Client(base_url=TEST_URL, headers=headers, timeout=30.0) as client:
        yield client


@pytest.fixture(autouse=True, scope="session")
def _warmup_api(client):
    """Fire a warmup request to absorb Cloud Run cold start while unit tests run."""
    thread = threading.Thread(
        target=lambda: client.get("/", timeout=120.0),
        daemon=True,
    )
    print("Warming up API")
    thread.start()
    yield
    thread.join(timeout=0)


@pytest.fixture(scope="session")
def firestore_client():
    """Session-scoped Firestore client for direct database operations."""
    return firestore.Client()


@pytest.fixture(scope="session")
def domain_for_testing(firestore_client, test_owner_id):
    """A domain owned by test-owner, available for any test that needs a domain."""
    domain_data = make_domain_data(name="Shared Test Domain")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def domain_with_different_owner(firestore_client):
    """A domain owned by different-owner for ownership validation tests."""
    domain_data = make_domain_data(
        owner_id="different-owner",
        name="Other User's Domain",
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def application_for_testing(firestore_client, test_owner_id):
    """An application owned by test-owner, available for any test that needs one."""
    app_data = make_application_data(name="Shared Test Application")
    doc_ref = firestore_client.collection(APPLICATIONS_COLLECTION).document(
        app_data["id"]
    )
    doc_ref.set(app_data)
    yield app_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def application_with_different_owner(firestore_client):
    """An application owned by different-owner for ownership validation tests."""
    app_data = make_application_data(
        owner_id="different-owner",
        name="Other User's Application",
    )
    doc_ref = firestore_client.collection(APPLICATIONS_COLLECTION).document(
        app_data["id"]
    )
    doc_ref.set(app_data)
    yield app_data
    doc_ref.delete()
