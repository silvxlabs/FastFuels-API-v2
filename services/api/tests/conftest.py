"""
Pytest configuration for API tests.

Provides shared fixtures:
- cleanup_stale_test_data: Autouse session fixture that removes leftover test data
- client: HTTP client for API requests
- firestore_client: Firestore client for direct database operations
- domain_for_testing: A domain owned by test-owner for tests that need a domain
- domain_with_different_owner: A domain owned by different-owner for ownership tests
"""

import os

import gcsfs
import pytest
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from httpx import Client

from lib.config import (
    APPLICATIONS_COLLECTION,
    DEV_API_KEY,
    DEV_OWNER_ID,
    DOMAINS_COLLECTION,
    EXPORTS_BUCKET,
    EXPORTS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    KEYS_COLLECTION,
)
from tests.fixtures import make_application_data, make_domain_data

TEST_URL = os.getenv("TEST_API_URL", "http://127.0.0.1:8080")
print(f"\nRunning tests at {TEST_URL}\n")

TEST_KEY = os.getenv("TEST_API_KEY", DEV_API_KEY)
HEADERS = {"API-KEY": TEST_KEY}
CLIENT_ARGS = {"base_url": TEST_URL, "headers": HEADERS}

TEST_OWNER_IDS = [DEV_OWNER_ID, "different-owner"]
COLLECTIONS = [
    DOMAINS_COLLECTION,
    GRIDS_COLLECTION,
    EXPORTS_COLLECTION,
    APPLICATIONS_COLLECTION,
    KEYS_COLLECTION,
]


def _delete_docs_by_owner(fs_client, collection: str, owner_id: str) -> list[str]:
    """Delete all docs in a collection matching owner_id. Returns deleted doc IDs."""
    query = fs_client.collection(collection).where(
        filter=FieldFilter("owner_id", "==", owner_id)
    )
    docs = list(query.stream())
    if not docs:
        return []

    doc_ids = [doc.id for doc in docs]
    for i in range(0, len(docs), 500):
        batch = fs_client.batch()
        for doc in docs[i : i + 500]:
            batch.delete(doc.reference)
        batch.commit()
    return doc_ids


@pytest.fixture(scope="session", autouse=True)
def cleanup_stale_test_data():
    """Remove leftover test data from previous runs before tests start.

    Queries each collection for documents owned by test users and deletes
    them. For grids, also removes the corresponding GCS directories.
    """
    fs_client = firestore.Client()
    gcs = gcsfs.GCSFileSystem()

    # Collect grid and export IDs before deleting so we can clean up GCS
    grid_ids = []
    export_ids = []
    for owner_id in TEST_OWNER_IDS:
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
        count = 0
        for owner_id in TEST_OWNER_IDS:
            count += len(_delete_docs_by_owner(fs_client, collection, owner_id))
        if count:
            deleted_counts[collection] = count

    # Also clean keys by creator_id (application keys have owner_id = app ID)
    for owner_id in TEST_OWNER_IDS:
        query = fs_client.collection(KEYS_COLLECTION).where(
            filter=FieldFilter("creator_id", "==", owner_id)
        )
        docs = list(query.stream())
        for i in range(0, len(docs), 500):
            batch = fs_client.batch()
            for doc in docs[i : i + 500]:
                batch.delete(doc.reference)
            batch.commit()

    if deleted_counts:
        print(f"\nCleaned up stale test data: {deleted_counts}")

    yield


@pytest.fixture(scope="session")
def client():
    """Session-scoped HTTP client for API tests."""
    with Client(**CLIENT_ARGS) as client:
        yield client


@pytest.fixture(scope="session")
def firestore_client():
    """Session-scoped Firestore client for direct database operations."""
    return firestore.Client()


@pytest.fixture(scope="session")
def domain_for_testing(firestore_client):
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
def application_for_testing(firestore_client):
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
