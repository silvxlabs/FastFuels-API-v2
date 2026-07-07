"""
Pytest configuration for API tests.

Provides shared fixtures:
- test_owner_id: Lazy lookup of owner from pre-seeded test key in Firestore
- client: HTTP client for API requests
- firestore_client: Firestore client for direct database operations
- domain_for_testing: A domain owned by test-owner for tests that need a domain
- domain_with_different_owner: A domain owned by different-owner for ownership tests

Leaked ``test-``-prefixed resources are reclaimed nightly by walle (by id
prefix, with an age guard), so there is no pre-run owner sweep here. A broad
``owner_id ==`` delete with no age guard raced concurrent suites (#353); the
suite tolerates accumulation (list assertions use ``>=`` / membership).
"""

import os
import threading

import pytest
from api.auth import hash_api_key
from google.cloud import firestore
from httpx import Client

from lib.config import (
    APPLICATIONS_COLLECTION,
    DOMAINS_COLLECTION,
    KEYS_COLLECTION,
)
from tests import fixtures
from tests.fixtures import make_application_data, make_domain_data

TEST_URL = os.getenv("TEST_API_URL", "http://127.0.0.1:8080")
TEST_API_KEY = os.environ.get("TEST_API_KEY", "")


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
def client(test_owner_id):
    """Session-scoped HTTP client for API tests.

    Depends on ``test_owner_id`` so ``fixtures.DEFAULT_OWNER_ID`` is populated
    before any test seeds data through the shared client.
    """
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
def second_domain(firestore_client, test_owner_id):
    """An extra domain owned by test-owner, for tests that need multiple domains."""
    domain_data = make_domain_data(name="Extra Domain for Testing")
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
