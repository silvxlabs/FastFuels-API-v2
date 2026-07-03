"""
Integration tests for create-time quota enforcement (api.quota).

Requires a running local API server (http://127.0.0.1:8080) and Firestore —
same setup as the other router tests. Active (pending) resource docs are seeded
directly in Firestore, then the create endpoint is asserted to reject with 429
once the active-jobs limit is reached.

Features are the exercised resource: they have the simplest create (only a
domain is required, no source resource) and a default active limit of 10. The
active-jobs count is owner-wide, so each test starts from a clean slate of the
owner's feature docs to keep the counts exact.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from api.quota import Quotas
from google.cloud.firestore_v1.base_query import FieldFilter
from httpx import Client

from lib.config import (
    DOMAINS_COLLECTION,
    FEATURES_COLLECTION,
    KEYS_COLLECTION,
    USERS_COLLECTION,
)
from tests.conftest import TEST_URL
from tests.fixtures import make_domain_data, make_feature_data, make_key_data

FEATURE_ACTIVE_LIMIT = Quotas().max_active_features
ROAD_ROUTE = "/domains/{domain_id}/features/road/osm"


@pytest.fixture(autouse=True)
def mock_create_task():
    """Match the existing feature-router tests: don't enqueue a real background
    job when a create is allowed through."""
    with patch("api.resources.features.road.router.create_http_task_async") as mock:
        yield mock


@pytest.fixture
def seed_features(firestore_client, test_owner_id):
    """Start from a clean slate of the owner's feature docs, provide a seeder,
    and remove everything the test seeded or created on teardown."""

    def _clear():
        query = firestore_client.collection(FEATURES_COLLECTION).where(
            filter=FieldFilter("owner_id", "==", test_owner_id)
        )
        for doc in query.stream():
            doc.reference.delete()

    def _seed(count, status, domain_id):
        for _ in range(count):
            data = make_feature_data(domain_id=domain_id, status=status)
            firestore_client.collection(FEATURES_COLLECTION).document(data["id"]).set(
                data
            )

    _clear()
    yield _seed
    _clear()


class TestActiveJobQuota:
    def test_over_active_limit_returns_429(
        self, client, domain_for_testing, seed_features
    ):
        """At the active-jobs limit, a create is rejected with a structured 429
        carrying Retry-After and the correct quota/current/limit."""
        domain_id = domain_for_testing["id"]
        seed_features(FEATURE_ACTIVE_LIMIT, "pending", domain_id)

        response = client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "60"
        detail = response.json()["detail"]
        assert detail["reason"] == "QUOTA_EXCEEDED"
        assert detail["quota"] == "max_active_features"
        assert detail["current"] == FEATURE_ACTIVE_LIMIT
        assert detail["limit"] == FEATURE_ACTIVE_LIMIT

    def test_running_also_counts_as_active(
        self, client, domain_for_testing, seed_features
    ):
        """`running` jobs count toward the active limit, not just `pending`."""
        domain_id = domain_for_testing["id"]
        seed_features(FEATURE_ACTIVE_LIMIT, "running", domain_id)

        response = client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 429
        assert response.json()["detail"]["quota"] == "max_active_features"

    def test_completed_and_failed_do_not_count(
        self, client, domain_for_testing, seed_features, firestore_client
    ):
        """Inactive (completed/failed) docs do not count toward the active
        limit, so a create is allowed even well past the limit in raw count."""
        domain_id = domain_for_testing["id"]
        seed_features(FEATURE_ACTIVE_LIMIT, "completed", domain_id)
        seed_features(FEATURE_ACTIVE_LIMIT, "failed", domain_id)

        response = client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 201
        # The teardown clears all owner feature docs, including this one.

    def test_under_active_limit_is_allowed(
        self, client, domain_for_testing, seed_features
    ):
        """One below the limit still creates successfully."""
        domain_id = domain_for_testing["id"]
        seed_features(FEATURE_ACTIVE_LIMIT - 1, "pending", domain_id)

        response = client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 201


@pytest.fixture
def owner_env(firestore_client):
    """Build isolated owners, each with a personal key, a domain, and an optional
    users-v2 quota doc, so every test resolves a fresh (uncached) quota. Removes
    all seeded docs and closes all clients on teardown.
    """
    created: list[tuple[str, str]] = []
    clients: list[Client] = []

    def _make(users_doc: dict | None = None, features: dict | None = None):
        owner_id = f"test-{uuid4().hex}"

        key = make_key_data(owner_id=owner_id, scopes=["read", "write"])
        secret = key.pop("_test_secret")
        firestore_client.collection(KEYS_COLLECTION).document(key["id"]).set(key)
        created.append((KEYS_COLLECTION, key["id"]))

        domain = make_domain_data(owner_id=owner_id)
        firestore_client.collection(DOMAINS_COLLECTION).document(domain["id"]).set(
            domain
        )
        created.append((DOMAINS_COLLECTION, domain["id"]))

        if users_doc is not None:
            firestore_client.collection(USERS_COLLECTION).document(owner_id).set(
                users_doc
            )
            created.append((USERS_COLLECTION, owner_id))

        for feature_status, count in (features or {}).items():
            for _ in range(count):
                feat = make_feature_data(
                    domain_id=domain["id"], owner_id=owner_id, status=feature_status
                )
                firestore_client.collection(FEATURES_COLLECTION).document(
                    feat["id"]
                ).set(feat)
                created.append((FEATURES_COLLECTION, feat["id"]))

        owner_client = Client(base_url=TEST_URL, headers={"API-KEY": secret})
        clients.append(owner_client)
        return owner_client, domain["id"]

    yield _make

    for owner_client in clients:
        owner_client.close()
    for collection, doc_id in created:
        firestore_client.collection(collection).document(doc_id).delete()


class TestOwnerQuotaConfig:
    """Phase 2: resolve_quotas reads the owner's users-v2 doc (tier / overrides).

    Each test uses a brand-new owner so the server's quota cache has no prior
    entry; all state is seeded before the first request as that owner.
    """

    def test_override_lowers_active_limit(self, owner_env):
        """A quota_overrides entry changes the enforced limit."""
        owner_client, domain_id = owner_env(
            users_doc={"quota_overrides": {"max_active_features": 1}},
            features={"pending": 1},
        )
        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert response.status_code == 429
        detail = response.json()["detail"]
        assert detail["quota"] == "max_active_features"
        assert detail["limit"] == 1
        assert detail["current"] == 1

    def test_malformed_doc_falls_back_to_defaults(self, owner_env):
        """A malformed owner doc degrades to the default limits — never a 500."""
        owner_client, domain_id = owner_env(
            users_doc={
                "tier": "bogus",
                "quota_overrides": {"not_a_quota": 5, "max_active_features": "x"},
            },
            features={"pending": FEATURE_ACTIVE_LIMIT},
        )
        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert response.status_code == 429
        detail = response.json()["detail"]
        assert detail["quota"] == "max_active_features"
        assert detail["limit"] == FEATURE_ACTIVE_LIMIT

    def test_suspended_blocks_create_but_allows_read_and_delete(self, owner_env):
        """The suspended tier zeroes create limits while GET / DELETE keep working."""
        owner_client, domain_id = owner_env(users_doc={"tier": "suspended"})

        create = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert create.status_code == 429
        assert create.json()["detail"]["quota"] == "max_active_features"

        assert owner_client.get(f"/domains/{domain_id}").status_code == 200
        assert owner_client.delete(f"/domains/{domain_id}").status_code == 204
