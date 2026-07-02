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

import pytest
from api.quota import Quotas
from google.cloud.firestore_v1.base_query import FieldFilter

from lib.config import FEATURES_COLLECTION
from tests.fixtures import make_feature_data

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
