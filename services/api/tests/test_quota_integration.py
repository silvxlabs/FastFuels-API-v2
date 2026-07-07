"""
Integration tests for create-time quota enforcement (api.quota).

Requires a running local API server (http://127.0.0.1:8080) and Firestore —
same setup as the other router tests. Active (pending) resource docs are seeded
directly in Firestore, then the create endpoint is asserted to reject with 429
once the active-jobs limit is reached.

Features are the exercised resource: they have the simplest create (only a
domain is required, no source resource) and a default active limit of 10. The
active-jobs count is owner-wide, so every test uses a fresh, isolated owner (see
``owner_env``) — never the shared test owner — to keep the counts exact and
independent of that owner's tier.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from api.quota import Quotas
from api.resources.domains.examples import EXAMPLE_WGS84_DEFAULT
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


class TestActiveJobQuota:
    """Active-jobs limit, exercised on a fresh isolated owner (standard limits).

    Uses ``owner_env`` rather than the shared test owner so the assertions never
    depend on that owner's tier — the shared owner is quota-exempt, which would
    otherwise let an over-limit create through.
    """

    def test_over_active_limit_returns_429(self, owner_env):
        """At the active-jobs limit, a create is rejected with a structured 429
        carrying Retry-After and the correct quota/current/limit."""
        owner_client, domain_id = owner_env(features={"pending": FEATURE_ACTIVE_LIMIT})

        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "60"
        detail = response.json()["detail"]
        assert detail["reason"] == "QUOTA_EXCEEDED"
        assert detail["quota"] == "max_active_features"
        assert detail["current"] == FEATURE_ACTIVE_LIMIT
        assert detail["limit"] == FEATURE_ACTIVE_LIMIT

    def test_running_also_counts_as_active(self, owner_env):
        """`running` jobs count toward the active limit, not just `pending`."""
        owner_client, domain_id = owner_env(features={"running": FEATURE_ACTIVE_LIMIT})

        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 429
        assert response.json()["detail"]["quota"] == "max_active_features"

    def test_completed_and_failed_do_not_count(self, owner_env):
        """Inactive (completed/failed) docs do not count toward the active
        limit, so a create is allowed even well past the limit in raw count."""
        owner_client, domain_id = owner_env(
            features={"completed": FEATURE_ACTIVE_LIMIT, "failed": FEATURE_ACTIVE_LIMIT}
        )

        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )

        assert response.status_code == 201

    def test_under_active_limit_is_allowed(self, owner_env):
        """One below the limit still creates successfully."""
        owner_client, domain_id = owner_env(
            features={"pending": FEATURE_ACTIVE_LIMIT - 1}
        )

        response = owner_client.post(
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

    def _make(
        users_doc: dict | None = None,
        features: dict | None = None,
        feature_size_bytes: int | None = None,
    ):
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
                if feature_size_bytes is not None:
                    feat["size_bytes"] = feature_size_bytes
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


class TestCountAndStorageQuota:
    """Phase 3: total-count and per-type storage limits.

    Fresh owner per test (uncached quotas). The feature cases exercise the
    combined count + sum(size_bytes) aggregation, which needs an
    (owner_id, size_bytes) composite index on features-v2; a missing index
    surfaces here as a 500 rather than the expected 429.
    """

    def test_count_over_limit_returns_429_without_retry_after(self, owner_env):
        """A completed resource doesn't count as an active job, so the
        total-count limit (not the active-jobs limit) is what rejects."""
        owner_client, domain_id = owner_env(
            users_doc={"quota_overrides": {"max_features": 1}},
            features={"completed": 1},
        )
        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert response.status_code == 429
        assert "Retry-After" not in response.headers
        detail = response.json()["detail"]
        assert detail["quota"] == "max_features"
        assert detail["limit"] == 1
        assert detail["current"] == 1

    def test_storage_over_limit_returns_429_without_retry_after(self, owner_env):
        """Summed size_bytes over the per-type storage limit rejects the create."""
        owner_client, domain_id = owner_env(
            users_doc={"quota_overrides": {"max_feature_storage_bytes": 1000}},
            features={"completed": 1},
            feature_size_bytes=2000,
        )
        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert response.status_code == 429
        assert "Retry-After" not in response.headers
        detail = response.json()["detail"]
        assert detail["quota"] == "max_feature_storage_bytes"
        assert detail["limit"] == 1000
        assert detail["current"] == 2000

    def test_domain_count_over_limit_returns_429(self, owner_env):
        """owner_env already creates one domain, so max_domains=1 rejects the
        next domain create — proving domain create is wired to the quota check."""
        owner_client, _ = owner_env(users_doc={"quota_overrides": {"max_domains": 1}})
        response = owner_client.post("/domains", json=EXAMPLE_WGS84_DEFAULT)
        assert response.status_code == 429
        assert "Retry-After" not in response.headers
        detail = response.json()["detail"]
        assert detail["quota"] == "max_domains"
        assert detail["limit"] == 1
        assert detail["current"] == 1

    def test_active_jobs_rejection_keeps_retry_after(self, owner_env):
        """Regression: the active-jobs limit still carries Retry-After with the
        new count/storage checks in place."""
        owner_client, domain_id = owner_env(
            users_doc={"quota_overrides": {"max_active_features": 1}},
            features={"pending": 1},
        )
        response = owner_client.post(
            ROAD_ROUTE.format(domain_id=domain_id), json={"type": "road"}
        )
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "60"
        assert response.json()["detail"]["quota"] == "max_active_features"
