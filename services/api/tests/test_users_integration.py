"""
Integration tests for the quota read API (GET /users/me, GET /users/me/usage).

Requires a running local API server (http://127.0.0.1:8080) and Firestore, like
the other router tests. Each test builds a brand-new owner — a user or an
application — so the server's in-process quota cache has no prior entry, seeds a
known set of resources, and asserts the read endpoints reflect that state. All
seeded docs are removed on teardown.
"""

from uuid import uuid4

import pytest
from httpx import Client

from lib.config import (
    APPLICATIONS_COLLECTION,
    DOMAINS_COLLECTION,
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    KEYS_COLLECTION,
    USERS_COLLECTION,
)
from tests.conftest import TEST_URL
from tests.fixtures import (
    make_application_data,
    make_domain_data,
    make_feature_data,
    make_grid_data,
    make_key_data,
)


@pytest.fixture
def owner_env(firestore_client):
    """Build an isolated owner (user or application) with a key, a parent domain,
    an optional owner config doc, and optional seeded resources. Returns an
    authenticated client, the owner id, and the domain id. Removes every seeded
    doc and closes every client on teardown.
    """
    created: list[tuple[str, str]] = []
    clients: list[Client] = []

    def _make(
        *,
        access: str = "personal",
        config: dict | None = None,  # tier / quota_overrides on the owner doc
        grids: dict | None = None,  # status -> count
        grid_size_bytes: int | None = None,
        features: dict | None = None,  # status -> count
    ):
        owner_id = f"test-{uuid4().hex}"

        if access == "application":
            app = make_application_data(owner_id=f"test-human-{uuid4().hex}")
            app["id"] = owner_id
            if config:
                app.update(config)
            firestore_client.collection(APPLICATIONS_COLLECTION).document(owner_id).set(
                app
            )
            created.append((APPLICATIONS_COLLECTION, owner_id))
            key = make_key_data(
                owner_id=owner_id,
                creator_id=f"test-human-{uuid4().hex}",
                access="application",
                application_id=owner_id,
                scopes=["read", "write"],
            )
        else:
            if config is not None:
                firestore_client.collection(USERS_COLLECTION).document(owner_id).set(
                    config
                )
                created.append((USERS_COLLECTION, owner_id))
            key = make_key_data(owner_id=owner_id, scopes=["read", "write"])

        secret = key.pop("_test_secret")
        firestore_client.collection(KEYS_COLLECTION).document(key["id"]).set(key)
        created.append((KEYS_COLLECTION, key["id"]))

        domain = make_domain_data(owner_id=owner_id)
        firestore_client.collection(DOMAINS_COLLECTION).document(domain["id"]).set(
            domain
        )
        created.append((DOMAINS_COLLECTION, domain["id"]))

        for grid_status, count in (grids or {}).items():
            for _ in range(count):
                grid = make_grid_data(
                    domain_id=domain["id"], owner_id=owner_id, status=grid_status
                )
                if grid_size_bytes is not None:
                    grid["size_bytes"] = grid_size_bytes
                firestore_client.collection(GRIDS_COLLECTION).document(grid["id"]).set(
                    grid
                )
                created.append((GRIDS_COLLECTION, grid["id"]))

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
        return owner_client, owner_id, domain["id"]

    yield _make

    for owner_client in clients:
        owner_client.close()
    for collection, doc_id in created:
        firestore_client.collection(collection).document(doc_id).delete()


class TestUsersMe:
    def test_personal_reports_user_kind_tier_and_overrides(self, owner_env):
        owner_client, owner_id, _ = owner_env(
            config={"tier": "standard", "quota_overrides": {"max_active_grids": 3}},
        )
        response = owner_client.get("/users/me")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == owner_id
        assert body["kind"] == "user"
        assert body["tier"] == "standard"
        # Overrides win; untouched fields keep their defaults.
        assert body["quotas"]["max_active_grids"] == 3
        assert body["quotas"]["max_grids"] == 1000

    def test_application_reports_application_kind_and_tier(self, owner_env):
        owner_client, owner_id, _ = owner_env(
            access="application", config={"tier": "application"}
        )
        response = owner_client.get("/users/me")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == owner_id
        assert body["kind"] == "application"
        assert body["tier"] == "application"
        assert body["quotas"]["max_active_grids"] == 100  # application preset
        assert body["quotas"]["resource_ttl_days"] is None  # never expires

    def test_unknown_stored_tier_reports_standard(self, owner_env):
        owner_client, _, _ = owner_env(config={"tier": "platinum"})
        response = owner_client.get("/users/me")
        assert response.status_code == 200
        assert response.json()["tier"] == "standard"


class TestUsersMeUsage:
    def test_usage_matches_seeded_state(self, owner_env):
        owner_client, _, _ = owner_env(
            grids={"completed": 3},
            grid_size_bytes=2000,
            features={"pending": 2},
        )
        response = owner_client.get("/users/me/usage")
        assert response.status_code == 200
        usage = response.json()

        assert usage["grids"]["total"] == {"usage": 3, "limit": 1000}
        assert usage["grids"]["active"]["usage"] == 0  # completed is not active
        assert usage["grids"]["storage"]["usage_bytes"] == 6000
        assert usage["grids"]["storage"]["limit_bytes"] == 50 * 2**30

        # A pending job counts toward both total and active.
        assert usage["features"]["total"]["usage"] == 2
        assert usage["features"]["active"]["usage"] == 2

        # Count-only types carry total alone.
        assert usage["domains"]["total"] == {"usage": 1, "limit": 50}
        assert set(usage["domains"]) == {"total"}
        assert usage["api_keys"]["total"] == {"usage": 1, "limit": 50}
        assert set(usage["api_keys"]) == {"total"}

        # Untouched types read zero.
        assert usage["exports"]["total"]["usage"] == 0
        assert usage["pointclouds"]["storage"]["usage_bytes"] == 0

        assert usage["lifecycle"]["resource_ttl_days"] == 180
        assert usage["lifecycle"]["failed_resource_ttl_days"] == 14
        assert usage["lifecycle"]["next_expiry_on"] is None

    def test_usage_is_scoped_to_the_application_owner(self, owner_env):
        owner_client, _, _ = owner_env(
            access="application",
            config={"tier": "application"},
            grids={"completed": 1},
            grid_size_bytes=1234,
        )
        response = owner_client.get("/users/me/usage")
        assert response.status_code == 200
        usage = response.json()
        assert usage["grids"]["total"]["usage"] == 1
        assert usage["grids"]["storage"]["usage_bytes"] == 1234
        assert usage["domains"]["total"]["usage"] == 1
        # The application key is metered by owner_id == application id.
        assert usage["api_keys"]["total"]["usage"] == 1
        # Application tier: general TTL is null (never expires); the limit
        # reflects the application preset.
        assert usage["grids"]["total"]["limit"] == 10_000
        assert usage["lifecycle"]["resource_ttl_days"] is None
        assert usage["lifecycle"]["failed_resource_ttl_days"] == 14
