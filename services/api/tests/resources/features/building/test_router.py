"""
Integration tests for api/v2/resources/features/building/router.py

Tests the POST /domains/{domain_id}/features/building/overture endpoint.
These tests make real HTTP requests to the API and interact with Firestore.
"""

import pytest
from api.resources.features.building.examples import BUILDING_EXAMPLE_VALUES

from lib.config import FEATURES_COLLECTION


@pytest.fixture
def created_features(firestore_client):
    """Collects created feature ids and deletes them on teardown."""
    ids = []
    yield ids
    for feature_id in ids:
        firestore_client.collection(FEATURES_COLLECTION).document(feature_id).delete()


class TestCreateOvertureBuildingFeature:
    """Test the POST /domains/{domain_id}/features/building/overture endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features/building/overture"

    def create(self, client, domain_id, body, created_features):
        response = client.post(self.route(domain_id), json=body)
        if response.status_code == 201:
            created_features.append(response.json()["id"])
        return response

    def test_minimal_request_creates_feature(
        self, client, domain_for_testing, created_features
    ):
        """Minimal request creates a feature document in pending state."""
        response = self.create(
            client, domain_for_testing["id"], {"type": "building"}, created_features
        )

        assert response.status_code == 201

        data = response.json()
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "building"
        assert data["status"] == "pending"
        assert data["georeference"] is None
        assert data["attributions"] == []
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["source"] == {
            "product": "overture",
            "description": "Overture Maps building footprints",
            "extent_buffer_m": 0,
            "release": None,
        }

    def test_feature_document_persisted(
        self, client, firestore_client, domain_for_testing, created_features
    ):
        """The feature document is written to Firestore."""
        response = self.create(
            client, domain_for_testing["id"], {"type": "building"}, created_features
        )
        assert response.status_code == 201

        snapshot = (
            firestore_client.collection(FEATURES_COLLECTION)
            .document(response.json()["id"])
            .get()
        )
        assert snapshot.exists
        doc = snapshot.to_dict()
        assert doc["type"] == "building"
        assert doc["status"] == "pending"
        assert doc["domain_id"] == domain_for_testing["id"]
        assert doc["source"]["product"] == "overture"

    def test_request_with_extent_buffer_m(
        self, client, domain_for_testing, created_features
    ):
        """extent_buffer_m is persisted into the source dict."""
        response = self.create(
            client,
            domain_for_testing["id"],
            {"type": "building", "extent_buffer_m": 25},
            created_features,
        )

        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_m"] == 25

    @pytest.mark.parametrize(
        "body",
        [
            {"type": "building", "extent_buffer_m": 101},
            {"type": "building", "extent_buffer_m": -1},
            {"type": "road"},
        ],
    )
    def test_invalid_request_rejected(
        self, client, domain_for_testing, created_features, body
    ):
        """Out-of-range buffers and the wrong type return 422."""
        response = self.create(client, domain_for_testing["id"], body, created_features)

        assert response.status_code == 422

    def test_request_with_metadata(self, client, domain_for_testing, created_features):
        """Request accepts name, description, and tags."""
        request_body = {
            "type": "building",
            "name": "Project Alpha Buildings",
            "description": "Building footprints",
            "tags": ["overture", "buildings", "alpha"],
        }

        response = self.create(
            client, domain_for_testing["id"], request_body, created_features
        )

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Project Alpha Buildings"
        assert data["description"] == "Building footprints"
        assert data["tags"] == ["overture", "buildings", "alpha"]

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"), json={"type": "building"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]), json={"type": "building"}
        )

        assert response.status_code == 404

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, created_features
    ):
        """Response should not expose the owner_id field to the client."""
        response = self.create(
            client, domain_for_testing["id"], {"type": "building"}, created_features
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    def test_listed_with_type_filter(
        self, client, domain_for_testing, created_features
    ):
        """GET /features?type=building returns the created feature."""
        response = self.create(
            client, domain_for_testing["id"], {"type": "building"}, created_features
        )
        assert response.status_code == 201
        feature_id = response.json()["id"]

        listed = client.get(
            f"/domains/{domain_for_testing['id']}/features",
            params={"type": "building", "size": 1000},
        )
        assert listed.status_code == 200
        features = listed.json()["features"]
        assert all(f["type"] == "building" for f in features)
        assert feature_id in {f["id"] for f in features}

    @pytest.mark.parametrize("example_name,example_value", BUILDING_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_for_testing, created_features, example_name, example_value
    ):
        """Each documented example payload should successfully create a feature."""
        response = self.create(
            client, domain_for_testing["id"], example_value, created_features
        )

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["type"] == "building"
        assert data["source"]["product"] == "overture"
