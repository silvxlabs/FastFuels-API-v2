"""
Integration tests for api/v2/resources/features/router.py

Tests the standard CRUD endpoints (GET, LIST, PATCH, DELETE).
"""

from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

import pytest

from lib.config import FEATURES_COLLECTION


def make_feature_data(
    domain_id: str,
    owner_id: str = "test-owner",
    type: str = "road",
    name: str = "",
    product: str = "osm",
    tags: list[str] = None,
) -> dict:
    """Helper to generate valid feature document dictionaries."""
    return {
        "id": f"test-feat-{uuid4().hex[:8]}",
        "domain_id": domain_id,
        "owner_id": owner_id,
        "type": type,
        "name": name,
        "description": "",
        "status": "completed",
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "source": {"product": product},
        "tags": tags or [],
        "georeference": {"crs": "EPSG:4326", "bounds": [-120.0, 40.0, -119.0, 41.0]},
    }


# Mocks
# Mock out GCS blob deletion so background tasks don't hang the test server
@pytest.fixture(autouse=True)
def mock_gcs_delete():
    with patch("api.resources.features.router.delete_document_async") as mock:
        yield mock


# Fixtures


@pytest.fixture(scope="session")
def feature_in_firestore(firestore_client, test_owner_id, domain_for_testing):
    """Create a feature document directly in Firestore, yield it, then delete."""
    feature_data = make_feature_data(
        domain_id=domain_for_testing["id"],
        owner_id=test_owner_id,
        name="Test Feature for GET",
        tags=["test", "fixture"],
    )
    doc_ref = firestore_client.collection(FEATURES_COLLECTION).document(
        feature_data["id"]
    )
    doc_ref.set(feature_data)
    yield feature_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def feature_with_different_owner(firestore_client, domain_with_different_owner):
    """Create a feature owned by a different user for ownership validation tests."""
    feature_data = make_feature_data(
        domain_id=domain_with_different_owner["id"],
        owner_id="different-owner",
        name="Other User's Feature",
    )
    doc_ref = firestore_client.collection(FEATURES_COLLECTION).document(
        feature_data["id"]
    )
    doc_ref.set(feature_data)
    yield feature_data
    doc_ref.delete()


# GET /domains/{domain_id}/features/{feature_id} Tests


class TestGetFeature:
    """Test the GET /domains/{domain_id}/features/{feature_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features"

    def test_get_existing_feature(
        self, client, feature_in_firestore, domain_for_testing
    ):
        feature_id = feature_in_firestore["id"]
        response = client.get(f"{self.route(domain_for_testing['id'])}/{feature_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == feature_id
        assert data["name"] == "Test Feature for GET"
        assert data["tags"] == ["test", "fixture"]
        assert "source" in data
        assert "georeference" in data

    def test_get_nonexistent_feature_returns_404(self, client, domain_for_testing):
        fake_id = "00000000000000000000000000000000"
        response = client.get(f"{self.route(domain_for_testing['id'])}/{fake_id}")
        assert response.status_code == 404

    def test_get_feature_wrong_owner_returns_404(
        self, client, feature_with_different_owner, domain_with_different_owner
    ):
        feature_id = feature_with_different_owner["id"]
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{feature_id}"
        )
        assert response.status_code == 404


# GET /domains/-/features (Wildcard List) Tests


class TestListFeaturesWildcard:
    """Test GET /domains/-/features returns features across all domains."""

    @pytest.fixture(scope="class")
    def features_across_domains(
        self, firestore_client, test_owner_id, domain_for_testing, second_domain
    ):
        features = []
        for domain_id in [domain_for_testing["id"], second_domain["id"]]:
            feat_data = make_feature_data(
                domain_id=domain_id, owner_id=test_owner_id, name=f"Feat in {domain_id}"
            )
            firestore_client.collection(FEATURES_COLLECTION).document(
                feat_data["id"]
            ).set(feat_data)
            features.append(feat_data)
        yield features
        for feat in features:
            firestore_client.collection(FEATURES_COLLECTION).document(
                feat["id"]
            ).delete()

    def route(self):
        return "/domains/-/features"

    def test_wildcard_returns_features_from_all_domains(
        self, client, features_across_domains
    ):
        response = client.get(self.route())
        assert response.status_code == 200

        feature_ids = [f["id"] for f in response.json()["features"]]
        for feat in features_across_domains:
            assert feat["id"] in feature_ids

    def test_wildcard_excludes_other_users_features(
        self, client, feature_with_different_owner
    ):
        response = client.get(self.route())
        feature_ids = [f["id"] for f in response.json()["features"]]
        assert feature_with_different_owner["id"] not in feature_ids


# GET /domains/{domain_id}/features (List) Tests


class TestListFeatures:
    """Test the GET /domains/{domain_id}/features endpoint (list all features)."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features"

    @pytest.fixture(scope="class")
    def features_for_listing(self, firestore_client, test_owner_id, domain_for_testing):
        features = []
        configs = [
            {
                "name": "Alpha Road",
                "type": "road",
                "product": "osm",
                "tags": ["list-test"],
            },
            {
                "name": "Beta Water",
                "type": "water",
                "product": "osm",
                "tags": ["list-test", "hydro"],
            },
            {
                "name": "Gamma Layer",
                "type": "layerset",
                "product": "custom",
                "tags": ["custom"],
            },
        ]

        for config in configs:
            feat_data = make_feature_data(
                domain_id=domain_for_testing["id"], owner_id=test_owner_id, **config
            )
            firestore_client.collection(FEATURES_COLLECTION).document(
                feat_data["id"]
            ).set(feat_data)
            features.append(feat_data)

        yield features

        for feat in features:
            firestore_client.collection(FEATURES_COLLECTION).document(
                feat["id"]
            ).delete()

    def test_list_returns_paginated_response(self, client, domain_for_testing):
        response = client.get(self.route(domain_for_testing["id"]))
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "current_page" in data
        assert "total_items" in data

    def test_list_sorting_by_name(
        self, client, features_for_listing, domain_for_testing
    ):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=ascending"
        )
        data = response.json()
        names = [
            f["name"]
            for f in data["features"]
            if f["name"] in ["Alpha Road", "Beta Water", "Gamma Layer"]
        ]
        assert names == sorted(names)

    def test_list_filter_by_type(
        self, client, features_for_listing, domain_for_testing
    ):
        response = client.get(f"{self.route(domain_for_testing['id'])}?type=water")
        data = response.json()
        for feat in data["features"]:
            assert feat["type"] == "water"

    def test_list_filter_by_product(
        self, client, features_for_listing, domain_for_testing
    ):
        response = client.get(f"{self.route(domain_for_testing['id'])}?product=custom")
        data = response.json()
        for feat in data["features"]:
            assert feat["source"]["product"] == "custom"

    def test_list_filter_by_tag(self, client, features_for_listing, domain_for_testing):
        response = client.get(f"{self.route(domain_for_testing['id'])}?tag=list-test")
        data = response.json()
        for feat in data["features"]:
            assert "list-test" in feat["tags"]


# PATCH /domains/{domain_id}/features/{feature_id} Tests


class TestUpdateFeature:
    """Test the PATCH /domains/{domain_id}/features/{feature_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features"

    @pytest.fixture(scope="class")
    def feature_for_update(self, firestore_client, test_owner_id, domain_for_testing):
        feat_data = make_feature_data(
            domain_id=domain_for_testing["id"],
            owner_id=test_owner_id,
            name="Original Name",
            tags=["original"],
        )
        doc_ref = firestore_client.collection(FEATURES_COLLECTION).document(
            feat_data["id"]
        )
        doc_ref.set(feat_data)
        yield feat_data
        doc_ref.delete()

    def test_update_name_and_tags(self, client, feature_for_update, domain_for_testing):
        feat_id = feature_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{feat_id}",
            json={"name": "Updated Name", "tags": ["new-tag"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["tags"] == ["new-tag"]

    def test_update_preserves_immutable_fields(
        self, client, feature_for_update, domain_for_testing
    ):
        feat_id = feature_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{feat_id}",
            json={"name": "Immutable Test"},
        )
        data = response.json()
        assert data["id"] == feat_id
        assert data["domain_id"] == feature_for_update["domain_id"]
        assert data["type"] == feature_for_update["type"]


# DELETE /domains/{domain_id}/features/{feature_id} Tests


class TestDeleteFeature:
    """Test the DELETE /domains/{domain_id}/features/{feature_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features"

    @pytest.fixture(scope="function")
    def feature_for_delete(self, firestore_client, test_owner_id, domain_for_testing):
        feat_data = make_feature_data(
            domain_id=domain_for_testing["id"],
            owner_id=test_owner_id,
            name="Feature to Delete",
        )
        doc_ref = firestore_client.collection(FEATURES_COLLECTION).document(
            feat_data["id"]
        )
        doc_ref.set(feat_data)
        yield feat_data
        if doc_ref.get().exists:
            doc_ref.delete()

    def test_delete_existing_feature(
        self, client, feature_for_delete, firestore_client, domain_for_testing
    ):
        feat_id = feature_for_delete["id"]
        response = client.delete(f"{self.route(domain_for_testing['id'])}/{feat_id}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify feature is deleted from DB
        doc = firestore_client.collection(FEATURES_COLLECTION).document(feat_id).get()
        assert not doc.exists

    def test_delete_twice_returns_404_second_time(
        self, client, feature_for_delete, domain_for_testing
    ):
        feat_id = feature_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{feat_id}")
        assert response.status_code == 204

        response_2 = client.delete(f"{self.route(domain_for_testing['id'])}/{feat_id}")
        assert response_2.status_code == 404
