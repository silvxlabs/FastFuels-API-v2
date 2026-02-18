"""
Integration tests for api/v2/resources/keys/router.py

These tests make real HTTP requests to the API and interact with Firestore.
Requires a running local API server (http://127.0.0.1:8080).
"""

import pytest
from api.auth import hash_api_key

from lib.config import KEYS_COLLECTION


class TestCreateKey:
    route = "/keys"

    def test_personal_key(self, client):
        response = client.post(
            self.route,
            json={"name": "Personal Test Key"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Personal Test Key"
        assert data["access"] == "personal"
        assert "secret" in data

    def test_secret_is_64_char_hex(self, client):
        response = client.post(self.route, json={"name": "Hex Check Key"})
        assert response.status_code == 201

        secret = response.json()["secret"]
        assert len(secret) == 64
        int(secret, 16)  # Should not raise — valid hex

    def test_id_is_sha256_of_secret(self, client):
        response = client.post(self.route, json={"name": "Hash Check Key"})
        assert response.status_code == 201

        data = response.json()
        expected_id = hash_api_key(data["secret"])
        assert data["id"] == expected_id

    def test_custom_scopes(self, client):
        response = client.post(
            self.route,
            json={"name": "RW Key", "scopes": ["read", "write"]},
        )
        assert response.status_code == 201
        assert response.json()["scopes"] == ["read", "write"]

    def test_custom_valid_days(self, client):
        response = client.post(
            self.route,
            json={"name": "Short Key", "valid_days": 7},
        )
        assert response.status_code == 201
        assert response.json()["valid_days"] == 7

    def test_application_key(self, client, application_for_testing):
        app_id = application_for_testing["id"]
        response = client.post(
            self.route,
            json={
                "name": "App Key",
                "access": "application",
                "application_id": app_id,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["access"] == "application"
        assert data["application_id"] == app_id

    def test_application_key_owner_is_app_id(self, client, application_for_testing):
        app_id = application_for_testing["id"]
        response = client.post(
            self.route,
            json={
                "name": "App Owner Key",
                "access": "application",
                "application_id": app_id,
            },
        )
        assert response.status_code == 201
        assert response.json()["owner_id"] == app_id

    def test_nonexistent_app_returns_404(self, client):
        response = client.post(
            self.route,
            json={
                "name": "Bad App Key",
                "access": "application",
                "application_id": "00000000000000000000000000000000",
            },
        )
        assert response.status_code == 404

    def test_unowned_app_returns_404(self, client, application_with_different_owner):
        app_id = application_with_different_owner["id"]
        response = client.post(
            self.route,
            json={
                "name": "Unowned App Key",
                "access": "application",
                "application_id": app_id,
            },
        )
        assert response.status_code == 404

    def test_missing_application_id_returns_422(self, client):
        response = client.post(
            self.route,
            json={"name": "Missing App ID Key", "access": "application"},
        )
        assert response.status_code == 422


class TestListKeys:
    route = "/keys"

    @pytest.fixture(scope="class")
    def keys_for_listing(self, client, application_for_testing):
        """Create a personal key and an application key for list tests."""
        personal = client.post(self.route, json={"name": "List Personal Key"})
        assert personal.status_code == 201

        app_key = client.post(
            self.route,
            json={
                "name": "List App Key",
                "access": "application",
                "application_id": application_for_testing["id"],
            },
        )
        assert app_key.status_code == 201

        return {
            "personal": personal.json(),
            "application": app_key.json(),
        }

    def test_returns_paginated_response(self, client):
        response = client.get(self.route)
        assert response.status_code == 200

        data = response.json()
        assert "keys" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["keys"], list)

    def test_includes_personal_and_application_keys(self, client, keys_for_listing):
        response = client.get(self.route)
        assert response.status_code == 200

        data = response.json()
        key_ids = [k["id"] for k in data["keys"]]
        assert keys_for_listing["personal"]["id"] in key_ids
        assert keys_for_listing["application"]["id"] in key_ids

    def test_page_param(self, client):
        response = client.get(f"{self.route}?page=0&size=1")
        assert response.status_code == 200

        data = response.json()
        assert data["current_page"] == 0
        assert len(data["keys"]) <= 1

    def test_size_param(self, client):
        response = client.get(f"{self.route}?size=2")
        assert response.status_code == 200

        data = response.json()
        assert data["page_size"] == 2
        assert len(data["keys"]) <= 2


class TestGetKey:
    route = "/keys"

    @pytest.fixture(scope="class")
    def personal_key(self, client):
        response = client.post(self.route, json={"name": "Get Test Key"})
        assert response.status_code == 201
        return response.json()

    @pytest.fixture(scope="class")
    def application_key(self, client, application_for_testing):
        response = client.post(
            self.route,
            json={
                "name": "Get App Key",
                "access": "application",
                "application_id": application_for_testing["id"],
            },
        )
        assert response.status_code == 201
        return response.json()

    def test_personal_key(self, client, personal_key):
        key_id = personal_key["id"]
        response = client.get(f"{self.route}/{key_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == key_id
        assert data["name"] == "Get Test Key"
        # Secret should NOT be in the response
        assert "secret" not in data

    def test_application_key_via_parent_ownership(self, client, application_key):
        key_id = application_key["id"]
        response = client.get(f"{self.route}/{key_id}")
        assert response.status_code == 200
        assert response.json()["id"] == key_id

    def test_nonexistent_returns_404(self, client):
        response = client.get(f"{self.route}/00000000000000000000000000000000")
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, firestore_client):
        """A key owned by a different user should return 404."""
        from tests.fixtures import make_key_data

        key_data = make_key_data(owner_id="different-owner", name="Other User Key")
        key_data.pop("_test_secret")
        key_ref = firestore_client.collection(KEYS_COLLECTION).document(key_data["id"])
        key_ref.set(key_data)

        try:
            response = client.get(f"{self.route}/{key_data['id']}")
            assert response.status_code == 404
        finally:
            key_ref.delete()

    def test_unowned_app_key_returns_404(
        self, client, firestore_client, application_with_different_owner
    ):
        """A key for an app owned by a different user should return 404."""
        from tests.fixtures import make_key_data

        app_id = application_with_different_owner["id"]
        key_data = make_key_data(
            owner_id=app_id,
            name="Unowned App Key",
            access="application",
            application_id=app_id,
        )
        key_data.pop("_test_secret")
        key_ref = firestore_client.collection(KEYS_COLLECTION).document(key_data["id"])
        key_ref.set(key_data)

        try:
            response = client.get(f"{self.route}/{key_data['id']}")
            assert response.status_code == 404
        finally:
            key_ref.delete()


class TestDeleteKey:
    route = "/keys"

    @pytest.fixture
    def key_for_delete(self, client):
        response = client.post(
            self.route,
            json={"name": "Delete Target Key", "scopes": ["read", "write"]},
        )
        assert response.status_code == 201
        return response.json()

    def test_happy_path(self, client, key_for_delete):
        key_id = key_for_delete["id"]
        response = client.delete(f"{self.route}/{key_id}")
        assert response.status_code == 204

    def test_nonexistent_returns_404(self, client):
        response = client.delete(f"{self.route}/00000000000000000000000000000000")
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, firestore_client):
        from tests.fixtures import make_key_data

        key_data = make_key_data(owner_id="different-owner", name="Other User Key")
        key_data.pop("_test_secret")
        key_ref = firestore_client.collection(KEYS_COLLECTION).document(key_data["id"])
        key_ref.set(key_data)

        try:
            response = client.delete(f"{self.route}/{key_data['id']}")
            assert response.status_code == 404
        finally:
            key_ref.delete()

    def test_double_delete_returns_404(self, client, key_for_delete):
        key_id = key_for_delete["id"]

        # First delete
        response1 = client.delete(f"{self.route}/{key_id}")
        assert response1.status_code == 204

        # Second delete
        response2 = client.delete(f"{self.route}/{key_id}")
        assert response2.status_code == 404
