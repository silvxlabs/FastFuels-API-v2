"""
Integration tests for api/v2/resources/applications/router.py

These tests make real HTTP requests to the API and interact with Firestore.
Requires a running local API server (http://127.0.0.1:8080).
"""

import time

import pytest
from httpx import Client

from lib.config import APPLICATIONS_COLLECTION, KEYS_COLLECTION
from tests.fixtures import make_application_data, make_key_data


class TestCreateApplication:
    route = "/applications"

    def test_create_with_name_and_description(self, client):
        response = client.post(
            self.route,
            json={"name": "Test App", "description": "A test application"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Test App"
        assert data["description"] == "A test application"
        assert len(data["id"]) == 32
        assert "created_on" in data
        assert "modified_on" in data

    def test_create_minimal(self, client):
        response = client.post(self.route, json={"name": "Minimal App"})
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Minimal App"
        assert data.get("description") is None

    def test_owner_id_matches_requester(self, client, test_owner_id):
        response = client.post(self.route, json={"name": "Owner Check App"})
        assert response.status_code == 201

        data = response.json()
        assert data["owner_id"] == test_owner_id

    def test_id_is_32_char_hex(self, client):
        response = client.post(self.route, json={"name": "ID Check App"})
        assert response.status_code == 201

        app_id = response.json()["id"]
        assert len(app_id) == 32
        int(app_id, 16)  # Should not raise — valid hex

    def test_rejects_quota_config(self, client):
        """tier / quota_overrides are admin-only and cannot be set at create (422)."""
        response = client.post(
            self.route, json={"name": "Quota App", "tier": "partner"}
        )
        assert response.status_code == 422

    def test_application_access_forbidden(
        self, client, firestore_client, test_owner_id
    ):
        """Application-access keys cannot create applications (403)."""
        # Create an application to serve as the key owner
        app_data = {
            "id": "testapp403",
            "owner_id": test_owner_id,
            "name": "App For 403 Test",
        }
        app_ref = firestore_client.collection(APPLICATIONS_COLLECTION).document(
            app_data["id"]
        )
        app_ref.set(app_data)

        # Create an application-scoped key in Firestore
        key_data = make_key_data(
            owner_id="testapp403",
            name="App-scoped Key",
            access="application",
            application_id="testapp403",
            scopes=["read", "write"],
        )
        secret = key_data.pop("_test_secret")
        key_ref = firestore_client.collection(KEYS_COLLECTION).document(key_data["id"])
        key_ref.set(key_data)

        try:
            # Use a separate client with the application key
            from tests.conftest import TEST_URL

            with Client(
                base_url=TEST_URL,
                headers={"API-KEY": secret},
            ) as app_client:
                response = app_client.post(self.route, json={"name": "Should Fail"})
                assert response.status_code == 403
        finally:
            key_ref.delete()
            app_ref.delete()


class TestListApplications:
    route = "/applications"

    def test_returns_paginated_response(self, client):
        response = client.get(self.route)
        assert response.status_code == 200

        data = response.json()
        assert "applications" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["applications"], list)

    def test_includes_user_apps(self, isolated_owner):
        # Fresh isolated owner: the listing is bounded to the seeded app and
        # never buried by the shared owner's accumulated test applications.
        client, owner_id, seed = isolated_owner
        app = seed(APPLICATIONS_COLLECTION, make_application_data(owner_id=owner_id))

        response = client.get(self.route)
        assert response.status_code == 200

        data = response.json()
        app_ids = [a["id"] for a in data["applications"]]
        assert app["id"] in app_ids

    def test_excludes_other_users_apps(self, client, application_with_different_owner):
        response = client.get(self.route)
        assert response.status_code == 200

        data = response.json()
        app_ids = [a["id"] for a in data["applications"]]
        assert application_with_different_owner["id"] not in app_ids

    def test_page_param(self, client):
        response = client.get(f"{self.route}?page=0&size=1")
        assert response.status_code == 200

        data = response.json()
        assert data["current_page"] == 0
        assert len(data["applications"]) <= 1

    def test_size_param(self, client):
        response = client.get(f"{self.route}?size=2")
        assert response.status_code == 200

        data = response.json()
        assert data["page_size"] == 2
        assert len(data["applications"]) <= 2


class TestGetApplication:
    route = "/applications"

    def test_happy_path(self, client, application_for_testing):
        app_id = application_for_testing["id"]
        response = client.get(f"{self.route}/{app_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == app_id
        assert data["name"] == application_for_testing["name"]

    def test_nonexistent_returns_404(self, client):
        response = client.get(f"{self.route}/00000000000000000000000000000000")
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, application_with_different_owner):
        app_id = application_with_different_owner["id"]
        response = client.get(f"{self.route}/{app_id}")
        assert response.status_code == 404

    def test_surfaces_quota_config(self, client, firestore_client, test_owner_id):
        """tier / quota_overrides stored on the document are returned by GET."""
        app = make_application_data(owner_id=test_owner_id, name="Quota Config App")
        app["tier"] = "application"
        app["quota_overrides"] = {"max_active_grids": 100}
        ref = firestore_client.collection(APPLICATIONS_COLLECTION).document(app["id"])
        ref.set(app)
        try:
            response = client.get(f"{self.route}/{app['id']}")
            assert response.status_code == 200
            data = response.json()
            assert data["tier"] == "application"
            assert data["quota_overrides"] == {"max_active_grids": 100}
        finally:
            ref.delete()


class TestUpdateApplication:
    route = "/applications"

    @pytest.fixture(scope="class")
    def app_for_update(self, client):
        """Create an application for update tests via the API."""
        response = client.post(
            self.route,
            json={"name": "Update Target", "description": "Original description"},
        )
        assert response.status_code == 201
        return response.json()

    def test_update_name(self, client, app_for_update):
        app_id = app_for_update["id"]
        response = client.patch(f"{self.route}/{app_id}", json={"name": "Updated Name"})
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    def test_update_description(self, client, app_for_update):
        app_id = app_for_update["id"]
        response = client.patch(
            f"{self.route}/{app_id}", json={"description": "Updated description"}
        )
        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    def test_update_both(self, client, app_for_update):
        app_id = app_for_update["id"]
        response = client.patch(
            f"{self.route}/{app_id}",
            json={"name": "Both Name", "description": "Both Desc"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Both Name"
        assert data["description"] == "Both Desc"

    def test_modified_on_changes(self, client, app_for_update):
        app_id = app_for_update["id"]
        original_modified = app_for_update["modified_on"]

        time.sleep(0.1)
        response = client.patch(
            f"{self.route}/{app_id}", json={"name": "Timestamp Test"}
        )
        assert response.status_code == 200
        assert response.json()["modified_on"] != original_modified

    def test_empty_body(self, client, app_for_update):
        app_id = app_for_update["id"]
        response = client.patch(f"{self.route}/{app_id}", json={})
        assert response.status_code == 200

    def test_patch_rejects_quota_config(self, client, firestore_client, app_for_update):
        """tier / quota_overrides are admin-only: a PATCH carrying them is a 422,
        and the stored document is left untouched."""
        app_id = app_for_update["id"]
        response = client.patch(
            f"{self.route}/{app_id}",
            json={"tier": "partner", "quota_overrides": {"max_active_grids": 9999}},
        )
        assert response.status_code == 422

        stored = (
            firestore_client.collection(APPLICATIONS_COLLECTION)
            .document(app_id)
            .get()
            .to_dict()
        )
        assert stored.get("tier") is None
        assert stored.get("quota_overrides") is None

    def test_nonexistent_returns_404(self, client):
        response = client.patch(
            f"{self.route}/00000000000000000000000000000000",
            json={"name": "Should Fail"},
        )
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, application_with_different_owner):
        app_id = application_with_different_owner["id"]
        response = client.patch(f"{self.route}/{app_id}", json={"name": "Should Fail"})
        assert response.status_code == 404


class TestDeleteApplication:
    route = "/applications"

    @pytest.fixture
    def app_for_delete(self, client, firestore_client):
        """Create an application for delete tests via the API."""
        response = client.post(
            self.route, json={"name": "Delete Target", "description": "To be deleted"}
        )
        assert response.status_code == 201
        data = response.json()
        yield data
        # Cleanup if not deleted
        doc = (
            firestore_client.collection(APPLICATIONS_COLLECTION)
            .document(data["id"])
            .get()
        )
        if doc.exists:
            firestore_client.collection(APPLICATIONS_COLLECTION).document(
                data["id"]
            ).delete()

    def test_happy_path(self, client, app_for_delete):
        app_id = app_for_delete["id"]
        response = client.delete(f"{self.route}/{app_id}")
        assert response.status_code == 204

    def test_cascade_deletes_keys(self, client, firestore_client, app_for_delete):
        """Deleting an application should cascade-delete its keys."""
        app_id = app_for_delete["id"]

        # Create a key owned by this application
        key_data = make_key_data(
            owner_id=app_id,
            name="App Key",
            access="application",
            application_id=app_id,
        )
        key_data.pop("_test_secret")
        key_ref = firestore_client.collection(KEYS_COLLECTION).document(key_data["id"])
        key_ref.set(key_data)

        # Delete the application
        response = client.delete(f"{self.route}/{app_id}")
        assert response.status_code == 204

        # Key should be gone too
        key_doc = key_ref.get()
        assert not key_doc.exists

    def test_nonexistent_returns_404(self, client):
        response = client.delete(f"{self.route}/00000000000000000000000000000000")
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, application_with_different_owner):
        app_id = application_with_different_owner["id"]
        response = client.delete(f"{self.route}/{app_id}")
        assert response.status_code == 404

    def test_deleted_app_not_in_list(self, client, app_for_delete):
        app_id = app_for_delete["id"]

        # Delete it
        client.delete(f"{self.route}/{app_id}")

        # Verify it's not in the list
        list_response = client.get(self.route)
        assert list_response.status_code == 200
        app_ids = [a["id"] for a in list_response.json()["applications"]]
        assert app_id not in app_ids
