"""
Integration tests for api/v2/resources/point_clouds/router.py

Tests the standard CRUD endpoints (LIST, GET, PATCH, DELETE). These
tests make real HTTP requests to the API and interact with Firestore. There is
no create endpoint yet (#328/#329), so documents are seeded directly.
"""

import pytest

from lib.config import POINT_CLOUDS_COLLECTION
from tests.fixtures import make_point_cloud_data

# GET /domains/{domain_id}/pointclouds/{point_cloud_id} Tests


class TestGetPointCloud:
    """Test the GET /domains/{domain_id}/pointclouds/{point_cloud_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/pointclouds"

    def test_get_existing_point_cloud(
        self, client, point_cloud_in_firestore, domain_for_testing
    ):
        """Successfully retrieve a point cloud that exists."""
        pc_id = point_cloud_in_firestore["id"]

        response = client.get(f"{self.route(domain_for_testing['id'])}/{pc_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == pc_id
        assert data["name"] == "Test Point Cloud for GET"
        assert data["type"] == "als"
        assert data["tags"] == ["test", "fixture"]
        assert "source" in data
        assert "georeference" in data
        assert "created_on" in data
        assert "modified_on" in data

    def test_get_nonexistent_point_cloud_returns_404(self, client, domain_for_testing):
        """Fetching a non-existent point cloud returns 404."""
        response = client.get(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_point_cloud_wrong_owner_returns_404(
        self, client, point_cloud_with_different_owner, domain_with_different_owner
    ):
        """Fetching a point cloud owned by another user returns 404."""
        pc_id = point_cloud_with_different_owner["id"]
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{pc_id}"
        )
        assert response.status_code == 404

    def test_get_point_cloud_wrong_domain_returns_404(
        self,
        client,
        point_cloud_in_firestore,
        domain_with_different_owner,
    ):
        """Fetching a point cloud under the wrong domain returns 404."""
        pc_id = point_cloud_in_firestore["id"]
        response = client.get(
            f"{self.route(domain_with_different_owner['id'])}/{pc_id}"
        )
        assert response.status_code == 404

    def test_get_point_cloud_excludes_owner_id(
        self, client, point_cloud_in_firestore, domain_for_testing
    ):
        """Response should not expose the owner_id field."""
        pc_id = point_cloud_in_firestore["id"]
        response = client.get(f"{self.route(domain_for_testing['id'])}/{pc_id}")
        assert response.status_code == 200
        assert "owner_id" not in response.json()


# GET /domains/-/pointclouds (Wildcard List) Tests


class TestListPointCloudsWildcard:
    """Test GET /domains/-/pointclouds returns point clouds across all domains."""

    @pytest.fixture(scope="class")
    def point_clouds_across_domains(
        self, firestore_client, domain_for_testing, second_domain
    ):
        """Point clouds spread across two domains, both owned by test-owner."""
        point_clouds = []
        for domain_id in [domain_for_testing["id"], second_domain["id"]]:
            pc_data = make_point_cloud_data(
                domain_id=domain_id, name=f"Point cloud in {domain_id}"
            )
            doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
                pc_data["id"]
            )
            doc_ref.set(pc_data)
            point_clouds.append(pc_data)
        yield point_clouds
        for pc in point_clouds:
            firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
                pc["id"]
            ).delete()

    def route(self):
        return "/domains/-/pointclouds"

    def test_wildcard_returns_200(self, client):
        response = client.get(self.route())
        assert response.status_code == 200

    def test_wildcard_returns_point_clouds_from_all_domains(
        self, client, point_clouds_across_domains
    ):
        """Point clouds from multiple domains are all returned."""
        response = client.get(self.route())
        assert response.status_code == 200

        pc_ids = [p["id"] for p in response.json()["point_clouds"]]
        for pc in point_clouds_across_domains:
            assert pc["id"] in pc_ids

    def test_wildcard_excludes_other_users_point_clouds(
        self, client, point_cloud_with_different_owner
    ):
        """Wildcard list does not return point clouds owned by other users."""
        response = client.get(self.route())
        assert response.status_code == 200

        pc_ids = [p["id"] for p in response.json()["point_clouds"]]
        assert point_cloud_with_different_owner["id"] not in pc_ids

    def test_wildcard_excludes_owner_id(self, client, point_clouds_across_domains):
        """Wildcard list does not expose owner_id."""
        response = client.get(self.route())
        assert response.status_code == 200

        for pc in response.json()["point_clouds"]:
            assert "owner_id" not in pc

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_wildcard_sorting_matrix_returns_200(
        self, client, point_clouds_across_domains, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        url = f"{self.route()}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200


# GET /domains/{domain_id}/pointclouds (List) Tests


class TestListPointClouds:
    """Test the GET /domains/{domain_id}/pointclouds endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/pointclouds"

    @pytest.fixture(scope="class")
    def point_clouds_for_listing(self, firestore_client, domain_for_testing):
        """Create multiple point clouds for list testing.

        Mixes als/tls and sources so type/source filters have something to bite
        on: two als-from-3dep and one tls-from-upload.
        """
        specs = [
            ("Alpha Cloud", "als", {"name": "3dep"}),
            ("Beta Cloud", "als", {"name": "3dep"}),
            ("Gamma Cloud", "tls", {"name": "upload"}),
        ]
        point_clouds = []
        for name, pc_type, source in specs:
            pc_data = make_point_cloud_data(
                domain_id=domain_for_testing["id"],
                name=name,
                point_cloud_type=pc_type,
                source=source,
                tags=["list-test"],
            )
            doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
                pc_data["id"]
            )
            doc_ref.set(pc_data)
            point_clouds.append(pc_data)

        yield point_clouds

        for pc in point_clouds:
            firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
                pc["id"]
            ).delete()

    def test_list_returns_200(self, client, domain_for_testing):
        response = client.get(self.route(domain_for_testing["id"]))
        assert response.status_code == 200

    def test_list_returns_paginated_response(self, client, domain_for_testing):
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        data = response.json()

        assert "point_clouds" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["point_clouds"], list)

    def test_list_returns_user_point_clouds(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        pc_ids = [p["id"] for p in response.json()["point_clouds"]]
        for pc in point_clouds_for_listing:
            assert pc["id"] in pc_ids

    def test_list_excludes_other_users_point_clouds(
        self, client, point_cloud_with_different_owner, domain_for_testing
    ):
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        pc_ids = [p["id"] for p in response.json()["point_clouds"]]
        assert point_cloud_with_different_owner["id"] not in pc_ids

    def test_list_excludes_owner_id(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(self.route(domain_for_testing["id"]))

        assert response.status_code == 200
        for pc in response.json()["point_clouds"]:
            assert "owner_id" not in pc

    def test_list_pagination_page_param(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        base = self.route(domain_for_testing["id"])

        response1 = client.get(f"{base}?page=0&size=1")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["current_page"] == 0
        assert len(data1["point_clouds"]) == 1

        response2 = client.get(f"{base}?page=1&size=1")
        assert response2.status_code == 200
        assert response2.json()["current_page"] == 1

    def test_list_pagination_size_param(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 2
        assert len(data["point_clouds"]) <= 2

    def test_list_sorting_by_name_ascending(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=ascending"
        )

        assert response.status_code == 200
        names = [p["name"] for p in response.json()["point_clouds"]]
        assert names == sorted(names)

    def test_list_sorting_by_name_descending(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=name&sort_order=descending"
        )

        assert response.status_code == 200
        names = [p["name"] for p in response.json()["point_clouds"]]
        assert names == sorted(names, reverse=True)

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_list_sorting_matrix_returns_200(
        self, client, point_clouds_for_listing, domain_for_testing, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        url = f"{self.route(domain_for_testing['id'])}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
        assert response.status_code == 200

    def test_list_filter_by_type(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        """Filter by acquisition type returns only matching point clouds."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?type=tls")

        assert response.status_code == 200
        data = response.json()
        for pc in data["point_clouds"]:
            assert pc["type"] == "tls"
        # The single tls fixture must be present.
        tls_names = [p["name"] for p in data["point_clouds"]]
        assert "Gamma Cloud" in tls_names

    def test_list_filter_by_source(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        """Filter by source name returns only matching point clouds."""
        response = client.get(f"{self.route(domain_for_testing['id'])}?source=3dep")

        assert response.status_code == 200
        for pc in response.json()["point_clouds"]:
            assert pc["source"]["name"] == "3dep"

    def test_list_filter_by_source_no_results(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?source=nonexistent_source"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["point_clouds"] == []
        assert data["total_items"] == 0

    def test_list_filter_by_tag(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(f"{self.route(domain_for_testing['id'])}?tag=list-test")

        assert response.status_code == 200
        data = response.json()
        for pc in data["point_clouds"]:
            assert "list-test" in pc["tags"]
        pc_ids = [p["id"] for p in data["point_clouds"]]
        for pc in point_clouds_for_listing:
            assert pc["id"] in pc_ids

    def test_list_filter_combined_type_and_tag(
        self, client, point_clouds_for_listing, domain_for_testing
    ):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?type=als&tag=list-test"
        )

        assert response.status_code == 200
        for pc in response.json()["point_clouds"]:
            assert pc["type"] == "als"
            assert "list-test" in pc["tags"]

    def test_list_invalid_page_returns_422(self, client, domain_for_testing):
        response = client.get(f"{self.route(domain_for_testing['id'])}?page=-1")
        assert response.status_code == 422

    def test_list_invalid_size_too_large_returns_422(self, client, domain_for_testing):
        response = client.get(f"{self.route(domain_for_testing['id'])}?size=1001")
        assert response.status_code == 422

    def test_list_invalid_type_returns_422(self, client, domain_for_testing):
        response = client.get(f"{self.route(domain_for_testing['id'])}?type=mls")
        assert response.status_code == 422

    def test_list_invalid_sort_by_returns_422(self, client, domain_for_testing):
        response = client.get(
            f"{self.route(domain_for_testing['id'])}?sort_by=invalid_field"
        )
        assert response.status_code == 422


# PATCH /domains/{domain_id}/pointclouds/{point_cloud_id} Tests


class TestUpdatePointCloud:
    """Test the PATCH /domains/{domain_id}/pointclouds/{point_cloud_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/pointclouds"

    @pytest.fixture(scope="class")
    def point_cloud_for_update(self, firestore_client, domain_for_testing):
        pc_data = make_point_cloud_data(
            domain_id=domain_for_testing["id"],
            name="Original Name",
            description="Original Description",
            tags=["original"],
            status="completed",
        )
        doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
            pc_data["id"]
        )
        doc_ref.set(pc_data)
        yield pc_data
        doc_ref.delete()

    def test_update_name(self, client, point_cloud_for_update, domain_for_testing):
        pc_id = point_cloud_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original Description"
        assert data["tags"] == ["original"]

    def test_update_tags(self, client, point_cloud_for_update, domain_for_testing):
        pc_id = point_cloud_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"tags": ["new", "tags"]},
        )

        assert response.status_code == 200
        assert response.json()["tags"] == ["new", "tags"]

    def test_update_modifies_modified_on(
        self, client, point_cloud_for_update, domain_for_testing
    ):
        pc_id = point_cloud_for_update["id"]
        original_modified_on = point_cloud_for_update["modified_on"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"name": "Timestamp Test"},
        )

        assert response.status_code == 200
        assert response.json()["modified_on"] != original_modified_on.isoformat()

    def test_update_does_not_change_checksum(
        self, client, point_cloud_for_update, domain_for_testing
    ):
        """Metadata-only edits must not change the content checksum."""
        pc_id = point_cloud_for_update["id"]
        original_checksum = point_cloud_for_update["checksum"]

        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"name": "Checksum Test"},
        )

        assert response.status_code == 200
        assert response.json()["checksum"] == original_checksum

    def test_update_preserves_immutable_fields(
        self, client, point_cloud_for_update, domain_for_testing
    ):
        pc_id = point_cloud_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"name": "Immutable Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == pc_id
        assert data["domain_id"] == point_cloud_for_update["domain_id"]
        assert data["type"] == "als"
        assert "source" in data

    def test_update_nonexistent_returns_404(self, client, domain_for_testing):
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000",
            json={"name": "Should Fail"},
        )
        assert response.status_code == 404

    def test_update_excludes_owner_id_from_response(
        self, client, point_cloud_for_update, domain_for_testing
    ):
        pc_id = point_cloud_for_update["id"]
        response = client.patch(
            f"{self.route(domain_for_testing['id'])}/{pc_id}",
            json={"name": "Owner Test"},
        )

        assert response.status_code == 200
        assert "owner_id" not in response.json()


# DELETE /domains/{domain_id}/pointclouds/{point_cloud_id} Tests


class TestDeletePointCloud:
    """Test the DELETE /domains/{domain_id}/pointclouds/{point_cloud_id} endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/pointclouds"

    @pytest.fixture(scope="function")
    def point_cloud_for_delete(self, firestore_client, domain_for_testing):
        pc_data = make_point_cloud_data(
            domain_id=domain_for_testing["id"],
            name="Point Cloud to Delete",
            tags=["delete-test"],
        )
        doc_ref = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
            pc_data["id"]
        )
        doc_ref.set(pc_data)
        yield pc_data
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.delete()

    def test_delete_existing_point_cloud(
        self, client, point_cloud_for_delete, firestore_client, domain_for_testing
    ):
        pc_id = point_cloud_for_delete["id"]

        response = client.delete(f"{self.route(domain_for_testing['id'])}/{pc_id}")

        assert response.status_code == 204
        assert response.content == b""

        doc = firestore_client.collection(POINT_CLOUDS_COLLECTION).document(pc_id).get()
        assert not doc.exists

    def test_delete_nonexistent_returns_404(self, client, domain_for_testing):
        response = client.delete(
            f"{self.route(domain_for_testing['id'])}/00000000000000000000000000000000"
        )
        assert response.status_code == 404

    def test_delete_wrong_owner_returns_404(
        self, client, point_cloud_with_different_owner, domain_with_different_owner
    ):
        pc_id = point_cloud_with_different_owner["id"]
        response = client.delete(
            f"{self.route(domain_with_different_owner['id'])}/{pc_id}"
        )
        assert response.status_code == 404

    def test_delete_is_permanent(
        self, client, point_cloud_for_delete, domain_for_testing
    ):
        pc_id = point_cloud_for_delete["id"]

        delete_response = client.delete(
            f"{self.route(domain_for_testing['id'])}/{pc_id}"
        )
        assert delete_response.status_code == 204

        get_response = client.get(f"{self.route(domain_for_testing['id'])}/{pc_id}")
        assert get_response.status_code == 404

    def test_delete_twice_returns_404_second_time(
        self, client, point_cloud_for_delete, domain_for_testing
    ):
        pc_id = point_cloud_for_delete["id"]

        response1 = client.delete(f"{self.route(domain_for_testing['id'])}/{pc_id}")
        assert response1.status_code == 204

        response2 = client.delete(f"{self.route(domain_for_testing['id'])}/{pc_id}")
        assert response2.status_code == 404
