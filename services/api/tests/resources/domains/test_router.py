"""
Integration tests for api/v2/resources/domains/router.py

Tests the domain creation endpoint with all documented examples to ensure
that our API documentation examples actually work. This is critical because
users rely on these examples to understand how to use the API.

These tests make real HTTP requests to the API and interact with Firestore.
Validation edge cases are tested in test_validate.py - this file focuses on
happy paths and example verification.
"""

import json
from pathlib import Path

import pytest
from api.resources.domains.examples import ALL_EXAMPLE_VALUES
from google.cloud import firestore

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data

# Path to v2 test data directory
DATA_DIR = Path(__file__).parent / "data"

# Track created domains for cleanup/reuse
DOMAINS = []


@pytest.fixture(scope="session")
def firestore_client():
    """Session-scoped Firestore client for direct database operations."""
    return firestore.Client()


@pytest.fixture(scope="session")
def domain_in_firestore(firestore_client):
    """Create a domain document directly in Firestore, yield it, then delete."""
    domain_data = make_domain_data(
        name="Test Domain for GET",
        description="Created by fixture for GET endpoint tests",
        tags=["test", "fixture"],
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def domain_with_different_owner(firestore_client):
    """Create a domain owned by a different user for ownership validation tests."""
    domain_data = make_domain_data(
        owner_id="different-owner",
        name="Other User's Domain",
        description="Owned by different-owner",
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


class TestCreateDomainExamples:
    """Test that all documented examples create valid domains.

    These tests are critical for documentation accuracy. If an example
    fails here, it means our documentation is showing users broken examples.
    """

    route = "/domains"

    @pytest.mark.parametrize("example_name,example_value", ALL_EXAMPLE_VALUES)
    def test_example_creates_domain(self, client, example_name, example_value):
        """Each documented example should successfully create a domain."""
        response = client.post(self.route, json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()

        # Verify required response fields
        assert "id" in data
        assert len(data["id"]) == 32  # UUID hex format
        assert "created_on" in data
        assert "modified_on" in data
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) > 0

        # Name and description should match input
        assert data["name"] == example_value.get("name", "")
        assert data["description"] == example_value.get("description", "")

        # Track for other tests
        DOMAINS.append(data)


class TestCreateDomainFromFiles:
    """Test domain creation using GeoJSON files from the test data directory.

    These tests verify that real-world GeoJSON files work with the API.
    """

    route = "/domains"

    def test_blue_mountain_feature_collection(self, client):
        """Create domain from Blue Mountain FeatureCollection."""
        with open(DATA_DIR / "blue_mountain_feature_4326.geojson") as f:
            request_body = json.load(f)

        # Wrap in FeatureCollection if it's a Feature
        if request_body.get("type") == "Feature":
            request_body = {
                "type": "FeatureCollection",
                "features": [request_body],
                "name": "Blue Mountain",
                "description": "Test domain from GeoJSON file",
            }
        else:
            request_body["name"] = "Blue Mountain Feature Collection"
            request_body["description"] = "A test feature collection"

        response = client.post(self.route, json=request_body)

        assert response.status_code == 201
        DOMAINS.append(response.json())

    def test_polygon_utm_preserves_crs(self, client):
        """UTM coordinates should preserve the input CRS."""
        with open(DATA_DIR / "polygon_utm.geojson") as f:
            request_body = json.load(f)

        # Wrap in FeatureCollection if needed
        if request_body.get("type") == "Feature":
            original_crs = request_body.get("crs")
            request_body = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": request_body["geometry"],
                        "properties": request_body.get("properties", {}),
                    }
                ],
                "crs": original_crs,
                "name": "UTM Polygon",
            }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 201

        response_body = response.json()
        # UTM CRS should be preserved (not reprojected)
        assert response_body["crs"] == request_body["crs"]

        DOMAINS.append(response_body)

    def test_tags_are_preserved(self, client):
        """Tags provided in request should be returned in response."""
        with open(DATA_DIR / "blue_mountain_feature_4326.geojson") as f:
            feature = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [feature]
                if feature.get("type") == "Feature"
                else feature.get("features", [])
            ),
            "name": "Tagged Domain",
            "tags": ["test", "integration", "v2"],
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 201

        response_body = response.json()
        assert response_body["tags"] == ["test", "integration", "v2"]

        DOMAINS.append(response_body)


class TestCreateDomainValidationErrors:
    """Test validation error responses.

    Note: Detailed validation logic is tested in test_validate.py.
    These tests verify the API returns appropriate HTTP errors.
    """

    route = "/domains"

    def test_zero_area_returns_422(self, client):
        """Point geometry (zero area) should return 422."""
        with open(DATA_DIR / "point.geojson") as f:
            point = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [point] if point.get("type") == "Feature" else point.get("features", [])
            ),
            "name": "Point Domain",
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "area greater than zero" in response.json()["detail"]

    def test_oversized_returns_422(self, client):
        """Large domain (> 16 sq km) should return 422."""
        with open(DATA_DIR / "saint_mary_5070.geojson") as f:
            request_body = json.load(f)

        # Ensure it's a FeatureCollection
        if request_body.get("type") == "Feature":
            request_body = {
                "type": "FeatureCollection",
                "features": [request_body],
                "crs": request_body.get("crs"),
            }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "16 square kilometers" in response.json()["detail"]

    def test_outside_conus_returns_422(self, client):
        """Domain outside CONUS should return 422."""
        with open(DATA_DIR / "polygon_in_alaska.geojson") as f:
            alaska = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [alaska]
                if alaska.get("type") == "Feature"
                else alaska.get("features", [])
            ),
            "name": "Alaska Domain",
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "within CONUS" in response.json()["detail"]

    def test_invalid_crs_returns_422(self, client):
        """Invalid CRS should return 422."""
        request_body = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-114.0, 46.8],
                                [-114.01, 46.8],
                                [-114.01, 46.79],
                                [-114.0, 46.79],
                                [-114.0, 46.8],
                            ]
                        ],
                    },
                }
            ],
            "crs": {"type": "name", "properties": {"name": "INVALID:CRS"}},
            "name": "Invalid CRS Domain",
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "Invalid CRS" in response.json()["detail"]


class TestGetDomain:
    """Test the GET /domains/{domain_id} endpoint.

    Uses fixtures that create documents directly in Firestore to ensure
    test independence from the POST endpoint.
    """

    route = "/domains"

    def test_get_existing_domain(self, client, domain_in_firestore):
        """Successfully retrieve a domain that exists."""
        domain_id = domain_in_firestore["id"]

        response = client.get(f"{self.route}/{domain_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == domain_id
        assert data["name"] == "Test Domain for GET"
        assert data["description"] == "Created by fixture for GET endpoint tests"
        assert data["type"] == "FeatureCollection"
        assert data["tags"] == ["test", "fixture"]
        assert "features" in data
        assert "created_on" in data
        assert "modified_on" in data
        assert "crs" in data

    def test_get_nonexistent_domain_returns_404(self, client):
        """Fetching a non-existent domain should return 404."""
        fake_domain_id = "00000000000000000000000000000000"

        response = client.get(f"{self.route}/{fake_domain_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_domain_wrong_owner_returns_404(
        self, client, domain_with_different_owner
    ):
        """Fetching a domain owned by another user should return 404.

        This tests that ownership validation works and returns 404
        (not 403) to avoid leaking document existence information.
        """
        domain_id = domain_with_different_owner["id"]

        response = client.get(f"{self.route}/{domain_id}")

        # Should return 404, not 403
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_domain_deserializes_coordinates(self, client, domain_in_firestore):
        """Fetched domain should have properly deserialized coordinates.

        Coordinates are stored as JSON strings in Firestore but should
        be returned as nested arrays in the API response.
        """
        domain_id = domain_in_firestore["id"]

        response = client.get(f"{self.route}/{domain_id}")
        assert response.status_code == 200

        data = response.json()
        features = data["features"]

        assert len(features) > 0
        for feature in features:
            geometry = feature["geometry"]
            assert "coordinates" in geometry
            # Coordinates should be a list, not a string
            assert isinstance(geometry["coordinates"], list)
            # Should be nested arrays for polygon
            assert isinstance(geometry["coordinates"][0], list)
            assert isinstance(geometry["coordinates"][0][0], list)

    def test_get_domain_excludes_owner_id(self, client, domain_in_firestore):
        """Response should not expose the owner_id field.

        owner_id is internal for access control and should not be
        returned in the API response.
        """
        domain_id = domain_in_firestore["id"]

        response = client.get(f"{self.route}/{domain_id}")
        assert response.status_code == 200

        data = response.json()
        assert "owner_id" not in data


class TestListDomains:
    """Test the GET /domains endpoint (list all domains).

    Uses fixtures that create documents directly in Firestore to ensure
    test independence from the POST endpoint.
    """

    route = "/domains"

    @pytest.fixture(scope="class")
    def domains_for_listing(self, firestore_client):
        """Create multiple domains for list testing."""
        domains = []
        for i in range(3):
            domain_data = make_domain_data(
                name=f"List Test Domain {chr(65 + i)}",  # A, B, C
                description=f"Domain {i} for list testing",
                tags=["list-test"],
            )
            doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
                domain_data["id"]
            )
            doc_ref.set(domain_data)
            domains.append(domain_data)

        yield domains

        # Cleanup
        for domain in domains:
            firestore_client.collection(DOMAINS_COLLECTION).document(
                domain["id"]
            ).delete()

    def test_list_returns_200(self, client):
        """List endpoint returns 200 OK."""
        response = client.get(self.route)

        assert response.status_code == 200

    def test_list_returns_paginated_response(self, client):
        """Response includes pagination metadata."""
        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()

        assert "domains" in data
        assert "current_page" in data
        assert "page_size" in data
        assert "total_items" in data
        assert isinstance(data["domains"], list)

    def test_list_returns_user_domains(self, client, domains_for_listing):
        """List returns domains belonging to the authenticated user."""
        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()

        # Should have at least the 3 domains we created
        assert data["total_items"] >= 3

        # Check that our test domains are in the results
        domain_ids = [d["id"] for d in data["domains"]]
        for domain in domains_for_listing:
            assert domain["id"] in domain_ids

    def test_list_excludes_other_users_domains(
        self, client, domain_with_different_owner
    ):
        """List does not return domains owned by other users."""
        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()

        domain_ids = [d["id"] for d in data["domains"]]
        assert domain_with_different_owner["id"] not in domain_ids

    def test_list_pagination_page_param(self, client, domains_for_listing):
        """Page parameter controls which page is returned."""
        # Get first page with size 1
        response1 = client.get(f"{self.route}?page=0&size=1")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["current_page"] == 0
        assert len(data1["domains"]) == 1

        # Get second page with size 1
        response2 = client.get(f"{self.route}?page=1&size=1")
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["current_page"] == 1

        # Should be different domains
        if len(data2["domains"]) > 0:
            assert data1["domains"][0]["id"] != data2["domains"][0]["id"]

    def test_list_pagination_size_param(self, client, domains_for_listing):
        """Size parameter controls how many domains per page."""
        response = client.get(f"{self.route}?size=2")

        assert response.status_code == 200
        data = response.json()

        assert data["page_size"] == 2
        assert len(data["domains"]) <= 2

    def test_list_sorting_by_name_ascending(self, client, domains_for_listing):
        """Sorting by name ascending returns alphabetical order."""
        response = client.get(f"{self.route}?sort_by=name&sort_order=ascending")

        assert response.status_code == 200
        data = response.json()

        names = [d["name"] for d in data["domains"]]
        assert names == sorted(names)

    def test_list_sorting_by_name_descending(self, client, domains_for_listing):
        """Sorting by name descending returns reverse alphabetical order."""
        response = client.get(f"{self.route}?sort_by=name&sort_order=descending")

        assert response.status_code == 200
        data = response.json()

        names = [d["name"] for d in data["domains"]]
        assert names == sorted(names, reverse=True)

    def test_list_sorting_by_created_on(self, client, domains_for_listing):
        """Sorting by created_on is accepted."""
        response = client.get(f"{self.route}?sort_by=created_on&sort_order=descending")

        assert response.status_code == 200

    def test_list_sorting_by_modified_on(self, client, domains_for_listing):
        """Sorting by modified_on is accepted."""
        response = client.get(f"{self.route}?sort_by=modified_on&sort_order=ascending")

        assert response.status_code == 200

    def test_list_invalid_page_returns_422(self, client):
        """Negative page number returns 422."""
        response = client.get(f"{self.route}?page=-1")

        assert response.status_code == 422

    def test_list_invalid_size_too_small_returns_422(self, client):
        """Size less than 1 returns 422."""
        response = client.get(f"{self.route}?size=0")

        assert response.status_code == 422

    def test_list_invalid_size_too_large_returns_422(self, client):
        """Size greater than 1000 returns 422."""
        response = client.get(f"{self.route}?size=1001")

        assert response.status_code == 422

    def test_list_invalid_sort_by_returns_422(self, client):
        """Invalid sort_by field returns 422."""
        response = client.get(f"{self.route}?sort_by=invalid_field")

        assert response.status_code == 422

    def test_list_invalid_sort_order_returns_422(self, client):
        """Invalid sort_order returns 422."""
        response = client.get(f"{self.route}?sort_order=invalid")

        assert response.status_code == 422

    def test_list_domains_deserialize_coordinates(self, client, domains_for_listing):
        """Listed domains have properly deserialized coordinates."""
        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()

        for domain in data["domains"]:
            for feature in domain["features"]:
                coords = feature["geometry"]["coordinates"]
                # Should be a list, not a string
                assert isinstance(coords, list)

    def test_list_domains_exclude_owner_id(self, client, domains_for_listing):
        """Listed domains do not expose owner_id."""
        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()

        for domain in data["domains"]:
            assert "owner_id" not in domain

    def test_list_empty_result(self, client, firestore_client):
        """List returns empty array when user has no domains.

        Note: This test would need a way to authenticate as a different user
        with no domains. For now, we verify the response structure is correct
        even with an empty result set by checking a page beyond existing data.
        """
        # Request a page far beyond any reasonable data
        response = client.get(f"{self.route}?page=9999")

        assert response.status_code == 200
        data = response.json()

        assert data["domains"] == []
        assert data["current_page"] == 9999


class TestUpdateDomain:
    """Test the PATCH /domains/{domain_id} endpoint.

    Uses fixtures that create documents directly in Firestore to ensure
    test independence from other endpoints.
    """

    route = "/domains"

    @pytest.fixture(scope="class")
    def domain_for_update(self, firestore_client):
        """Create a domain for update tests."""
        domain_data = make_domain_data(
            name="Original Name",
            description="Original Description",
            tags=["original"],
        )
        doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        doc_ref.set(domain_data)
        yield domain_data
        doc_ref.delete()

    def test_update_name(self, client, domain_for_update):
        """Update only the name field."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original Description"
        assert data["tags"] == ["original"]

    def test_update_description(self, client, domain_for_update):
        """Update only the description field."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"description": "Updated Description"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["description"] == "Updated Description"

    def test_update_tags(self, client, domain_for_update):
        """Update only the tags field."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"tags": ["new", "tags"]},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["tags"] == ["new", "tags"]

    def test_update_multiple_fields(self, client, domain_for_update):
        """Update multiple fields at once."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={
                "name": "Multi Update Name",
                "description": "Multi Update Description",
                "tags": ["multi", "update"],
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Multi Update Name"
        assert data["description"] == "Multi Update Description"
        assert data["tags"] == ["multi", "update"]

    def test_update_modifies_modified_on(self, client, domain_for_update):
        """Update should change the modified_on timestamp."""
        domain_id = domain_for_update["id"]
        original_modified_on = domain_for_update["modified_on"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Timestamp Test"},
        )

        assert response.status_code == 200

        data = response.json()
        # modified_on should be different from original
        assert data["modified_on"] != original_modified_on.isoformat()

    def test_update_preserves_immutable_fields(self, client, domain_for_update):
        """Update should not change id, created_on, features, or crs."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Immutable Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == domain_id
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) > 0
        assert "crs" in data

    def test_update_returns_full_domain(self, client, domain_for_update):
        """Update response includes all domain fields."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Full Response Test"},
        )

        assert response.status_code == 200

        data = response.json()
        # Verify all expected fields are present
        assert "id" in data
        assert "type" in data
        assert "name" in data
        assert "description" in data
        assert "created_on" in data
        assert "modified_on" in data
        assert "tags" in data
        assert "crs" in data
        assert "features" in data

    def test_update_nonexistent_domain_returns_404(self, client):
        """Update a non-existent domain should return 404."""
        fake_domain_id = "00000000000000000000000000000000"

        response = client.patch(
            f"{self.route}/{fake_domain_id}",
            json={"name": "Should Fail"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_wrong_owner_returns_404(self, client, domain_with_different_owner):
        """Update a domain owned by another user should return 404.

        Returns 404 (not 403) to avoid leaking document existence information.
        """
        domain_id = domain_with_different_owner["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Should Fail"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_empty_body(self, client, domain_for_update):
        """Update with empty body should succeed (only updates modified_on)."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={},
        )

        assert response.status_code == 200

    def test_update_excludes_owner_id_from_response(self, client, domain_for_update):
        """Response should not expose the owner_id field."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Owner Test"},
        )

        assert response.status_code == 200

        data = response.json()
        assert "owner_id" not in data

    def test_update_deserializes_coordinates(self, client, domain_for_update):
        """Updated domain should have properly deserialized coordinates."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": "Coords Test"},
        )

        assert response.status_code == 200

        data = response.json()
        for feature in data["features"]:
            coords = feature["geometry"]["coordinates"]
            # Should be a list, not a string
            assert isinstance(coords, list)

    def test_update_empty_string_name(self, client, domain_for_update):
        """Should accept empty string as name value."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"name": ""},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["name"] == ""

    def test_update_empty_tags_list(self, client, domain_for_update):
        """Should accept empty list for tags (clears all tags)."""
        domain_id = domain_for_update["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"tags": []},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["tags"] == []


class TestDeleteDomain:
    """Test the DELETE /domains/{domain_id} endpoint.

    Uses fixtures that create documents directly in Firestore to ensure
    test independence from other endpoints.
    """

    route = "/domains"

    @pytest.fixture(scope="function")
    def domain_for_delete(self, firestore_client):
        """Create a domain for delete tests.

        Function-scoped because the domain gets deleted during the test.
        """
        domain_data = make_domain_data(
            name="Domain to Delete",
            description="This domain will be deleted",
            tags=["delete-test"],
        )
        doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        doc_ref.set(domain_data)
        yield domain_data
        # Cleanup if not already deleted
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.delete()

    def test_delete_existing_domain(self, client, domain_for_delete, firestore_client):
        """Successfully delete an existing domain."""
        domain_id = domain_for_delete["id"]

        response = client.delete(f"{self.route}/{domain_id}")

        assert response.status_code == 204
        assert response.content == b""

        # Verify domain is actually deleted from Firestore
        doc = firestore_client.collection(DOMAINS_COLLECTION).document(domain_id).get()
        assert not doc.exists

    def test_delete_nonexistent_domain_returns_404(self, client):
        """Delete a non-existent domain should return 404."""
        fake_domain_id = "00000000000000000000000000000000"

        response = client.delete(f"{self.route}/{fake_domain_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_wrong_owner_returns_404(self, client, domain_with_different_owner):
        """Delete a domain owned by another user should return 404.

        Returns 404 (not 403) to avoid leaking document existence information.
        """
        domain_id = domain_with_different_owner["id"]

        response = client.delete(f"{self.route}/{domain_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_is_permanent(self, client, domain_for_delete):
        """Deleted domain cannot be retrieved."""
        domain_id = domain_for_delete["id"]

        # Delete the domain
        delete_response = client.delete(f"{self.route}/{domain_id}")
        assert delete_response.status_code == 204

        # Try to get it
        get_response = client.get(f"{self.route}/{domain_id}")
        assert get_response.status_code == 404

    def test_delete_twice_returns_404_second_time(self, client, domain_for_delete):
        """Deleting the same domain twice returns 404 on second attempt."""
        domain_id = domain_for_delete["id"]

        # First delete succeeds
        response1 = client.delete(f"{self.route}/{domain_id}")
        assert response1.status_code == 204

        # Second delete returns 404
        response2 = client.delete(f"{self.route}/{domain_id}")
        assert response2.status_code == 404

    def test_delete_does_not_appear_in_list(self, client, domain_for_delete):
        """Deleted domain does not appear in list endpoint."""
        domain_id = domain_for_delete["id"]

        # Verify it appears in list before delete
        list_response_before = client.get(self.route)
        domain_ids_before = [d["id"] for d in list_response_before.json()["domains"]]
        assert domain_id in domain_ids_before

        # Delete the domain
        client.delete(f"{self.route}/{domain_id}")

        # Verify it no longer appears in list
        list_response_after = client.get(self.route)
        domain_ids_after = [d["id"] for d in list_response_after.json()["domains"]]
        assert domain_id not in domain_ids_after

    def test_delete_returns_no_body(self, client, domain_for_delete):
        """Delete returns 204 with no response body."""
        domain_id = domain_for_delete["id"]

        response = client.delete(f"{self.route}/{domain_id}")

        assert response.status_code == 204
        assert response.content == b""
        # No JSON body to parse
        assert response.headers.get("content-length", "0") == "0"

    def test_delete_domain_with_children_returns_412(self, client, firestore_client):
        """Delete domain with child grids without force returns 412."""
        # Create a domain
        domain_data = make_domain_data(name="Domain with children")
        domain_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        domain_ref.set(domain_data)

        # Create a child grid
        grid_data = make_grid_data(domain_id=domain_data["id"])
        grid_ref = firestore_client.collection(GRIDS_COLLECTION).document(
            grid_data["id"]
        )
        grid_ref.set(grid_data)

        try:
            response = client.delete(f"{self.route}/{domain_data['id']}")
            assert response.status_code == 412
            assert "child resources" in response.json()["detail"].lower()
        finally:
            # Cleanup
            grid_ref.delete()
            doc = domain_ref.get()
            if doc.exists:
                domain_ref.delete()

    def test_delete_domain_with_children_force_cascades(self, client, firestore_client):
        """Delete domain with force=true cascade-deletes child grids."""
        # Create a domain
        domain_data = make_domain_data(name="Domain for cascade delete")
        domain_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        domain_ref.set(domain_data)

        # Create child grids
        grid_ids = []
        for i in range(3):
            grid_data = make_grid_data(
                domain_id=domain_data["id"], name=f"Child grid {i}"
            )
            grid_ref = firestore_client.collection(GRIDS_COLLECTION).document(
                grid_data["id"]
            )
            grid_ref.set(grid_data)
            grid_ids.append(grid_data["id"])

        response = client.delete(f"{self.route}/{domain_data['id']}?force=true")
        assert response.status_code == 204

        # Verify domain is deleted
        doc = domain_ref.get()
        assert not doc.exists

        # Verify all child grids are deleted
        for grid_id in grid_ids:
            grid_doc = (
                firestore_client.collection(GRIDS_COLLECTION).document(grid_id).get()
            )
            assert not grid_doc.exists

    def test_delete_empty_domain_without_force_succeeds(self, client, firestore_client):
        """Delete domain with no children succeeds without force."""
        domain_data = make_domain_data(name="Empty domain")
        domain_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        domain_ref.set(domain_data)

        response = client.delete(f"{self.route}/{domain_data['id']}")
        assert response.status_code == 204

        doc = domain_ref.get()
        assert not doc.exists
