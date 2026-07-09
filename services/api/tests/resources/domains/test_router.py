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
from api.resources.domains.examples import (
    ALL_EXAMPLE_VALUES,
    EXAMPLE_PADDED,
    EXAMPLE_WGS84_DEFAULT,
)
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

        # Two-feature response: "domain" (working extent) and "input" (original polygon)
        assert len(data["features"]) >= 2
        names = [f["properties"]["name"] for f in data["features"]]
        assert "domain" in names
        assert "input" in names

        # bbox field should be populated and equal the "domain" feature's bounds
        assert "bbox" in data
        assert data["bbox"] is not None
        assert len(data["bbox"]) == 4

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

    def test_pad_to_resolution_snaps_domain_bbox(self, client):
        """pad_to_resolution=30 should snap the 'domain' feature bbox to multiples of 30."""
        with open(DATA_DIR / "blue_mountain_feature_4326.geojson") as f:
            feature = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [feature]
                if feature.get("type") == "Feature"
                else feature.get("features", [])
            ),
            "name": "Padded Domain",
            "pad_to_resolution": 30,
        }

        response = client.post("/domains", json=request_body)

        assert response.status_code == 201
        data = response.json()

        # Response should echo pad_to_resolution
        assert data["pad_to_resolution"] == 30

        # Two features: domain + input
        assert len(data["features"]) == 2
        names = [f["properties"]["name"] for f in data["features"]]
        assert names == ["domain", "input"]

        # bbox field should be present and snapped to multiples of 30
        assert data["bbox"] is not None
        for value in data["bbox"]:
            assert value % 30 == 0, f"bbox value {value} is not a multiple of 30"

        # The "domain" feature's geometry should match the bbox
        domain_feature = data["features"][0]
        assert domain_feature["properties"]["name"] == "domain"
        coords = domain_feature["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        assert (min(xs), min(ys), max(xs), max(ys)) == tuple(data["bbox"])

        DOMAINS.append(data)

    def test_pad_to_resolution_none_omitted_from_response(self, client):
        """When pad_to_resolution is not provided, it should be omitted from the response."""
        with open(DATA_DIR / "blue_mountain_feature_4326.geojson") as f:
            feature = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [feature]
                if feature.get("type") == "Feature"
                else feature.get("features", [])
            ),
            "name": "Unpadded Domain",
        }

        response = client.post("/domains", json=request_body)

        assert response.status_code == 201
        data = response.json()

        # response_model_exclude_none should drop pad_to_resolution when None
        assert "pad_to_resolution" not in data

        # Should still have two features and bbox
        assert len(data["features"]) == 2
        assert "bbox" in data

        DOMAINS.append(data)


class TestPreviewDomain:
    """Test the POST /domains/preview endpoint.

    Preview runs the same validation + projection pipeline as create but
    returns id="preview" and never writes to Firestore.
    """

    route = "/domains/preview"

    def test_preview_returns_200(self, client):
        """Preview returns 200 (not 201 — nothing is created)."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)

        assert response.status_code == 200

    def test_preview_id_is_preview(self, client):
        """Preview response always has id='preview'."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)

        assert response.status_code == 200
        assert response.json()["id"] == "preview"

    def test_preview_returns_two_features(self, client):
        """Preview returns the two-feature FeatureCollection (domain + input)."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)

        assert response.status_code == 200
        data = response.json()

        assert len(data["features"]) == 2
        names = [f["properties"]["name"] for f in data["features"]]
        assert "domain" in names
        assert "input" in names

    def test_preview_returns_bbox(self, client):
        """Preview response includes a valid 4-element bbox."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)

        assert response.status_code == 200
        data = response.json()

        assert "bbox" in data
        assert len(data["bbox"]) == 4

    def test_preview_pad_to_resolution_honored(self, client):
        """pad_to_resolution snaps the domain bbox in preview, same as create."""
        response = client.post(self.route, json=EXAMPLE_PADDED)

        assert response.status_code == 200
        data = response.json()

        assert data["pad_to_resolution"] == 30
        for value in data["bbox"]:
            assert value % 30 == 0, f"bbox value {value} is not a multiple of 30"

    def test_preview_no_document_written(self, client, firestore_client):
        """Preview must not create any Firestore document."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)
        assert response.status_code == 200

        doc = firestore_client.collection(DOMAINS_COLLECTION).document("preview").get()
        assert not doc.exists

    def test_preview_zero_area_returns_422(self, client):
        """Point geometry (zero area) propagates as 422."""
        with open(DATA_DIR / "point.geojson") as f:
            point = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [point] if point.get("type") == "Feature" else point.get("features", [])
            ),
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "area greater than zero" in response.json()["detail"]

    def test_preview_outside_conus_returns_422(self, client):
        """Domain outside CONUS propagates as 422."""
        with open(DATA_DIR / "polygon_in_alaska.geojson") as f:
            alaska = json.load(f)

        request_body = {
            "type": "FeatureCollection",
            "features": (
                [alaska]
                if alaska.get("type") == "Feature"
                else alaska.get("features", [])
            ),
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "within CONUS" in response.json()["detail"]

    def test_preview_invalid_crs_returns_422(self, client):
        """Invalid CRS propagates as 422."""
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
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422
        assert "Invalid CRS" in response.json()["detail"]


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

    def test_pad_to_resolution_zero_returns_422(self, client):
        """pad_to_resolution=0 should return 422 (gt=0 constraint)."""
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
            "pad_to_resolution": 0,
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422

    def test_pad_to_resolution_negative_returns_422(self, client):
        """pad_to_resolution=-1 should return 422 (gt=0 constraint)."""
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
            "pad_to_resolution": -1,
        }

        response = client.post(self.route, json=request_body)

        assert response.status_code == 422


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


class TestGetDomainLattice:
    """Test the GET /domains/{domain_id}/lattice endpoint.

    Uses the same ``domain_in_firestore`` fixture as ``TestGetDomain``.
    The fixture domain is a 1000m x 1000m square in EPSG:32611 with
    ``bbox = [500000, 5200000, 501000, 5201000]``.
    """

    @staticmethod
    def route(domain_id: str) -> str:
        return f"/domains/{domain_id}/lattice"

    def test_returns_lattice_at_30m(self, client, domain_in_firestore):
        """30m lattice over 1000m square: ceil(1000/30) = 34 cells per side."""
        response = client.get(
            self.route(domain_in_firestore["id"]), params={"resolution": 30}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["crs"] == "EPSG:32611"
        assert data["resolution"] == 30.0
        assert data["num_buffer_cells"] == 0
        assert data["shape"] == [34, 34]
        # transform: a=30, b=0, c=minx=500000, d=0, e=-30, f=miny+height*res
        assert data["transform"] == [30.0, 0.0, 500000.0, 0.0, -30.0, 5201020.0]

    def test_returns_lattice_at_50m(self, client, domain_in_firestore):
        """50m divides evenly into 1000m: 20 cells per side, snug bounds."""
        response = client.get(
            self.route(domain_in_firestore["id"]), params={"resolution": 50}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["shape"] == [20, 20]
        assert data["transform"] == [50.0, 0.0, 500000.0, 0.0, -50.0, 5201000.0]

    def test_buffer_cells_expands_lattice(self, client, domain_in_firestore):
        """num_buffer_cells=2 at 30m adds 60m on each side; shape grows by 4."""
        response = client.get(
            self.route(domain_in_firestore["id"]),
            params={"resolution": 30, "num_buffer_cells": 2},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["num_buffer_cells"] == 2
        # Expanded bounds: 1120m wide → ceil(1120/30) = 38 cells per side.
        assert data["shape"] == [38, 38]
        # Origin shifts left/up by 2*30=60m. c = 500000-60, f = miny-60 + 38*30.
        assert data["transform"] == [30.0, 0.0, 499940.0, 0.0, -30.0, 5201080.0]

    def test_missing_resolution_returns_422(self, client, domain_in_firestore):
        """resolution is required."""
        response = client.get(self.route(domain_in_firestore["id"]))
        assert response.status_code == 422

    def test_zero_resolution_returns_422(self, client, domain_in_firestore):
        """resolution must be > 0."""
        response = client.get(
            self.route(domain_in_firestore["id"]), params={"resolution": 0}
        )
        assert response.status_code == 422

    def test_negative_resolution_returns_422(self, client, domain_in_firestore):
        """Negative resolution is rejected."""
        response = client.get(
            self.route(domain_in_firestore["id"]), params={"resolution": -10}
        )
        assert response.status_code == 422

    def test_negative_buffer_returns_422(self, client, domain_in_firestore):
        """num_buffer_cells must be >= 0."""
        response = client.get(
            self.route(domain_in_firestore["id"]),
            params={"resolution": 30, "num_buffer_cells": -1},
        )
        assert response.status_code == 422

    def test_nonexistent_domain_returns_404(self, client):
        """Unknown domain id returns 404."""
        response = client.get(
            self.route("00000000000000000000000000000000"),
            params={"resolution": 30},
        )
        assert response.status_code == 404

    def test_wrong_owner_returns_404(self, client, domain_with_different_owner):
        """Domain owned by a different user returns 404, not 403."""
        response = client.get(
            self.route(domain_with_different_owner["id"]),
            params={"resolution": 30},
        )
        assert response.status_code == 404


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

    def test_list_returns_user_domains(self, isolated_owner):
        """List returns exactly the authenticated owner's domains.

        Runs on a fresh isolated owner so the result is bounded to the seeded
        set and never buried by the shared owner's accumulated test data.
        """
        client, owner_id, seed = isolated_owner
        seeded = [
            seed(
                DOMAINS_COLLECTION,
                make_domain_data(
                    owner_id=owner_id, name=f"List Test Domain {chr(65 + i)}"
                ),
            )
            for i in range(3)
        ]

        response = client.get(self.route)

        assert response.status_code == 200
        data = response.json()
        assert data["total_items"] == 3
        domain_ids = [d["id"] for d in data["domains"]]
        for domain in seeded:
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

    @pytest.mark.parametrize("sort_by", ["created_on", "modified_on", "name"])
    @pytest.mark.parametrize("sort_order", [None, "ascending", "descending"])
    def test_list_sorting_matrix_returns_200(
        self, client, domains_for_listing, sort_by, sort_order
    ):
        """Every sort field/direction combination is served (issue #321)."""
        url = f"{self.route}?sort_by={sort_by}"
        if sort_order:
            url += f"&sort_order={sort_order}"
        response = client.get(url)
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


class TestCreateDomainStyle:
    """Style behavior on POST /domains."""

    route = "/domains"

    def test_omitted_style_is_absent_from_response(self, client):
        """Without an explicit style, no style is stored or returned."""
        response = client.post(self.route, json=EXAMPLE_WGS84_DEFAULT)

        assert response.status_code == 201
        data = response.json()

        # response_model_exclude_none=True drops the field entirely when None.
        assert "style" not in data

        DOMAINS.append(data)

    def test_explicit_full_style_round_trips(self, client):
        body = {
            **EXAMPLE_WGS84_DEFAULT,
            "name": "Explicit Style Domain",
            "style": {
                "stroke_color": "#123456",
                "stroke_opacity": 0.75,
                "stroke_width": 3,
                "fill_color": "#abcdef",
                "fill_opacity": 0.25,
            },
        }
        response = client.post(self.route, json=body)

        assert response.status_code == 201
        style = response.json()["style"]
        assert style == {
            "stroke_color": "#123456",
            "stroke_opacity": 0.75,
            "stroke_width": 3,
            "fill_color": "#abcdef",
            "fill_opacity": 0.25,
        }

        DOMAINS.append(response.json())

    def test_partial_style_does_not_get_filled_in(self, client):
        """A user-supplied partial style is honored as-is (no auto-fill)."""
        body = {
            **EXAMPLE_WGS84_DEFAULT,
            "name": "Partial Style Domain",
            "style": {"fill_color": "#abcdef"},
        }
        response = client.post(self.route, json=body)

        assert response.status_code == 201
        style = response.json()["style"]
        # response_model_exclude_none drops null sub-fields entirely.
        assert style == {"fill_color": "#abcdef"}

        DOMAINS.append(response.json())

    def test_accepts_named_color(self, client):
        """Color strings aren't format-validated — named colors etc. round-trip."""
        body = {
            **EXAMPLE_WGS84_DEFAULT,
            "name": "Named Color Domain",
            "style": {"fill_color": "red", "stroke_color": "rgb(0, 0, 0)"},
        }
        response = client.post(self.route, json=body)
        assert response.status_code == 201
        style = response.json()["style"]
        assert style["fill_color"] == "red"
        assert style["stroke_color"] == "rgb(0, 0, 0)"
        DOMAINS.append(response.json())

    def test_overlong_color_returns_422(self, client):
        body = {
            **EXAMPLE_WGS84_DEFAULT,
            "name": "Overlong Color Domain",
            "style": {"fill_color": "a" * 65},
        }
        response = client.post(self.route, json=body)
        assert response.status_code == 422

    def test_out_of_range_opacity_returns_422(self, client):
        body = {
            **EXAMPLE_WGS84_DEFAULT,
            "name": "Bad Opacity Domain",
            "style": {"fill_opacity": 1.5},
        }
        response = client.post(self.route, json=body)
        assert response.status_code == 422


class TestUpdateDomainStyle:
    """Style merge semantics on PATCH /domains/{id}."""

    route = "/domains"

    @pytest.fixture
    def styled_domain(self, client):
        """Create a domain via the API so it has a server-populated default style."""
        response = client.post(
            self.route,
            json={
                **EXAMPLE_WGS84_DEFAULT,
                "name": "Style PATCH Domain",
                "style": {
                    "stroke_color": "#111111",
                    "stroke_opacity": 1.0,
                    "stroke_width": 2,
                    "fill_color": "#222222",
                    "fill_opacity": 0.5,
                },
            },
        )
        assert response.status_code == 201
        domain = response.json()
        DOMAINS.append(domain)
        return domain

    def test_patch_merges_partial_style(self, client, styled_domain):
        """PATCH with one sub-field updates that field and preserves the others."""
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_color": "#abcdef"}},
        )
        assert response.status_code == 200

        style = response.json()["style"]
        assert style["fill_color"] == "#abcdef"
        # Untouched sub-fields are preserved.
        assert style["stroke_color"] == "#111111"
        assert style["stroke_opacity"] == 1.0
        assert style["stroke_width"] == 2
        assert style["fill_opacity"] == 0.5

    def test_patch_updates_multiple_fields_at_once(self, client, styled_domain):
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_color": "#aaaaaa", "fill_opacity": 0.1}},
        )
        assert response.status_code == 200

        style = response.json()["style"]
        assert style["fill_color"] == "#aaaaaa"
        assert style["fill_opacity"] == 0.1
        # Stroke side untouched.
        assert style["stroke_color"] == "#111111"
        assert style["stroke_opacity"] == 1.0
        assert style["stroke_width"] == 2

    def test_patch_geometry_unchanged_after_style_update(self, client, styled_domain):
        """Updating style must not modify the FeatureCollection."""
        domain_id = styled_domain["id"]
        original_features = styled_domain["features"]
        original_bbox = styled_domain["bbox"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_color": "#dddddd"}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["features"] == original_features
        assert data["bbox"] == original_bbox

    def test_patch_style_alongside_other_fields(self, client, styled_domain):
        """Style and metadata can be updated in the same PATCH."""
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={
                "name": "Renamed and Restyled",
                "style": {"fill_color": "#cccccc"},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Renamed and Restyled"
        assert data["style"]["fill_color"] == "#cccccc"
        # Untouched style sub-fields preserved.
        assert data["style"]["stroke_color"] == "#111111"

    def test_patch_overlong_color_returns_422(self, client, styled_domain):
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_color": "a" * 65}},
        )
        assert response.status_code == 422

    def test_patch_out_of_range_opacity_returns_422(self, client, styled_domain):
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_opacity": 2.0}},
        )
        assert response.status_code == 422

    def test_patch_negative_stroke_width_returns_422(self, client, styled_domain):
        domain_id = styled_domain["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"stroke_width": -1}},
        )
        assert response.status_code == 422

    def test_patch_style_on_legacy_domain_with_no_style(self, client, firestore_client):
        """A domain stored before the style field existed accepts PATCH style."""
        domain_data = make_domain_data(name="Legacy unstyled domain")
        # Ensure no style is stored; make_domain_data omits it already.
        assert "style" not in domain_data
        doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
            domain_data["id"]
        )
        doc_ref.set(domain_data)

        try:
            response = client.patch(
                f"{self.route}/{domain_data['id']}",
                json={"style": {"fill_color": "#abcdef"}},
            )
            assert response.status_code == 200
            style = response.json()["style"]
            assert style["fill_color"] == "#abcdef"
        finally:
            doc_ref.delete()

    def test_patch_style_wrong_owner_returns_404(
        self, client, domain_with_different_owner
    ):
        """Style PATCH respects ownership — 404 (not 403) for other users' domains."""
        domain_id = domain_with_different_owner["id"]

        response = client.patch(
            f"{self.route}/{domain_id}",
            json={"style": {"fill_color": "#abcdef"}},
        )
        assert response.status_code == 404

    def test_patch_nonexistent_domain_style_returns_404(self, client):
        fake_id = "00000000000000000000000000000000"

        response = client.patch(
            f"{self.route}/{fake_id}",
            json={"style": {"fill_color": "#abcdef"}},
        )
        assert response.status_code == 404


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

    def test_delete_does_not_appear_in_list(self, isolated_owner):
        """Deleted domain does not appear in list endpoint.

        Runs on a fresh isolated owner so the pre-delete presence check isn't
        defeated by the shared owner's accumulated test data.
        """
        client, owner_id, seed = isolated_owner
        domain_id = seed(
            DOMAINS_COLLECTION,
            make_domain_data(owner_id=owner_id, name="Domain to Delete"),
        )["id"]

        # Verify it appears in list before delete
        domain_ids_before = [d["id"] for d in client.get(self.route).json()["domains"]]
        assert domain_id in domain_ids_before

        # Delete the domain
        client.delete(f"{self.route}/{domain_id}")

        # Verify it no longer appears in list
        domain_ids_after = [d["id"] for d in client.get(self.route).json()["domains"]]
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


class TestReprojectDomain:
    """Test the POST /domains/reproject endpoint.

    Stateless reprojection — no Firestore reads or writes.
    """

    route = "/domains/reproject"

    # Small WGS84 polygon in western Montana (UTM zone 12N / EPSG:32612)
    WGS84_FC = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "test"},
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
    }

    def test_reproject_returns_200(self, client):
        """Reproject returns 200."""
        response = client.post(
            self.route, params={"target_epsg": 5070}, json=self.WGS84_FC
        )
        assert response.status_code == 200

    def test_round_trip_4326_to_5070_to_4326(self, client):
        """Round-trip 4326 → EPSG:5070 → 4326 returns approximately original coordinates."""
        response_5070 = client.post(
            self.route, params={"target_epsg": 5070}, json=self.WGS84_FC
        )
        assert response_5070.status_code == 200

        response_4326 = client.post(
            self.route, params={"target_epsg": 4326}, json=response_5070.json()
        )
        assert response_4326.status_code == 200

        orig_coords = self.WGS84_FC["features"][0]["geometry"]["coordinates"][0]
        result_coords = response_4326.json()["features"][0]["geometry"]["coordinates"][
            0
        ]
        for orig, result in zip(orig_coords, result_coords):
            assert abs(orig[0] - result[0]) < 1e-5
            assert abs(orig[1] - result[1]) < 1e-5

    def test_preserves_feature_properties(self, client):
        """Reprojected features retain original properties."""
        response = client.post(
            self.route, params={"target_epsg": 5070}, json=self.WGS84_FC
        )
        assert response.status_code == 200
        data = response.json()
        assert data["features"][0]["properties"]["name"] == "test"

    def test_preserves_multi_feature_input(self, client):
        """All features in a multi-feature FeatureCollection are reprojected and returned."""
        multi_fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
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
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-114.02, 46.81],
                                [-114.03, 46.81],
                                [-114.03, 46.80],
                                [-114.02, 46.80],
                                [-114.02, 46.81],
                            ]
                        ],
                    },
                },
            ],
        }
        response = client.post(self.route, params={"target_epsg": 5070}, json=multi_fc)
        assert response.status_code == 200
        data = response.json()
        assert len(data["features"]) == 2
        ids = {f["properties"]["id"] for f in data["features"]}
        assert ids == {1, 2}

    def test_response_crs_reflects_target(self, client):
        """Response FeatureCollection crs field is set to the target EPSG."""
        response = client.post(
            self.route, params={"target_epsg": 5070}, json=self.WGS84_FC
        )
        assert response.status_code == 200
        data = response.json()
        assert data["crs"]["properties"]["name"] == "EPSG:5070"

    def test_invalid_target_epsg_returns_422(self, client):
        """Invalid target EPSG returns 422."""
        response = client.post(
            self.route, params={"target_epsg": 999999}, json=self.WGS84_FC
        )
        assert response.status_code == 422
        assert "Invalid CRS" in response.json()["detail"]

    def test_invalid_source_crs_returns_422(self, client):
        """Invalid source CRS in the request body returns 422."""
        fc = {
            **self.WGS84_FC,
            "crs": {"type": "name", "properties": {"name": "INVALID:9999"}},
        }
        response = client.post(self.route, params={"target_epsg": 5070}, json=fc)
        assert response.status_code == 422
        assert "Invalid CRS" in response.json()["detail"]
