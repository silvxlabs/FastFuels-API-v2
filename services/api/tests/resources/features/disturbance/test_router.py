"""
Integration tests for api/v2/resources/features/disturbance/router.py

Tests the POST /domains/{domain_id}/features/disturbance/geojson endpoint.
These tests make real HTTP requests to the API and interact with Firestore.

The disturbance endpoint is synchronous like layerset: it writes the Parquet to
GCS and the record to Firestore in-request and returns status="completed".

The documented examples are placed over the Blackfoot area, but the test domain
is elsewhere. Payloads are translated onto the test domain's bbox and declared
in its CRS first, so the upload's domain-overlap check passes.
"""

import copy

import pytest
from api.resources.features.disturbance.examples import (
    DISTURBANCE_EXAMPLE_VALUES,
    EXAMPLE_DISTURBANCE_MINIMAL,
)
from pyproj import Transformer


def _points(payload):
    return [
        point
        for feature in payload["features"]
        for ring in feature["geometry"]["coordinates"]
        for point in ring
    ]


def _bounds(payload):
    points = _points(payload)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _fit_to_domain(payload, domain):
    """Translate a payload onto the domain's bbox, declared in the domain's CRS."""
    payload = copy.deepcopy(payload)
    minx, miny, _, _ = domain["bbox"]
    left, bottom, _, _ = _bounds(payload)
    dx, dy = minx + 100.0 - left, miny + 100.0 - bottom
    for point in _points(payload):
        point[0] += dx
        point[1] += dy
    payload["crs"] = copy.deepcopy(domain["crs"])
    return payload


class TestCreateDisturbance:
    """Test the POST /domains/{domain_id}/features/disturbance/geojson endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features/disturbance/geojson"

    def test_minimal_request_creates_feature(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "disturbance"
        assert data["status"] == "completed"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []
        assert data["source"]["product"] == "Upload"
        assert data["source"]["mapping_year"] == body["mapping_year"]

    def test_request_with_metadata(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        body.update(
            name="Fire and thinning history",
            description="Fires and thinning",
            tags=["disturbance", "fire"],
        )

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Fire and thinning history"
        assert data["description"] == "Fires and thinning"
        assert data["tags"] == ["disturbance", "fire"]

    def test_invalid_domain_returns_404(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)

        response = client.post(
            self.route("00000000000000000000000000000000"), json=body
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_with_different_owner)

        response = client.post(self.route(domain_with_different_owner["id"]), json=body)

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        assert "owner_id" not in response.json()

    def test_georeference_is_the_union_bounds_in_the_files_crs(
        self, client, domain_for_testing
    ):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        georeference = response.json()["georeference"]
        assert georeference["crs"] == domain_for_testing["crs"]["properties"]["name"]
        assert georeference["bounds"] == pytest.approx(_bounds(body))

    def test_lat_lon_polygons_are_accepted(self, client, domain_for_testing):
        """With no crs block the coordinates are read as EPSG:4326."""
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        transformer = Transformer.from_crs(
            domain_for_testing["crs"]["properties"]["name"], "EPSG:4326", always_xy=True
        )
        del body["crs"]
        for point in _points(body):
            point[0], point[1] = transformer.transform(point[0], point[1])

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        assert response.json()["georeference"]["crs"] == "EPSG:4326"

    def test_empty_feature_collection_returns_422(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        body["features"] = []

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    def test_missing_mapping_year_returns_422(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        del body["mapping_year"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    @pytest.mark.parametrize("year", [1899, 2101])
    def test_mapping_year_out_of_range_returns_422(
        self, client, domain_for_testing, year
    ):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        body["mapping_year"] = year

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    @pytest.mark.parametrize("code", [0, 144, 811])
    def test_invalid_fdist_returns_422(self, client, domain_for_testing, code):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        body["features"][0]["properties"]["fdist"] = code

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    def test_polygons_outside_the_domain_return_422(self, client, domain_for_testing):
        body = _fit_to_domain(EXAMPLE_DISTURBANCE_MINIMAL, domain_for_testing)
        for feature in body["features"]:
            feature["geometry"]["coordinates"] = [
                [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]]
            ]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 422
        assert "intersect" in response.json()["detail"].lower()

    @pytest.mark.parametrize("example_name,example_value", DISTURBANCE_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented example, placed on the test domain, creates a feature."""
        body = _fit_to_domain(example_value, domain_for_testing)

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["type"] == "disturbance"
        assert data["status"] == "completed"
        assert data["source"]["product"] == "Upload"
