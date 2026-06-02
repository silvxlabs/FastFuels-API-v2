"""
Integration tests for api/v2/resources/features/layerset/router.py

Tests the POST /domains/{domain_id}/features/layerset/geojson endpoint.
These tests make real HTTP requests to the API and interact with Firestore.

The layerset endpoint differs from road/water:
- Synchronous: writes GeoJSON to GCS and Firestore in-request (no Cloud Tasks).
- Returns status="completed" immediately, not "pending".
- Includes a computed georeference in the response.
"""

import pytest
from api.resources.features.layerset.examples import LAYERSET_EXAMPLE_VALUES

# NOTE: Unlike road/water, this endpoint synchronously writes the GeoJSON
# payload to GCS via lib.gcs.blobs.upload_json. Tests hit the running API
# server in a separate process, so a unittest.mock.patch in the test process
# can't intercept that upload. Each test consequently writes a real blob to
# FEATURES_BUCKET. (This mirrors how the existing road/water mock_create_task
# fixtures are vestigial in practice — the real Cloud Tasks dispatch happens.)

# A minimal valid upload payload. The schema requires at least one Feature
# (empty collections can't be rasterized) and the router requires a projected
# CRS (geographic CRSes break rasterization in meters).
_PROJECTED_CRS_BLOCK = {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:EPSG::32612"},
}
_MINIMAL_FEATURE = {
    "type": "Feature",
    "properties": {
        "fuel_type": "shrub",
        "fuel_loading": 1.0,
        "fuel_height": 1.0,
        "percent_cover": 50,
        "distribution": "homogeneous",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [294000.0, 5199000.0],
                [294100.0, 5199000.0],
                [294100.0, 5199100.0],
                [294000.0, 5199000.0],
            ]
        ],
    },
}
_MINIMAL_PAYLOAD = {
    "type": "FeatureCollection",
    "crs": _PROJECTED_CRS_BLOCK,
    "features": [_MINIMAL_FEATURE],
}


class TestCreateLayerset:
    """Test the POST /domains/{domain_id}/features/layerset/geojson endpoint."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/features/layerset/geojson"

    def test_minimal_request_creates_feature(self, client, domain_for_testing):
        """Minimal request creates a feature document in completed state."""
        response = client.post(
            self.route(domain_for_testing["id"]), json=_MINIMAL_PAYLOAD
        )

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["type"] == "layerset"

        # Synchronous: returns already-completed
        assert data["status"] == "completed"

        # Defaults
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Source records that this is a user-uploaded layerset
        assert data["source"]["product"] == "Upload"

    def test_request_with_metadata(self, client, domain_for_testing):
        """Request accepts name, description, and tags."""
        request_body = {
            **_MINIMAL_PAYLOAD,
            "name": "Custom Surface Fuels",
            "description": "Hand-drawn fuelbed scenario",
            "tags": ["surface-fuels", "custom"],
        }

        response = client.post(self.route(domain_for_testing["id"]), json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Custom Surface Fuels"
        assert data["description"] == "Hand-drawn fuelbed scenario"
        assert data["tags"] == ["surface-fuels", "custom"]

    def test_invalid_domain_returns_404(self, client):
        """Non-existent domain_id returns 404."""
        response = client.post(
            self.route("00000000000000000000000000000000"), json=_MINIMAL_PAYLOAD
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_wrong_owner_domain_returns_404(self, client, domain_with_different_owner):
        """Domain owned by another user returns 404."""
        response = client.post(
            self.route(domain_with_different_owner["id"]), json=_MINIMAL_PAYLOAD
        )

        assert response.status_code == 404

    def test_response_excludes_owner_id(self, client, domain_for_testing):
        """Response should not expose the owner_id field to the client."""
        response = client.post(
            self.route(domain_for_testing["id"]), json=_MINIMAL_PAYLOAD
        )
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    def test_extracts_georeference_from_geojson(self, client, domain_for_testing):
        """Response georeference reflects the union bounds of every feature's geometry.

        The example payload declares ``crs == EPSG:32612`` (UTM 12N meters)
        on the FeatureCollection. The union bounds across all seven features
        span the Blackfoot-area UTM rectangle below (Lubrecht polygon shapes
        translated into the Blackfoot example domain); both the CRS and the
        bounds round-trip into the stored Feature's ``georeference``.
        """
        # Use the documented "minimal" example with all 7 features.
        payload = LAYERSET_EXAMPLE_VALUES[0][1]  # ("minimal", EXAMPLE_LAYERSET_MINIMAL)

        response = client.post(self.route(domain_for_testing["id"]), json=payload)
        assert response.status_code == 201

        data = response.json()
        assert data["georeference"] is not None
        assert data["georeference"]["crs"] == "EPSG:32612"

        bounds = data["georeference"]["bounds"]
        assert len(bounds) == 4
        assert bounds[0] == pytest.approx(294029.28510358, abs=0.01)
        assert bounds[1] == pytest.approx(5198853.44471689, abs=0.01)
        assert bounds[2] == pytest.approx(294849.82095037, abs=0.01)
        assert bounds[3] == pytest.approx(5199877.74955579, abs=0.01)

    def test_empty_feature_collection_returns_422(self, client, domain_for_testing):
        """An empty FeatureCollection is rejected at schema validation."""
        payload = {
            "type": "FeatureCollection",
            "crs": _PROJECTED_CRS_BLOCK,
            "features": [],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=payload)
        assert response.status_code == 422

    def test_no_crs_block_returns_422(self, client, domain_for_testing):
        """Omitting the GeoJSON `crs` block defaults to geographic EPSG:4326
        per the GeoJSON spec, which the rasterizer rejects. Surface this at
        upload time so the user gets a 422 instead of a deferred worker crash.
        """
        payload = {"type": "FeatureCollection", "features": [_MINIMAL_FEATURE]}
        response = client.post(self.route(domain_for_testing["id"]), json=payload)
        assert response.status_code == 422
        assert "projected" in response.json()["detail"].lower()

    def test_geographic_crs_block_returns_422(self, client, domain_for_testing):
        """An explicit geographic CRS (e.g. EPSG:4326) is rejected."""
        payload = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
            },
            "features": [_MINIMAL_FEATURE],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=payload)
        assert response.status_code == 422
        assert "geographic" in response.json()["detail"].lower()

    def test_unparseable_crs_returns_422(self, client, domain_for_testing):
        """A CRS string that pyproj can't parse returns 422 with a clear message."""
        payload = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "not-a-real-crs"},
            },
            "features": [_MINIMAL_FEATURE],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=payload)
        assert response.status_code == 422

    @pytest.mark.parametrize("example_name,example_value", LAYERSET_EXAMPLE_VALUES)
    def test_documented_example_creates_feature(
        self, client, domain_for_testing, example_name, example_value
    ):
        """Each documented LAYERSET example payload should successfully create a feature."""
        response = client.post(self.route(domain_for_testing["id"]), json=example_value)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["type"] == "layerset"
        assert data["status"] == "completed"
        assert data["source"]["product"] == "Upload"
