"""
Integration tests for
POST /domains/{domain_id}/features/{road,water,building,disturbance}/geojson

These tests make real HTTP requests to the API and interact with Firestore and
GCS. The endpoint is synchronous: it writes the GeoParquet blob and the
Firestore document in-request and returns status="completed".
"""

import pytest
from api.resources.features.building.examples import BUILDING_GEOJSON_EXAMPLE_VALUES
from api.resources.features.disturbance.examples import (
    DISTURBANCE_GEOJSON_EXAMPLE_VALUES,
)
from api.resources.features.road.examples import (
    EXAMPLE_ROAD_GEOJSON,
    ROAD_GEOJSON_EXAMPLE_VALUES,
)
from api.resources.features.water.examples import WATER_GEOJSON_EXAMPLE_VALUES

from lib.config import FEATURES_COLLECTION

DOCUMENTED_EXAMPLES = (
    [("road", name, value) for name, value in ROAD_GEOJSON_EXAMPLE_VALUES]
    + [("water", name, value) for name, value in WATER_GEOJSON_EXAMPLE_VALUES]
    + [("building", name, value) for name, value in BUILDING_GEOJSON_EXAMPLE_VALUES]
    + [
        ("disturbance", name, value)
        for name, value in DISTURBANCE_GEOJSON_EXAMPLE_VALUES
    ]
)

# A polygon in the Pacific, far from every test domain.
_OUTSIDE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-140.0, 30.0], [-139.9, 30.0], [-139.9, 30.1], [-140.0, 30.0]]
                ],
            },
        }
    ],
}


@pytest.fixture
def created_features(firestore_client):
    """Collects created feature ids and deletes them on teardown."""
    ids = []
    yield ids
    for feature_id in ids:
        firestore_client.collection(FEATURES_COLLECTION).document(feature_id).delete()


def route(domain_id, feature_type):
    return f"/domains/{domain_id}/features/{feature_type}/geojson"


def upload(client, domain_id, feature_type, body, created_features):
    response = client.post(route(domain_id, feature_type), json=body)
    if response.status_code == 201:
        created_features.append(response.json()["id"])
    return response


class TestCreateGeojsonFeature:
    @pytest.mark.parametrize(
        "feature_type,example_name,example_value", DOCUMENTED_EXAMPLES
    )
    def test_documented_example_creates_feature(
        self,
        client,
        blackfoot_domain,
        created_features,
        feature_type,
        example_name,
        example_value,
    ):
        """Each documented example uploads, is stored, and reads back."""
        domain_id = blackfoot_domain["id"]
        response = upload(
            client, domain_id, feature_type, example_value, created_features
        )
        assert response.status_code == 201, (
            f"Example '{example_name}' failed: {response.json()}"
        )

        data = response.json()
        assert data["type"] == feature_type
        assert data["status"] == "completed"
        assert data["name"] == example_value["name"]
        assert data["tags"] == example_value["tags"]
        assert data["source"] == {
            "product": "geojson",
            "description": "User-uploaded GeoJSON",
            "extent_buffer_m": 0,
        }
        assert data["georeference"]["crs"] == "EPSG:32612"
        assert data["attributions"] == []
        assert "owner_id" not in data

        metadata = client.get(
            f"/domains/{domain_id}/features/{data['id']}/data/metadata"
        )
        assert metadata.status_code == 200
        assert metadata.json()["total_features"] == len(example_value["features"])

        rows = client.get(f"/domains/{domain_id}/features/{data['id']}/data/0")
        assert rows.status_code == 200
        properties = [f["properties"] for f in rows.json()["features"]]
        assert properties == [f["properties"] for f in example_value["features"]]

    def test_size_bytes_recorded(
        self, client, firestore_client, blackfoot_domain, created_features
    ):
        """The stored document carries size_bytes for the storage quota."""
        response = upload(
            client,
            blackfoot_domain["id"],
            "road",
            EXAMPLE_ROAD_GEOJSON,
            created_features,
        )
        assert response.status_code == 201
        doc = (
            firestore_client.collection(FEATURES_COLLECTION)
            .document(response.json()["id"])
            .get()
            .to_dict()
        )
        assert doc["size_bytes"] > 0

    def test_extent_buffer_m_recorded(self, client, blackfoot_domain, created_features):
        body = {**EXAMPLE_ROAD_GEOJSON, "extent_buffer_m": 25}
        response = upload(
            client, blackfoot_domain["id"], "road", body, created_features
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_m"] == 25

    def test_nothing_intersecting_returns_422(
        self, client, blackfoot_domain, created_features
    ):
        response = upload(
            client, blackfoot_domain["id"], "water", _OUTSIDE, created_features
        )
        assert response.status_code == 422
        assert "intersect" in response.json()["detail"]

    def test_line_geometry_returns_422(
        self, client, blackfoot_domain, created_features
    ):
        feature = EXAMPLE_ROAD_GEOJSON["features"][0]
        line = {
            "type": "LineString",
            "coordinates": feature["geometry"]["coordinates"][0],
        }
        body = {
            "type": "FeatureCollection",
            "features": [{**feature, "geometry": line}],
        }
        response = upload(
            client, blackfoot_domain["id"], "road", body, created_features
        )
        assert response.status_code == 422

    def test_mixed_property_types_return_422(
        self, client, blackfoot_domain, created_features
    ):
        first, second = EXAMPLE_ROAD_GEOJSON["features"]
        body = {
            "type": "FeatureCollection",
            "features": [
                {**first, "properties": {"width": 4}},
                {**second, "properties": {"width": "4 m"}},
            ],
        }
        response = upload(
            client, blackfoot_domain["id"], "road", body, created_features
        )
        assert response.status_code == 422
        assert "width" in response.json()["detail"]

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, created_features
    ):
        response = upload(
            client,
            domain_with_different_owner["id"],
            "road",
            EXAMPLE_ROAD_GEOJSON,
            created_features,
        )
        assert response.status_code == 404

    def test_crown_has_no_geojson_source(self, client, blackfoot_domain):
        response = client.post(
            route(blackfoot_domain["id"], "crown"), json=EXAMPLE_ROAD_GEOJSON
        )
        assert response.status_code in (404, 405)
