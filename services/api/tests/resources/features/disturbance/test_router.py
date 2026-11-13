"""
Integration tests for POST /domains/{domain_id}/features/disturbance/geojson

The upload itself is tested with the other GeoJSON uploads in
``tests/resources/features/test_geojson_router.py``. These tests cover only the
disturbance properties. They make real HTTP requests to the API.
"""

import copy

import pytest
from api.resources.features.disturbance.examples import EXAMPLE_DISTURBANCE_GEOJSON


def route(domain_id):
    return f"/domains/{domain_id}/features/disturbance/geojson"


def _with_properties(properties):
    body = copy.deepcopy(EXAMPLE_DISTURBANCE_GEOJSON)
    body["features"][0]["properties"] = properties
    return body


class TestCreateGeojsonDisturbanceFeature:
    @pytest.mark.parametrize("code", [0, 144, 811])
    def test_invalid_fdist_returns_422(self, client, blackfoot_domain, code):
        response = client.post(
            route(blackfoot_domain["id"]), json=_with_properties({"fdist": code})
        )
        assert response.status_code == 422

    def test_missing_fdist_returns_422(self, client, blackfoot_domain):
        response = client.post(
            route(blackfoot_domain["id"]), json=_with_properties({"name": "Fire"})
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("year", [1899, 2101, 2**63])
    def test_year_out_of_range_returns_422(self, client, blackfoot_domain, year):
        response = client.post(
            route(blackfoot_domain["id"]),
            json=_with_properties({"fdist": 122, "year": year}),
        )
        assert response.status_code == 422
