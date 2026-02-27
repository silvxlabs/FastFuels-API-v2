"""
Tests for lib.domain_utils.

Uses real domain JSON fixtures (same ones used by griddle integration tests)
to test the full parsing pipeline: stringified coordinates, GeoJSON CRS
objects, multiple features, error handling.
"""

import copy
import json

import geopandas as gpd
import pytest

from lib.domain_utils import (
    EmptyDomainError,
    InvalidGeometryError,
    parse_domain_gdf,
)
from lib.testing import SHARED_TEST_DOMAINS_DIR

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


def _load_domain_json(filename: str) -> dict:
    """Load a domain JSON file from griddle test data."""
    with open(DOMAINS_DIR / filename) as f:
        return json.load(f)


def _stringify_coordinates(domain_data: dict) -> dict:
    """Simulate Firestore serialization: stringify nested coordinate arrays."""
    data = copy.deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


class TestWithBlueMountainDomain:
    """Tests using the Blue Mountain domain (EPSG:32611, ~1km², single polygon)."""

    @pytest.fixture
    def blue_mtn_raw(self):
        return _load_domain_json("blue_mtn.json")

    @pytest.fixture
    def blue_mtn_firestore(self, blue_mtn_raw):
        """Blue Mountain domain as it would be stored in Firestore."""
        return _stringify_coordinates(blue_mtn_raw)

    def test_returns_geodataframe(self, blue_mtn_firestore):
        result = parse_domain_gdf(blue_mtn_firestore)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_parsed_from_geojson_object(self, blue_mtn_firestore):
        result = parse_domain_gdf(blue_mtn_firestore)
        assert result.crs.to_epsg() == 32611

    def test_single_feature(self, blue_mtn_firestore):
        result = parse_domain_gdf(blue_mtn_firestore)
        assert len(result) == 1

    def test_geometry_is_polygon(self, blue_mtn_firestore):
        result = parse_domain_gdf(blue_mtn_firestore)
        assert result.geometry.iloc[0].geom_type == "Polygon"

    def test_bounds_in_utm_range(self, blue_mtn_firestore):
        """Bounds should be in UTM Zone 11N range (~720k-722k E, ~5189k-5191k N)."""
        result = parse_domain_gdf(blue_mtn_firestore)
        bounds = result.total_bounds  # [minx, miny, maxx, maxy]
        assert 720000 < bounds[0] < 722000
        assert 5189000 < bounds[1] < 5191000

    def test_unstringified_coordinates_also_work(self, blue_mtn_raw):
        """Coordinates as native arrays (not stringified) should also parse."""
        result = parse_domain_gdf(blue_mtn_raw)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 32611


class TestWithTileBoundaryDomain:
    """Tests using the tile boundary domain (EPSG:32612, 1km square)."""

    @pytest.fixture
    def tile_boundary_firestore(self):
        return _stringify_coordinates(_load_domain_json("meta_chm_2_tiles.json"))

    def test_different_utm_zone(self, tile_boundary_firestore):
        result = parse_domain_gdf(tile_boundary_firestore)
        assert result.crs.to_epsg() == 32612

    def test_square_domain_bounds(self, tile_boundary_firestore):
        """1km square domain should have ~1000m extent in both dimensions."""
        result = parse_domain_gdf(tile_boundary_firestore)
        bounds = result.total_bounds
        x_extent = bounds[2] - bounds[0]
        y_extent = bounds[3] - bounds[1]
        assert pytest.approx(x_extent, rel=0.01) == 1000.0
        assert pytest.approx(y_extent, rel=0.01) == 1000.0


class TestCrsHandling:
    """Tests for CRS edge cases."""

    def test_crs_as_plain_string(self):
        """CRS stored as a plain string (not GeoJSON object)."""
        data = {
            "crs": "EPSG:4326",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 38]]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        result = parse_domain_gdf(data)
        assert result.crs.to_epsg() == 4326

    def test_missing_crs_defaults_to_4326(self):
        data = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 38]]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        result = parse_domain_gdf(data)
        assert result.crs.to_epsg() == 4326

    def test_crs_none_defaults_to_4326(self):
        data = {
            "crs": None,
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 38]]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        result = parse_domain_gdf(data)
        assert result.crs.to_epsg() == 4326


class TestMultipleFeatures:
    """Tests for domains with multiple polygon features."""

    def test_two_polygons(self):
        data = {
            "crs": {"type": "name", "properties": {"name": "EPSG:32610"}},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                        ),
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]]
                        ),
                    },
                    "properties": {},
                },
            ],
        }
        result = parse_domain_gdf(data)
        assert len(result) == 2
        bounds = result.total_bounds
        assert bounds[0] == 0  # minx
        assert bounds[1] == 0  # miny
        assert bounds[2] == 30  # maxx
        assert bounds[3] == 30  # maxy


class TestErrorHandling:
    """Tests for error conditions."""

    def test_empty_features_raises(self):
        with pytest.raises(EmptyDomainError):
            parse_domain_gdf({"crs": "EPSG:4326", "features": []})

    def test_missing_features_key_raises(self):
        with pytest.raises(EmptyDomainError):
            parse_domain_gdf({"crs": "EPSG:4326"})

    def test_invalid_json_coordinates_raises(self):
        data = {
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": "not-valid-json{{{",
                    },
                    "properties": {},
                }
            ],
        }
        with pytest.raises(InvalidGeometryError):
            parse_domain_gdf(data)
