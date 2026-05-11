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

from lib.alignment import lattice_from_bounds
from lib.domain_utils import (
    EmptyDomainError,
    InvalidGeometryError,
    domain_anchored_transform,
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


class TestTwoFeatureFormat:
    """Tests for the two-feature format produced by API v2 domain creation.

    A v2 domain is stored with two named features: "domain" (the working
    extent / bbox rectangle, possibly snapped via pad_to_resolution) and
    "input" (the user's original polygon). The bbox rectangle contains the
    input polygon by construction, so loading both features into a single
    GeoDataFrame yields a ``total_bounds`` equal to the working extent
    without any filtering.
    """

    @pytest.fixture
    def two_feature_data(self):
        return {
            "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "domain"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
                        ),
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"name": "input"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
                        ),
                    },
                },
            ],
        }

    def test_loads_both_features(self, two_feature_data):
        result = parse_domain_gdf(two_feature_data)
        assert len(result) == 2

    def test_total_bounds_equal_working_extent(self, two_feature_data):
        result = parse_domain_gdf(two_feature_data)
        bounds = result.total_bounds
        # The bbox rectangle (0..100) contains the input polygon (10..90),
        # so total_bounds reduces to the working extent without filtering.
        assert bounds[0] == 0
        assert bounds[1] == 0
        assert bounds[2] == 100
        assert bounds[3] == 100

    def test_padded_extent_drives_total_bounds(self):
        """When the "domain" feature is snapped outward, total_bounds reflects it."""
        data = {
            "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "domain"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            # Snapped outward beyond the input's tight bbox
                            [[[0, 0], [120, 0], [120, 120], [0, 120], [0, 0]]]
                        ),
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"name": "input"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": json.dumps(
                            [[[10, 10], [115, 10], [115, 115], [10, 115], [10, 10]]]
                        ),
                    },
                },
            ],
        }
        result = parse_domain_gdf(data)
        bounds = result.total_bounds
        # Padded extent (120 × 120), not the input's tight bbox (105 × 105)
        assert bounds[2] == 120
        assert bounds[3] == 120


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


class TestDomainAnchoredTransform:
    """Tests for ``domain_anchored_transform`` — the thin wrapper that ties
    a domain GeoDataFrame to ``lattice_from_bounds``."""

    @pytest.fixture
    def blue_mtn_gdf(self):
        return parse_domain_gdf(_load_domain_json("blue_mtn.json"))

    def test_matches_lattice_from_bounds(self, blue_mtn_gdf):
        transform_a, shape_a = domain_anchored_transform(blue_mtn_gdf, 30.0)
        transform_b, shape_b = lattice_from_bounds(
            tuple(blue_mtn_gdf.total_bounds), 30.0
        )
        assert tuple(transform_a)[:6] == tuple(transform_b)[:6]
        assert shape_a == shape_b

    def test_resolution_drives_cell_size(self, blue_mtn_gdf):
        transform, _ = domain_anchored_transform(blue_mtn_gdf, 5.0)
        assert transform.a == pytest.approx(5.0)
        assert transform.e == pytest.approx(-5.0)
