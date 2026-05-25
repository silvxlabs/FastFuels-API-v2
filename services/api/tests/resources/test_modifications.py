"""
Unit tests for api/resources/modifications.py coordinate (de)serialization
helpers.

Inline-geometry spatial conditions (``source == "geometry"``) carry a GeoJSON
Polygon/MultiPolygon whose ``coordinates`` is a nested array. Firestore does
not support nested arrays, so the API JSON-encodes them before write and
decodes them on read. These are pure unit tests with no external dependencies.
"""

import json

from api.resources.modifications import (
    parse_modification_coordinates,
    stringify_modification_coordinates,
)

POLYGON_COORDS = [
    [
        [-120.0, 38.0],
        [-119.5, 38.0],
        [-119.5, 38.5],
        [-120.0, 38.5],
        [-120.0, 38.0],
    ]
]


def _geometry_mod():
    return {
        "conditions": [
            {
                "source": "geometry",
                "operator": "within",
                "geometry": {"type": "Polygon", "coordinates": POLYGON_COORDS},
            }
        ],
        "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
    }


def _feature_mod():
    return {
        "conditions": [
            {"source": "feature", "operator": "intersects", "feature_id": "feat_abc"}
        ],
        "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
    }


class TestStringifyModificationCoordinates:
    def test_geometry_condition_coordinates_become_json_string(self):
        mods = [_geometry_mod()]
        result = stringify_modification_coordinates(mods)
        coords = result[0]["conditions"][0]["geometry"]["coordinates"]
        assert isinstance(coords, str)
        assert json.loads(coords) == POLYGON_COORDS

    def test_feature_condition_untouched(self):
        mods = [_feature_mod()]
        result = stringify_modification_coordinates(mods)
        assert result[0]["conditions"][0] == {
            "source": "feature",
            "operator": "intersects",
            "feature_id": "feat_abc",
        }

    def test_idempotent(self):
        mods = [_geometry_mod()]
        once = stringify_modification_coordinates(mods)
        twice = stringify_modification_coordinates(once)
        coords = twice[0]["conditions"][0]["geometry"]["coordinates"]
        assert isinstance(coords, str)
        assert json.loads(coords) == POLYGON_COORDS

    def test_attribute_only_modification_untouched(self):
        mods = [
            {
                "conditions": [{"band": "fbfm", "operator": "eq", "value": 91}],
                "actions": [{"band": "fbfm", "modifier": "replace", "value": 99}],
            }
        ]
        result = stringify_modification_coordinates(mods)
        assert result == mods


class TestParseModificationCoordinates:
    def test_string_coordinates_become_nested_list(self):
        mods = stringify_modification_coordinates([_geometry_mod()])
        result = parse_modification_coordinates(mods)
        coords = result[0]["conditions"][0]["geometry"]["coordinates"]
        assert coords == POLYGON_COORDS

    def test_idempotent_on_already_parsed(self):
        mods = [_geometry_mod()]
        result = parse_modification_coordinates(mods)
        assert result[0]["conditions"][0]["geometry"]["coordinates"] == POLYGON_COORDS


class TestRoundTrip:
    def test_stringify_then_parse_restores_original(self):
        original = [_geometry_mod(), _feature_mod()]
        restored = parse_modification_coordinates(
            stringify_modification_coordinates([_geometry_mod(), _feature_mod()])
        )
        assert restored == original
