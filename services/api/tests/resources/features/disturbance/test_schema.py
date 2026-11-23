"""
Unit tests for api/v2/resources/features/disturbance/schema.py

Tests the disturbance properties and the upload request. Pure unit tests with
no external dependencies.
"""

import pytest
from api.resources.features.disturbance.examples import (
    DISTURBANCE_GEOJSON_EXAMPLE_VALUES,
)
from api.resources.features.disturbance.schema import (
    CreateDisturbanceFeatureRequest,
    DisturbanceProperties,
)
from pydantic import ValidationError

from lib.landfire.config import VALID_FDIST_CODES


class TestDisturbanceProperties:
    def test_every_valid_code_is_accepted(self):
        for code in VALID_FDIST_CODES:
            assert DisturbanceProperties(fdist=code).fdist == code

    @pytest.mark.parametrize("code", [0, -121, 100, 114, 141, 811, 1111])
    def test_invalid_codes_are_rejected(self, code):
        """0, a negative, and codes with a type, severity or time out of range."""
        with pytest.raises(ValidationError):
            DisturbanceProperties(fdist=code)

    @pytest.mark.parametrize("code", ["122", 122.0, 122.5])
    def test_non_integer_codes_are_rejected(self, code):
        with pytest.raises(ValidationError):
            DisturbanceProperties(fdist=code)

    def test_fdist_is_required(self):
        with pytest.raises(ValidationError):
            DisturbanceProperties()

    def test_labels_are_optional(self):
        properties = DisturbanceProperties(fdist=122)
        assert properties.name is None
        assert properties.type is None
        assert properties.year is None
        assert properties.severity is None

    def test_labels_are_kept_as_given(self):
        properties = DisturbanceProperties(
            fdist=122, name="Example Fire", type="wildfire", year=2022, severity="high"
        )
        assert properties.name == "Example Fire"
        assert properties.type == "wildfire"
        assert properties.year == 2022
        assert properties.severity == "high"

    def test_labels_are_not_checked_against_the_code(self):
        """311 is a mechanical removal, but the label says wildfire."""
        properties = DisturbanceProperties(fdist=311, type="wildfire")
        assert properties.type == "wildfire"

    def test_year_must_be_an_integer(self):
        with pytest.raises(ValidationError):
            DisturbanceProperties(fdist=122, year="2022")

    @pytest.mark.parametrize("year", [1899, 2101, 2**63])
    def test_year_out_of_range_is_rejected(self, year):
        with pytest.raises(ValidationError):
            DisturbanceProperties(fdist=122, year=year)

    def test_other_properties_are_kept(self):
        properties = DisturbanceProperties(fdist=122, acres=10)
        assert properties.model_dump()["acres"] == 10


_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
}


def _body(**overrides):
    body = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"fdist": 122}, "geometry": _POLYGON}
        ],
    }
    body.update(overrides)
    return body


class TestCreateDisturbanceFeatureRequest:
    def test_minimal_body_has_default_metadata(self):
        body = CreateDisturbanceFeatureRequest(**_body())
        assert body.name == ""
        assert body.description == ""
        assert body.tags == []

    def test_metadata_is_accepted(self):
        body = CreateDisturbanceFeatureRequest(
            **_body(name="History", description="Fires", tags=["a", "b"])
        )
        assert body.name == "History"
        assert body.description == "Fires"
        assert body.tags == ["a", "b"]

    @pytest.mark.parametrize("name,value", DISTURBANCE_GEOJSON_EXAMPLE_VALUES)
    def test_documented_examples_parse(self, name, value):
        body = CreateDisturbanceFeatureRequest(**value)
        assert len(body.features) == 2

    def test_empty_feature_collection_is_rejected(self):
        with pytest.raises(ValidationError):
            CreateDisturbanceFeatureRequest(**_body(features=[]))

    def test_multipolygon_geometry_is_accepted(self):
        multipolygon = {
            "type": "MultiPolygon",
            "coordinates": [_POLYGON["coordinates"]],
        }
        payload = _body(
            features=[
                {
                    "type": "Feature",
                    "properties": {"fdist": 122},
                    "geometry": multipolygon,
                }
            ]
        )
        CreateDisturbanceFeatureRequest(**payload)

    def test_point_geometry_is_rejected(self):
        payload = _body(
            features=[
                {
                    "type": "Feature",
                    "properties": {"fdist": 122},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                }
            ]
        )
        with pytest.raises(ValidationError):
            CreateDisturbanceFeatureRequest(**payload)

    def test_invalid_code_inside_a_feature_is_rejected(self):
        payload = _body(
            features=[
                {"type": "Feature", "properties": {"fdist": 0}, "geometry": _POLYGON}
            ]
        )
        with pytest.raises(ValidationError) as exc_info:
            CreateDisturbanceFeatureRequest(**payload)
        assert any("fdist" in str(error["loc"]) for error in exc_info.value.errors())

    def test_null_properties_are_rejected(self):
        payload = _body(
            features=[{"type": "Feature", "properties": None, "geometry": _POLYGON}]
        )
        with pytest.raises(ValidationError):
            CreateDisturbanceFeatureRequest(**payload)

    def test_null_geometry_is_rejected(self):
        payload = _body(
            features=[
                {"type": "Feature", "properties": {"fdist": 122}, "geometry": None}
            ]
        )
        with pytest.raises(ValidationError):
            CreateDisturbanceFeatureRequest(**payload)

    def test_geometry_property_is_rejected(self):
        payload = _body(
            features=[
                {
                    "type": "Feature",
                    "properties": {"fdist": 122, "geometry": "x"},
                    "geometry": _POLYGON,
                }
            ]
        )
        with pytest.raises(ValidationError, match="geometry"):
            CreateDisturbanceFeatureRequest(**payload)

    def test_properties_are_stored_as_given(self):
        """Unset labels aren't written as null columns; extra properties are kept."""
        payload = _body(
            features=[
                {
                    "type": "Feature",
                    "properties": {"fdist": 122, "acres": 10},
                    "geometry": _POLYGON,
                }
            ]
        )
        body = CreateDisturbanceFeatureRequest(**payload)
        dumped = body.features[0].model_dump(mode="json")
        assert dumped["properties"] == {"fdist": 122, "acres": 10}
