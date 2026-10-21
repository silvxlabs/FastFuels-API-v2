"""
Unit tests for api/v2/resources/features/disturbance/schema.py

Tests the Disturbance schema models, source metadata, and the flat GeoJSON
FeatureCollection shape. Pure unit tests with no external dependencies.
"""

import pytest
from api.resources.features.disturbance.examples import DISTURBANCE_EXAMPLE_VALUES
from api.resources.features.disturbance.schema import (
    CreateDisturbanceRequestBody,
    DisturbanceProperties,
    DisturbanceSource,
)
from pydantic import ValidationError

from lib.landfire import VALID_FDIST_CODES


class TestDisturbanceSource:
    def test_defaults(self):
        source = DisturbanceSource(mapping_year=2024)
        assert source.product == "Upload"
        assert source.description == "User-uploaded disturbance polygons"

    def test_mapping_year_is_required(self):
        with pytest.raises(ValidationError):
            DisturbanceSource()

    def test_model_dump(self):
        assert DisturbanceSource(mapping_year=2024).model_dump() == {
            "product": "Upload",
            "description": "User-uploaded disturbance polygons",
            "mapping_year": 2024,
        }


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

    def test_other_properties_are_ignored(self):
        properties = DisturbanceProperties(fdist=122, acres=10)
        assert "acres" not in properties.model_dump()


_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
}


def _body(**overrides):
    body = {
        "type": "FeatureCollection",
        "mapping_year": 2024,
        "features": [
            {"type": "Feature", "properties": {"fdist": 122}, "geometry": _POLYGON}
        ],
    }
    body.update(overrides)
    return body


class TestCreateDisturbanceRequestBody:
    def test_minimal_body_has_default_metadata(self):
        body = CreateDisturbanceRequestBody(**_body())
        assert body.name == ""
        assert body.description == ""
        assert body.tags == []

    def test_metadata_is_accepted(self):
        body = CreateDisturbanceRequestBody(
            **_body(name="History", description="Fires", tags=["a", "b"])
        )
        assert body.name == "History"
        assert body.description == "Fires"
        assert body.tags == ["a", "b"]

    @pytest.mark.parametrize("name,value", DISTURBANCE_EXAMPLE_VALUES)
    def test_documented_examples_parse(self, name, value):
        body = CreateDisturbanceRequestBody(**value)
        assert len(body.features) == 2

    def test_empty_feature_collection_is_rejected(self):
        with pytest.raises(ValidationError):
            CreateDisturbanceRequestBody(**_body(features=[]))

    def test_mapping_year_is_required(self):
        payload = _body()
        del payload["mapping_year"]
        with pytest.raises(ValidationError):
            CreateDisturbanceRequestBody(**payload)

    @pytest.mark.parametrize("year", [1900, 2024, 2100])
    def test_mapping_year_in_range_is_accepted(self, year):
        assert (
            CreateDisturbanceRequestBody(**_body(mapping_year=year)).mapping_year
            == year
        )

    @pytest.mark.parametrize("year", [0, 24, 1899, 2101, 20244])
    def test_mapping_year_out_of_range_is_rejected(self, year):
        with pytest.raises(ValidationError):
            CreateDisturbanceRequestBody(**_body(mapping_year=year))

    def test_mapping_year_must_be_an_integer(self):
        with pytest.raises(ValidationError):
            CreateDisturbanceRequestBody(**_body(mapping_year="2024"))

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
        CreateDisturbanceRequestBody(**payload)

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
            CreateDisturbanceRequestBody(**payload)

    def test_invalid_code_inside_a_feature_is_rejected(self):
        payload = _body(
            features=[
                {"type": "Feature", "properties": {"fdist": 0}, "geometry": _POLYGON}
            ]
        )
        with pytest.raises(ValidationError) as exc_info:
            CreateDisturbanceRequestBody(**payload)
        assert any("fdist" in str(error["loc"]) for error in exc_info.value.errors())
