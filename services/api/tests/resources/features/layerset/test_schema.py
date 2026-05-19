"""
Unit tests for api/v2/resources/features/layerset/schema.py

Tests the Layerset schema models, source metadata, and hierarchical GeoJSON
structure. These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.features.layerset.examples import EXAMPLE_LAYERSET_MINIMAL
from api.resources.features.layerset.schema import (
    CreateLayersetRequestBody,
    LayersetFeatureCollection,
    LayersetSource,
)
from api.resources.features.schema import FeatureType
from pydantic import ValidationError


class TestLayersetSource:
    """Tests for LayersetSource model.

    Note: Unlike OsmRoadSource / OsmWaterSource, this schema uses plain `str`
    defaults rather than `Literal`, so overriding is technically permitted.
    These tests pin the *default* values, which is what the router records on
    every created document.
    """

    def test_product_default_is_upload(self):
        """The product field defaults to 'Upload'."""
        source = LayersetSource()
        assert source.product == "Upload"

    def test_description_default(self):
        """The description has a useful default."""
        source = LayersetSource()
        assert source.description == "User-uploaded layerset"

    def test_model_dump(self):
        """Model serializes with the expected keys."""
        source = LayersetSource()
        data = source.model_dump()
        assert data == {
            "product": "Upload",
            "description": "User-uploaded layerset",
        }


class TestCreateLayersetRequestBody:
    """Tests for CreateLayersetRequestBody model."""

    def test_minimal_valid_request(self):
        """Minimal request requires type='layerset' and a geojson FeatureCollection."""
        request = CreateLayersetRequestBody(
            type="layerset",
            geojson={"type": "FeatureCollection", "features": []},
        )
        assert request.type == FeatureType.layerset
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.geojson.features == []

    def test_geojson_is_required(self):
        """The geojson field cannot be omitted."""
        with pytest.raises(ValidationError):
            CreateLayersetRequestBody(type="layerset")

    def test_type_must_be_layerset(self):
        """The type field cannot be set to any other feature type."""
        with pytest.raises(ValidationError):
            CreateLayersetRequestBody(
                type="road",
                geojson={"type": "FeatureCollection", "features": []},
            )

    def test_full_request_with_all_fields(self):
        """Full request with all optional metadata fields."""
        request = CreateLayersetRequestBody(
            type="layerset",
            name="Test Layerset",
            description="A test custom layerset.",
            tags=["custom", "test"],
            geojson={"type": "FeatureCollection", "features": []},
        )
        assert request.type == FeatureType.layerset
        assert request.name == "Test Layerset"
        assert request.description == "A test custom layerset."
        assert request.tags == ["custom", "test"]


class TestLayersetFeatureCollection:
    """Smoke tests for the flat-GeoJSON schema using the documented example."""

    def test_documented_example_payload_parses(self):
        """The example payload from examples.py parses into the model."""
        request = CreateLayersetRequestBody(**EXAMPLE_LAYERSET_MINIMAL)

        assert isinstance(request.geojson, LayersetFeatureCollection)
        assert len(request.geojson.features) == 7

        # Every feature has the rasterizer's required columns on properties
        for feature in request.geojson.features:
            assert feature.properties.fuel_type
            assert feature.properties.distribution.value in {
                "homogeneous",
                "uniform_random",
                "random_clusters",
            }

        # fuel_type values are a small canonical set on the example
        fuel_types = {f.properties.fuel_type for f in request.geojson.features}
        assert fuel_types == {"shrub", "herb", "litter"}

        # The team's per-fuelbed traceability identifier is preserved
        strata_fbs = {f.properties.strata_fb for f in request.geojson.features}
        assert "Shrub1_52" in strata_fbs
        assert "LitterLichenMoss_53" in strata_fbs

    def test_top_level_crs_block_is_preserved(self):
        """The optional crs block on the FeatureCollection round-trips."""
        request = CreateLayersetRequestBody(**EXAMPLE_LAYERSET_MINIMAL)
        assert request.geojson.crs is not None
        assert request.geojson.crs.properties["name"] == "urn:ogc:def:crs:EPSG::32612"

    def test_random_clusters_requires_patch_size(self):
        """A random_clusters feature without patch_size is rejected at validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            CreateLayersetRequestBody(
                type="layerset",
                geojson={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "fuel_type": "shrub",
                                "fuel_loading": 1.0,
                                "fuel_height": 1.0,
                                "percent_cover": 50,
                                "distribution": "random_clusters",
                                # patch_size deliberately omitted
                            },
                            "geometry": {
                                "type": "MultiPolygon",
                                "coordinates": [],
                            },
                        }
                    ],
                },
            )
        assert "patch_size" in str(exc_info.value)
