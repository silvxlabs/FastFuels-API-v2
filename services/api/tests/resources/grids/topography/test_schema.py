"""
Unit tests for api/v2/resources/grids/topography/schema.py

Tests the Topography schema models and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType
from api.resources.grids.topography.schema import (
    TOPOGRAPHY_BAND_DEFS,
    CreateLandfireTopographyRequest,
    LandfireTopographySource,
    TopographyBand,
    build_topography_bands,
)
from pydantic import ValidationError


class TestTopographyBand:
    """Tests for TopographyBand enum."""

    def test_has_elevation(self):
        assert TopographyBand.elevation == "elevation"

    def test_has_slope(self):
        assert TopographyBand.slope == "slope"

    def test_has_aspect(self):
        assert TopographyBand.aspect == "aspect"

    def test_has_exactly_three_members(self):
        assert len(TopographyBand) == 3


class TestLandfireTopographySource:
    """Tests for LandfireTopographySource model."""

    def test_product_is_always_topography(self):
        source = LandfireTopographySource(
            version="2020", bands=[TopographyBand.elevation]
        )
        assert source.product == "topography"

    def test_product_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            LandfireTopographySource(
                product="other", version="2020", bands=[TopographyBand.elevation]
            )

    def test_name_is_always_landfire(self):
        source = LandfireTopographySource(
            version="2020", bands=[TopographyBand.elevation]
        )
        assert source.name == "landfire"

    def test_bands_are_stored(self):
        source = LandfireTopographySource(
            version="2020",
            bands=[TopographyBand.elevation, TopographyBand.slope],
        )
        assert source.bands == [TopographyBand.elevation, TopographyBand.slope]

    def test_version_is_required(self):
        with pytest.raises(ValidationError):
            LandfireTopographySource(bands=[TopographyBand.elevation])

    def test_bands_are_required(self):
        with pytest.raises(ValidationError):
            LandfireTopographySource(version="2020")

    def test_model_dump(self):
        source = LandfireTopographySource(
            version="2020",
            bands=[TopographyBand.elevation, TopographyBand.aspect],
        )
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "topography"
        assert data["version"] == "2020"
        assert data["bands"] == ["elevation", "aspect"]


class TestCreateLandfireTopographyRequest:
    """Tests for CreateLandfireTopographyRequest model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        request = CreateLandfireTopographyRequest()
        assert request.version == "2020"
        assert request.bands == [
            TopographyBand.elevation,
            TopographyBand.slope,
            TopographyBand.aspect,
        ]
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_version_defaults_to_2020(self):
        request = CreateLandfireTopographyRequest()
        assert request.version == "2020"

    def test_version_can_be_overridden(self):
        request = CreateLandfireTopographyRequest(version="2022")
        assert request.version == "2022"

    def test_bands_default_to_all_three(self):
        request = CreateLandfireTopographyRequest()
        assert len(request.bands) == 3

    def test_bands_can_be_subset(self):
        request = CreateLandfireTopographyRequest(bands=["elevation"])
        assert request.bands == [TopographyBand.elevation]

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(bands=[])

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(bands=["invalid"])

    def test_full_request_with_all_fields(self):
        request = CreateLandfireTopographyRequest(
            version="2020",
            name="Terrain data",
            description="Test terrain",
            tags=["topo"],
            bands=["elevation", "slope"],
        )
        assert request.name == "Terrain data"
        assert request.description == "Test terrain"
        assert request.tags == ["topo"]
        assert request.bands == [TopographyBand.elevation, TopographyBand.slope]


class TestTopographyBandDefs:
    """Tests for TOPOGRAPHY_BAND_DEFS constant."""

    def test_elevation_key(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.elevation]["key"] == "elevation"

    def test_elevation_type(self):
        assert (
            TOPOGRAPHY_BAND_DEFS[TopographyBand.elevation]["type"]
            == BandType.continuous
        )

    def test_elevation_unit(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.elevation]["unit"] == "m"

    def test_slope_key(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.slope]["key"] == "slope"

    def test_slope_unit(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.slope]["unit"] == "degrees"

    def test_aspect_key(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.aspect]["key"] == "aspect"

    def test_aspect_unit(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.aspect]["unit"] == "degrees"

    def test_all_bands_are_continuous(self):
        for band_def in TOPOGRAPHY_BAND_DEFS.values():
            assert band_def["type"] == BandType.continuous


class TestBuildTopographyBands:
    """Tests for build_topography_bands function."""

    def test_all_bands(self):
        bands = build_topography_bands(
            [
                TopographyBand.elevation,
                TopographyBand.slope,
                TopographyBand.aspect,
            ]
        )
        assert len(bands) == 3
        assert bands[0].key == "elevation"
        assert bands[0].index == 0
        assert bands[1].key == "slope"
        assert bands[1].index == 1
        assert bands[2].key == "aspect"
        assert bands[2].index == 2

    def test_single_band(self):
        bands = build_topography_bands([TopographyBand.slope])
        assert len(bands) == 1
        assert bands[0].key == "slope"
        assert bands[0].index == 0
        assert bands[0].unit == "degrees"

    def test_subset_preserves_order(self):
        bands = build_topography_bands(
            [
                TopographyBand.aspect,
                TopographyBand.elevation,
            ]
        )
        assert bands[0].key == "aspect"
        assert bands[0].index == 0
        assert bands[1].key == "elevation"
        assert bands[1].index == 1
