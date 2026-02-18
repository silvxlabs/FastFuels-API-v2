"""
Unit tests for api/v2/resources/grids/landfire/schema.py

Tests the LANDFIRE schema models and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.landfire.schema import (
    FBFM40_BAND,
    TOPOGRAPHY_BAND_DEFS,
    CreateLandfireFbfm40Request,
    CreateLandfireTopographyRequest,
    LandfireFbfm40Source,
    LandfireSource,
    LandfireTopographySource,
    TopographyBand,
    build_topography_bands,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestLandfireSource:
    """Tests for LandfireSource base model."""

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireSource(product="fbfm40", version="2022")
        assert source.name == "landfire"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'landfire'."""
        with pytest.raises(ValidationError):
            LandfireSource(name="other", product="fbfm40", version="2022")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            LandfireSource(version="2022")

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireSource(product="fbfm40")

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = LandfireSource(product="fbfm40", version="2022")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = LandfireSource(
            product="fbfm40",
            version="2022",
            description="Test description",
        )
        assert source.description == "Test description"


class TestLandfireFbfm40Source:
    """Tests for LandfireFbfm40Source model."""

    def test_product_is_always_fbfm40(self):
        """The product field is always 'fbfm40'."""
        source = LandfireFbfm40Source(version="2022")
        assert source.product == "fbfm40"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'fbfm40'."""
        with pytest.raises(ValidationError):
            LandfireFbfm40Source(product="other", version="2022")

    def test_name_is_always_landfire(self):
        """The name field is always 'landfire'."""
        source = LandfireFbfm40Source(version="2022")
        assert source.name == "landfire"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = LandfireFbfm40Source(version="2022")
        assert "FBFM40" in source.description
        assert "Scott-Burgan" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            LandfireFbfm40Source()

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LandfireFbfm40Source(version="2022")
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "fbfm40"
        assert data["version"] == "2022"
        assert "description" in data


class TestCreateLandfireFbfm40Request:
    """Tests for CreateLandfireFbfm40Request model.

    domain_id now comes from the URL path parameter, not the request body.
    """

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateLandfireFbfm40Request()
        assert request.version == "2022"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2022(self):
        """version defaults to '2022'."""
        request = CreateLandfireFbfm40Request()
        assert request.version == "2022"

    def test_version_can_be_overridden(self):
        """version can be set to a different value."""
        request = CreateLandfireFbfm40Request(version="2020")
        assert request.version == "2020"

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateLandfireFbfm40Request(
            version="2022",
            name="Test Grid",
            description="A test grid",
            tags=["test", "fuel"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "fuel"]


class TestFbfm40Band:
    """Tests for FBFM40_BAND constant."""

    def test_key_is_fbfm(self):
        """Band key is 'fbfm'."""
        assert FBFM40_BAND.key == "fbfm"

    def test_type_is_categorical(self):
        """Band type is categorical."""
        assert FBFM40_BAND.type == BandType.categorical

    def test_unit_is_none(self):
        """Band unit is None (categorical has no unit)."""
        assert FBFM40_BAND.unit is None

    def test_index_is_zero(self):
        """Band index is 0."""
        assert FBFM40_BAND.index == 0


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
