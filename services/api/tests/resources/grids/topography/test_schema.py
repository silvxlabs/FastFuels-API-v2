"""
Unit tests for api/v2/resources/grids/topography/schema.py

Tests the Topography schema models and constants.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType, TileMetadata
from api.resources.grids.topography.schema import (
    TOPOGRAPHY_BAND_DEFS,
    CreateLandfireTopographyRequest,
    CreateThreeDepTopographyRequest,
    LandfireTopographySource,
    LandfireTopographyVersion,
    ThreeDepResolution,
    ThreeDepTopographySource,
    TopographyBand,
    build_topography_bands,
)
from pydantic import ValidationError

from lib.units import validate_unit


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


class TestLandfireTopographyVersion:
    """Tests for LandfireTopographyVersion enum."""

    def test_has_2020(self):
        assert LandfireTopographyVersion.v2020 == "2020"

    def test_has_exactly_one_member(self):
        assert len(LandfireTopographyVersion) == 1


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

    def test_invalid_version_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(version="2022")

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

    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(bands=["elevation", "elevation"])

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

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateLandfireTopographyRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateLandfireTopographyRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateLandfireTopographyRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateLandfireTopographyRequest(extent_buffer_cells=11)


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
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.slope]["unit"] == "deg"

    def test_aspect_key(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.aspect]["key"] == "aspect"

    def test_aspect_unit(self):
        assert TOPOGRAPHY_BAND_DEFS[TopographyBand.aspect]["unit"] == "deg"

    def test_all_bands_are_continuous(self):
        for band_def in TOPOGRAPHY_BAND_DEFS.values():
            assert band_def["type"] == BandType.continuous

    def test_all_units_are_canonical(self):
        for band_def in TOPOGRAPHY_BAND_DEFS.values():
            validate_unit(band_def.get("unit"))


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
        assert bands[0].name == "Slope"
        assert bands[0].description
        assert bands[0].index == 0
        assert bands[0].unit == "deg"

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


class TestThreeDepResolution:
    """Tests for ThreeDepResolution enum."""

    def test_has_exactly_three_members(self):
        assert len(ThreeDepResolution) == 3

    def test_one_meter(self):
        assert ThreeDepResolution.one_meter == 1

    def test_ten_meter(self):
        assert ThreeDepResolution.ten_meter == 10

    def test_thirty_meter(self):
        assert ThreeDepResolution.thirty_meter == 30


class TestThreeDepTopographySource:
    """Tests for ThreeDepTopographySource model."""

    def test_product_is_always_topography(self):
        source = ThreeDepTopographySource(
            source_resolution=10, bands=[TopographyBand.elevation]
        )
        assert source.product == "topography"

    def test_product_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            ThreeDepTopographySource(
                product="other", source_resolution=10, bands=[TopographyBand.elevation]
            )

    def test_name_is_always_3dep(self):
        source = ThreeDepTopographySource(
            source_resolution=10, bands=[TopographyBand.elevation]
        )
        assert source.name == "3dep"

    def test_bands_are_stored(self):
        source = ThreeDepTopographySource(
            source_resolution=10,
            bands=[TopographyBand.elevation, TopographyBand.slope],
        )
        assert source.bands == [TopographyBand.elevation, TopographyBand.slope]

    def test_resolution_is_required(self):
        with pytest.raises(ValidationError):
            ThreeDepTopographySource(bands=[TopographyBand.elevation])

    def test_bands_are_required(self):
        with pytest.raises(ValidationError):
            ThreeDepTopographySource(source_resolution=10)

    def test_metadata_fields_default_to_none(self):
        source = ThreeDepTopographySource(
            source_resolution=10, bands=[TopographyBand.elevation]
        )
        assert source.tile_metadata is None

    def test_model_dump(self):
        source = ThreeDepTopographySource(
            source_resolution=10,
            bands=[TopographyBand.elevation, TopographyBand.aspect],
        )
        data = source.model_dump()
        assert data["name"] == "3dep"
        assert data["product"] == "topography"
        assert data["source_resolution"] == 10
        assert data["bands"] == ["elevation", "aspect"]
        assert data["tile_metadata"] is None

    def test_model_dump_with_metadata(self):
        source = ThreeDepTopographySource(
            source_resolution=10,
            bands=[TopographyBand.elevation],
            tile_metadata=TileMetadata(
                tiles=["https://example.com/tile.tif"],
                tile_source="s1m",
                tile_count=1,
                native_crs="EPSG:4326",
                acquisition_dates=["20230515"],
            ),
        )
        data = source.model_dump()
        assert data["tile_metadata"]["tiles"] == ["https://example.com/tile.tif"]
        assert data["tile_metadata"]["tile_source"] == "s1m"
        assert data["tile_metadata"]["tile_count"] == 1
        assert data["tile_metadata"]["native_crs"] == "EPSG:4326"
        assert data["tile_metadata"]["acquisition_dates"] == ["20230515"]


class TestCreateThreeDepTopographyRequest:
    """Tests for CreateThreeDepTopographyRequest model."""

    def test_minimal_valid_request(self):
        request = CreateThreeDepTopographyRequest()
        assert request.source_resolution == ThreeDepResolution.ten_meter
        assert request.bands == [TopographyBand.elevation]
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_resolution_defaults_to_10m(self):
        request = CreateThreeDepTopographyRequest()
        assert request.source_resolution == 10

    def test_resolution_can_be_1m(self):
        request = CreateThreeDepTopographyRequest(source_resolution=1)
        assert request.source_resolution == ThreeDepResolution.one_meter

    def test_resolution_can_be_30m(self):
        request = CreateThreeDepTopographyRequest(source_resolution=30)
        assert request.source_resolution == ThreeDepResolution.thirty_meter

    def test_invalid_resolution_rejected(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(source_resolution=5)

    def test_bands_default_to_elevation_only(self):
        request = CreateThreeDepTopographyRequest()
        assert request.bands == [TopographyBand.elevation]

    def test_bands_can_be_all_three(self):
        request = CreateThreeDepTopographyRequest(
            bands=["elevation", "slope", "aspect"]
        )
        assert len(request.bands) == 3

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(bands=[])

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(bands=["invalid"])

    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(bands=["elevation", "elevation"])

    def test_full_request_with_all_fields(self):
        request = CreateThreeDepTopographyRequest(
            source_resolution=1,
            name="High-res terrain",
            description="1m DEM",
            tags=["3dep"],
            bands=["elevation", "slope"],
        )
        assert request.name == "High-res terrain"
        assert request.description == "1m DEM"
        assert request.tags == ["3dep"]
        assert request.source_resolution == 1
        assert request.bands == [TopographyBand.elevation, TopographyBand.slope]

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateThreeDepTopographyRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateThreeDepTopographyRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateThreeDepTopographyRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_accepts_maximum(self):
        request = CreateThreeDepTopographyRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateThreeDepTopographyRequest(extent_buffer_cells=11)
