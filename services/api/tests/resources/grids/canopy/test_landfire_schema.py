"""
Unit tests for the LANDFIRE canopy schemas in
api/v2/resources/grids/canopy/schema.py.

Covers the canopy band vocabulary (LandfireCanopyFuelBand, LANDFIRE_CANOPY_BAND_DEFS,
build_landfire_canopy_bands), the LANDFIRE source/request models, and validators.
"""

import pytest
from api.resources.grids.canopy.schema import (
    LANDFIRE_CANOPY_BAND_DEFS,
    CreateLandfireCanopyRequest,
    LandfireCanopyFuelBand,
    LandfireCanopySource,
    LandfireCanopyVersion,
    build_landfire_canopy_bands,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError

from lib.units import validate_unit


class TestLandfireCanopyFuelBand:
    """Tests for the LandfireCanopyFuelBand enum."""

    def test_has_chm(self):
        assert LandfireCanopyFuelBand.chm == "chm"

    def test_has_cbd(self):
        assert LandfireCanopyFuelBand.cbd == "cbd"

    def test_has_cbh(self):
        assert LandfireCanopyFuelBand.cbh == "cbh"

    def test_has_cc(self):
        assert LandfireCanopyFuelBand.cc == "cc"

    def test_has_exactly_four_members(self):
        assert len(LandfireCanopyFuelBand) == 4

    def test_members_match_band_defs(self):
        assert set(LANDFIRE_CANOPY_BAND_DEFS.keys()) == set(LandfireCanopyFuelBand)


class TestLandfireCanopyFuelBandDefs:
    """Tests for the LANDFIRE_CANOPY_BAND_DEFS mapping."""

    def test_chm_key(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.chm]["key"] == "chm"

    def test_chm_unit(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.chm]["unit"] == "m"

    def test_cbd_key(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cbd]["key"] == "cbd"

    def test_cbd_unit_is_kg_per_m3(self):
        assert (
            LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cbd]["unit"] == "kg/m**3"
        )

    def test_cbh_key(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cbh]["key"] == "cbh"

    def test_cbh_unit_is_meters(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cbh]["unit"] == "m"

    def test_cc_key(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cc]["key"] == "cc"

    def test_cc_unit_is_percent(self):
        assert LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand.cc]["unit"] == "%"

    def test_all_bands_are_continuous(self):
        for band_def in LANDFIRE_CANOPY_BAND_DEFS.values():
            assert band_def["type"] == BandType.continuous

    def test_all_units_are_canonical(self):
        for band_def in LANDFIRE_CANOPY_BAND_DEFS.values():
            validate_unit(band_def.get("unit"))


class TestBuildLandfireCanopyFuelBands:
    """Tests for build_landfire_canopy_bands."""

    def test_all_bands(self):
        bands = build_landfire_canopy_bands(
            [
                LandfireCanopyFuelBand.chm,
                LandfireCanopyFuelBand.cbd,
                LandfireCanopyFuelBand.cbh,
                LandfireCanopyFuelBand.cc,
            ]
        )
        assert [b.key for b in bands] == ["chm", "cbd", "cbh", "cc"]
        assert [b.index for b in bands] == [0, 1, 2, 3]

    def test_single_band(self):
        bands = build_landfire_canopy_bands([LandfireCanopyFuelBand.cbd])
        assert len(bands) == 1
        assert bands[0].key == "cbd"
        assert bands[0].index == 0
        assert bands[0].unit == "kg/m**3"

    def test_subset_preserves_order(self):
        bands = build_landfire_canopy_bands(
            [LandfireCanopyFuelBand.cc, LandfireCanopyFuelBand.chm]
        )
        assert bands[0].key == "cc"
        assert bands[0].index == 0
        assert bands[1].key == "chm"
        assert bands[1].index == 1

    def test_all_bands_are_continuous(self):
        bands = build_landfire_canopy_bands(list(LandfireCanopyFuelBand))
        for band in bands:
            assert band.type == BandType.continuous


class TestLandfireCanopyVersion:
    """Tests for the LandfireCanopyVersion enum."""

    def test_has_2024(self):
        assert LandfireCanopyVersion.v2024 == "2024"

    def test_has_exactly_one_member(self):
        assert len(LandfireCanopyVersion) == 1


class TestLandfireCanopySource:
    """Tests for LandfireCanopySource."""

    def test_product_is_always_landfire(self):
        source = LandfireCanopySource(
            version="2024", bands=[LandfireCanopyFuelBand.chm]
        )
        assert source.product == "landfire"

    def test_product_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            LandfireCanopySource(
                product="other", version="2024", bands=[LandfireCanopyFuelBand.chm]
            )

    def test_name_is_always_canopy(self):
        source = LandfireCanopySource(
            version="2024", bands=[LandfireCanopyFuelBand.chm]
        )
        assert source.name == "canopy"

    def test_bands_are_stored(self):
        source = LandfireCanopySource(
            version="2024",
            bands=[LandfireCanopyFuelBand.cbd, LandfireCanopyFuelBand.cbh],
        )
        assert source.bands == [LandfireCanopyFuelBand.cbd, LandfireCanopyFuelBand.cbh]

    def test_version_is_required(self):
        with pytest.raises(ValidationError):
            LandfireCanopySource(bands=[LandfireCanopyFuelBand.chm])

    def test_bands_are_required(self):
        with pytest.raises(ValidationError):
            LandfireCanopySource(version="2024")

    def test_invalid_version_rejected(self):
        with pytest.raises(ValidationError):
            LandfireCanopySource(version="2099", bands=[LandfireCanopyFuelBand.chm])

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            LandfireCanopySource(version="2024", bands=["nope"])

    def test_model_dump(self):
        source = LandfireCanopySource(
            version="2024",
            bands=[LandfireCanopyFuelBand.chm, LandfireCanopyFuelBand.cc],
        )
        data = source.model_dump()
        assert data["name"] == "canopy"
        assert data["product"] == "landfire"
        assert data["version"] == "2024"
        assert data["bands"] == ["chm", "cc"]


class TestCreateLandfireCanopyRequest:
    """Tests for CreateLandfireCanopyRequest."""

    def test_minimal_valid_request(self):
        request = CreateLandfireCanopyRequest()
        assert request.version == "2024"
        assert request.bands == [
            LandfireCanopyFuelBand.chm,
            LandfireCanopyFuelBand.cbd,
            LandfireCanopyFuelBand.cbh,
            LandfireCanopyFuelBand.cc,
        ]
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_version_defaults_to_2024(self):
        request = CreateLandfireCanopyRequest()
        assert request.version == "2024"

    def test_invalid_version_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(version="2022")

    def test_bands_default_to_all_four(self):
        request = CreateLandfireCanopyRequest()
        assert len(request.bands) == 4

    def test_bands_can_be_subset(self):
        request = CreateLandfireCanopyRequest(bands=["cbd"])
        assert request.bands == [LandfireCanopyFuelBand.cbd]

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(bands=[])

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(bands=["invalid"])

    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(bands=["cbd", "cbd"])

    def test_full_request_with_all_fields(self):
        request = CreateLandfireCanopyRequest(
            version="2024",
            name="Canopy fuels",
            description="Test canopy",
            tags=["canopy", "landfire"],
            bands=["chm", "cbd"],
        )
        assert request.name == "Canopy fuels"
        assert request.description == "Test canopy"
        assert request.tags == ["canopy", "landfire"]
        assert request.bands == [LandfireCanopyFuelBand.chm, LandfireCanopyFuelBand.cbd]

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateLandfireCanopyRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateLandfireCanopyRequest(extent_buffer_cells=6)
        assert request.extent_buffer_cells == 6

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateLandfireCanopyRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateLandfireCanopyRequest(extent_buffer_cells=11)
