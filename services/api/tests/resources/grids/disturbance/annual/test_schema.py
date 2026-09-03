"""
Unit tests for api/v2/resources/grids/disturbance/annual/schema.py
and api/v2/resources/grids/providers/landfire.py

Tests the Limited Annual Disturbance schema models, LandfireSource base,
and constants. These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.disturbance.annual.schema import (
    DISTURBANCE_BAND,
    CreateLandfireDisturbanceRequest,
    LandfireDisturbanceSource,
)
from api.resources.grids.schema import BandType
from pydantic import ValidationError

from lib.landfire import LANDFIRE_VERSIONS


class TestLandfireDisturbanceSource:
    """Tests for LandfireDisturbanceSource model."""

    def test_product_is_always_annual_disturbance(self):
        source = LandfireDisturbanceSource(version="2025")
        assert source.product == "annual_disturbance"

    def test_product_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            LandfireDisturbanceSource(product="other", version="2025")

    def test_name_is_always_landfire(self):
        source = LandfireDisturbanceSource(version="2025")
        assert source.name == "landfire"

    def test_description_is_fixed(self):
        source = LandfireDisturbanceSource(version="2025")
        assert "Disturbance" in source.description

    def test_version_is_required(self):
        with pytest.raises(ValidationError):
            LandfireDisturbanceSource()

    def test_model_dump(self):
        source = LandfireDisturbanceSource(version="2025")
        data = source.model_dump()
        assert data["name"] == "landfire"
        assert data["product"] == "annual_disturbance"
        assert data["version"] == "2025"
        assert "description" in data


class TestCreateLandfireDisturbanceRequest:
    """domain_id comes from the URL path parameter, not the request body."""

    def test_minimal_valid_request(self):
        request = CreateLandfireDisturbanceRequest()
        assert request.version == LANDFIRE_VERSIONS["annual_disturbance"]["default"]
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_registry_default(self):
        request = CreateLandfireDisturbanceRequest()
        assert request.version == LANDFIRE_VERSIONS["annual_disturbance"]["default"]

    def test_version_can_be_overridden(self):
        version = LANDFIRE_VERSIONS["annual_disturbance"]["lfps_available"][0]
        request = CreateLandfireDisturbanceRequest(version=version)
        assert request.version == version

    def test_invalid_version_rejected(self):
        with pytest.raises(ValidationError):
            CreateLandfireDisturbanceRequest(version="2019")

    def test_full_request_with_all_fields(self):
        request = CreateLandfireDisturbanceRequest(
            name="Test Grid",
            description="A test grid",
            tags=["test", "disturbance"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "disturbance"]

    def test_alignment_defaults_to_domain_target(self):
        request = CreateLandfireDisturbanceRequest()
        assert request.alignment.target == "domain"

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateLandfireDisturbanceRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateLandfireDisturbanceRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateLandfireDisturbanceRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateLandfireDisturbanceRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateLandfireDisturbanceRequest(extent_buffer_cells=11)


class TestDisturbanceBand:
    def test_key_is_annual_disturbance(self):
        """Must match the dataset variable name griddle's
        fetch_annual_disturbance() writes."""
        assert DISTURBANCE_BAND.key == "annual_disturbance"

    def test_type_is_categorical(self):
        assert DISTURBANCE_BAND.type == BandType.categorical

    def test_unit_is_none(self):
        assert DISTURBANCE_BAND.unit is None

    def test_index_is_zero(self):
        assert DISTURBANCE_BAND.index == 0

    def test_has_name_and_description(self):
        assert DISTURBANCE_BAND.name
        assert DISTURBANCE_BAND.description
