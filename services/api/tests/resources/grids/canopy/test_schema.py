"""
Unit tests for api/v2/resources/grids/canopy/schema.py
and api/v2/resources/grids/providers/canopy.py

Tests the Meta/NAIP canopy schema models, CanopySource base, and band definitions.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.canopy.schema import (
    Attribution,
    CreateMetaChmRequest,
    CreateNaipChmRequest,
    MetaChmSource,
    NaipChmSource,
    build_chm_bands,
)
from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestCanopySource:
    """Tests for CanopySource base model."""

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = CanopySource(product="meta")
        assert source.name == "canopy"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'canopy'."""
        with pytest.raises(ValidationError):
            CanopySource(name="other", product="meta")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            CanopySource()

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = CanopySource(product="meta")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = CanopySource(
            product="meta",
            description="Test description",
        )
        assert source.description == "Test description"

    def test_extent_buffer_cells_defaults_to_zero(self):
        source = CanopySource(product="meta")
        assert source.extent_buffer_cells == 0

    def test_extent_buffer_cells_can_be_set(self):
        source = CanopySource(product="meta", extent_buffer_cells=10)
        assert source.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CanopySource(product="meta", extent_buffer_cells=-1)


class TestMetaChmSource:
    """Tests for MetaChmSource model."""

    def test_product_is_always_meta(self):
        """The product field is always 'meta'."""
        source = MetaChmSource(version="2")
        assert source.product == "meta"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'meta'."""
        with pytest.raises(ValidationError):
            MetaChmSource(product="other", version="2")

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = MetaChmSource(version="2")
        assert source.name == "canopy"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = MetaChmSource(version="2")
        assert "Meta" in source.description
        assert "canopy height" in source.description

    def test_model_dump(self):
        """Model serializes correctly."""
        source = MetaChmSource(version="2")
        data = source.model_dump()
        assert data["name"] == "canopy"
        assert data["product"] == "meta"
        assert "description" in data

    def test_attribution_defaults_to_none(self):
        """Attribution is None by default."""
        source = MetaChmSource(version="2")
        assert source.attribution is None

    def test_attribution_accepted(self):
        """MetaChmSource accepts an Attribution object."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Test citation",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        source = MetaChmSource(version="2", attribution=attr)
        assert source.attribution.license_name == "CC-BY-4.0"

    def test_attribution_serialized_in_model_dump(self):
        """Attribution is included in model_dump output."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Test citation",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        source = MetaChmSource(version="2", attribution=attr)
        data = source.model_dump()
        assert data["attribution"]["license_name"] == "CC-BY-4.0"
        assert data["attribution"]["accessed_on"] == "2026-02-27"


class TestCreateMetaChmRequest:
    """Tests for CreateMetaChmRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateMetaChmRequest()
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []
        assert request.version == "2"

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateMetaChmRequest(
            name="Test Grid",
            description="A test grid",
            tags=["test", "chm"],
            version="2",
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "chm"]
        assert request.version == "2"

    def test_version_can_be_set_to_v1(self):
        """Version can be explicitly set to v1."""
        request = CreateMetaChmRequest(version="1")
        assert request.version == "1"

    def test_extent_buffer_cells_defaults_to_none(self):
        request = CreateMetaChmRequest()
        assert request.extent_buffer_cells is None

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateMetaChmRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateMetaChmRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateMetaChmRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateMetaChmRequest(extent_buffer_cells=11)

    def test_resolved_extent_buffer_cells_uses_default_when_omitted(self):
        request = CreateMetaChmRequest()
        assert request.resolved_extent_buffer_cells(0) == 0

    def test_resolved_extent_buffer_cells_preserves_zero(self):
        request = CreateMetaChmRequest(extent_buffer_cells=0)
        assert request.resolved_extent_buffer_cells(0) == 0


class TestAttribution:
    """Tests for Attribution model."""

    def test_valid_attribution(self):
        """All fields accepted."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Some citation text",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        assert attr.license_name == "CC-BY-4.0"
        assert attr.access_url == "https://example.com"

    def test_missing_field_rejected(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            Attribution(
                license_name="CC-BY-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                # missing citation, access_url, accessed_on
            )


class TestChmBands:
    """Tests for build_chm_bands helper and band definitions."""

    def test_single_chm_band(self):
        """build_chm_bands returns a single band."""
        bands = build_chm_bands()
        assert len(bands) == 1

    def test_band_key_is_chm(self):
        """The band key is 'chm'."""
        bands = build_chm_bands()
        assert bands[0].key == "chm"

    def test_band_type_is_continuous(self):
        """The band type is continuous."""
        bands = build_chm_bands()
        assert bands[0].type == BandType.continuous

    def test_band_unit_is_meters(self):
        """The band unit is 'm'."""
        bands = build_chm_bands()
        assert bands[0].unit == "m"

    def test_band_index_is_zero(self):
        """The band index is 0."""
        bands = build_chm_bands()
        assert bands[0].index == 0


class TestNaipChmSource:
    """Tests for NaipChmSource model."""

    def test_product_is_always_naip(self):
        """The product field is always 'naip'."""
        source = NaipChmSource()
        assert source.product == "naip"

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = NaipChmSource()
        assert source.name == "canopy"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = NaipChmSource()
        assert "NAIP" in source.description
        assert "0.6m resolution" in source.description

    def test_model_dump(self):
        """Model serializes correctly."""
        source = NaipChmSource()
        data = source.model_dump()
        assert data["product"] == "naip"
        assert "description" in data


class TestCreateNaipChmRequest:
    """Tests for CreateNaipChmRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateNaipChmRequest()
        # Assumes you updated the default to 2023 in the schema!
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateNaipChmRequest(
            name="Test NAIP Grid",
            description="A test grid",
            tags=["test", "chm", "naip"],
        )
        assert request.name == "Test NAIP Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "chm", "naip"]

    def test_extent_buffer_cells_defaults_to_none(self):
        request = CreateNaipChmRequest()
        assert request.extent_buffer_cells is None

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateNaipChmRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateNaipChmRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateNaipChmRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateNaipChmRequest(extent_buffer_cells=11)

    def test_resolved_extent_buffer_cells_uses_default_when_omitted(self):
        request = CreateNaipChmRequest()
        assert request.resolved_extent_buffer_cells(0) == 0
