"""
Unit tests for api/v2/resources/grids/chm/schema.py
and api/v2/resources/grids/providers/chm.py

Tests the Meta CHM schema models, ChmSource base, and band definitions.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.chm.schema import (
    CreateMetaChmRequest,
    MetaChmSource,
    MetaChmVersion,
    build_chm_bands,
)
from api.resources.grids.providers.chm import ChmSource
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestChmSource:
    """Tests for ChmSource base model."""

    def test_name_is_always_chm(self):
        """The name field is always 'chm'."""
        source = ChmSource(product="meta", version="2024")
        assert source.name == "chm"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'chm'."""
        with pytest.raises(ValidationError):
            ChmSource(name="other", product="meta", version="2024")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            ChmSource(version="2024")

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            ChmSource(product="meta")

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = ChmSource(product="meta", version="2024")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = ChmSource(
            product="meta",
            version="2024",
            description="Test description",
        )
        assert source.description == "Test description"


class TestMetaChmSource:
    """Tests for MetaChmSource model."""

    def test_product_is_always_meta(self):
        """The product field is always 'meta'."""
        source = MetaChmSource(version="2024")
        assert source.product == "meta"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'meta'."""
        with pytest.raises(ValidationError):
            MetaChmSource(product="other", version="2024")

    def test_name_is_always_chm(self):
        """The name field is always 'chm'."""
        source = MetaChmSource(version="2024")
        assert source.name == "chm"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = MetaChmSource(version="2024")
        assert "Meta" in source.description
        assert "canopy height" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            MetaChmSource()

    def test_invalid_version_rejected(self):
        """Invalid version string is rejected."""
        with pytest.raises(ValidationError):
            MetaChmSource(version="9999")

    def test_model_dump(self):
        """Model serializes correctly."""
        source = MetaChmSource(version="2024")
        data = source.model_dump()
        assert data["name"] == "chm"
        assert data["product"] == "meta"
        assert data["version"] == "2024"
        assert "description" in data


class TestCreateMetaChmRequest:
    """Tests for CreateMetaChmRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateMetaChmRequest()
        assert request.version == "2024"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2024(self):
        """version defaults to '2024'."""
        request = CreateMetaChmRequest()
        assert request.version == "2024"

    def test_invalid_version_rejected(self):
        """Invalid version string is rejected."""
        with pytest.raises(ValidationError):
            CreateMetaChmRequest(version="9999")

    def test_all_versions_accepted(self):
        """All defined Meta CHM versions are accepted."""
        for version in MetaChmVersion:
            request = CreateMetaChmRequest(version=version.value)
            assert request.version == version

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateMetaChmRequest(
            version="2024",
            name="Test Grid",
            description="A test grid",
            tags=["test", "chm"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "chm"]


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
