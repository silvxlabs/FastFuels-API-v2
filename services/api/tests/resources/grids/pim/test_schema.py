"""
Unit tests for api/v2/resources/grids/pim/schema.py
and api/v2/resources/grids/providers/pim.py

Tests the TreeMap schema models, PimSource base, and band definitions.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.pim.schema import (
    CreateTreeMapRequest,
    TreeMapBand,
    TreeMapSource,
    TreeMapVersion,
    build_treemap_bands,
)
from api.resources.grids.providers.pim import PimSource
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestPimSource:
    """Tests for PimSource base model."""

    def test_name_is_always_pim(self):
        """The name field is always 'pim'."""
        source = PimSource(product="treemap", version="2022")
        assert source.name == "pim"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'pim'."""
        with pytest.raises(ValidationError):
            PimSource(name="other", product="treemap", version="2022")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            PimSource(version="2022")

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            PimSource(product="treemap")

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = PimSource(product="treemap", version="2022")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = PimSource(
            product="treemap",
            version="2022",
            description="Test description",
        )
        assert source.description == "Test description"


class TestTreeMapSource:
    """Tests for TreeMapSource model."""

    def test_product_is_always_treemap(self):
        """The product field is always 'treemap'."""
        source = TreeMapSource(version="2022", bands=["tm_id"])
        assert source.product == "treemap"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'treemap'."""
        with pytest.raises(ValidationError):
            TreeMapSource(product="other", version="2022", bands=["tm_id"])

    def test_name_is_always_pim(self):
        """The name field is always 'pim'."""
        source = TreeMapSource(version="2022", bands=["tm_id"])
        assert source.name == "pim"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = TreeMapSource(version="2022", bands=["tm_id"])
        assert "TreeMap" in source.description
        assert "plot imputation" in source.description

    def test_version_is_required(self):
        """The version field is required."""
        with pytest.raises(ValidationError):
            TreeMapSource(bands=["tm_id"])

    def test_bands_is_required(self):
        """The bands field is required."""
        with pytest.raises(ValidationError):
            TreeMapSource(version="2022")

    def test_invalid_version_rejected(self):
        """Invalid version string is rejected."""
        with pytest.raises(ValidationError):
            TreeMapSource(version="9999", bands=["tm_id"])

    def test_model_dump(self):
        """Model serializes correctly."""
        source = TreeMapSource(version="2022", bands=["tm_id"])
        data = source.model_dump()
        assert data["name"] == "pim"
        assert data["product"] == "treemap"
        assert data["version"] == "2022"
        assert data["bands"] == ["tm_id"]
        assert "description" in data

    def test_both_bands(self):
        """Model accepts both bands."""
        source = TreeMapSource(version="2022", bands=["tm_id", "plt_cn"])
        data = source.model_dump()
        assert data["bands"] == ["tm_id", "plt_cn"]


class TestCreateTreeMapRequest:
    """Tests for CreateTreeMapRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateTreeMapRequest()
        assert request.version == "2022"
        assert request.bands == [TreeMapBand.tm_id]
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_version_defaults_to_2022(self):
        """version defaults to '2022'."""
        request = CreateTreeMapRequest()
        assert request.version == "2022"

    def test_version_can_be_overridden(self):
        """version can be set to a different value."""
        request = CreateTreeMapRequest(version="2016")
        assert request.version == "2016"

    def test_invalid_version_rejected(self):
        """Invalid version string is rejected."""
        with pytest.raises(ValidationError):
            CreateTreeMapRequest(version="9999")

    def test_all_versions_accepted(self):
        """All defined TreeMap versions are accepted."""
        for version in TreeMapVersion:
            request = CreateTreeMapRequest(version=version.value)
            assert request.version == version

    def test_bands_default_to_tm_id_only(self):
        """bands defaults to just tm_id."""
        request = CreateTreeMapRequest()
        assert request.bands == [TreeMapBand.tm_id]

    def test_bands_can_include_both(self):
        """bands can include both tm_id and plt_cn."""
        request = CreateTreeMapRequest(bands=["tm_id", "plt_cn"])
        assert request.bands == [TreeMapBand.tm_id, TreeMapBand.plt_cn]

    def test_bands_cannot_be_empty(self):
        """bands must have at least one entry."""
        with pytest.raises(ValidationError):
            CreateTreeMapRequest(bands=[])

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateTreeMapRequest(
            version="2022",
            bands=["tm_id", "plt_cn"],
            name="Test Grid",
            description="A test grid",
            tags=["test", "pim"],
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "pim"]
        assert request.bands == [TreeMapBand.tm_id, TreeMapBand.plt_cn]


class TestTreeMapBands:
    """Tests for build_treemap_bands helper and band definitions."""

    def test_single_tm_id_band(self):
        """Single tm_id band builds correctly."""
        bands = build_treemap_bands([TreeMapBand.tm_id])
        assert len(bands) == 1
        assert bands[0].key == "tm_id"
        assert bands[0].type == BandType.categorical
        assert bands[0].unit is None
        assert bands[0].index == 0

    def test_single_plt_cn_band(self):
        """Single plt_cn band builds correctly."""
        bands = build_treemap_bands([TreeMapBand.plt_cn])
        assert len(bands) == 1
        assert bands[0].key == "plt_cn"
        assert bands[0].type == BandType.categorical
        assert bands[0].unit is None
        assert bands[0].index == 0

    def test_both_bands(self):
        """Both bands build with correct indices."""
        bands = build_treemap_bands([TreeMapBand.tm_id, TreeMapBand.plt_cn])
        assert len(bands) == 2
        assert bands[0].key == "tm_id"
        assert bands[0].index == 0
        assert bands[1].key == "plt_cn"
        assert bands[1].index == 1

    def test_reversed_order(self):
        """Bands build with indices matching request order."""
        bands = build_treemap_bands([TreeMapBand.plt_cn, TreeMapBand.tm_id])
        assert len(bands) == 2
        assert bands[0].key == "plt_cn"
        assert bands[0].index == 0
        assert bands[1].key == "tm_id"
        assert bands[1].index == 1
