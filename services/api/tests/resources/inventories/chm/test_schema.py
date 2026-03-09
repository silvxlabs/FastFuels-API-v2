"""
Unit tests for api/v2/resources/inventories/chm/schema.py

Tests the CHM inventory extraction schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.chm.schema import (
    ChmInventorySource,
    CreateChmInventoryRequest,
    StemIsolationLmf,
)
from pydantic import ValidationError


class TestStemIsolationLmf:
    """Tests for StemIsolationLmf model."""

    def test_default_values(self):
        """Model initializes with correct default values."""
        algo = StemIsolationLmf()
        assert algo.name == "lmf"
        assert algo.min_height == 2.0
        assert algo.footprint_size == 3

    def test_name_is_always_lmf(self):
        """The name field cannot be set to anything other than 'lmf'."""
        with pytest.raises(ValidationError):
            StemIsolationLmf(name="watershed")

    def test_custom_values(self):
        """Model accepts custom valid parameters."""
        algo = StemIsolationLmf(min_height=5.5, footprint_size=7)
        assert algo.min_height == 5.5
        assert algo.footprint_size == 7

    def test_model_dump(self):
        """Model serializes correctly."""
        algo = StemIsolationLmf(min_height=3.0, footprint_size=5)
        data = algo.model_dump()
        assert data["name"] == "lmf"
        assert data["min_height"] == 3.0
        assert data["footprint_size"] == 5


class TestChmInventorySource:
    """Tests for ChmInventorySource model."""

    def test_valid_initialization(self):
        """Model initializes successfully with required fields."""
        source = ChmInventorySource(
            source_chm_grid_id="grid123",
            algorithm=StemIsolationLmf(),
        )
        assert source.name == "chm"
        assert source.source_chm_grid_id == "grid123"
        assert source.algorithm.name == "lmf"

    def test_name_is_always_chm(self):
        """The name field cannot be overridden."""
        with pytest.raises(ValidationError):
            ChmInventorySource(
                name="pim",
                source_chm_grid_id="grid123",
                algorithm=StemIsolationLmf(),
            )

    def test_source_chm_grid_id_is_required(self):
        """The source_chm_grid_id field is required."""
        with pytest.raises(ValidationError):
            ChmInventorySource(algorithm=StemIsolationLmf())

    def test_algorithm_is_required(self):
        """The algorithm field is required."""
        with pytest.raises(ValidationError):
            ChmInventorySource(source_chm_grid_id="grid123")


class TestCreateChmInventoryRequest:
    """Tests for CreateChmInventoryRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with only the required source_chm_grid_id."""
        request = CreateChmInventoryRequest(source_chm_grid_id="grid123")

        # Check explicit fields
        assert request.source_chm_grid_id == "grid123"

        # Check inherited CreateInventoryRequestBase defaults
        assert request.type == "tree"
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

        # Check CHM specific defaults
        assert isinstance(request.algorithm, StemIsolationLmf)
        assert request.algorithm.min_height == 2.0
        assert request.modifications == []

    def test_full_request_with_all_fields(self):
        """Full request with all optional base and specific fields."""
        request = CreateChmInventoryRequest(
            source_chm_grid_id="grid123",
            algorithm=StemIsolationLmf(min_height=4.0, footprint_size=5),
            type="tree",
            name="Test CHM Inventory",
            description="Extracting trees from NAIP",
            tags=["chm", "lidar"],
            modifications=[],
        )
        assert request.source_chm_grid_id == "grid123"
        assert request.algorithm.min_height == 4.0
        assert request.algorithm.footprint_size == 5
        assert request.name == "Test CHM Inventory"
        assert request.description == "Extracting trees from NAIP"
        assert request.tags == ["chm", "lidar"]

    def test_missing_source_grid_id_rejected(self):
        """Missing required source_chm_grid_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateChmInventoryRequest(name="Failing request")
