"""
Unit tests for api/v2/resources/inventories/tree/allometry/gdam/schema.py

Tests the GDAM allometry inventory schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.schema import InventoryType
from api.resources.inventories.tree.allometry.gdam.schema import (
    CreateGdamInventoryRequest,
    GdamInventorySource,
)
from pydantic import ValidationError


class TestGdamInventorySource:
    """Tests for GdamInventorySource model."""

    def test_valid_initialization(self):
        """Model initializes successfully with the required field."""
        source = GdamInventorySource(source_tree_inventory_id="inv123")
        assert source.name == "gdam"
        assert source.source_tree_inventory_id == "inv123"

    def test_name_is_always_gdam(self):
        """The name field cannot be set to anything other than 'gdam'."""
        with pytest.raises(ValidationError):
            GdamInventorySource(name="pim", source_tree_inventory_id="inv123")

    def test_source_tree_inventory_id_is_required(self):
        """The source_tree_inventory_id field is required."""
        with pytest.raises(ValidationError):
            GdamInventorySource()

    def test_source_tree_inventory_checksum_defaults_to_none(self):
        """source_tree_inventory_checksum defaults to None when not captured."""
        source = GdamInventorySource(source_tree_inventory_id="inv123")
        assert source.source_tree_inventory_checksum is None

    def test_source_tree_inventory_checksum_round_trips(self):
        """source_tree_inventory_checksum is carried through serialization."""
        source = GdamInventorySource(
            source_tree_inventory_id="inv123",
            source_tree_inventory_checksum="sum123",
        )
        assert source.source_tree_inventory_checksum == "sum123"
        assert source.model_dump()["source_tree_inventory_checksum"] == "sum123"

    def test_model_dump(self):
        """Model serializes to the persisted source shape."""
        source = GdamInventorySource(source_tree_inventory_id="inv123")
        data = source.model_dump()
        assert data == {
            "name": "gdam",
            "source_tree_inventory_id": "inv123",
            "source_tree_inventory_checksum": None,
            "impute_columns": ["dbh", "crown_ratio", "fia_species_code"],
        }


class TestCreateGdamInventoryRequest:
    """Tests for CreateGdamInventoryRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request needs only source_tree_inventory_id; metadata defaults."""
        request = CreateGdamInventoryRequest(source_tree_inventory_id="inv123")
        assert request.source_tree_inventory_id == "inv123"
        assert request.type == InventoryType.tree
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_request_with_metadata(self):
        """Optional metadata fields are accepted and stored."""
        request = CreateGdamInventoryRequest(
            source_tree_inventory_id="inv123",
            name="My GDAM inventory",
            description="Filled in via GDAM",
            tags=["a", "b"],
        )
        assert request.name == "My GDAM inventory"
        assert request.description == "Filled in via GDAM"
        assert request.tags == ["a", "b"]

    def test_missing_source_tree_inventory_id_rejected(self):
        """Missing required source_tree_inventory_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateGdamInventoryRequest(name="Failing request")

    def test_field_set(self):
        """The request exposes only its intended fields — no algorithm/
        modifications/treatments knobs. Pins the contract so a future field
        addition is a deliberate, test-breaking change.
        """
        assert set(CreateGdamInventoryRequest.model_fields) == {
            "type",
            "name",
            "description",
            "tags",
            "source_tree_inventory_id",
            "impute_columns",
        }

    def test_unknown_fields_ignored(self):
        """Unknown fields (e.g. an algorithm block) are dropped, not stored."""
        request = CreateGdamInventoryRequest(
            source_tree_inventory_id="inv123",
            algorithm={"name": "lmf"},
        )
        assert not hasattr(request, "algorithm")

    def test_impute_columns_defaults_to_all(self):
        """Omitting impute_columns imputes all three morphology columns."""
        request = CreateGdamInventoryRequest(source_tree_inventory_id="inv123")
        assert request.impute_columns == ["dbh", "crown_ratio", "fia_species_code"]

    def test_impute_columns_subset_accepted(self):
        """A subset of the imputable columns is accepted and stored."""
        request = CreateGdamInventoryRequest(
            source_tree_inventory_id="inv123",
            impute_columns=["fia_species_code"],
        )
        assert request.impute_columns == ["fia_species_code"]

    def test_impute_columns_empty_rejected(self):
        """An empty impute_columns list is rejected."""
        with pytest.raises(ValidationError, match="at least one column"):
            CreateGdamInventoryRequest(
                source_tree_inventory_id="inv123", impute_columns=[]
            )

    def test_impute_columns_duplicates_rejected(self):
        """Duplicate columns in impute_columns are rejected."""
        with pytest.raises(ValidationError, match="duplicate"):
            CreateGdamInventoryRequest(
                source_tree_inventory_id="inv123",
                impute_columns=["dbh", "dbh"],
            )

    def test_impute_columns_unknown_rejected(self):
        """An unknown column name is rejected by the Literal type."""
        with pytest.raises(ValidationError):
            CreateGdamInventoryRequest(
                source_tree_inventory_id="inv123",
                impute_columns=["height"],
            )
