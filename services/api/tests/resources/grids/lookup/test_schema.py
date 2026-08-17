"""
Unit tests for api/v2/resources/grids/lookup/schema.py

Tests the shared LookupSource base model. Product-specific source, request,
and band tests live in each product's own test_schema.py.
"""

import pytest
from api.resources.grids.lookup.schema import LookupSource
from pydantic import ValidationError


class TestLookupSource:
    """Tests for LookupSource base model."""

    def test_name_is_always_lookup(self):
        """The name field is always 'lookup'."""
        source = LookupSource(
            table="fbfm40", source_grid_id="grid-123", source_band="fbfm"
        )
        assert source.name == "lookup"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'lookup'."""
        with pytest.raises(ValidationError):
            LookupSource(
                name="other",
                table="fbfm40",
                source_grid_id="grid-123",
                source_band="fbfm",
            )

    def test_table_is_required(self):
        """The table field is required."""
        with pytest.raises(ValidationError):
            LookupSource(source_grid_id="grid-123", source_band="fbfm")

    def test_source_grid_id_is_required(self):
        """The source_grid_id field is required."""
        with pytest.raises(ValidationError):
            LookupSource(table="fbfm40", source_band="fbfm")

    def test_source_band_is_required(self):
        """The source_band field is required."""
        with pytest.raises(ValidationError):
            LookupSource(table="fbfm40", source_grid_id="grid-123")

    def test_model_dump(self):
        """Model serializes correctly."""
        source = LookupSource(
            table="fbfm40", source_grid_id="grid-123", source_band="fbfm"
        )
        data = source.model_dump()
        assert data["name"] == "lookup"
        assert data["table"] == "fbfm40"
        assert data["source_grid_id"] == "grid-123"
        assert data["source_band"] == "fbfm"
