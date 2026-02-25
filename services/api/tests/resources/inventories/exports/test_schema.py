"""
Unit tests for inventory export request schema and examples.

These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.exports.schema import (
    ExportInventoryRequest,
    InventoryExportFormat,
    InventoryExportSource,
)
from api.resources.inventories.exports.examples import (
    ALL_INVENTORY_EXPORT_EXAMPLE_VALUES,
)
from pydantic import ValidationError


class TestInventoryExportFormat:
    """Validation tests for InventoryExportFormat enum."""

    def test_valid_formats(self):
        assert InventoryExportFormat("parquet") == InventoryExportFormat.parquet
        assert InventoryExportFormat("csv") == InventoryExportFormat.csv
        assert InventoryExportFormat("geojson") == InventoryExportFormat.geojson
        assert InventoryExportFormat("geopackage") == InventoryExportFormat.geopackage

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            InventoryExportFormat("xlsx")


class TestExportInventoryRequest:
    """Validation tests for ExportInventoryRequest."""

    def test_defaults(self):
        request = ExportInventoryRequest()
        assert request.columns is None
        assert request.expiration_days == 7
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_columns_list(self):
        request = ExportInventoryRequest(columns=["x", "y", "dbh"])
        assert request.columns == ["x", "y", "dbh"]

    def test_columns_none(self):
        request = ExportInventoryRequest(columns=None)
        assert request.columns is None

    def test_expiration_days_range(self):
        request = ExportInventoryRequest(expiration_days=1)
        assert request.expiration_days == 1

        request = ExportInventoryRequest(expiration_days=7)
        assert request.expiration_days == 7

    def test_expiration_days_below_min_rejected(self):
        with pytest.raises(ValidationError):
            ExportInventoryRequest(expiration_days=0)

    def test_expiration_days_above_max_rejected(self):
        with pytest.raises(ValidationError):
            ExportInventoryRequest(expiration_days=8)

    def test_name_max_length(self):
        request = ExportInventoryRequest(name="a" * 255)
        assert len(request.name) == 255

        with pytest.raises(ValidationError):
            ExportInventoryRequest(name="a" * 256)

    def test_description_max_length(self):
        request = ExportInventoryRequest(description="a" * 2000)
        assert len(request.description) == 2000

        with pytest.raises(ValidationError):
            ExportInventoryRequest(description="a" * 2001)

    def test_columns_empty_list(self):
        """Empty columns list is accepted (distinct from None / omitted)."""
        request = ExportInventoryRequest(columns=[])
        assert request.columns == []

    def test_columns_max_length(self):
        """Columns list is capped at 100 entries."""
        request = ExportInventoryRequest(columns=[f"col_{i}" for i in range(100)])
        assert len(request.columns) == 100

        with pytest.raises(ValidationError):
            ExportInventoryRequest(columns=[f"col_{i}" for i in range(101)])

    def test_tags(self):
        request = ExportInventoryRequest(tags=["a", "b"])
        assert request.tags == ["a", "b"]


class TestInventoryExportSource:
    """Validation tests for InventoryExportSource."""

    @pytest.mark.parametrize("fmt", ["parquet", "csv", "geojson", "geopackage"])
    def test_valid_format_names(self, fmt):
        source = InventoryExportSource(name=fmt, inventory_id="inv-123")
        assert source.name == fmt

    def test_invalid_format_name_rejected(self):
        with pytest.raises(ValidationError):
            InventoryExportSource(name="xlsx", inventory_id="inv-123")

    def test_inventory_id_required(self):
        with pytest.raises(ValidationError):
            InventoryExportSource(name="csv")

    def test_columns_none_vs_list(self):
        source_none = InventoryExportSource(
            name="csv", inventory_id="inv-123", columns=None
        )
        source_list = InventoryExportSource(
            name="csv", inventory_id="inv-123", columns=["x", "y"]
        )
        assert source_none.columns is None
        assert source_list.columns == ["x", "y"]

    def test_crs_optional(self):
        source = InventoryExportSource(name="geojson", inventory_id="inv-123")
        assert source.crs is None

        source = InventoryExportSource(
            name="geojson", inventory_id="inv-123", crs="EPSG:32611"
        )
        assert source.crs == "EPSG:32611"


class TestExampleValidation:
    """Validate that all documented examples pass schema validation."""

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_INVENTORY_EXPORT_EXAMPLE_VALUES
    )
    def test_example_is_valid(self, example_name, example_value):
        """Each example should pass schema validation."""
        ExportInventoryRequest(**example_value)
