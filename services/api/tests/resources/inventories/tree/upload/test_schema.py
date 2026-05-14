"""
Unit tests for api/v2/resources/inventories/tree/upload/schema.py

Tests the upload inventory schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.tree.upload.schema import (
    CreateInventoryUploadRequest,
    InventoryColumnMapping,
    InventoryUploadFormat,
)
from pydantic import ValidationError


class TestInventoryUploadFormat:
    def test_all_formats_valid(self):
        assert InventoryUploadFormat("csv") == InventoryUploadFormat.csv
        assert InventoryUploadFormat("geojson") == InventoryUploadFormat.geojson
        assert InventoryUploadFormat("geopackage") == InventoryUploadFormat.geopackage

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            InventoryUploadFormat("shapefile")


class TestInventoryColumnMapping:
    def test_defaults_all_none(self):
        mapping = InventoryColumnMapping()
        assert mapping.x is None
        assert mapping.y is None
        assert mapping.height is None
        assert mapping.fia_species_code is None
        assert mapping.fia_status_code is None
        assert mapping.dbh is None
        assert mapping.crown_ratio is None

    def test_partial_mapping_valid(self):
        mapping = InventoryColumnMapping(height="HT", fia_species_code="SPCD")
        assert mapping.height == "HT"
        assert mapping.fia_species_code == "SPCD"
        assert mapping.dbh is None

    def test_full_mapping_valid(self):
        mapping = InventoryColumnMapping(
            x="easting",
            y="northing",
            height="HT",
            fia_species_code="SPCD",
            fia_status_code="STATUSCD",
            dbh="DIA",
            crown_ratio="CR",
        )
        assert mapping.x == "easting"
        assert mapping.crown_ratio == "CR"

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            InventoryColumnMapping(weight="weight_col")

    def test_unknown_key_rejected_multiple(self):
        with pytest.raises(ValidationError):
            InventoryColumnMapping(biomass="bio", stem_count="n")

    def test_model_dump_excludes_none(self):
        mapping = InventoryColumnMapping(height="HT")
        dumped = mapping.model_dump(exclude_none=True)
        assert dumped == {"height": "HT"}


class TestCreateInventoryUploadRequest:
    def test_minimal_csv_request(self):
        req = CreateInventoryUploadRequest(format="csv")
        assert req.format == InventoryUploadFormat.csv
        assert req.columns == InventoryColumnMapping()
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []

    def test_minimal_geojson_request(self):
        req = CreateInventoryUploadRequest(format="geojson")
        assert req.format == InventoryUploadFormat.geojson

    def test_minimal_geopackage_request(self):
        req = CreateInventoryUploadRequest(format="geopackage")
        assert req.format == InventoryUploadFormat.geopackage

    def test_missing_format_raises(self):
        with pytest.raises(ValidationError):
            CreateInventoryUploadRequest()

    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError):
            CreateInventoryUploadRequest(format="shapefile")

    def test_with_column_mapping(self):
        req = CreateInventoryUploadRequest(
            format="csv",
            columns={"height": "HT", "fia_species_code": "SPCD"},
        )
        assert req.columns.height == "HT"
        assert req.columns.fia_species_code == "SPCD"

    def test_columns_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            CreateInventoryUploadRequest(
                format="csv",
                columns={"weight": "weight_col"},
            )

    def test_with_metadata(self):
        req = CreateInventoryUploadRequest(
            format="csv",
            name="Field Survey 2024",
            description="Plot measurements",
            tags=["field", "2024"],
        )
        assert req.name == "Field Survey 2024"
        assert req.tags == ["field", "2024"]
