"""
Unit tests for api/v2/resources/inventories/tree/allometry/crown_base_height/schema.py

These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.schema import InventoryType
from api.resources.inventories.tree.allometry.crown_base_height.examples import (
    ALL_CROWN_BASE_HEIGHT_EXAMPLE_VALUES,
)
from api.resources.inventories.tree.allometry.crown_base_height.schema import (
    CreateCrownBaseHeightInventoryRequest,
    CrownBaseHeightInventorySource,
    resolve_crown_base_height_columns,
)
from pydantic import ValidationError


class TestCreateCrownBaseHeightInventoryRequest:
    def test_defaults(self):
        request = CreateCrownBaseHeightInventoryRequest(
            source_tree_inventory_id="inv123"
        )
        assert request.source_tree_inventory_id == "inv123"
        assert request.model.name == "fia_national_logit_2026"
        assert request.overwrite is False
        assert request.type == InventoryType.tree
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_source_tree_inventory_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateCrownBaseHeightInventoryRequest()

    @pytest.mark.parametrize(
        "name",
        [
            "fia_national_logit_2026",
            "monleon_2004_eq1",
            "monleon_2004_eq2",
            "toney_reeves_2009",
            "randolph_2010",
            "westfall_2019",
        ],
    )
    def test_model_selected_by_name(self, name):
        request = CreateCrownBaseHeightInventoryRequest(
            source_tree_inventory_id="inv123", model={"name": name}
        )
        assert request.model.name == name

    @pytest.mark.parametrize("model", [{"name": "gdam"}, {"name": "bogus"}, {}])
    def test_unknown_model_rejected(self, model):
        with pytest.raises(ValidationError):
            CreateCrownBaseHeightInventoryRequest(
                source_tree_inventory_id="inv123", model=model
            )

    def test_overwrite_true(self):
        request = CreateCrownBaseHeightInventoryRequest(
            source_tree_inventory_id="inv123", overwrite=True
        )
        assert request.overwrite is True

    def test_input_requirements_list_model_inputs(self):
        schema = CreateCrownBaseHeightInventoryRequest.model_json_schema()
        (entry,) = schema["x-input-requirements"]
        assert entry["field"] == "source_tree_inventory_id"
        assert entry["columns"] == sorted(
            ["x", "y", "height", "dbh", "crown_ratio", "fia_species_code"]
        )

    @pytest.mark.parametrize("name,value", ALL_CROWN_BASE_HEIGHT_EXAMPLE_VALUES)
    def test_examples_validate(self, name, value):
        CreateCrownBaseHeightInventoryRequest(**value)


class TestCrownBaseHeightInventorySource:
    def test_defaults(self):
        source = CrownBaseHeightInventorySource(source_tree_inventory_id="inv123")
        assert source.name == "crown_base_height"
        assert source.source_tree_inventory_checksum is None
        assert source.model.name == "fia_national_logit_2026"
        assert source.overwrite is False

    def test_name_is_fixed(self):
        with pytest.raises(ValidationError):
            CrownBaseHeightInventorySource(name="gdam", source_tree_inventory_id="x")

    def test_dump_records_model_and_overwrite(self):
        source = CrownBaseHeightInventorySource(
            source_tree_inventory_id="inv123",
            source_tree_inventory_checksum="abc",
            model={"name": "westfall_2019"},
            overwrite=True,
        )
        assert source.model_dump() == {
            "name": "crown_base_height",
            "source_tree_inventory_id": "inv123",
            "source_tree_inventory_checksum": "abc",
            "model": {"name": "westfall_2019"},
            "overwrite": True,
        }


class TestResolveCrownBaseHeightColumns:
    _SOURCE = [
        {"key": "x", "type": "continuous", "unit": "m"},
        {"key": "y", "type": "continuous", "unit": "m"},
        {"key": "height", "type": "continuous", "unit": "m"},
        {"key": "dbh", "type": "continuous", "unit": "cm"},
        {"key": "crown_ratio", "type": "continuous"},
        {"key": "fia_species_code", "type": "categorical"},
    ]

    def test_appends_crown_base_height(self):
        resolved = resolve_crown_base_height_columns(self._SOURCE)
        assert resolved[:-1] == self._SOURCE
        assert resolved[-1] == {
            "key": "crown_base_height",
            "type": "continuous",
            "unit": "m",
            "summary": None,
        }

    def test_existing_column_kept_in_place(self):
        source = [
            {"key": "crown_base_height", "type": "continuous", "unit": "m"},
            *self._SOURCE,
        ]
        assert resolve_crown_base_height_columns(source) == source

    def test_does_not_mutate_source(self):
        source = [dict(c) for c in self._SOURCE]
        resolved = resolve_crown_base_height_columns(source)
        resolved[0]["key"] = "changed"
        assert source == self._SOURCE
