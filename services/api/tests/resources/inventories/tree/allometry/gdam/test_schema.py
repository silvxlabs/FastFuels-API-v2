"""
Unit tests for api/v2/resources/inventories/tree/allometry/gdam/schema.py

Tests the GDAM allometry inventory schema models and validation.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.inventories.schema import InventoryType
from api.resources.inventories.tree.allometry.gdam.schema import (
    CreateGdamAllometryRequest,
    GdamAllometrySource,
    crown_base_height_missing_inputs,
    gdam_conditioned_columns,
    resolve_gdam_columns,
)
from pydantic import ValidationError


class TestGdamAllometrySource:
    """Tests for GdamAllometrySource model."""

    def test_valid_initialization(self):
        """Model initializes successfully with the required field."""
        source = GdamAllometrySource(source_tree_inventory_id="inv123")
        assert source.name == "gdam"
        assert source.source_tree_inventory_id == "inv123"

    def test_name_is_always_gdam(self):
        """The name field cannot be set to anything other than 'gdam'."""
        with pytest.raises(ValidationError):
            GdamAllometrySource(name="pim", source_tree_inventory_id="inv123")

    def test_source_tree_inventory_id_is_required(self):
        """The source_tree_inventory_id field is required."""
        with pytest.raises(ValidationError):
            GdamAllometrySource()

    def test_source_tree_inventory_checksum_defaults_to_none(self):
        """source_tree_inventory_checksum defaults to None when not captured."""
        source = GdamAllometrySource(source_tree_inventory_id="inv123")
        assert source.source_tree_inventory_checksum is None

    def test_source_tree_inventory_checksum_round_trips(self):
        """source_tree_inventory_checksum is carried through serialization."""
        source = GdamAllometrySource(
            source_tree_inventory_id="inv123",
            source_tree_inventory_checksum="sum123",
        )
        assert source.source_tree_inventory_checksum == "sum123"
        assert source.model_dump()["source_tree_inventory_checksum"] == "sum123"

    def test_model_dump(self):
        """Model serializes to the persisted source shape."""
        source = GdamAllometrySource(source_tree_inventory_id="inv123")
        data = source.model_dump()
        assert data == {
            "name": "gdam",
            "source_tree_inventory_id": "inv123",
            "source_tree_inventory_checksum": None,
            "impute_columns": [
                "dbh",
                "crown_ratio",
                "fia_species_code",
                "crown_base_height",
            ],
            "overwrite": False,
            "crown_base_height_model": {"name": "fia_national_logit_2026"},
            "conditioned_columns": None,
        }

    def test_document_without_new_fields_validates(self):
        """A source stored before overwrite/model/conditioned_columns existed
        still reads, with the defaults."""
        source = GdamAllometrySource.model_validate(
            {
                "name": "gdam",
                "source_tree_inventory_id": "inv123",
                "source_tree_inventory_checksum": "sum123",
                "impute_columns": ["dbh"],
            }
        )
        assert source.overwrite is False
        assert source.crown_base_height_model.name == "fia_national_logit_2026"
        assert source.conditioned_columns is None

    def test_records_new_fields(self):
        source = GdamAllometrySource(
            source_tree_inventory_id="inv123",
            overwrite=True,
            crown_base_height_model={"name": "randolph_2010"},
            conditioned_columns=["dbh"],
        )
        data = source.model_dump()
        assert data["overwrite"] is True
        assert data["crown_base_height_model"] == {"name": "randolph_2010"}
        assert data["conditioned_columns"] == ["dbh"]


class TestCreateGdamAllometryRequest:
    """Tests for CreateGdamAllometryRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request needs only source_tree_inventory_id; metadata defaults."""
        request = CreateGdamAllometryRequest(source_tree_inventory_id="inv123")
        assert request.source_tree_inventory_id == "inv123"
        assert request.type == InventoryType.tree
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []

    def test_request_with_metadata(self):
        """Optional metadata fields are accepted and stored."""
        request = CreateGdamAllometryRequest(
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
            CreateGdamAllometryRequest(name="Failing request")

    def test_field_set(self):
        """The request exposes only its intended fields — no algorithm/
        modifications/treatments knobs. Pins the contract so a future field
        addition is a deliberate, test-breaking change.
        """
        assert set(CreateGdamAllometryRequest.model_fields) == {
            "type",
            "name",
            "description",
            "tags",
            "source_tree_inventory_id",
            "impute_columns",
            "overwrite",
            "crown_base_height_model",
        }

    def test_overwrite_defaults_to_false(self):
        request = CreateGdamAllometryRequest(source_tree_inventory_id="inv123")
        assert request.overwrite is False

    def test_crown_base_height_model_defaults_to_gdam(self):
        request = CreateGdamAllometryRequest(source_tree_inventory_id="inv123")
        assert request.crown_base_height_model.name == "gdam"
        assert request.model_dump()["crown_base_height_model"] == {"name": "gdam"}

    def test_crown_base_height_model_description_names_gdam(self):
        description = CreateGdamAllometryRequest.model_fields[
            "crown_base_height_model"
        ].description
        assert "`gdam` (the default)" in description
        assert "does not predict" not in description

    def test_crown_base_height_model_selected_by_name(self):
        request = CreateGdamAllometryRequest(
            source_tree_inventory_id="inv123",
            crown_base_height_model={"name": "westfall_2019"},
            overwrite=True,
        )
        assert request.crown_base_height_model.name == "westfall_2019"
        assert request.overwrite is True

    def test_unknown_crown_base_height_model_rejected(self):
        with pytest.raises(ValidationError):
            CreateGdamAllometryRequest(
                source_tree_inventory_id="inv123",
                crown_base_height_model={"name": "bogus"},
            )

    def test_unknown_fields_ignored(self):
        """Unknown fields (e.g. an algorithm block) are dropped, not stored."""
        request = CreateGdamAllometryRequest(
            source_tree_inventory_id="inv123",
            algorithm={"name": "lmf"},
        )
        assert not hasattr(request, "algorithm")

    def test_impute_columns_defaults_to_all(self):
        """Omitting impute_columns imputes all four morphology columns."""
        request = CreateGdamAllometryRequest(source_tree_inventory_id="inv123")
        assert request.impute_columns == [
            "dbh",
            "crown_ratio",
            "fia_species_code",
            "crown_base_height",
        ]

    def test_impute_columns_crown_base_height_only_accepted(self):
        """crown_base_height alone is a valid selection at the schema layer."""
        request = CreateGdamAllometryRequest(
            source_tree_inventory_id="inv123",
            impute_columns=["crown_base_height"],
        )
        assert request.impute_columns == ["crown_base_height"]

    def test_impute_columns_subset_accepted(self):
        """A subset of the imputable columns is accepted and stored."""
        request = CreateGdamAllometryRequest(
            source_tree_inventory_id="inv123",
            impute_columns=["fia_species_code"],
        )
        assert request.impute_columns == ["fia_species_code"]

    def test_impute_columns_empty_rejected(self):
        """An empty impute_columns list is rejected."""
        with pytest.raises(ValidationError, match="at least one column"):
            CreateGdamAllometryRequest(
                source_tree_inventory_id="inv123", impute_columns=[]
            )

    def test_impute_columns_duplicates_rejected(self):
        """Duplicate columns in impute_columns are rejected."""
        with pytest.raises(ValidationError, match="duplicate"):
            CreateGdamAllometryRequest(
                source_tree_inventory_id="inv123",
                impute_columns=["dbh", "dbh"],
            )

    def test_impute_columns_unknown_rejected(self):
        """An unknown column name is rejected by the Literal type."""
        with pytest.raises(ValidationError):
            CreateGdamAllometryRequest(
                source_tree_inventory_id="inv123",
                impute_columns=["height"],
            )


class TestResolveGdamColumns:
    """`resolve_gdam_columns` reflects exactly what the imputed parquet carries:
    the source columns plus the columns GDAM actually imputes — never a
    hardcoded full set that over-claims un-imputed or never-written columns."""

    CHM_COLUMNS = [
        {"key": "x", "type": "continuous", "unit": "m"},
        {"key": "y", "type": "continuous", "unit": "m"},
        {"key": "height", "type": "continuous", "unit": "m"},
    ]

    def _keys(self, columns):
        return [c["key"] for c in columns]

    def test_full_impute_adds_all_four_not_status(self):
        result = resolve_gdam_columns(
            self.CHM_COLUMNS,
            ["dbh", "crown_ratio", "fia_species_code", "crown_base_height"],
        )
        assert self._keys(result) == [
            "x",
            "y",
            "height",
            "dbh",
            "crown_ratio",
            "fia_species_code",
            "crown_base_height",
        ]
        # GDAM never writes the live/dead flag, so it must not be claimed.
        assert "fia_status_code" not in self._keys(result)

    def test_narrowed_impute_only_adds_requested(self):
        result = resolve_gdam_columns(self.CHM_COLUMNS, ["fia_species_code"])
        assert self._keys(result) == ["x", "y", "height", "fia_species_code"]
        assert "dbh" not in self._keys(result)

    def test_existing_column_not_duplicated(self):
        source = self.CHM_COLUMNS + [{"key": "dbh", "type": "continuous", "unit": "cm"}]
        result = resolve_gdam_columns(source, ["dbh", "fia_species_code"])
        assert self._keys(result).count("dbh") == 1

    def test_crown_base_height_carries_type_and_unit(self):
        result = resolve_gdam_columns(self.CHM_COLUMNS, ["crown_base_height"])
        cbh = next(c for c in result if c["key"] == "crown_base_height")
        assert cbh["type"] == "continuous"
        assert cbh["unit"] == "m"

    def test_imputed_column_carries_type_and_unit(self):
        result = resolve_gdam_columns(self.CHM_COLUMNS, ["dbh"])
        dbh = next(c for c in result if c["key"] == "dbh")
        assert dbh["type"] == "continuous"
        assert dbh["unit"] == "cm"

    def test_source_tree_id_carried(self):
        """GDAM preserves the source tree_id column (#611)."""
        source = [{"key": "tree_id", "type": "categorical"}, *self.CHM_COLUMNS]
        result = resolve_gdam_columns(source, ["dbh"])
        assert self._keys(result) == ["tree_id", "x", "y", "height", "dbh"]

    def test_legacy_source_does_not_gain_tree_id(self):
        """A source without tree_id doesn't gain one: GDAM never writes it."""
        result = resolve_gdam_columns(self.CHM_COLUMNS, ["dbh"])
        assert "tree_id" not in self._keys(result)


REFIT = "fia_national_logit_2026"


class TestCrownBaseHeightMissingInputs:
    """A crown base height equation needs species, crown ratio and diameter,
    each either imputed in the same request or present on the source; gdam
    needs none."""

    def test_not_requested_skips_check(self):
        missing = crown_base_height_missing_inputs({"x", "y", "height"}, ["dbh"], REFIT)
        assert missing == []

    def test_all_inputs_on_source(self):
        source = {"x", "y", "height", "dbh", "crown_ratio", "fia_species_code"}
        missing = crown_base_height_missing_inputs(source, ["crown_base_height"], REFIT)
        assert missing == []

    def test_all_inputs_imputed(self):
        impute = ["dbh", "crown_ratio", "fia_species_code", "crown_base_height"]
        missing = crown_base_height_missing_inputs({"x", "y", "height"}, impute, REFIT)
        assert missing == []

    def test_mix_of_source_and_imputed(self):
        source = {"x", "y", "height", "dbh"}
        impute = ["crown_ratio", "fia_species_code", "crown_base_height"]
        assert crown_base_height_missing_inputs(source, impute, REFIT) == []

    def test_missing_inputs_reported(self):
        source = {"x", "y", "height", "crown_ratio"}
        impute = ["crown_base_height"]
        assert crown_base_height_missing_inputs(source, impute, REFIT) == [
            "dbh",
            "fia_species_code",
        ]

    def test_gdam_needs_no_inputs(self):
        missing = crown_base_height_missing_inputs(
            {"x", "y", "height"}, ["crown_base_height"], "gdam"
        )
        assert missing == []


class TestGdamConditionedColumns:
    """GDAM conditions on every GDAM column the source carries, except the
    overwritten ones; crown_base_height is sent only with the gdam model."""

    ALL = ["dbh", "crown_ratio", "fia_species_code", "crown_base_height"]
    FULL_SOURCE = {
        "x",
        "y",
        "height",
        "dbh",
        "crown_ratio",
        "fia_species_code",
        "crown_base_height",
    }

    def test_position_height_source_conditions_on_nothing(self):
        assert (
            gdam_conditioned_columns({"x", "y", "height"}, self.ALL, False, REFIT) == []
        )

    def test_no_overwrite_conditions_on_all_present(self):
        assert gdam_conditioned_columns(self.FULL_SOURCE, self.ALL, False, REFIT) == [
            "dbh",
            "crown_ratio",
            "fia_species_code",
        ]

    def test_no_overwrite_conditions_on_columns_not_imputed(self):
        source = {"x", "y", "height", "dbh", "crown_ratio"}
        assert gdam_conditioned_columns(source, ["fia_species_code"], False, REFIT) == [
            "dbh",
            "crown_ratio",
        ]

    def test_overwrite_excludes_overwritten(self):
        result = gdam_conditioned_columns(
            self.FULL_SOURCE, ["crown_ratio", "crown_base_height"], True, REFIT
        )
        assert result == ["dbh", "fia_species_code"]

    def test_overwrite_all_conditions_on_nothing(self):
        assert gdam_conditioned_columns(self.FULL_SOURCE, self.ALL, True, REFIT) == []

    def test_crown_base_height_not_conditioned_with_equation(self):
        source = {"x", "y", "height", "crown_base_height"}
        assert gdam_conditioned_columns(source, ["dbh"], False, REFIT) == []

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_crown_base_height_alone_calls_no_gdam(self, overwrite):
        result = gdam_conditioned_columns(
            self.FULL_SOURCE, ["crown_base_height"], overwrite, REFIT
        )
        assert result == []

    def test_gdam_conditions_on_kept_crown_base_height(self):
        assert gdam_conditioned_columns(self.FULL_SOURCE, ["dbh"], False, "gdam") == [
            "dbh",
            "crown_ratio",
            "fia_species_code",
            "crown_base_height",
        ]

    def test_gdam_overwritten_crown_base_height_not_conditioned(self):
        result = gdam_conditioned_columns(
            self.FULL_SOURCE, ["crown_base_height"], True, "gdam"
        )
        assert result == ["dbh", "crown_ratio", "fia_species_code"]

    def test_gdam_crown_base_height_alone_calls_gdam(self):
        result = gdam_conditioned_columns(
            self.FULL_SOURCE, ["crown_base_height"], False, "gdam"
        )
        assert result == [
            "dbh",
            "crown_ratio",
            "fia_species_code",
            "crown_base_height",
        ]

    def test_gdam_position_height_source_conditions_on_nothing(self):
        source = {"x", "y", "height"}
        assert gdam_conditioned_columns(source, self.ALL, False, "gdam") == []
