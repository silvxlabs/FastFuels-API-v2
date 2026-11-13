"""
Unit tests for inventory silvicultural treatment schemas.

Tests the InventoryTreatment discriminated union (InventoryDiameterTreatment /
InventoryBasalAreaTreatment / InventoryCrownSpacingTreatment), the columns each
variant reads, and that treatments are accepted as a create-time field on the
PIM create request.
"""

import pytest
from api.resources.crown_radius import (
    CanopyAllometryMaxCrownRadiusSource,
    InventoryColumnMaxCrownRadiusSource,
)
from api.resources.inventories.treatment_models import (
    InventoryBasalAreaTreatment,
    InventoryCrownSpacingRank,
    InventoryCrownSpacingTreatment,
    InventoryDiameterTreatment,
    InventoryTreatment,
    InventoryTreatmentMethod,
)
from api.resources.inventories.tree.pim.examples import EXAMPLE_PIM_WITH_TREATMENT
from api.resources.inventories.tree.pim.schema import CreatePimInventoryRequest
from pydantic import TypeAdapter, ValidationError

TREATMENT_ADAPTER = TypeAdapter(InventoryTreatment)

GEOMETRY_CONDITION = {
    "source": "geometry",
    "operator": "within",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [[-120.0, 38.0], [-119.5, 38.0], [-119.5, 38.5], [-120.0, 38.0]]
        ],
    },
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
}

FEATURE_CONDITION = {
    "source": "feature",
    "operator": "within",
    "feature_id": "feat_abc",
    "buffer_m": 50,
}


class TestInventoryTreatmentMethod:
    def test_values(self):
        assert {m.value for m in InventoryTreatmentMethod} == {
            "from_below",
            "from_above",
            "proportional",
        }


class TestInventoryDiameterTreatment:
    @pytest.mark.parametrize("method", ["from_below", "from_above"])
    def test_valid_directional(self, method):
        t = InventoryDiameterTreatment(method=method, value=30.0)
        assert t.metric == "diameter"
        assert t.conditions == []

    def test_proportional_rejected(self):
        # proportional is not a member of the diameter variant's method Literal.
        with pytest.raises(ValidationError):
            InventoryDiameterTreatment(method="proportional", value=30.0)

    def test_non_positive_value_rejected(self):
        with pytest.raises(ValidationError):
            InventoryDiameterTreatment(method="from_below", value=0)

    def test_inches_unit(self):
        t = InventoryDiameterTreatment(method="from_below", value=12.0, unit="in")
        assert t.unit == "in"

    def test_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryDiameterTreatment(method="from_below", value=30.0, unit="kg")

    def test_non_canonical_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryDiameterTreatment(method="from_below", value=30.0, unit="ft^2")


class TestInventoryBasalAreaTreatment:
    @pytest.mark.parametrize("method", ["from_below", "from_above", "proportional"])
    def test_valid_methods(self, method):
        t = InventoryBasalAreaTreatment(method=method, value=25.0)
        assert t.metric == "basal_area"

    def test_non_positive_value_rejected(self):
        with pytest.raises(ValidationError):
            InventoryBasalAreaTreatment(method="proportional", value=-1.0)

    def test_imperial_unit(self):
        t = InventoryBasalAreaTreatment(
            method="from_above", value=80.0, unit="ft**2/acre"
        )
        assert t.unit == "ft**2/acre"

    def test_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryBasalAreaTreatment(method="from_below", value=25.0, unit="m")

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryBasalAreaTreatment(method="from_below", value=25.0, unit="bogus")

    def test_with_geometry_condition(self):
        t = InventoryBasalAreaTreatment(
            method="from_below", value=18.0, conditions=[GEOMETRY_CONDITION]
        )
        assert len(t.conditions) == 1

    def test_with_feature_condition(self):
        t = InventoryBasalAreaTreatment(
            method="from_above", value=20.0, conditions=[FEATURE_CONDITION]
        )
        assert t.conditions[0].feature_id == "feat_abc"

    def test_single_condition_converted_to_list(self):
        t = InventoryBasalAreaTreatment(
            method="from_below", value=18.0, conditions=GEOMETRY_CONDITION
        )
        assert isinstance(t.conditions, list)
        assert len(t.conditions) == 1

    def test_attribute_condition_rejected(self):
        # Treatments accept spatial conditions only.
        with pytest.raises(ValidationError):
            InventoryBasalAreaTreatment(
                method="from_below",
                value=18.0,
                conditions=[{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            )


class TestInventoryTreatmentDiscrimination:
    def test_diameter_variant(self):
        t = TREATMENT_ADAPTER.validate_python(
            {"metric": "diameter", "method": "from_below", "value": 30.0}
        )
        assert isinstance(t, InventoryDiameterTreatment)

    def test_basal_area_variant(self):
        t = TREATMENT_ADAPTER.validate_python(
            {"metric": "basal_area", "method": "proportional", "value": 25.0}
        )
        assert isinstance(t, InventoryBasalAreaTreatment)

    def test_proportional_diameter_rejected_through_union(self):
        with pytest.raises(ValidationError):
            TREATMENT_ADAPTER.validate_python(
                {"metric": "diameter", "method": "proportional", "value": 30.0}
            )

    def test_missing_metric_rejected(self):
        with pytest.raises(ValidationError):
            TREATMENT_ADAPTER.validate_python({"method": "from_below", "value": 30.0})

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValidationError):
            TREATMENT_ADAPTER.validate_python(
                {"metric": "trees_per_hectare", "method": "from_below", "value": 200}
            )


class TestCreateRequestAcceptsTreatments:
    def test_default_empty(self):
        req = CreatePimInventoryRequest(source_pim_grid_id="grid_1", seed=42)
        assert req.treatments == []

    def test_with_treatment(self):
        req = CreatePimInventoryRequest(
            source_pim_grid_id="grid_1",
            seed=42,
            treatments=[{"metric": "diameter", "method": "from_below", "value": 30.0}],
        )
        assert len(req.treatments) == 1
        assert isinstance(req.treatments[0], InventoryDiameterTreatment)
        assert req.treatments[0].value == 30.0

    def test_openapi_example_round_trips(self):
        req = CreatePimInventoryRequest(**EXAMPLE_PIM_WITH_TREATMENT)
        assert len(req.treatments) >= 1


class TestInventoryCrownSpacingTreatment:
    def test_defaults(self):
        t = TREATMENT_ADAPTER.validate_python({"metric": "crown_spacing", "value": 3.0})
        assert isinstance(t, InventoryCrownSpacingTreatment)
        assert t.rank_by == InventoryCrownSpacingRank.dbh
        assert isinstance(
            t.max_crown_radius_source, CanopyAllometryMaxCrownRadiusSource
        )
        assert t.max_crown_radius_source.equations == "purves"
        assert t.conditions == []
        assert t.native_unit == "m"

    def test_measured_radius_ranked_by_height(self):
        t = TREATMENT_ADAPTER.validate_python(
            {
                "metric": "crown_spacing",
                "value": 3.0,
                "rank_by": "height",
                "max_crown_radius_source": {
                    "type": "inventory_column",
                    "column": "crown_radius",
                },
            }
        )
        assert isinstance(
            t.max_crown_radius_source, InventoryColumnMaxCrownRadiusSource
        )
        assert t.max_crown_radius_source.column == "crown_radius"

    def test_feet_accepted(self):
        t = InventoryCrownSpacingTreatment(value=10.0, unit="ft")
        assert t.unit == "ft"

    @pytest.mark.parametrize("unit", ["cm**2", "kg", "m**2/ha"])
    def test_incompatible_unit_rejected(self, unit):
        with pytest.raises(ValidationError, match="not compatible"):
            InventoryCrownSpacingTreatment(value=3.0, unit=unit)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_value_rejected(self, value):
        with pytest.raises(ValidationError):
            InventoryCrownSpacingTreatment(value=value)

    def test_unknown_rank_by_rejected(self):
        with pytest.raises(ValidationError):
            InventoryCrownSpacingTreatment(value=3.0, rank_by="crown_ratio")

    def test_unknown_equations_rejected(self):
        with pytest.raises(ValidationError):
            InventoryCrownSpacingTreatment(
                value=3.0,
                max_crown_radius_source={"type": "allometry", "equations": "nope"},
            )

    def test_spatial_conditions_accepted(self):
        t = InventoryCrownSpacingTreatment(
            value=3.0, conditions=[FEATURE_CONDITION, FEATURE_CONDITION]
        )
        assert len(t.conditions) == 2

    def test_on_pim_create_request(self):
        req = CreatePimInventoryRequest(
            source_pim_grid_id="g",
            treatments=[{"metric": "crown_spacing", "value": 3.0}],
        )
        assert isinstance(req.treatments[0], InventoryCrownSpacingTreatment)


class TestRequiredColumns:
    """Each variant names the inventory columns it reads."""

    def test_diameter_and_basal_area_read_dbh(self):
        assert InventoryDiameterTreatment(
            method="from_below", value=10.0
        ).required_columns() == {"dbh"}
        assert InventoryBasalAreaTreatment(
            method="proportional", value=10.0
        ).required_columns() == {"dbh"}

    def test_crown_spacing_purves(self):
        t = InventoryCrownSpacingTreatment(value=3.0)
        assert t.required_columns() == {
            "x",
            "y",
            "dbh",
            "fia_species_code",
            "height",
            "crown_base_height",
        }

    def test_crown_spacing_crookston_stage(self):
        t = InventoryCrownSpacingTreatment(
            value=3.0,
            max_crown_radius_source={
                "type": "allometry",
                "equations": "crookston_stage",
            },
        )
        assert t.required_columns() == {"x", "y", "dbh", "fia_species_code", "height"}

    def test_crown_spacing_measured_radius_needs_no_dbh(self):
        """Null radius values fall back to Purves in standgen, but a complete
        radius column needs no allometry inputs, so none are required."""
        t = InventoryCrownSpacingTreatment(
            value=3.0,
            rank_by="height",
            max_crown_radius_source={
                "type": "inventory_column",
                "column": "crown_radius",
            },
        )
        assert t.required_columns() == {"x", "y", "height", "crown_radius"}
