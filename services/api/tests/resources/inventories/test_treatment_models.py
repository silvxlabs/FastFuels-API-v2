"""
Unit tests for inventory silvicultural treatment schemas.

Tests the InventoryTreatment discriminated union (InventoryDiameterTreatment /
InventoryBasalAreaTreatment) and that treatments are accepted as a create-time
field on the PIM create request.
"""

import pytest
from api.resources.inventories.treatment_models import (
    InventoryBasalAreaTreatment,
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
