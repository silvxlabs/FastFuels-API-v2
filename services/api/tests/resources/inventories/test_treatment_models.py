"""
Unit tests for inventory silvicultural treatment schemas.

Tests InventoryTreatmentMethod, InventoryTreatmentTarget, InventoryTreatment,
and that treatments are accepted as a create-time field on the PIM create
request.
"""

import pytest
from api.resources.inventories.treatment_models import (
    InventoryTreatment,
    InventoryTreatmentMethod,
    InventoryTreatmentTarget,
)
from api.resources.inventories.tree.pim.examples import EXAMPLE_PIM_WITH_TREATMENT
from api.resources.inventories.tree.pim.schema import CreatePimInventoryRequest
from pydantic import ValidationError

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


class TestInventoryTreatmentTarget:
    def test_diameter_only(self):
        target = InventoryTreatmentTarget(diameter=30.0)
        assert target.diameter == 30.0
        assert target.basal_area is None

    def test_basal_area_only(self):
        target = InventoryTreatmentTarget(basal_area=25.0)
        assert target.basal_area == 25.0
        assert target.diameter is None

    def test_both_metrics_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(diameter=30.0, basal_area=25.0)

    def test_no_metric_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget()

    @pytest.mark.parametrize("field", ["diameter", "basal_area"])
    def test_non_positive_rejected(self, field):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(**{field: 0})
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(**{field: -1.0})

    def test_diameter_inches_unit(self):
        target = InventoryTreatmentTarget(diameter=12.0, unit="in")
        assert target.unit == "in"

    def test_basal_area_imperial_unit(self):
        target = InventoryTreatmentTarget(basal_area=80.0, unit="ft**2/acre")
        assert target.unit == "ft**2/acre"

    def test_non_canonical_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(basal_area=80.0, unit="ft^2/acre")

    def test_diameter_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(diameter=30.0, unit="kg")

    def test_basal_area_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(basal_area=25.0, unit="m")

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatmentTarget(basal_area=25.0, unit="bogus")


class TestInventoryTreatment:
    def test_from_below_diameter(self):
        t = InventoryTreatment(method="from_below", target={"diameter": 30.0})
        assert t.method == InventoryTreatmentMethod.from_below
        assert t.conditions == []

    def test_from_above_basal_area(self):
        t = InventoryTreatment(method="from_above", target={"basal_area": 20.0})
        assert t.method == InventoryTreatmentMethod.from_above

    def test_proportional_basal_area(self):
        t = InventoryTreatment(method="proportional", target={"basal_area": 15.0})
        assert t.method == InventoryTreatmentMethod.proportional

    def test_proportional_diameter_rejected(self):
        # No proportional-to-diameter operation exists.
        with pytest.raises(ValidationError):
            InventoryTreatment(method="proportional", target={"diameter": 30.0})

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatment(method="random", target={"basal_area": 15.0})

    def test_missing_target_rejected(self):
        with pytest.raises(ValidationError):
            InventoryTreatment(method="from_below")

    def test_with_geometry_condition(self):
        t = InventoryTreatment(
            method="from_below",
            target={"basal_area": 18.0},
            conditions=[GEOMETRY_CONDITION],
        )
        assert len(t.conditions) == 1

    def test_with_feature_condition(self):
        t = InventoryTreatment(
            method="from_above",
            target={"basal_area": 20.0},
            conditions=[FEATURE_CONDITION],
        )
        assert t.conditions[0].feature_id == "feat_abc"

    def test_single_condition_converted_to_list(self):
        t = InventoryTreatment(
            method="from_below",
            target={"basal_area": 18.0},
            conditions=GEOMETRY_CONDITION,
        )
        assert isinstance(t.conditions, list)
        assert len(t.conditions) == 1

    def test_attribute_condition_rejected(self):
        # Treatments accept spatial conditions only.
        with pytest.raises(ValidationError):
            InventoryTreatment(
                method="from_below",
                target={"basal_area": 18.0},
                conditions=[{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            )


class TestCreateRequestAcceptsTreatments:
    def test_default_empty(self):
        req = CreatePimInventoryRequest(source_pim_grid_id="grid_1", seed=42)
        assert req.treatments == []

    def test_with_treatment(self):
        req = CreatePimInventoryRequest(
            source_pim_grid_id="grid_1",
            seed=42,
            treatments=[{"method": "from_below", "target": {"diameter": 30.0}}],
        )
        assert len(req.treatments) == 1
        assert req.treatments[0].target.diameter == 30.0

    def test_openapi_example_round_trips(self):
        req = CreatePimInventoryRequest(**EXAMPLE_PIM_WITH_TREATMENT)
        assert len(req.treatments) >= 1
