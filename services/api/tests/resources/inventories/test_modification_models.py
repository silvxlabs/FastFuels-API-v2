"""
Unit tests for inventory modification schemas.

Tests InventoryModificationCondition, InventoryExpressionCondition,
InventoryModificationAction, RemoveAction, and InventoryModification.
"""

import pytest
from api.resources.inventories.modification_models import (
    ATTRIBUTE_UNITS,
    InventoryAttribute,
    InventoryExpressionCondition,
    InventoryFeatureSpatialCondition,
    InventoryGeometrySpatialCondition,
    InventoryModification,
    InventoryModificationAction,
    InventoryModificationCondition,
    RemoveAction,
)
from api.resources.inventories.modifications.schema import (
    ApplyModificationsRequest,
    ModificationsInventorySource,
)
from pydantic import ValidationError


class TestInventoryAttribute:
    def test_all_values(self):
        assert InventoryAttribute.dbh == "dbh"
        assert InventoryAttribute.height == "height"
        assert InventoryAttribute.crown_ratio == "crown_ratio"
        assert InventoryAttribute.fia_species_code == "fia_species_code"

    def test_attribute_units_keys_match(self):
        for attr in InventoryAttribute:
            assert attr.value in ATTRIBUTE_UNITS


class TestInventoryModificationCondition:
    def test_basic_condition(self):
        cond = InventoryModificationCondition(attribute="dbh", operator="lt", value=5.0)
        assert cond.attribute == "dbh"
        assert cond.operator == "lt"
        assert cond.value == 5.0
        assert cond.unit is None

    def test_condition_with_unit(self):
        cond = InventoryModificationCondition(
            attribute="dbh", operator="lt", value=1.0, unit="in"
        )
        assert cond.unit == "in"

    def test_condition_with_list_value(self):
        cond = InventoryModificationCondition(
            attribute="fia_species_code", operator="eq", value=[93, 15]
        )
        assert cond.value == [93, 15]

    def test_species_only_eq_ne(self):
        # eq should work
        InventoryModificationCondition(
            attribute="fia_species_code", operator="eq", value=93
        )
        # ne should work
        InventoryModificationCondition(
            attribute="fia_species_code", operator="ne", value=93
        )
        # gt should fail
        with pytest.raises(ValidationError, match="fia_species_code"):
            InventoryModificationCondition(
                attribute="fia_species_code", operator="gt", value=93
            )

    def test_species_lt_rejected(self):
        with pytest.raises(ValidationError):
            InventoryModificationCondition(
                attribute="fia_species_code", operator="lt", value=93
            )

    def test_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError, match="not compatible"):
            InventoryModificationCondition(
                attribute="dbh", operator="lt", value=5.0, unit="kg"
            )

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValidationError, match="Unknown unit"):
            InventoryModificationCondition(
                attribute="dbh", operator="lt", value=5.0, unit="fathoms_of_nonsense"
            )

    def test_unit_on_species_rejected(self):
        with pytest.raises(ValidationError, match="not supported"):
            InventoryModificationCondition(
                attribute="fia_species_code", operator="eq", value=93, unit="cm"
            )

    def test_height_with_feet(self):
        cond = InventoryModificationCondition(
            attribute="height", operator="gt", value=65, unit="ft"
        )
        assert cond.unit == "ft"

    def test_crown_ratio_dimensionless(self):
        cond = InventoryModificationCondition(
            attribute="crown_ratio", operator="gt", value=0.5
        )
        assert cond.unit is None

    def test_crown_ratio_rejects_length_unit(self):
        with pytest.raises(ValidationError, match="not dimensionless"):
            InventoryModificationCondition(
                attribute="crown_ratio", operator="gt", value=0.5, unit="cm"
            )

    def test_list_value_with_gt_rejected(self):
        with pytest.raises(ValidationError, match="List values"):
            InventoryModificationCondition(
                attribute="dbh", operator="gt", value=[5.0, 10.0]
            )

    def test_list_value_with_lt_rejected(self):
        with pytest.raises(ValidationError, match="List values"):
            InventoryModificationCondition(
                attribute="dbh", operator="lt", value=[5.0, 10.0]
            )

    def test_list_value_with_eq_allowed(self):
        cond = InventoryModificationCondition(
            attribute="dbh", operator="eq", value=[5.0, 10.0]
        )
        assert cond.value == [5.0, 10.0]

    def test_list_value_with_ne_allowed(self):
        cond = InventoryModificationCondition(
            attribute="dbh", operator="ne", value=[5.0, 10.0]
        )
        assert cond.value == [5.0, 10.0]


class TestInventoryExpressionCondition:
    def test_valid_expression(self):
        cond = InventoryExpressionCondition(expression="dbh < 5 and height < 2")
        assert cond.expression == "dbh < 5 and height < 2"

    def test_expression_with_math(self):
        InventoryExpressionCondition(expression="height * crown_ratio < 1.0")

    def test_expression_with_division(self):
        InventoryExpressionCondition(expression="height / dbh > 100")

    def test_disallowed_name(self):
        with pytest.raises(ValidationError, match="disallowed name"):
            InventoryExpressionCondition(expression="fia_species_code == 93")

    def test_function_call_rejected(self):
        with pytest.raises(ValidationError, match="Function calls"):
            InventoryExpressionCondition(expression="abs(dbh) < 5")

    def test_invalid_syntax(self):
        with pytest.raises(ValidationError, match="Invalid expression"):
            InventoryExpressionCondition(expression="dbh <><> 5")

    def test_x_y_not_allowed(self):
        with pytest.raises(ValidationError, match="disallowed name"):
            InventoryExpressionCondition(expression="x > 0")

    def test_attribute_access_rejected(self):
        with pytest.raises(ValidationError, match="Attribute access"):
            InventoryExpressionCondition(expression="dbh.__class__")


class TestInventoryModificationAction:
    def test_basic_action(self):
        action = InventoryModificationAction(
            attribute="height", modifier="multiply", value=0.9
        )
        assert action.attribute == "height"
        assert action.modifier == "multiply"
        assert action.value == 0.9
        assert action.unit is None

    def test_action_with_unit(self):
        action = InventoryModificationAction(
            attribute="dbh", modifier="add", value=1.0, unit="in"
        )
        assert action.unit == "in"

    def test_all_modifiers(self):
        for modifier in ["multiply", "divide", "add", "subtract", "replace"]:
            InventoryModificationAction(
                attribute="height", modifier=modifier, value=1.0
            )

    def test_incompatible_unit_rejected(self):
        with pytest.raises(ValidationError, match="not compatible"):
            InventoryModificationAction(
                attribute="height", modifier="replace", value=5.0, unit="kg"
            )

    def test_divide_by_zero_rejected(self):
        with pytest.raises(ValidationError, match="divide by zero"):
            InventoryModificationAction(attribute="height", modifier="divide", value=0)

    def test_divide_by_nonzero_allowed(self):
        action = InventoryModificationAction(
            attribute="height", modifier="divide", value=2.0
        )
        assert action.value == 2.0


class TestRemoveAction:
    def test_default_modifier(self):
        action = RemoveAction()
        assert action.modifier == "remove"

    def test_serialization(self):
        action = RemoveAction()
        assert action.model_dump() == {"modifier": "remove"}


class TestInventoryModification:
    def test_basic_remove(self):
        mod = InventoryModification(
            conditions={"attribute": "dbh", "operator": "lt", "value": 2.54},
            actions={"modifier": "remove"},
        )
        assert len(mod.conditions) == 1
        assert len(mod.actions) == 1

    def test_single_condition_wrapped(self):
        """Single condition dict should be auto-wrapped in a list."""
        mod = InventoryModification(
            conditions={"attribute": "dbh", "operator": "lt", "value": 5.0},
            actions={"modifier": "remove"},
        )
        assert isinstance(mod.conditions, list)
        assert len(mod.conditions) == 1

    def test_single_action_wrapped(self):
        """Single action dict should be auto-wrapped in a list."""
        mod = InventoryModification(
            conditions=[{"attribute": "dbh", "operator": "gt", "value": 50}],
            actions={"attribute": "height", "modifier": "multiply", "value": 0.9},
        )
        assert isinstance(mod.actions, list)
        assert len(mod.actions) == 1

    def test_multiple_conditions(self):
        mod = InventoryModification(
            conditions=[
                {"attribute": "fia_species_code", "operator": "eq", "value": 202},
                {"attribute": "dbh", "operator": "lt", "value": 5.0},
            ],
            actions={"modifier": "remove"},
        )
        assert len(mod.conditions) == 2

    def test_expression_condition(self):
        mod = InventoryModification(
            conditions={"expression": "dbh < 5 and height < 2"},
            actions={"modifier": "remove"},
        )
        assert isinstance(mod.conditions[0], InventoryExpressionCondition)

    def test_mixed_conditions(self):
        mod = InventoryModification(
            conditions=[
                {"attribute": "dbh", "operator": "gt", "value": 10},
                {"expression": "height / dbh > 50"},
            ],
            actions={"modifier": "remove"},
        )
        assert isinstance(mod.conditions[0], InventoryModificationCondition)
        assert isinstance(mod.conditions[1], InventoryExpressionCondition)

    def test_remove_must_be_sole_action(self):
        with pytest.raises(ValidationError, match="sole action"):
            InventoryModification(
                conditions={"attribute": "dbh", "operator": "lt", "value": 5.0},
                actions=[
                    {"modifier": "remove"},
                    {"attribute": "height", "modifier": "multiply", "value": 0.9},
                ],
            )

    def test_empty_conditions_rejected(self):
        with pytest.raises(ValidationError):
            InventoryModification(
                conditions=[],
                actions={"modifier": "remove"},
            )

    def test_empty_actions_rejected(self):
        with pytest.raises(ValidationError):
            InventoryModification(
                conditions={"attribute": "dbh", "operator": "lt", "value": 5.0},
                actions=[],
            )

    def test_model_dump_roundtrip(self):
        mod = InventoryModification(
            conditions={"attribute": "dbh", "operator": "lt", "value": 2.54},
            actions={"modifier": "remove"},
        )
        data = mod.model_dump()
        assert data["conditions"][0]["attribute"] == "dbh"
        assert data["actions"][0]["modifier"] == "remove"
        # Can reconstruct from dump
        InventoryModification(**data)


class TestModificationsInventorySource:
    def test_name_is_modifications(self):
        source = ModificationsInventorySource(
            source_inventory_id="inv123",
            modifications=[{"conditions": [], "actions": []}],
        )
        assert source.name == "modifications"

    def test_name_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            ModificationsInventorySource(
                name="other",
                source_inventory_id="inv123",
                modifications=[],
            )


class TestApplyModificationsRequest:
    def test_basic_request(self):
        req = ApplyModificationsRequest(
            modifications=[
                {
                    "conditions": {"attribute": "dbh", "operator": "lt", "value": 5.0},
                    "actions": {"modifier": "remove"},
                }
            ]
        )
        assert len(req.modifications) == 1

    def test_empty_modifications_rejected(self):
        with pytest.raises(ValidationError):
            ApplyModificationsRequest(modifications=[])

    def test_inherits_base_fields(self):
        req = ApplyModificationsRequest(
            name="Modified trees",
            description="Removed small trees",
            tags=["modified"],
            modifications=[
                {
                    "conditions": {"attribute": "dbh", "operator": "lt", "value": 5.0},
                    "actions": {"modifier": "remove"},
                }
            ],
        )
        assert req.name == "Modified trees"
        assert req.tags == ["modified"]


# =============================================================================
# InventoryGeometrySpatialCondition Tests
# =============================================================================


SAMPLE_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [[-120.0, 38.0], [-119.0, 38.0], [-119.0, 39.0], [-120.0, 39.0], [-120.0, 38.0]]
    ],
}


class TestInventoryGeometrySpatialCondition:
    def test_minimal_valid(self):
        cond = InventoryGeometrySpatialCondition(
            source="geometry", operator="within", geometry=SAMPLE_POLYGON
        )
        assert cond.source == "geometry"
        assert cond.operator == "within"
        assert cond.geometry == SAMPLE_POLYGON
        assert cond.crs is None
        assert cond.buffer_m is None

    def test_source_required(self):
        with pytest.raises(ValidationError):
            InventoryGeometrySpatialCondition(
                operator="within", geometry=SAMPLE_POLYGON
            )

    def test_geometry_required(self):
        with pytest.raises(ValidationError):
            InventoryGeometrySpatialCondition(source="geometry", operator="within")

    def test_operator_required(self):
        with pytest.raises(ValidationError):
            InventoryGeometrySpatialCondition(
                source="geometry", geometry=SAMPLE_POLYGON
            )

    def test_buffer_m_accepts_zero(self):
        cond = InventoryGeometrySpatialCondition(
            source="geometry",
            operator="within",
            geometry=SAMPLE_POLYGON,
            buffer_m=0,
        )
        assert cond.buffer_m == 0

    def test_buffer_m_rejects_negative(self):
        with pytest.raises(ValidationError):
            InventoryGeometrySpatialCondition(
                source="geometry",
                operator="within",
                geometry=SAMPLE_POLYGON,
                buffer_m=-1.0,
            )


# =============================================================================
# InventoryFeatureSpatialCondition Tests
# =============================================================================


class TestInventoryFeatureSpatialCondition:
    def test_minimal_valid(self):
        cond = InventoryFeatureSpatialCondition(
            source="feature", operator="within", feature_id="feat_road_abc"
        )
        assert cond.source == "feature"
        assert cond.operator == "within"
        assert cond.feature_id == "feat_road_abc"
        assert cond.buffer_m is None

    def test_source_required(self):
        with pytest.raises(ValidationError):
            InventoryFeatureSpatialCondition(
                operator="within", feature_id="feat_road_abc"
            )

    def test_feature_id_required(self):
        with pytest.raises(ValidationError):
            InventoryFeatureSpatialCondition(source="feature", operator="within")

    def test_operator_required(self):
        with pytest.raises(ValidationError):
            InventoryFeatureSpatialCondition(
                source="feature", feature_id="feat_road_abc"
            )

    def test_buffer_m_accepts_positive(self):
        cond = InventoryFeatureSpatialCondition(
            source="feature",
            operator="within",
            feature_id="feat_water_xyz",
            buffer_m=5.0,
        )
        assert cond.buffer_m == 5.0

    def test_buffer_m_rejects_negative(self):
        with pytest.raises(ValidationError):
            InventoryFeatureSpatialCondition(
                source="feature",
                operator="within",
                feature_id="feat_road_abc",
                buffer_m=-2.5,
            )

    def test_no_target_field(self):
        """Inventory spatial conditions have no `target` field since trees are
        points."""
        cond = InventoryFeatureSpatialCondition(
            source="feature", operator="within", feature_id="feat_road_abc"
        )
        assert not hasattr(cond, "target")


# =============================================================================
# Spatial condition dispatch via InventoryModification
# =============================================================================


class TestInventorySpatialConditionDispatch:
    def test_feature_variant_via_source(self):
        mod = InventoryModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_water_xyz",
                    "buffer_m": 5,
                }
            ],
            actions=[{"modifier": "remove"}],
        )
        cond = mod.conditions[0]
        assert isinstance(cond, InventoryFeatureSpatialCondition)
        assert cond.feature_id == "feat_water_xyz"
        assert cond.buffer_m == 5

    def test_geometry_variant_via_source(self):
        mod = InventoryModification(
            conditions=[
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": SAMPLE_POLYGON,
                }
            ],
            actions=[{"modifier": "remove"}],
        )
        assert isinstance(mod.conditions[0], InventoryGeometrySpatialCondition)

    def test_spatial_with_attribute_condition_compound(self):
        """A spatial condition + an attribute condition can coexist in the
        same modification (AND semantics)."""
        mod = InventoryModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 4,
                },
                {"attribute": "dbh", "operator": "gt", "value": 30},
            ],
            actions=[{"modifier": "remove"}],
        )
        assert len(mod.conditions) == 2
        assert isinstance(mod.conditions[0], InventoryFeatureSpatialCondition)
        assert isinstance(mod.conditions[1], InventoryModificationCondition)

    def test_spatial_with_remove_action(self):
        """RemoveAction can pair with a spatial condition."""
        mod = InventoryModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_water_xyz",
                    "buffer_m": 5,
                }
            ],
            actions=[{"modifier": "remove"}],
        )
        assert isinstance(mod.actions[0], RemoveAction)

    def test_round_trip_via_model_dump(self):
        mod = InventoryModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 3,
                }
            ],
            actions=[{"modifier": "remove"}],
        )
        data = mod.model_dump()
        assert data["conditions"][0]["source"] == "feature"
        assert data["conditions"][0]["feature_id"] == "feat_road_abc"
        reparsed = InventoryModification.model_validate(data)
        assert isinstance(reparsed.conditions[0], InventoryFeatureSpatialCondition)
