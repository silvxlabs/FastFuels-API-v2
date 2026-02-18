"""
Unit tests for api/v2/resources/grids/modifications.py

Tests the grid-specific modification classes and base enums.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.modifications import (
    GridModification,
    GridModificationAction,
    GridModificationCondition,
    GridSpatialCondition,
)
from api.resources.modifications import (
    Modifier,
    Operator,
    SpatialOperator,
    SpatialTarget,
)
from pydantic import ValidationError

# =============================================================================
# Base Enum Tests (from api/v2/resources/modifications.py)
# =============================================================================


class TestOperator:
    """Tests for Operator enum."""

    def test_all_comparison_operators_exist(self):
        """All standard comparison operators are defined."""
        assert Operator.eq.value == "eq"
        assert Operator.ne.value == "ne"
        assert Operator.gt.value == "gt"
        assert Operator.lt.value == "lt"
        assert Operator.ge.value == "ge"
        assert Operator.le.value == "le"

    def test_enum_count(self):
        """Enum has exactly 6 members."""
        assert len(Operator) == 6

    def test_can_create_from_string(self):
        """Enum can be created from string value."""
        assert Operator("eq") == Operator.eq
        assert Operator("gt") == Operator.gt


class TestSpatialOperator:
    """Tests for SpatialOperator enum."""

    def test_all_spatial_operators_exist(self):
        """All spatial operators are defined."""
        assert SpatialOperator.within.value == "within"
        assert SpatialOperator.outside.value == "outside"
        assert SpatialOperator.intersects.value == "intersects"

    def test_enum_count(self):
        """Enum has exactly 3 members."""
        assert len(SpatialOperator) == 3


class TestSpatialTarget:
    """Tests for SpatialTarget enum."""

    def test_all_spatial_targets_exist(self):
        """All spatial targets are defined."""
        assert SpatialTarget.centroid.value == "centroid"
        assert SpatialTarget.cell.value == "cell"

    def test_enum_count(self):
        """Enum has exactly 2 members."""
        assert len(SpatialTarget) == 2


class TestModifier:
    """Tests for Modifier enum."""

    def test_all_modifiers_exist(self):
        """All modifiers are defined."""
        assert Modifier.multiply.value == "multiply"
        assert Modifier.divide.value == "divide"
        assert Modifier.add.value == "add"
        assert Modifier.subtract.value == "subtract"
        assert Modifier.replace.value == "replace"

    def test_enum_count(self):
        """Enum has exactly 5 members."""
        assert len(Modifier) == 5


# =============================================================================
# GridModificationCondition Tests
# =============================================================================


class TestGridModificationCondition:
    """Tests for GridModificationCondition model."""

    def test_minimal_valid_condition(self):
        """Minimal condition with required fields."""
        condition = GridModificationCondition(
            band="fuel_load.1hr",
            operator=Operator.gt,
            value=0,
        )
        assert condition.band == "fuel_load.1hr"
        assert condition.operator == Operator.gt
        assert condition.value == 0

    def test_band_is_required(self):
        """band field is required."""
        with pytest.raises(ValidationError):
            GridModificationCondition(operator=Operator.gt, value=0)

    def test_operator_is_required(self):
        """operator field is required."""
        with pytest.raises(ValidationError):
            GridModificationCondition(band="fuel_load.1hr", value=0)

    def test_value_is_required(self):
        """value field is required."""
        with pytest.raises(ValidationError):
            GridModificationCondition(band="fuel_load.1hr", operator=Operator.gt)

    def test_value_accepts_int(self):
        """value accepts integer."""
        condition = GridModificationCondition(
            band="fbfm", operator=Operator.eq, value=91
        )
        assert condition.value == 91

    def test_value_accepts_float(self):
        """value accepts float."""
        condition = GridModificationCondition(
            band="fuel_load.1hr", operator=Operator.gt, value=0.5
        )
        assert condition.value == 0.5

    def test_value_accepts_string(self):
        """value accepts string (for categorical comparisons)."""
        condition = GridModificationCondition(
            band="fbfm", operator=Operator.eq, value="GR1"
        )
        assert condition.value == "GR1"

    def test_value_accepts_list(self):
        """value accepts list (for 'in' comparisons)."""
        condition = GridModificationCondition(
            band="fbfm", operator=Operator.eq, value=["GR1", "GR2", "GR3"]
        )
        assert condition.value == ["GR1", "GR2", "GR3"]

    def test_operator_accepts_string(self):
        """operator accepts string that maps to enum."""
        condition = GridModificationCondition(
            band="fuel_load.1hr", operator="gt", value=0
        )
        assert condition.operator == Operator.gt

    def test_dot_notation_band_preserved(self):
        """Dot notation in band key is preserved."""
        condition = GridModificationCondition(
            band="savr.live_herb", operator=Operator.gt, value=1000
        )
        assert condition.band == "savr.live_herb"


# =============================================================================
# GridSpatialCondition Tests
# =============================================================================


class TestGridSpatialCondition:
    """Tests for GridSpatialCondition model."""

    @pytest.fixture
    def sample_polygon(self):
        """Sample GeoJSON polygon."""
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [-120.0, 38.0],
                    [-119.0, 38.0],
                    [-119.0, 39.0],
                    [-120.0, 39.0],
                    [-120.0, 38.0],
                ]
            ],
        }

    def test_minimal_valid_condition(self, sample_polygon):
        """Minimal spatial condition with required fields."""
        condition = GridSpatialCondition(
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.operator == SpatialOperator.within
        assert condition.geometry == sample_polygon
        assert condition.crs is None
        assert condition.target == SpatialTarget.centroid

    def test_operator_is_required(self, sample_polygon):
        """operator field is required."""
        with pytest.raises(ValidationError):
            GridSpatialCondition(geometry=sample_polygon)

    def test_geometry_is_required(self):
        """geometry field is required."""
        with pytest.raises(ValidationError):
            GridSpatialCondition(operator=SpatialOperator.within)

    def test_crs_defaults_to_none(self, sample_polygon):
        """crs field defaults to None (domain CRS)."""
        condition = GridSpatialCondition(
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.crs is None

    def test_crs_can_be_set(self, sample_polygon):
        """crs field can be set explicitly."""
        crs = {"type": "name", "properties": {"name": "EPSG:4326"}}
        condition = GridSpatialCondition(
            operator=SpatialOperator.within,
            geometry=sample_polygon,
            crs=crs,
        )
        assert condition.crs == crs

    def test_target_defaults_to_centroid(self, sample_polygon):
        """target field defaults to centroid."""
        condition = GridSpatialCondition(
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.target == SpatialTarget.centroid

    def test_target_can_be_cell(self, sample_polygon):
        """target can be set to cell."""
        condition = GridSpatialCondition(
            operator=SpatialOperator.within,
            geometry=sample_polygon,
            target=SpatialTarget.cell,
        )
        assert condition.target == SpatialTarget.cell

    def test_all_spatial_operators(self, sample_polygon):
        """All spatial operators are accepted."""
        for op in SpatialOperator:
            condition = GridSpatialCondition(
                operator=op,
                geometry=sample_polygon,
            )
            assert condition.operator == op


# =============================================================================
# GridModificationAction Tests
# =============================================================================


class TestGridModificationAction:
    """Tests for GridModificationAction model."""

    def test_minimal_valid_action(self):
        """Minimal action with required fields."""
        action = GridModificationAction(
            band="fuel_load.1hr",
            modifier=Modifier.multiply,
            value=0.5,
        )
        assert action.band == "fuel_load.1hr"
        assert action.modifier == Modifier.multiply
        assert action.value == 0.5

    def test_band_is_required(self):
        """band field is required."""
        with pytest.raises(ValidationError):
            GridModificationAction(modifier=Modifier.multiply, value=0.5)

    def test_modifier_is_required(self):
        """modifier field is required."""
        with pytest.raises(ValidationError):
            GridModificationAction(band="fuel_load.1hr", value=0.5)

    def test_value_is_required(self):
        """value field is required."""
        with pytest.raises(ValidationError):
            GridModificationAction(band="fuel_load.1hr", modifier=Modifier.multiply)

    def test_value_accepts_int(self):
        """value accepts integer."""
        action = GridModificationAction(
            band="fbfm", modifier=Modifier.replace, value=91
        )
        assert action.value == 91

    def test_value_accepts_float(self):
        """value accepts float."""
        action = GridModificationAction(
            band="fuel_load.1hr", modifier=Modifier.multiply, value=0.5
        )
        assert action.value == 0.5

    def test_value_accepts_string(self):
        """value accepts string (for categorical replace)."""
        action = GridModificationAction(
            band="fbfm", modifier=Modifier.replace, value="NB1"
        )
        assert action.value == "NB1"

    def test_all_modifiers(self):
        """All modifiers are accepted."""
        for mod in Modifier:
            action = GridModificationAction(
                band="fuel_load.1hr",
                modifier=mod,
                value=1.0,
            )
            assert action.modifier == mod

    def test_modifier_accepts_string(self):
        """modifier accepts string that maps to enum."""
        action = GridModificationAction(
            band="fuel_load.1hr", modifier="multiply", value=0.5
        )
        assert action.modifier == Modifier.multiply


# =============================================================================
# GridModification Tests
# =============================================================================


class TestGridModification:
    """Tests for GridModification model."""

    def test_minimal_valid_modification(self):
        """Minimal modification with one condition and one action."""
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        assert len(modification.conditions) == 1
        assert len(modification.actions) == 1

    def test_conditions_is_required(self):
        """conditions field is required."""
        with pytest.raises(ValidationError):
            GridModification(
                actions=[
                    {"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}
                ]
            )

    def test_actions_is_required(self):
        """actions field is required."""
        with pytest.raises(ValidationError):
            GridModification(
                conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}]
            )

    def test_single_condition_converted_to_list(self):
        """Single condition dict is converted to list."""
        modification = GridModification(
            conditions={"band": "fuel_load.1hr", "operator": "gt", "value": 0},
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        assert isinstance(modification.conditions, list)
        assert len(modification.conditions) == 1

    def test_single_action_converted_to_list(self):
        """Single action dict is converted to list."""
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            actions={"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5},
        )
        assert isinstance(modification.actions, list)
        assert len(modification.actions) == 1

    def test_multiple_conditions(self):
        """Multiple conditions are ANDed together."""
        modification = GridModification(
            conditions=[
                {"band": "fuel_load.1hr", "operator": "gt", "value": 0},
                {"band": "fuel_load.1hr", "operator": "lt", "value": 1.0},
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        assert len(modification.conditions) == 2

    def test_multiple_actions(self):
        """Multiple actions can be applied."""
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            actions=[
                {"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5},
                {"band": "fuel_load.10hr", "modifier": "multiply", "value": 0.5},
            ],
        )
        assert len(modification.actions) == 2

    def test_mixed_conditions(self):
        """Can mix band conditions with spatial conditions."""
        modification = GridModification(
            conditions=[
                {"band": "fuel_load.1hr", "operator": "gt", "value": 0},
                {
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 39], [-120, 38]]
                        ],
                    },
                },
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        assert len(modification.conditions) == 2
        assert isinstance(modification.conditions[0], GridModificationCondition)
        assert isinstance(modification.conditions[1], GridSpatialCondition)

    def test_conditions_parsed_as_correct_types(self):
        """Conditions are parsed as the correct model types."""
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        assert isinstance(modification.conditions[0], GridModificationCondition)
        assert isinstance(modification.actions[0], GridModificationAction)

    def test_model_dump_serialization(self):
        """Model serializes correctly."""
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}],
        )
        data = modification.model_dump()
        assert data["conditions"][0]["band"] == "fuel_load.1hr"
        assert data["conditions"][0]["operator"] == "gt"
        assert data["actions"][0]["modifier"] == "multiply"
