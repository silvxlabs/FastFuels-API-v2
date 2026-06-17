"""
Unit tests for api/v2/resources/grids/modification_models.py

Tests the grid-specific modification classes and base enums.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.modification_models import (
    GridFeatureSpatialCondition,
    GridGeometrySpatialCondition,
    GridModification,
    GridModificationAction,
    GridModificationCondition,
    GridSpatialTarget,
)
from api.resources.grids.utils import resolve_modification_fuel_model_labels
from api.resources.modifications import (
    Modifier,
    Operator,
    SpatialOperator,
)
from fastapi import HTTPException
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


class TestGridSpatialTarget:
    """Tests for GridSpatialTarget enum."""

    def test_all_spatial_targets_exist(self):
        """All spatial targets are defined."""
        assert GridSpatialTarget.centroid.value == "centroid"
        assert GridSpatialTarget.cell.value == "cell"

    def test_enum_count(self):
        """Enum has exactly 2 members."""
        assert len(GridSpatialTarget) == 2


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
# GridGeometrySpatialCondition Tests
# =============================================================================


class TestGridGeometrySpatialCondition:
    """Tests for GridGeometrySpatialCondition model."""

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
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.source == "geometry"
        assert condition.operator == SpatialOperator.within
        assert condition.geometry == sample_polygon
        assert condition.crs is None
        assert condition.target == GridSpatialTarget.centroid

    def test_source_is_required(self, sample_polygon):
        """source field is required (discriminator)."""
        with pytest.raises(ValidationError):
            GridGeometrySpatialCondition(
                operator=SpatialOperator.within, geometry=sample_polygon
            )

    def test_operator_is_required(self, sample_polygon):
        """operator field is required."""
        with pytest.raises(ValidationError):
            GridGeometrySpatialCondition(source="geometry", geometry=sample_polygon)

    def test_geometry_is_required(self):
        """geometry field is required."""
        with pytest.raises(ValidationError):
            GridGeometrySpatialCondition(
                source="geometry", operator=SpatialOperator.within
            )

    def test_crs_defaults_to_none(self, sample_polygon):
        """crs field defaults to None (domain CRS)."""
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.crs is None

    def test_crs_can_be_set(self, sample_polygon):
        """crs field can be set explicitly."""
        crs = {"type": "name", "properties": {"name": "EPSG:4326"}}
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
            crs=crs,
        )
        assert condition.crs == crs

    def test_target_defaults_to_centroid(self, sample_polygon):
        """target field defaults to centroid."""
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.target == GridSpatialTarget.centroid

    def test_target_can_be_cell(self, sample_polygon):
        """target can be set to cell."""
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
            target=GridSpatialTarget.cell,
        )
        assert condition.target == GridSpatialTarget.cell

    def test_all_spatial_operators(self, sample_polygon):
        """All spatial operators are accepted."""
        for op in SpatialOperator:
            condition = GridGeometrySpatialCondition(
                source="geometry",
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
                    "source": "geometry",
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
        assert isinstance(modification.conditions[1], GridGeometrySpatialCondition)

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


# =============================================================================
# GridFeatureSpatialCondition Tests
# =============================================================================


class TestGridFeatureSpatialCondition:
    """Tests for GridFeatureSpatialCondition model."""

    def test_minimal_valid_condition(self):
        """Minimal feature spatial condition with required fields."""
        condition = GridFeatureSpatialCondition(
            source="feature",
            operator=SpatialOperator.intersects,
            feature_id="feat_road_abc",
        )
        assert condition.source == "feature"
        assert condition.operator == SpatialOperator.intersects
        assert condition.feature_id == "feat_road_abc"
        assert condition.buffer_m is None
        assert condition.target == GridSpatialTarget.centroid

    def test_source_is_required(self):
        """source field is required (discriminator)."""
        with pytest.raises(ValidationError):
            GridFeatureSpatialCondition(
                operator=SpatialOperator.intersects, feature_id="feat_road_abc"
            )

    def test_feature_id_is_required(self):
        """feature_id field is required."""
        with pytest.raises(ValidationError):
            GridFeatureSpatialCondition(
                source="feature", operator=SpatialOperator.intersects
            )

    def test_operator_is_required(self):
        """operator field is required."""
        with pytest.raises(ValidationError):
            GridFeatureSpatialCondition(source="feature", feature_id="feat_road_abc")

    def test_buffer_m_accepts_zero(self):
        """buffer_m accepts zero."""
        condition = GridFeatureSpatialCondition(
            source="feature",
            operator=SpatialOperator.intersects,
            feature_id="feat_road_abc",
            buffer_m=0,
        )
        assert condition.buffer_m == 0

    def test_buffer_m_accepts_positive(self):
        """buffer_m accepts positive values."""
        condition = GridFeatureSpatialCondition(
            source="feature",
            operator=SpatialOperator.intersects,
            feature_id="feat_water_xyz",
            buffer_m=5.0,
        )
        assert condition.buffer_m == 5.0

    def test_buffer_m_rejects_negative(self):
        """buffer_m rejects negative values."""
        with pytest.raises(ValidationError):
            GridFeatureSpatialCondition(
                source="feature",
                operator=SpatialOperator.intersects,
                feature_id="feat_road_abc",
                buffer_m=-1.0,
            )

    def test_target_can_be_cell(self):
        """target can be set to cell."""
        condition = GridFeatureSpatialCondition(
            source="feature",
            operator=SpatialOperator.intersects,
            feature_id="feat_road_abc",
            target=GridSpatialTarget.cell,
        )
        assert condition.target == GridSpatialTarget.cell


# =============================================================================
# Buffer field on GridGeometrySpatialCondition
# =============================================================================


class TestGridGeometrySpatialConditionBuffer:
    """Buffer-specific tests for GridGeometrySpatialCondition."""

    @pytest.fixture
    def sample_polygon(self):
        return {
            "type": "Polygon",
            "coordinates": [
                [[-120, 38], [-119, 38], [-119, 39], [-120, 39], [-120, 38]]
            ],
        }

    def test_buffer_m_defaults_to_none(self, sample_polygon):
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
        )
        assert condition.buffer_m is None

    def test_buffer_m_accepts_zero(self, sample_polygon):
        condition = GridGeometrySpatialCondition(
            source="geometry",
            operator=SpatialOperator.within,
            geometry=sample_polygon,
            buffer_m=0,
        )
        assert condition.buffer_m == 0

    def test_buffer_m_rejects_negative(self, sample_polygon):
        with pytest.raises(ValidationError):
            GridGeometrySpatialCondition(
                source="geometry",
                operator=SpatialOperator.within,
                geometry=sample_polygon,
                buffer_m=-2.5,
            )


# =============================================================================
# Discriminated Union dispatch via GridModification
# =============================================================================


class TestGridSpatialConditionDispatch:
    """Tests that GridModification.conditions correctly routes dict payloads
    to the right spatial-condition variant."""

    def test_feature_variant_via_source_field(self):
        modification = GridModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 3,
                }
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
        )
        cond = modification.conditions[0]
        assert isinstance(cond, GridFeatureSpatialCondition)
        assert cond.feature_id == "feat_road_abc"
        assert cond.buffer_m == 3

    def test_geometry_variant_via_source_field(self):
        modification = GridModification(
            conditions=[
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 39], [-120, 38]]
                        ],
                    },
                }
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
        )
        assert isinstance(modification.conditions[0], GridGeometrySpatialCondition)

    def test_source_field_required_for_spatial_condition(self):
        """A spatial-looking condition without `source` fails the discriminator."""
        with pytest.raises(ValidationError):
            GridModification(
                conditions=[
                    {
                        "operator": "within",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-120, 38],
                                    [-119, 38],
                                    [-119, 39],
                                    [-120, 39],
                                    [-120, 38],
                                ]
                            ],
                        },
                    }
                ],
                actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
            )

    def test_mixed_geometry_and_feature_conditions(self):
        """A single modification can mix geometry and feature variants."""
        modification = GridModification(
            conditions=[
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-120, 38], [-119, 38], [-119, 39], [-120, 39], [-120, 38]]
                        ],
                    },
                },
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_road_abc",
                },
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
        )
        assert len(modification.conditions) == 2
        assert isinstance(modification.conditions[0], GridGeometrySpatialCondition)
        assert isinstance(modification.conditions[1], GridFeatureSpatialCondition)

    def test_missing_geometry_and_feature_id_rejected(self):
        """A spatial-looking condition with neither geometry nor feature_id
        cannot be discriminated and fails validation."""
        with pytest.raises(ValidationError):
            GridModification(
                conditions=[{"operator": "intersects"}],
                actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
            )

    def test_feature_variant_round_trip(self):
        """model_dump emits `source` so the payload round-trips."""
        modification = GridModification(
            conditions=[
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 4,
                }
            ],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.1}],
        )
        data = modification.model_dump()
        assert data["conditions"][0]["source"] == "feature"
        assert data["conditions"][0]["feature_id"] == "feat_road_abc"
        assert data["conditions"][0]["buffer_m"] == 4

        # Re-parse and confirm identical
        reparsed = GridModification.model_validate(data)
        assert isinstance(reparsed.conditions[0], GridFeatureSpatialCondition)
        assert reparsed.conditions[0].feature_id == "feat_road_abc"


class TestFuelModelLabelResolution:
    """resolve_modification_fuel_model_labels normalizes FBFM labels to codes."""

    _BAND_TYPES = {"fbfm": "categorical", "fuel_load.1hr": "continuous"}

    def test_resolves_condition_and_action_labels_in_place(self):
        modification = GridModification(
            conditions=[
                {"band": "fbfm", "operator": "eq", "value": ["GR1", "GR2"]},
            ],
            actions=[{"band": "fbfm", "modifier": "replace", "value": "GR3"}],
        )

        resolve_modification_fuel_model_labels([modification], self._BAND_TYPES)

        assert modification.conditions[0].value == [101, 102]
        assert modification.actions[0].value == 103

    def test_passes_numeric_values_through(self):
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "gt", "value": 0.5}],
            actions=[{"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.1}],
        )

        resolve_modification_fuel_model_labels([modification], self._BAND_TYPES)

        assert modification.conditions[0].value == 0.5
        assert modification.actions[0].value == 0.1

    def test_unknown_label_raises_422(self):
        modification = GridModification(
            conditions=[{"band": "fbfm", "operator": "eq", "value": "GRX"}],
            actions=[{"band": "fbfm", "modifier": "replace", "value": 102}],
        )

        with pytest.raises(HTTPException) as exc_info:
            resolve_modification_fuel_model_labels([modification], self._BAND_TYPES)
        assert exc_info.value.status_code == 422

    def test_string_value_on_continuous_band_raises_422(self):
        # A label on a continuous band must be rejected clearly, not
        # silently mis-resolved to a fuel-model code.
        modification = GridModification(
            conditions=[{"band": "fuel_load.1hr", "operator": "eq", "value": "GR1"}],
            actions=[{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.1}],
        )

        with pytest.raises(HTTPException) as exc_info:
            resolve_modification_fuel_model_labels([modification], self._BAND_TYPES)
        assert exc_info.value.status_code == 422
        assert "continuous" in exc_info.value.detail.lower()
