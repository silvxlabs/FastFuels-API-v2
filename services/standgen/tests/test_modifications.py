"""
Unit tests for standgen/modifications.py

Tests the modification processing logic: conditions, actions,
unit conversion, clamping, and the full apply_modifications pipeline.
"""

import pandas as pd
import pytest
from standgen.modifications import (
    apply_action,
    apply_modifications,
    apply_single_modification,
    build_condition_mask,
    convert_value,
    evaluate_attribute_condition,
    evaluate_expression,
)


@pytest.fixture
def sample_df():
    """Sample tree inventory DataFrame."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fia_species_code": [202, 93, 122, 15, 202],
            "fia_status_code": [1, 1, 1, 2, 1],
            "dbh": [2.0, 10.0, 5.0, 1.0, 30.0],  # cm
            "height": [1.5, 15.0, 8.0, 0.5, 25.0],  # m
            "crown_ratio": [0.3, 0.6, 0.4, 0.2, 0.5],
        }
    )


class TestConvertValue:
    def test_no_unit_passthrough(self):
        assert convert_value("dbh", 5.0, None) == 5.0

    def test_inches_to_cm(self):
        result = convert_value("dbh", 1.0, "in")
        assert pytest.approx(result, rel=1e-3) == 2.54

    def test_feet_to_meters(self):
        result = convert_value("height", 100.0, "ft")
        assert pytest.approx(result, rel=1e-3) == 30.48

    def test_mm_to_cm(self):
        result = convert_value("dbh", 100.0, "mm")
        assert pytest.approx(result, rel=1e-3) == 10.0

    def test_dimensionless_passthrough(self):
        result = convert_value("crown_ratio", 0.5, None)
        assert result == 0.5

    def test_unknown_attribute_passthrough(self):
        result = convert_value("fia_species_code", 93, None)
        assert result == 93


class TestEvaluateAttributeCondition:
    def test_lt(self, sample_df):
        cond = {"attribute": "dbh", "operator": "lt", "value": 5.0}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [True, False, False, True, False]

    def test_gt(self, sample_df):
        cond = {"attribute": "dbh", "operator": "gt", "value": 5.0}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [False, True, False, False, True]

    def test_eq(self, sample_df):
        cond = {"attribute": "fia_species_code", "operator": "eq", "value": 202}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [True, False, False, False, True]

    def test_ne(self, sample_df):
        cond = {"attribute": "fia_species_code", "operator": "ne", "value": 202}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [False, True, True, True, False]

    def test_eq_list(self, sample_df):
        cond = {"attribute": "fia_species_code", "operator": "eq", "value": [93, 15]}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [False, True, False, True, False]

    def test_ne_list(self, sample_df):
        cond = {"attribute": "fia_species_code", "operator": "ne", "value": [93, 15]}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [True, False, True, False, True]

    def test_with_unit_conversion(self, sample_df):
        # 1 inch = 2.54 cm, so dbh < 1 inch → dbh < 2.54 cm
        cond = {"attribute": "dbh", "operator": "lt", "value": 1.0, "unit": "in"}
        mask = evaluate_attribute_condition(sample_df, cond)
        # dbh values: [2.0, 10.0, 5.0, 1.0, 30.0]
        # 2.0 < 2.54 → True, 1.0 < 2.54 → True
        assert mask.tolist() == [True, False, False, True, False]

    def test_ge(self, sample_df):
        cond = {"attribute": "dbh", "operator": "ge", "value": 5.0}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [False, True, True, False, True]

    def test_le(self, sample_df):
        cond = {"attribute": "dbh", "operator": "le", "value": 5.0}
        mask = evaluate_attribute_condition(sample_df, cond)
        assert mask.tolist() == [True, False, True, True, False]


class TestEvaluateExpression:
    def test_simple_expression(self, sample_df):
        mask = evaluate_expression(sample_df, "dbh < 5")
        assert mask.tolist() == [True, False, False, True, False]

    def test_compound_expression(self, sample_df):
        mask = evaluate_expression(sample_df, "dbh < 5 and height < 2")
        assert mask.tolist() == [True, False, False, True, False]

    def test_math_expression(self, sample_df):
        mask = evaluate_expression(sample_df, "height * crown_ratio < 1.0")
        # 1.5*0.3=0.45, 15*0.6=9, 8*0.4=3.2, 0.5*0.2=0.1, 25*0.5=12.5
        assert mask.tolist() == [True, False, False, True, False]


class TestBuildConditionMask:
    def test_single_condition(self, sample_df):
        conditions = [{"attribute": "dbh", "operator": "lt", "value": 5.0}]
        mask = build_condition_mask(sample_df, conditions)
        assert mask.tolist() == [True, False, False, True, False]

    def test_multiple_conditions_and(self, sample_df):
        conditions = [
            {"attribute": "fia_species_code", "operator": "eq", "value": 202},
            {"attribute": "dbh", "operator": "lt", "value": 5.0},
        ]
        mask = build_condition_mask(sample_df, conditions)
        # species 202: [True, False, False, False, True]
        # dbh < 5: [True, False, False, True, False]
        # AND: [True, False, False, False, False]
        assert mask.tolist() == [True, False, False, False, False]

    def test_expression_condition(self, sample_df):
        conditions = [{"expression": "dbh < 5 and height < 2"}]
        mask = build_condition_mask(sample_df, conditions)
        assert mask.tolist() == [True, False, False, True, False]

    def test_mixed_conditions(self, sample_df):
        conditions = [
            {"attribute": "fia_species_code", "operator": "ne", "value": 15},
            {"expression": "dbh < 5"},
        ]
        mask = build_condition_mask(sample_df, conditions)
        # ne 15: [True, True, True, False, True]
        # dbh < 5: [True, False, False, True, False]
        # AND: [True, False, False, False, False]
        assert mask.tolist() == [True, False, False, False, False]


class TestApplyAction:
    def test_multiply(self, sample_df):
        action = {"attribute": "height", "modifier": "multiply", "value": 0.5}
        mask = pd.Series([False, True, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[1, "height"] == pytest.approx(7.5)
        # Unchanged rows
        assert result.loc[0, "height"] == pytest.approx(1.5)

    def test_divide(self, sample_df):
        action = {"attribute": "dbh", "modifier": "divide", "value": 2.0}
        mask = pd.Series([True, False, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[0, "dbh"] == pytest.approx(1.0)

    def test_add(self, sample_df):
        action = {"attribute": "height", "modifier": "add", "value": 5.0}
        mask = pd.Series([True, False, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[0, "height"] == pytest.approx(6.5)

    def test_subtract(self, sample_df):
        action = {"attribute": "height", "modifier": "subtract", "value": 1.0}
        mask = pd.Series([True, False, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[0, "height"] == pytest.approx(0.5)

    def test_replace(self, sample_df):
        action = {"attribute": "dbh", "modifier": "replace", "value": 99.0}
        mask = pd.Series([True, False, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[0, "dbh"] == pytest.approx(99.0)

    def test_clamp_crown_ratio(self, sample_df):
        action = {"attribute": "crown_ratio", "modifier": "add", "value": 2.0}
        mask = pd.Series([True, True, True, True, True])
        result = apply_action(sample_df.copy(), action, mask)
        # All values should be clamped to 1.0
        assert (result["crown_ratio"] <= 1.0).all()

    def test_clamp_dbh_non_negative(self, sample_df):
        action = {"attribute": "dbh", "modifier": "subtract", "value": 100.0}
        mask = pd.Series([True, True, True, True, True])
        result = apply_action(sample_df.copy(), action, mask)
        assert (result["dbh"] >= 0).all()

    def test_action_with_unit(self, sample_df):
        # Add 1 inch (2.54 cm) to dbh
        action = {"attribute": "dbh", "modifier": "add", "value": 1.0, "unit": "in"}
        mask = pd.Series([True, False, False, False, False])
        result = apply_action(sample_df.copy(), action, mask)
        assert result.loc[0, "dbh"] == pytest.approx(2.0 + 2.54, rel=1e-3)


class TestApplySingleModification:
    def test_remove(self, sample_df):
        mod = {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            "actions": [{"modifier": "remove"}],
        }
        result = apply_single_modification(sample_df.copy(), mod)
        # Rows with dbh >= 5 remain
        assert len(result) == 3
        assert (result["dbh"] >= 5.0).all()

    def test_modify_attribute(self, sample_df):
        mod = {
            "conditions": [{"attribute": "height", "operator": "gt", "value": 20.0}],
            "actions": [{"attribute": "height", "modifier": "multiply", "value": 0.9}],
        }
        result = apply_single_modification(sample_df.copy(), mod)
        # Only row 4 (height=25) should be modified
        assert result.loc[4, "height"] == pytest.approx(22.5)
        assert result.loc[0, "height"] == pytest.approx(1.5)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["dbh", "height", "crown_ratio", "fia_species_code"])
        mod = {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            "actions": [{"modifier": "remove"}],
        }
        result = apply_single_modification(df, mod)
        assert len(result) == 0


class TestApplyModifications:
    def test_single_modification(self, sample_df):
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        result = apply_modifications(sample_df.copy(), mods)
        assert len(result) == 3

    def test_multiple_modifications(self, sample_df):
        mods = [
            {
                "conditions": [{"attribute": "dbh", "operator": "lt", "value": 2.0}],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [
                    {"attribute": "height", "operator": "gt", "value": 20.0}
                ],
                "actions": [
                    {"attribute": "height", "modifier": "multiply", "value": 0.9}
                ],
            },
        ]
        result = apply_modifications(sample_df.copy(), mods)
        # Row with dbh=1.0 removed (index 3), then height>20 (row 4) multiplied
        assert len(result) == 4
        # Find the row that had height=25 (dbh=30)
        tall_row = result[result["dbh"] == 30.0]
        assert tall_row["height"].values[0] == pytest.approx(22.5)

    def test_empty_modifications(self, sample_df):
        result = apply_modifications(sample_df.copy(), [])
        assert len(result) == len(sample_df)

    def test_expression_remove(self, sample_df):
        mods = [
            {
                "conditions": [{"expression": "height * crown_ratio < 1.0"}],
                "actions": [{"modifier": "remove"}],
            }
        ]
        result = apply_modifications(sample_df.copy(), mods)
        # Removes rows where h*cr < 1: 0.45 and 0.1
        assert len(result) == 3

    def test_unit_conversion_in_condition(self, sample_df):
        mods = [
            {
                "conditions": [
                    {"attribute": "dbh", "operator": "lt", "value": 1.0, "unit": "in"}
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        result = apply_modifications(sample_df.copy(), mods)
        # 1 inch = 2.54 cm, removes dbh < 2.54 → rows with dbh 2.0 and 1.0
        assert len(result) == 3
