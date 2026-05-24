"""
Unit tests for standgen/modifications.py

Tests the modification processing logic: conditions, actions,
unit conversion, clamping, and the full apply_modifications pipeline.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box, mapping
from standgen.modifications import (
    _resolve_feature_geometry,
    _resolve_inline_geometry,
    apply_action,
    apply_modifications,
    apply_single_modification,
    build_condition_mask,
    convert_value,
    evaluate_attribute_condition,
    evaluate_expression,
    evaluate_spatial_condition,
    resolve_spatial_conditions,
)

from lib.errors import ProcessingError


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


# Tree points in sample_df sit on the diagonal: (1,1), (2,2), (3,3), (4,4), (5,5).
UTM_CRS = "EPSG:32611"


class _FakeSnapshot:
    """Stand-in for a Firestore snapshot returned by get_document."""

    def __init__(self, doc: dict):
        self._doc = doc

    def to_dict(self) -> dict:
        return self._doc


class TestEvaluateSpatialCondition:
    """Per-partition point-in-geometry tests against a pre-resolved geometry."""

    def test_within(self, sample_df):
        # Box covers the first three diagonal points.
        cond = {
            "source": "geometry",
            "operator": "within",
            "_resolved_geometry": box(0.5, 0.5, 3.5, 3.5),
        }
        mask = evaluate_spatial_condition(sample_df, cond)
        assert mask.tolist() == [True, True, True, False, False]

    def test_outside_is_complement_of_within(self, sample_df):
        cond = {
            "source": "geometry",
            "operator": "outside",
            "_resolved_geometry": box(0.5, 0.5, 3.5, 3.5),
        }
        mask = evaluate_spatial_condition(sample_df, cond)
        assert mask.tolist() == [False, False, False, True, True]

    def test_intersects_interior(self, sample_df):
        cond = {
            "source": "geometry",
            "operator": "intersects",
            "_resolved_geometry": box(0.5, 0.5, 3.5, 3.5),
        }
        mask = evaluate_spatial_condition(sample_df, cond)
        assert mask.tolist() == [True, True, True, False, False]

    def test_boundary_within_excludes_edge(self, sample_df):
        # box(1,1,3,3): (1,1) and (3,3) sit on the boundary, (2,2) is interior.
        cond = {
            "source": "geometry",
            "operator": "within",
            "_resolved_geometry": box(1.0, 1.0, 3.0, 3.0),
        }
        mask = evaluate_spatial_condition(sample_df, cond)
        # within excludes boundary points → only the interior (2,2).
        assert mask.tolist() == [False, True, False, False, False]

    def test_boundary_intersects_includes_edge(self, sample_df):
        cond = {
            "source": "geometry",
            "operator": "intersects",
            "_resolved_geometry": box(1.0, 1.0, 3.0, 3.0),
        }
        mask = evaluate_spatial_condition(sample_df, cond)
        # intersects includes boundary points → (1,1), (2,2), (3,3).
        assert mask.tolist() == [True, True, True, False, False]

    def test_buffer_captures_more_points(self, sample_df):
        # Unbuffered box contains only (2,2).
        unbuffered = _resolve_inline_geometry(
            {"geometry": mapping(box(1.5, 1.5, 2.5, 2.5))}, 0.0, UTM_CRS
        )
        unbuffered_cond = {
            "source": "geometry",
            "operator": "within",
            "_resolved_geometry": unbuffered,
        }
        assert evaluate_spatial_condition(sample_df, unbuffered_cond).tolist() == [
            False,
            True,
            False,
            False,
            False,
        ]

        # Buffering by 1 m pulls in the neighbouring (1,1) and (3,3) points.
        buffered = _resolve_inline_geometry(
            {"geometry": mapping(box(1.5, 1.5, 2.5, 2.5))}, 1.0, UTM_CRS
        )
        buffered_cond = {
            "source": "geometry",
            "operator": "within",
            "_resolved_geometry": buffered,
        }
        assert evaluate_spatial_condition(sample_df, buffered_cond).tolist() == [
            True,
            True,
            True,
            False,
            False,
        ]

    def test_unknown_operator_raises(self, sample_df):
        cond = {
            "source": "geometry",
            "operator": "nearby",
            "_resolved_geometry": box(0.5, 0.5, 3.5, 3.5),
        }
        with pytest.raises(ProcessingError) as exc:
            evaluate_spatial_condition(sample_df, cond)
        assert exc.value.code == "INVALID_SPATIAL_OPERATOR"

    def test_unresolved_geometry_raises(self, sample_df):
        cond = {"source": "feature", "operator": "within", "feature_id": "f1"}
        with pytest.raises(ProcessingError) as exc:
            evaluate_spatial_condition(sample_df, cond)
        assert exc.value.code == "SPATIAL_CONDITION_UNRESOLVED"


class TestBuildConditionMaskSpatial:
    def test_spatial_and_attribute(self, sample_df):
        conditions = [
            {
                "source": "geometry",
                "operator": "within",
                "_resolved_geometry": box(0.5, 0.5, 3.5, 3.5),
            },
            {"attribute": "dbh", "operator": "gt", "value": 5.0},
        ]
        mask = build_condition_mask(sample_df, conditions)
        # within box: [T, T, T, F, F]; dbh > 5 ([2,10,5,1,30]): [F, T, F, F, T]
        # AND: [F, T, F, F, F]
        assert mask.tolist() == [False, True, False, False, False]


class TestResolveSpatialConditions:
    def test_geometry_variant_injects_and_deepcopies(self):
        mods = [
            {
                "conditions": [
                    {
                        "source": "geometry",
                        "operator": "within",
                        "geometry": mapping(box(0.5, 0.5, 3.5, 3.5)),
                    }
                ],
                "actions": [{"modifier": "remove"}],
            }
        ]
        resolved = resolve_spatial_conditions(mods, "domain-1", UTM_CRS)

        geom = resolved[0]["conditions"][0]["_resolved_geometry"]
        assert geom.equals(box(0.5, 0.5, 3.5, 3.5))
        # Deep copy: the caller's original dict is untouched.
        assert "_resolved_geometry" not in mods[0]["conditions"][0]

    def test_feature_variant_resolves_and_caches(self, monkeypatch):
        doc = {"domain_id": "domain-1", "status": "completed"}
        monkeypatch.setattr(
            "standgen.modifications.get_document",
            lambda collection, fid: (None, _FakeSnapshot(doc)),
        )
        read_calls = {"n": 0}

        def fake_read_parquet(path):
            read_calls["n"] += 1
            return gpd.GeoDataFrame(geometry=[box(0.5, 0.5, 3.5, 3.5)], crs=UTM_CRS)

        monkeypatch.setattr(
            "standgen.modifications.gpd.read_parquet", fake_read_parquet
        )

        # Two conditions referencing the same (feature_id, buffer_m) pair.
        mods = [
            {
                "conditions": [
                    {"source": "feature", "operator": "within", "feature_id": "feat-1"}
                ],
                "actions": [{"modifier": "remove"}],
            },
            {
                "conditions": [
                    {"source": "feature", "operator": "outside", "feature_id": "feat-1"}
                ],
                "actions": [{"attribute": "dbh", "modifier": "multiply", "value": 0.5}],
            },
        ]
        resolved = resolve_spatial_conditions(mods, "domain-1", UTM_CRS)

        assert resolved[0]["conditions"][0]["_resolved_geometry"].equals(
            box(0.5, 0.5, 3.5, 3.5)
        )
        # Cache hit on the second condition → parquet read exactly once.
        assert read_calls["n"] == 1

    def test_feature_not_found(self, monkeypatch):
        from lib.firestore import DocumentNotFoundError

        def raise_not_found(collection, fid):
            raise DocumentNotFoundError(fid)

        monkeypatch.setattr("standgen.modifications.get_document", raise_not_found)
        with pytest.raises(ProcessingError) as exc:
            _resolve_feature_geometry("domain-1", "feat-1", 0.0, {}, UTM_CRS)
        assert exc.value.code == "FEATURE_NOT_FOUND"

    def test_feature_domain_mismatch(self, monkeypatch):
        doc = {"domain_id": "other-domain", "status": "completed"}
        monkeypatch.setattr(
            "standgen.modifications.get_document",
            lambda collection, fid: (None, _FakeSnapshot(doc)),
        )
        with pytest.raises(ProcessingError) as exc:
            _resolve_feature_geometry("domain-1", "feat-1", 0.0, {}, UTM_CRS)
        assert exc.value.code == "FEATURE_DOMAIN_MISMATCH"

    def test_feature_not_ready(self, monkeypatch):
        doc = {"domain_id": "domain-1", "status": "running"}
        monkeypatch.setattr(
            "standgen.modifications.get_document",
            lambda collection, fid: (None, _FakeSnapshot(doc)),
        )
        with pytest.raises(ProcessingError) as exc:
            _resolve_feature_geometry("domain-1", "feat-1", 0.0, {}, UTM_CRS)
        assert exc.value.code == "FEATURE_NOT_READY"

    def test_feature_empty(self, monkeypatch):
        doc = {"domain_id": "domain-1", "status": "completed"}
        monkeypatch.setattr(
            "standgen.modifications.get_document",
            lambda collection, fid: (None, _FakeSnapshot(doc)),
        )
        monkeypatch.setattr(
            "standgen.modifications.gpd.read_parquet",
            lambda path: gpd.GeoDataFrame(geometry=[], crs=UTM_CRS),
        )
        with pytest.raises(ProcessingError) as exc:
            _resolve_feature_geometry("domain-1", "feat-1", 0.0, {}, UTM_CRS)
        assert exc.value.code == "FEATURE_EMPTY"
