"""
Unit tests for standgen/treatments.py

Tests the treatment processing logic: unit conversion, the fastfuels-core
thinner factory, region-area computation, spatial scoping/recombination, and the
full apply_treatments pipeline (including reproducible proportional thinning).
"""

import math

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastfuels_core.treatments import (
    DirectionalThinToDiameterLimit,
    DirectionalThinToStandBasalArea,
    ProportionalThinToBasalArea,
    ThinningDirection,
)
from shapely.geometry import box
from standgen.treatments import (
    apply_single_treatment,
    apply_treatments,
    build_thinner,
    compute_region_area_ha,
    convert_treatment_value,
)

from lib.errors import ProcessingError

UTM_CRS = "EPSG:32611"


def _ba(dbh_cm: float) -> float:
    """Basal area (m²) for a tree of the given diameter (cm)."""
    return dbh_cm**2 * (math.pi / 40_000)


@pytest.fixture
def sample_df():
    """Five trees on the (1,1)..(5,5) diagonal, dbh in cm."""
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


@pytest.fixture
def domain_gdf():
    """A 10 m × 10 m domain (100 m² = 0.01 ha) containing the sample points."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=UTM_CRS)


@pytest.fixture
def big_domain_gdf():
    """A 5 km × 5 km domain = 25 km², above the 16 km² basal-area area limit."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 5000, 5000)], crs=UTM_CRS)


def _within(geom):
    """A resolved 'within' spatial condition wrapping a geometry."""
    return {"source": "geometry", "operator": "within", "_resolved_geometry": geom}


def _ddf(df, npartitions=2):
    """Wrap a pandas frame as a dask DataFrame for the orchestrator path."""
    return dd.from_pandas(df, npartitions=npartitions)


class TestConvertTreatmentValue:
    def test_no_unit_passthrough_diameter(self):
        assert convert_treatment_value("diameter", 5.0, None) == 5.0

    def test_no_unit_passthrough_basal_area(self):
        assert convert_treatment_value("basal_area", 20.0, None) == 20.0

    def test_diameter_inches_to_cm(self):
        assert (
            pytest.approx(convert_treatment_value("diameter", 1.0, "in"), rel=1e-3)
            == 2.54
        )

    def test_diameter_mm_to_cm(self):
        assert (
            pytest.approx(convert_treatment_value("diameter", 100.0, "mm"), rel=1e-3)
            == 10.0
        )

    def test_basal_area_ft2_per_acre_to_m2_per_ha(self):
        # 1 ft²/acre ≈ 0.2296 m²/ha
        result = convert_treatment_value("basal_area", 1.0, "ft**2/acre")
        assert pytest.approx(result, rel=1e-3) == 0.22957


class TestBuildThinner:
    def test_diameter_from_below_maps_below(self):
        t = build_thinner(
            {"metric": "diameter", "method": "from_below", "value": 5.0}, None
        )
        assert isinstance(t, DirectionalThinToDiameterLimit)
        assert t.direction == ThinningDirection.BELOW
        assert t.limit == 5.0

    def test_diameter_from_above_maps_above(self):
        t = build_thinner(
            {"metric": "diameter", "method": "from_above", "value": 5.0}, None
        )
        assert isinstance(t, DirectionalThinToDiameterLimit)
        assert t.direction == ThinningDirection.ABOVE

    def test_diameter_unit_conversion_applied(self):
        t = build_thinner(
            {"metric": "diameter", "method": "from_below", "value": 2.0, "unit": "in"},
            None,
        )
        assert pytest.approx(t.limit, rel=1e-3) == 5.08

    def test_basal_area_directional_below(self):
        t = build_thinner(
            {"metric": "basal_area", "method": "from_below", "value": 20.0}, 0.5
        )
        assert isinstance(t, DirectionalThinToStandBasalArea)
        assert t.direction == ThinningDirection.BELOW
        assert t.target == 0.5

    def test_basal_area_directional_above(self):
        t = build_thinner(
            {"metric": "basal_area", "method": "from_above", "value": 20.0}, 0.5
        )
        assert isinstance(t, DirectionalThinToStandBasalArea)
        assert t.direction == ThinningDirection.ABOVE

    def test_basal_area_proportional(self):
        t = build_thinner(
            {"metric": "basal_area", "method": "proportional", "value": 20.0}, 0.5
        )
        assert isinstance(t, ProportionalThinToBasalArea)
        assert t.target == 0.5

    def test_unsupported_metric_raises(self):
        with pytest.raises(ProcessingError) as exc:
            build_thinner(
                {"metric": "spacing", "method": "from_below", "value": 1.0}, None
            )
        assert exc.value.code == "UNSUPPORTED_TREATMENT_METRIC"

    def test_invalid_method_raises(self):
        with pytest.raises(ProcessingError) as exc:
            build_thinner(
                {"metric": "diameter", "method": "proportional", "value": 1.0}, None
            )
        assert exc.value.code == "INVALID_TREATMENT_METHOD"


class TestComputeRegionAreaHa:
    def test_inventory_wide_is_full_domain(self, domain_gdf):
        assert compute_region_area_ha(domain_gdf, []) == pytest.approx(0.01)

    def test_within_clips_to_geometry(self, domain_gdf):
        conditions = [_within(box(0, 0, 5, 10))]  # 50 m²
        assert compute_region_area_ha(domain_gdf, conditions) == pytest.approx(0.005)

    def test_outside_subtracts_geometry(self, domain_gdf):
        cond = {
            "source": "geometry",
            "operator": "outside",
            "_resolved_geometry": box(0, 0, 5, 10),
        }
        assert compute_region_area_ha(domain_gdf, [cond]) == pytest.approx(0.005)

    def test_intersects_same_as_within_for_area(self, domain_gdf):
        cond = {
            "source": "geometry",
            "operator": "intersects",
            "_resolved_geometry": box(0, 0, 5, 10),
        }
        assert compute_region_area_ha(domain_gdf, [cond]) == pytest.approx(0.005)

    def test_multiple_conditions_anded(self, domain_gdf):
        conditions = [
            _within(box(0, 0, 5, 10)),
            _within(box(0, 0, 10, 5)),
        ]  # ∩ = box(0,0,5,5)
        assert compute_region_area_ha(domain_gdf, conditions) == pytest.approx(0.0025)

    def test_unresolved_geometry_raises(self, domain_gdf):
        cond = {"source": "feature", "operator": "within", "feature_id": "f1"}
        with pytest.raises(ProcessingError) as exc:
            compute_region_area_ha(domain_gdf, [cond])
        assert exc.value.code == "SPATIAL_CONDITION_UNRESOLVED"

    def test_unknown_operator_raises(self, domain_gdf):
        cond = {
            "source": "geometry",
            "operator": "nearby",
            "_resolved_geometry": box(0, 0, 5, 5),
        }
        with pytest.raises(ProcessingError) as exc:
            compute_region_area_ha(domain_gdf, [cond])
        assert exc.value.code == "INVALID_SPATIAL_OPERATOR"


class TestApplySingleTreatmentDiameter:
    def test_from_below_keeps_ge_limit(self, sample_df, domain_gdf):
        treatment = {"metric": "diameter", "method": "from_below", "value": 5.0}
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(result["dbh"].tolist()) == [5.0, 10.0, 30.0]

    def test_from_above_keeps_lt_limit(self, sample_df, domain_gdf):
        treatment = {"metric": "diameter", "method": "from_above", "value": 5.0}
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(result["dbh"].tolist()) == [1.0, 2.0]

    def test_unit_limit(self, sample_df, domain_gdf):
        # 1 in = 2.54 cm → from_below keeps dbh >= 2.54 (drops 2.0 and 1.0).
        treatment = {
            "metric": "diameter",
            "method": "from_below",
            "value": 1.0,
            "unit": "in",
        }
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(result["dbh"].tolist()) == [5.0, 10.0, 30.0]

    def test_spatial_within_only_thins_subset(self, sample_df, domain_gdf):
        # Box covers the first three points (dbh 2, 10, 5). from_below limit 5
        # drops dbh 2 in-region; the out-of-region dbh 1 (point 4,4) stays.
        treatment = {
            "metric": "diameter",
            "method": "from_below",
            "value": 5.0,
            "conditions": [_within(box(0.5, 0.5, 3.5, 3.5))],
        }
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(result["dbh"].tolist()) == [1.0, 5.0, 10.0, 30.0]

    def test_empty_subset_is_noop(self, sample_df, domain_gdf):
        treatment = {
            "metric": "diameter",
            "method": "from_below",
            "value": 5.0,
            "conditions": [_within(box(100, 100, 200, 200))],  # matches nothing
        }
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert len(result) == len(sample_df)

    def test_result_index_is_rangeindex(self, sample_df, domain_gdf):
        treatment = {"metric": "diameter", "method": "from_below", "value": 5.0}
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert result.index.equals(pd.RangeIndex(len(result)))

    def test_empty_df(self, domain_gdf):
        empty = pd.DataFrame(columns=["x", "y", "dbh"])
        treatment = {"metric": "diameter", "method": "from_below", "value": 5.0}
        result = apply_single_treatment(empty, treatment, domain_gdf)
        assert len(result) == 0


class TestApplySingleTreatmentBasalArea:
    def test_directional_below_reaches_target(self, sample_df, domain_gdf):
        # Domain 0.01 ha. value 7.5 m²/ha → target 0.075 m². from_below keeps the
        # largest trees by cumulative BA; only dbh 30 (BA≈0.0707) fits under 0.075.
        treatment = {"metric": "basal_area", "method": "from_below", "value": 7.5}
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert result["dbh"].tolist() == [30.0]
        assert result["dbh"].apply(_ba).sum() <= 0.075

    def test_target_larger_than_stand_is_noop(self, sample_df, domain_gdf):
        treatment = {"metric": "basal_area", "method": "from_below", "value": 1000.0}
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert len(result) == len(sample_df)

    def test_spatial_scopes_target_and_subset(self, sample_df, domain_gdf):
        # Box covers points 0,1,2 (dbh 2,10,5). Clipped region area = 0.003 ha
        # (30 m²). value 250 m²/ha → target 0.75 m² >> in-region BA, so the subset
        # is retained whole and out-of-region trees are untouched → no removals.
        treatment = {
            "metric": "basal_area",
            "method": "from_below",
            "value": 250.0,
            "conditions": [_within(box(0.5, 0.5, 3.5, 3.5))],
        }
        result = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert len(result) == len(sample_df)


class TestApplyTreatments:
    def test_empty_treatments_noop(self, sample_df, domain_gdf):
        result = apply_treatments(_ddf(sample_df), [], domain_gdf, seed=0)
        assert len(result.compute()) == len(sample_df)

    def test_returns_lazy_dask_dataframe(self, sample_df, domain_gdf):
        treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
        result = apply_treatments(_ddf(sample_df), treatments, domain_gdf, seed=0)
        assert isinstance(result, dd.DataFrame)

    def test_multiple_sequential_treatments(self, sample_df, domain_gdf):
        # from_below 3 cm drops dbh 2 and 1 → [10,5,30]; then from_above 6 cm
        # keeps dbh < 6 → [5].
        treatments = [
            {"metric": "diameter", "method": "from_below", "value": 3.0},
            {"metric": "diameter", "method": "from_above", "value": 6.0},
        ]
        result = apply_treatments(_ddf(sample_df), treatments, domain_gdf, seed=0)
        assert result.compute()["dbh"].tolist() == [5.0]

    def test_empty_dataframe(self, domain_gdf):
        empty = pd.DataFrame(
            {
                "x": pd.Series(dtype=float),
                "y": pd.Series(dtype=float),
                "dbh": pd.Series(dtype=float),
            }
        )
        treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
        result = apply_treatments(
            _ddf(empty, npartitions=1), treatments, domain_gdf, seed=0
        )
        assert len(result.compute()) == 0


class TestApplyTreatmentsDaskPaths:
    """The lazy/materialized paths must produce the same result as a single
    pandas application of the same treatment."""

    def test_diameter_lazy_matches_pandas(self, sample_df, domain_gdf):
        treatment = {"metric": "diameter", "method": "from_below", "value": 5.0}
        lazy = apply_treatments(
            _ddf(sample_df), [treatment], domain_gdf, seed=0
        ).compute()
        eager = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(lazy["dbh"].tolist()) == sorted(eager["dbh"].tolist())

    def test_basal_area_inventory_wide_matches_pandas(self, sample_df, domain_gdf):
        treatment = {"metric": "basal_area", "method": "from_below", "value": 7.5}
        lazy = apply_treatments(
            _ddf(sample_df), [treatment], domain_gdf, seed=0
        ).compute()
        eager = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(lazy["dbh"].tolist()) == sorted(eager["dbh"].tolist())

    def test_spatial_basal_area_partial_materialize_matches_pandas(
        self, sample_df, domain_gdf
    ):
        # Box covers points 0,1,2 (dbh 2,10,5); region 9 m² = 0.0009 ha.
        # value 10 m²/ha → target 0.009 m² keeps only dbh 10 in-region; the
        # out-of-region dbh 1 and 30 are untouched.
        treatment = {
            "metric": "basal_area",
            "method": "from_below",
            "value": 10.0,
            "conditions": [_within(box(0.5, 0.5, 3.5, 3.5))],
        }
        lazy = apply_treatments(
            _ddf(sample_df), [treatment], domain_gdf, seed=0
        ).compute()
        eager = apply_single_treatment(sample_df, treatment, domain_gdf)
        assert sorted(lazy["dbh"].tolist()) == sorted(eager["dbh"].tolist())
        assert sorted(lazy["dbh"].tolist()) == [1.0, 10.0, 30.0]


class TestAreaLimit:
    def test_inventory_wide_basal_area_over_limit_raises(
        self, sample_df, big_domain_gdf
    ):
        treatment = {"metric": "basal_area", "method": "from_below", "value": 20.0}
        with pytest.raises(ProcessingError) as exc:
            apply_treatments(_ddf(sample_df), [treatment], big_domain_gdf, seed=0)
        assert exc.value.code == "TREATMENT_AREA_TOO_LARGE"

    def test_scoped_basal_area_over_limit_raises(self, sample_df, big_domain_gdf):
        # The condition spans the whole 25 km² domain → region exceeds the limit.
        treatment = {
            "metric": "basal_area",
            "method": "proportional",
            "value": 20.0,
            "conditions": [_within(box(0, 0, 5000, 5000))],
        }
        with pytest.raises(ProcessingError) as exc:
            apply_treatments(_ddf(sample_df), [treatment], big_domain_gdf, seed=0)
        assert exc.value.code == "TREATMENT_AREA_TOO_LARGE"

    def test_diameter_over_limit_is_allowed(self, sample_df, big_domain_gdf):
        # Diameter never materializes the stand, so the area limit does not apply.
        treatment = {"metric": "diameter", "method": "from_below", "value": 5.0}
        result = apply_treatments(_ddf(sample_df), [treatment], big_domain_gdf, seed=0)
        assert sorted(result.compute()["dbh"].tolist()) == [5.0, 10.0, 30.0]


def _uniform_stand(n=1000):
    """A stand of identical-diameter trees with distinguishable ids."""
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "x": np.linspace(0.1, 9.9, n),
            "y": np.linspace(0.1, 9.9, n),
            "dbh": [10.0] * n,
        }
    )


class TestProportionalDeterminism:
    def test_same_seed_identical_result(self, domain_gdf):
        stand = _uniform_stand()
        # value chosen so target ≈ half the total BA → a real removal.
        treatment = {"metric": "basal_area", "method": "proportional", "value": 390.0}
        r1 = apply_treatments(
            _ddf(stand, 4), [treatment], domain_gdf, seed=42
        ).compute()
        r2 = apply_treatments(
            _ddf(stand, 4), [treatment], domain_gdf, seed=42
        ).compute()
        assert sorted(r1["id"].tolist()) == sorted(r2["id"].tolist())
        # It actually removed a meaningful fraction.
        assert 0 < len(r1) < len(stand)

    def test_different_seed_differs(self, domain_gdf):
        stand = _uniform_stand()
        treatment = {"metric": "basal_area", "method": "proportional", "value": 390.0}
        r1 = apply_treatments(_ddf(stand, 4), [treatment], domain_gdf, seed=1).compute()
        r2 = apply_treatments(_ddf(stand, 4), [treatment], domain_gdf, seed=2).compute()
        assert sorted(r1["id"].tolist()) != sorted(r2["id"].tolist())
