import numpy as np
import pandas as pd
import pytest
from fastfuels_core.allometry.crown_ratio import estimate_crown_base_height
from standgen.crown_base import crown_base_height


def test_matches_core():
    expected = estimate_crown_base_height(
        np.array([202]), np.array([0.4]), np.array([30.0]), np.array([20.0])
    )
    result = crown_base_height([202], [0.4], [30.0], [20.0], False)
    np.testing.assert_allclose(result, expected)
    assert 0 < result[0] < 20.0


def test_zero_crown_ratio_is_height():
    result = crown_base_height([202], [0.0], [30.0], [20.0], False)
    assert result[0] == pytest.approx(20.0)


def test_missing_dbh_is_nan():
    result = crown_base_height([202], [0.4], [np.nan], [20.0], False)
    assert np.isnan(result[0])


def test_missing_and_unknown_species_are_nan():
    result = crown_base_height(
        [np.nan, 12345, 202], [0.4, 0.4, 0.4], [30.0] * 3, [20.0] * 3, False
    )
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isfinite(result[2])


def test_crown_ratio_above_one_is_clipped():
    result = crown_base_height([202, 202], [1.2, 1.0], [30.0] * 2, [20.0] * 2, False)
    assert result[0] == pytest.approx(result[1])


def test_pacific_northwest_coefficients():
    args = ([202], [0.4], [30.0], [20.0])
    assert crown_base_height(*args, True)[0] != pytest.approx(
        crown_base_height(*args, False)[0]
    )


def test_nullable_int_species_series():
    species = pd.Series([202, pd.NA], dtype="Int64", index=[10, 20])
    crown_ratio = pd.Series([0.4, 0.4], index=[10, 20])
    dbh = pd.Series([30.0, 30.0], index=[10, 20])
    height = pd.Series([20.0, 20.0], index=[10, 20])
    result = crown_base_height(species, crown_ratio, dbh, height, False)
    assert isinstance(result, np.ndarray)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_published_model_matches_core():
    expected = estimate_crown_base_height(
        np.array([202]),
        np.array([0.4]),
        np.array([30.0]),
        np.array([20.0]),
        equation="monleon_2004_eq1",
    )
    result = crown_base_height(
        [202], [0.4], [30.0], [20.0], False, model="monleon_2004_eq1"
    )
    np.testing.assert_allclose(result, expected)


def test_pacific_northwest_ignored_for_published_model():
    args = ([202], [0.4], [30.0], [20.0])
    with_flag = crown_base_height(*args, True, model="monleon_2004_eq1")
    without_flag = crown_base_height(*args, False, model="monleon_2004_eq1")
    np.testing.assert_allclose(with_flag, without_flag)


def test_published_model_uncovered_species_is_nan():
    # Loblolly pine is outside the Interior West equation's species list
    args = ([131], [0.4], [30.0], [20.0], False)
    assert np.isnan(crown_base_height(*args, model="toney_reeves_2009")[0])
    assert np.isfinite(crown_base_height(*args)[0])
