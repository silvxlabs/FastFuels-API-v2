"""
Integration tests for uniform grid processing.

Tests the full griddle pipeline with constant-value grids. These are
faster than LANDFIRE tests since no remote data fetching is needed.
"""

import numpy as np


def test_uniform_fuel_moisture(griddle_runner):
    """Uniform grid should produce constant-value bands."""
    result = griddle_runner("blue_mtn.json", "uniform.json", timeout=60)
    ds = result.ds

    expected = {
        "fuel_moisture.1hr": 6.0,
        "fuel_moisture.10hr": 8.0,
    }

    for var_name, expected_value in expected.items():
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")
        assert ds[var_name].dtype == np.float32, (
            f"{var_name} should be float32, got {ds[var_name].dtype}"
        )
        assert ds[var_name].rio.nodata is not None
        np.testing.assert_allclose(
            ds[var_name].values,
            expected_value,
            err_msg=f"{var_name} should be constant {expected_value}",
        )

    assert "32611" in str(ds.rio.crs)

    # ~1300m / 2m = ~650 cells per side
    assert ds.rio.height > 100
    assert ds.rio.width > 100
