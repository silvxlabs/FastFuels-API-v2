"""
Integration tests for lookup grid processing.

Tests the full griddle pipeline with a lookup handler that converts
FBFM40 codes to fuel parameters using the SB40 lookup table.

Requires static test data in GCS (created by services/api/tests/e2e/).
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
)
def test_fbfm40_lookup(griddle_runner, source_grid):
    """Lookup should produce fuel parameter bands from FBFM40 source grid."""
    result = griddle_runner(
        "blue_mtn.json",
        "lookup_fbfm40.json",
        source_overrides={"source_grid_id": source_grid},
    )
    ds = result.ds

    for var in [
        "fuel_load.1hr",
        "fuel_load.10hr",
        "fuel_depth",
    ]:
        assert var in ds.data_vars, f"Missing variable: {var}"
        assert ds[var].dims == ("y", "x")
        assert ds[var].dtype == np.float32, (
            f"{var} should be float32, got {ds[var].dtype}"
        )

    assert (ds["fuel_depth"].values >= 0).all()
