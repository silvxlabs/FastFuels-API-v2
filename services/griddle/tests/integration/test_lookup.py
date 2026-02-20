"""
Integration tests for lookup grid processing.

Tests the full griddle pipeline with a lookup handler that converts
FBFM40 codes to fuel parameters using the SB40 lookup table.

Requires static test data in GCS (created by services/api/tests/e2e/).
"""

import pytest


@pytest.mark.parametrize(
    "source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True
)
def test_fbfm40_lookup(griddle_runner, source_grid):
    """Lookup should produce fuel parameter bands from FBFM40 source grid."""
    ds = griddle_runner(
        "blue_mtn.json",
        "lookup_fbfm40.json",
        source_overrides={"source_grid_id": source_grid},
    )

    for var in [
        "fuel_load.1hr",
        "fuel_load.10hr",
        "fuel_depth",
        "moisture_of_extinction",
    ]:
        assert var in ds.data_vars, f"Missing variable: {var}"
        assert ds[var].dims == ("y", "x")

    assert (ds["fuel_depth"].values >= 0).all()
