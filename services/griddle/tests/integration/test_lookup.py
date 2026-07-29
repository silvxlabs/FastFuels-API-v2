"""
Integration tests for lookup grid processing.

Tests the full griddle pipeline with a lookup handler that converts
FBFM40 codes to fuel parameters using the SB40 lookup table.

Requires static test data in GCS (created by services/api/tests/e2e/).
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "source_grid", ["static-test-blue-mtn-landfire-fbfm13"], indirect=True
)
def test_fbfm13_lookup(griddle_runner, source_grid):
    """Lookup should produce fuel parameter bands from FBFM13 source grid."""
    result = griddle_runner(
        "blue_mtn.json",
        "lookup_fbfm13.json",
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


def test_fccs_lookup_table_schema(griddle_runner):
    """Sanity check: the real fccs_parameter_lookup.parquet in TABLES_BUCKET
    has every column fccs_lookup expects.

    Pins the contract between the handler and the out-of-band parquet
    artifact — catches a renamed/dropped column or a missing file that
    every mocked unit test in tests/handlers/test_lookup.py would miss.
    """
    from griddle.handlers.lookup import FCCS_QUANTITY_COLUMNS, _load_fccs_table

    table = _load_fccs_table()

    assert len(table["codes"]) > 0
    for col in FCCS_QUANTITY_COLUMNS:
        assert col in table, f"Missing column in real table: {col}"


@pytest.mark.parametrize(
    "source_grid", ["static-test-blue-mtn-landfire-fccs"], indirect=True
)
def test_fccs_lookup(griddle_runner, source_grid):
    """Lookup should produce fuel parameter bands from FCCS source grid."""
    result = griddle_runner(
        "blue_mtn.json",
        "lookup_fccs.json",
        source_overrides={"source_grid_id": source_grid},
    )
    ds = result.ds

    for var in [
        "fuel_load.litter",
        "fuel_load.duff",
        "duff_depth",
    ]:
        assert var in ds.data_vars, f"Missing variable: {var}"
        assert ds[var].dims == ("y", "x")
        assert ds[var].dtype == np.float32, (
            f"{var} should be float32, got {ds[var].dtype}"
        )

    # Real fixture data may include NaN cells (nodata, or a real FCCS code
    # with no matching row in the FOFEM table) — unlike the FBFM tests,
    # don't assume full coverage. Only assert non-negativity where valid.
    depth_vals = ds["duff_depth"].values
    valid = ~np.isnan(depth_vals)
    assert valid.any(), "Expected at least some valid (non-NaN) duff_depth cells"
    assert (depth_vals[valid] >= 0).all()
