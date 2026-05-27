"""
Integration tests for PIM grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output. Uses the Blue Mountain domain (~1 sq km in Montana).

These tests hit real TreeMap COGs and tree table parquet files, and write to
real GCS/Firestore, so they require valid credentials and may take a few minutes.
"""

import numpy as np


def test_treemap_tm_id(griddle_runner):
    """TreeMap grid with tm_id only should produce a zarr with a tm_id variable."""
    result = griddle_runner("blue_mtn.json", "pim_treemap.json")
    ds = result.ds

    assert "tm_id" in ds.data_vars
    assert "plt_cn" not in ds.data_vars
    assert ds["tm_id"].dims == ("y", "x")
    assert "32611" in str(ds.rio.crs)
    assert ds.rio.height > 10
    assert ds.rio.width > 10

    # TM_ID values should be non-negative integers
    values = ds["tm_id"].values
    assert values.dtype in (np.uint16, np.uint32, np.int16, np.int32, np.int64)


def test_treemap_both_bands(griddle_runner):
    """TreeMap grid with both bands should produce tm_id and plt_cn variables."""
    result = griddle_runner("blue_mtn.json", "pim_treemap_both_bands.json")
    ds = result.ds

    for var_name in ["tm_id", "plt_cn"]:
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")
        assert ds[var_name].rio.nodata is not None

    assert "32611" in str(ds.rio.crs)

    # Both bands should have the same shape
    assert ds["tm_id"].shape == ds["plt_cn"].shape

    # PLT_CN should be int64 (FIA condition numbers are large integers)
    assert ds["plt_cn"].values.dtype == np.int64

    # Where tm_id > 0, plt_cn should generally be non-zero (mapped from tree table)
    valid_mask = ds["tm_id"].values > 0
    if valid_mask.any():
        mapped_plt_cn = ds["plt_cn"].values[valid_mask]
        assert (mapped_plt_cn > 0).any(), "Expected some mapped PLT_CN values > 0"
