"""
Integration tests for resample grid processing.

Tests the full griddle pipeline with a resample handler that changes
grid resolution using bilinear interpolation.

Requires static test data in GCS (created by services/api/tests/e2e/).
"""

import pytest


@pytest.mark.parametrize(
    "source_grid", ["static-test-blue-mtn-landfire-topography"], indirect=True
)
def test_resample_topography(griddle_runner, source_grid):
    """Resample should produce topographic bands at the target resolution."""
    result = griddle_runner(
        "blue_mtn.json",
        "resample_bilinear.json",
        source_overrides={"source_grid_id": source_grid},
    )
    ds = result.ds

    for var in ["elevation", "slope", "aspect"]:
        assert var in ds.data_vars, f"Missing variable: {var}"
        assert ds[var].dims == ("y", "x")
        assert ds[var].rio.nodata is not None

    assert ds.rio.height > 0
    assert ds.rio.width > 0
