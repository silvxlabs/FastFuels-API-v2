"""
Integration tests for LANDFIRE grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output. Uses the Blue Mountain domain (~1 sq km in Montana).

These tests hit real LANDFIRE COGs and write to real GCS/Firestore,
so they require valid credentials and may take a few minutes.
"""


def test_fbfm40(griddle_runner):
    """LANDFIRE FBFM40 grid should produce a zarr with an fbfm variable."""
    ds = griddle_runner("blue_mtn.json", "landfire_fbfm40.json")

    assert "fbfm" in ds.data_vars
    assert ds["fbfm"].dims == ("y", "x")
    assert "32611" in str(ds.rio.crs)
    assert ds.rio.height > 10
    assert ds.rio.width > 10


def test_topography(griddle_runner):
    """LANDFIRE topography grid should produce elevation, slope, aspect variables."""
    ds = griddle_runner("blue_mtn.json", "landfire_topography.json")

    for var_name in ["elevation", "slope", "aspect"]:
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")

    assert "32611" in str(ds.rio.crs)

    # All bands should have the same shape
    shapes = {(ds[v].sizes["y"], ds[v].sizes["x"]) for v in ds.data_vars}
    assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"
