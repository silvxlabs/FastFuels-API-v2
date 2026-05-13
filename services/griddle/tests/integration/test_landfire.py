"""
Integration tests for LANDFIRE grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output. Uses the Blue Mountain domain (~1 sq km in Montana).

These tests hit real LANDFIRE COGs and write to real GCS/Firestore,
so they require valid credentials and may take a few minutes.
"""


def test_fbfm40(griddle_runner):
    """LANDFIRE FBFM40 grid should produce a zarr with an fbfm variable."""
    result = griddle_runner("blue_mtn.json", "landfire_fbfm40.json")

    assert "fbfm" in result.ds.data_vars
    assert result.ds["fbfm"].dims == ("y", "x")
    assert "32611" in str(result.ds.rio.crs)
    assert result.ds.rio.height > 10
    assert result.ds.rio.width > 10


def test_topography(griddle_runner):
    """LANDFIRE topography grid should produce elevation, slope, aspect variables."""
    result = griddle_runner("blue_mtn.json", "landfire_topography.json")
    ds = result.ds

    for var_name in ["elevation", "slope", "aspect"]:
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")

    assert "32611" in str(ds.rio.crs)

    # All bands should have the same shape
    shapes = {(ds[v].sizes["y"], ds[v].sizes["x"]) for v in ds.data_vars}
    assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"


def test_canopy_landfire_all_bands(griddle_runner):
    """LANDFIRE canopy with all four bands produces chm, cbd, cbh, cc as floats."""
    result = griddle_runner("blue_mtn.json", "canopy_landfire_all.json")
    ds = result.ds

    for var_name in ["chm", "cbd", "cbh", "cc"]:
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")
        assert ds[var_name].dtype.kind == "f"  # post-scaling float

    assert "32611" in str(ds.rio.crs)
    shapes = {(ds[v].sizes["y"], ds[v].sizes["x"]) for v in ds.data_vars}
    assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"


def test_canopy_landfire_crown_inputs(griddle_runner):
    """LANDFIRE canopy crown subset returns just cbd and cbh."""
    result = griddle_runner("blue_mtn.json", "canopy_landfire_crown.json")
    ds = result.ds
    assert set(ds.data_vars) == {"cbd", "cbh"}


def test_canopy_landfire_chm_only(griddle_runner):
    """LANDFIRE canopy single-band path returns only chm."""
    result = griddle_runner("blue_mtn.json", "canopy_landfire_chm_only.json")
    ds = result.ds
    assert set(ds.data_vars) == {"chm"}
