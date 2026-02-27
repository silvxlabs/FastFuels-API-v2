"""
Integration tests for 3DEP grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output. Uses the Blue Mountain domain (~1 sq km in Montana),
Bondurant domain (~0.25 sq km in Wyoming), and multi-tile domains that
straddle 1-degree tile boundaries in central Montana.

These tests hit real 3DEP COGs via AWS S3 and write to real GCS/Firestore,
so they require valid credentials and may take a few minutes.
"""

from uuid import uuid4

import numpy as np

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from lib.firestore.documents import delete_document, get_document, set_document
from tests.integration.conftest import (
    DOMAINS_DIR,
    GRIDS_DIR,
    MockRequest,
    _stringify_coordinates,
    load_json,
)


def _assert_valid_data(ds, band, min_valid_frac=0.95):
    """Assert that a band has enough valid (non-nodata, non-NaN) pixels.

    Returns the array of valid values for further assertions.
    """
    values = ds[band].values.ravel()

    nodata = ds[band].rio.nodata
    if nodata is None:
        nodata = ds[band].encoding.get("_FillValue")

    if nodata is not None:
        valid_mask = ~np.isnan(values) & (values != nodata)
    else:
        valid_mask = ~np.isnan(values)

    valid_count = valid_mask.sum()
    total_count = len(values)
    valid_frac = valid_count / total_count

    assert valid_frac >= min_valid_frac, (
        f"{band}: valid fraction {valid_frac:.3f} < {min_valid_frac} "
        f"({valid_count}/{total_count} pixels)"
    )

    return values[valid_mask]


def test_topography_10m(griddle_runner):
    """3DEP 10m topography should produce elevation, slope, aspect with valid data."""
    result = griddle_runner("blue_mtn.json", "threedep_topography_10m.json")
    ds = result.ds

    for var_name in ("elevation", "slope", "aspect"):
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")

    assert "32611" in str(ds.rio.crs)

    # All bands same shape
    shapes = {(ds[v].sizes["y"], ds[v].sizes["x"]) for v in ds.data_vars}
    assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"

    # Data quality: majority of pixels should be valid
    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 800, f"Min elevation {elev_valid.min()} too low"
    assert elev_valid.max() <= 2000, f"Max elevation {elev_valid.max()} too high"

    slope_valid = _assert_valid_data(ds, "slope")
    assert slope_valid.min() >= 0
    assert slope_valid.max() <= 90

    aspect_valid = _assert_valid_data(ds, "aspect")
    assert aspect_valid.min() >= 0
    assert aspect_valid.max() < 360


def test_topography_30m(griddle_runner):
    """3DEP 30m topography should produce elevation with valid data."""
    result = griddle_runner("blue_mtn.json", "threedep_topography_30m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32611" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 800
    assert elev_valid.max() <= 2000


def test_topography_1m_with_coverage(griddle_runner):
    """3DEP 1m topography for a domain with known S1M coverage (Bondurant, WY)."""
    result = griddle_runner("bondurant.json", "threedep_topography_1m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 1800
    assert elev_valid.max() <= 3000


def test_tile_metadata_written_to_firestore(griddle_runner):
    """After processing, the Firestore source field should contain tile metadata.

    This verifies the metadata write-back path: handler returns tile metadata,
    dispatch merges it into the source dict, and main.py writes it to Firestore.
    """
    result = griddle_runner("blue_mtn.json", "threedep_topography_10m.json")

    _, snapshot = get_document(GRIDS_COLLECTION, result.grid_id)
    grid = snapshot.to_dict()
    source = grid["source"]
    tm = source["tile_metadata"]
    assert tm["tile_count"] >= 1
    assert isinstance(tm["tiles"], list) and len(tm["tiles"]) >= 1
    assert tm["native_crs"] is not None
    assert all(url.endswith(".tif") for url in tm["tiles"])


def _assert_tile_metadata(grid_id, expected_tile_count):
    """Read Firestore and assert tile_metadata was written correctly."""
    _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    grid = snapshot.to_dict()
    tm = grid["source"]["tile_metadata"]
    assert tm is not None, "tile_metadata should be populated after processing"
    assert tm["tile_count"] == expected_tile_count
    assert isinstance(tm["tiles"], list)
    assert len(tm["tiles"]) == expected_tile_count
    assert tm["native_crs"] is not None
    assert all(url.endswith(".tif") for url in tm["tiles"])


# Multi-tile tests: arc-second (10m/30m)
# 10m/30m tiles are 1x1 degree in EPSG:4326. These domains straddle
# integer degree boundaries to exercise the merge_arrays mosaic path.


def test_topography_10m_arc_second_2_tiles(griddle_runner):
    """3DEP 10m across 2 arc-second tiles: domain straddling latitude 47.0N."""
    result = griddle_runner("threedep_2_tiles.json", "threedep_topography_10m.json")
    ds = result.ds

    for var_name in ("elevation", "slope", "aspect"):
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")

    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 500
    assert elev_valid.max() <= 3000

    _assert_valid_data(ds, "slope")
    _assert_valid_data(ds, "aspect")

    _assert_tile_metadata(result.grid_id, expected_tile_count=2)


def test_topography_10m_arc_second_4_tiles(griddle_runner):
    """3DEP 10m across 4 arc-second tiles: domain straddling lat 47.0N and lon 112.0W."""
    result = griddle_runner("threedep_4_tiles.json", "threedep_topography_10m.json")
    ds = result.ds

    for var_name in ("elevation", "slope", "aspect"):
        assert var_name in ds.data_vars, f"Missing variable: {var_name}"
        assert ds[var_name].dims == ("y", "x")

    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 500
    assert elev_valid.max() <= 3000

    _assert_valid_data(ds, "slope")
    _assert_valid_data(ds, "aspect")

    _assert_tile_metadata(result.grid_id, expected_tile_count=4)


def test_topography_30m_arc_second_2_tiles(griddle_runner):
    """3DEP 30m across 2 arc-second tiles: domain straddling latitude 47.0N."""
    result = griddle_runner("threedep_2_tiles.json", "threedep_topography_30m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 500
    assert elev_valid.max() <= 3000

    _assert_tile_metadata(result.grid_id, expected_tile_count=2)


def test_topography_30m_arc_second_4_tiles(griddle_runner):
    """3DEP 30m across 4 arc-second tiles: domain straddling lat 47.0N and lon 112.0W."""
    result = griddle_runner("threedep_4_tiles.json", "threedep_topography_30m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 500
    assert elev_valid.max() <= 3000

    _assert_tile_metadata(result.grid_id, expected_tile_count=4)


# Multi-tile tests: S1M (1m)
# S1M tiles are 10km x 10km in EPSG:6350. This domain straddles a 10km
# tile boundary near Bondurant, WY to exercise the S1M mosaic path.


def test_topography_1m_s1m_2_tiles(griddle_runner):
    """3DEP 1m across 2 S1M tiles: domain straddling a 10km tile boundary."""
    result = griddle_runner("threedep_s1m_2_tiles.json", "threedep_topography_1m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 1800
    assert elev_valid.max() <= 3000

    _assert_tile_metadata(result.grid_id, expected_tile_count=2)


def test_topography_1m_s1m_4_tiles(griddle_runner):
    """3DEP 1m across 4 S1M tiles: domain straddling a 10km tile corner."""
    result = griddle_runner("threedep_s1m_4_tiles.json", "threedep_topography_1m.json")
    ds = result.ds

    assert "elevation" in ds.data_vars
    assert ds["elevation"].dims == ("y", "x")
    assert "32612" in str(ds.rio.crs)

    elev_valid = _assert_valid_data(ds, "elevation")
    assert elev_valid.min() >= 1800
    assert elev_valid.max() <= 3500

    _assert_tile_metadata(result.grid_id, expected_tile_count=4)


def test_topography_1m_no_coverage():
    """3DEP 1m topography records COVERAGE_ERROR when S1M tiles don't exist.

    Blue Mountain has no S1M coverage. The full pipeline should mark the
    grid as failed with a COVERAGE_ERROR code in Firestore.
    """
    from griddle.main import process_grid_request

    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    grid_data = load_json(GRIDS_DIR / "threedep_topography_1m.json")
    grid_data["domain_id"] = domain_id
    grid_id = f"test-{uuid4().hex}"
    grid_data["id"] = grid_id
    set_document(GRIDS_COLLECTION, grid_id, grid_data)

    try:
        request = MockRequest(data={"id": grid_id})
        response, status_code = process_grid_request(request)

        assert status_code == 200

        _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
        grid = snapshot.to_dict()
        assert grid["status"] == "failed"
        assert grid["error"]["code"] == "COVERAGE_ERROR"
    finally:
        delete_document(GRIDS_COLLECTION, grid_id)
        delete_document(DOMAINS_COLLECTION, domain_id)
