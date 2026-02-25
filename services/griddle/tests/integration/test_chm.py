"""
Integration tests for CHM grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output.

These tests hit real Meta CHM tiles via S3 and write to real GCS/Firestore,
so they require valid credentials and may take a few minutes.
"""

import numpy as np


def _assert_valid_chm(ds, expected_crs_code):
    """Shared assertions for CHM output datasets."""
    assert "chm" in ds.data_vars
    assert ds["chm"].dims == ("y", "x")
    assert expected_crs_code in str(ds.rio.crs)
    assert ds.rio.height > 10
    assert ds.rio.width > 10

    values = ds["chm"].values
    assert values.dtype in (np.float32, np.float64)
    assert np.nanmin(values) >= 0


def test_meta_chm(griddle_runner):
    """Meta CHM single-tile: Blue Mountain domain (~1 sq km in Montana)."""
    ds = griddle_runner("blue_mtn.json", "chm_meta.json")
    _assert_valid_chm(ds, "32611")


def test_meta_chm_2_tiles(griddle_runner):
    """Meta CHM across 2 tiles: domain on an E/W tile boundary."""
    ds = griddle_runner("tile_boundary_2.json", "chm_meta.json")
    _assert_valid_chm(ds, "32612")


def test_meta_chm_4_tiles(griddle_runner):
    """Meta CHM across 4 tiles: domain on a tile corner."""
    ds = griddle_runner("tile_boundary_4.json", "chm_meta.json")
    _assert_valid_chm(ds, "32612")


def test_naip_chm(griddle_runner):
    """NAIP CHM single-tile: Blue Mountain domain (~1 sq km in Montana)."""
    ds = griddle_runner("blue_mtn.json", "chm_naip.json")
    _assert_valid_chm(ds, "32611")


def test_naip_chm_2_tiles(griddle_runner):
    """NAIP CHM across 2 tiles: domain on an E/W tile boundary."""
    ds = griddle_runner("tile_boundary_2.json", "chm_naip.json")
    _assert_valid_chm(ds, "32612")


def test_naip_chm_4_tiles(griddle_runner):
    """NAIP CHM across 4 tiles: domain on a tile corner."""
    ds = griddle_runner("tile_boundary_4.json", "chm_naip.json")
    _assert_valid_chm(ds, "32612")
