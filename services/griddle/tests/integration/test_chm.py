"""
Integration tests for CHM grid processing.

Tests the full griddle pipeline: Firestore setup -> process_grid_request ->
verify GCS output.

These tests hit real Meta CHM tiles via S3 and write to real GCS/Firestore,
so they require valid credentials and may take a few minutes.
"""

import os
import tempfile

import numpy as np
import pytest

from lib.config import GRIDS_COLLECTION
from lib.firestore.documents import get_document


def _assert_valid_chm(ds, expected_crs_code):
    """Shared assertions for CHM output datasets."""
    assert "chm" in ds.data_vars
    assert ds["chm"].dims == ("y", "x")
    assert expected_crs_code in str(ds.rio.crs)
    assert ds.rio.height > 10
    assert ds.rio.width > 10

    values = ds["chm"].values
    assert values.dtype == np.float32
    assert np.nanmin(values) >= 0

    # The Exporter Contract: Verify the Dataset can be successfully written to a GeoTIFF
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.tif")
        ds.rio.to_raster(path)

        # Verify the TIFF was actually created and has bytes
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


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


def test_meta_chm(griddle_runner):
    """Meta CHM single-tile: Blue Mountain domain (~1 sq km in Montana)."""
    result = griddle_runner("blue_mtn.json", "chm_meta.json")
    _assert_valid_chm(result.ds, "32611")
    _assert_tile_metadata(result.grid_id, expected_tile_count=1)


@pytest.mark.parametrize("version", ["1", "2"])
def test_meta_chm_versions(griddle_runner, version):
    """Meta CHM single-tile for each version: Blue Mountain domain."""
    result = griddle_runner(
        "blue_mtn.json", "chm_meta.json", source_overrides={"version": version}
    )
    _assert_valid_chm(result.ds, "32611")
    _assert_tile_metadata(result.grid_id, expected_tile_count=1)


@pytest.mark.parametrize("version", ["1", "2"])
def test_meta_chm_2_tiles(griddle_runner, version):
    """Meta CHM across 2 tiles: domain on an E/W tile boundary."""
    result = griddle_runner(
        "meta_chm_2_tiles.json", "chm_meta.json", source_overrides={"version": version}
    )
    _assert_valid_chm(result.ds, "32612")
    _assert_tile_metadata(result.grid_id, expected_tile_count=2)


@pytest.mark.parametrize("version", ["1", "2"])
def test_meta_chm_4_tiles(griddle_runner, version):
    """Meta CHM across 4 tiles: domain on a tile corner."""
    result = griddle_runner(
        "meta_chm_4_tiles.json", "chm_meta.json", source_overrides={"version": version}
    )
    _assert_valid_chm(result.ds, "32612")
    _assert_tile_metadata(result.grid_id, expected_tile_count=4)


def test_naip_chm(griddle_runner):
    """NAIP CHM single-tile: Blue Mountain domain (~1 sq km in Montana)."""
    result = griddle_runner("blue_mtn.json", "chm_naip.json")
    _assert_valid_chm(result.ds, "32611")
    _assert_tile_metadata(result.grid_id, expected_tile_count=1)


def test_naip_chm_2_tiles(griddle_runner):
    """NAIP CHM across 2 tiles: domain on an E/W tile boundary."""
    result = griddle_runner("naip_chm_2_tiles.json", "chm_naip.json")
    _assert_valid_chm(result.ds, "32612")
    _assert_tile_metadata(result.grid_id, expected_tile_count=2)


def test_naip_chm_4_tiles(griddle_runner):
    """NAIP CHM across 4 tiles: domain on a tile corner."""
    result = griddle_runner("naip_chm_4_tiles.json", "chm_naip.json")
    _assert_valid_chm(result.ds, "32612")
    _assert_tile_metadata(result.grid_id, expected_tile_count=4)
