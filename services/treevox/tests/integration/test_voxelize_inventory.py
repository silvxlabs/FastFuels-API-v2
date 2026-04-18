"""Integration tests for the tree-inventory voxelization flow.

Happy-path tests use real static inventory fixtures produced by
``services/api/tests/e2e``. Semantics-specific tests (overlap resolution,
inventory-column biomass) upload crafted inventories via the ``trees=``
path because they need precise control over tree placement.

All tests are marked ``integration`` and require GCP auth + infrastructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# Real-fixture happy-path tests (use the static PIM/CHM inventories)


def test_voxelize_pim_inventory_all_bands(treevox_runner):
    """Voxelize the Blue Mountain PIM inventory into all 6 bands."""
    result = treevox_runner(
        static_inventory="static-test-blue-mtn-pim-inventory",
        bands=[
            "volume_fraction",
            "bulk_density.foliage",
            "savr.foliage",
            "fuel_moisture.live",
            "spcd",
            "tree_id",
        ],
        moisture_model={"method": "uniform", "live": 100.0},
    )

    ds = result.ds
    for key in (
        "volume_fraction",
        "bulk_density.foliage",
        "savr.foliage",
        "fuel_moisture.live",
        "spcd",
        "tree_id",
    ):
        assert key in ds.data_vars, f"{key} missing from zarr"

    assert ds["volume_fraction"].dtype == np.float32
    assert ds["bulk_density.foliage"].dtype == np.float32
    assert ds["spcd"].dtype == np.uint16
    assert ds["tree_id"].dtype == np.int32

    # Blue Mountain PIM has real trees → populated canopy.
    assert ds["volume_fraction"].values.sum() > 0
    assert ds["bulk_density.foliage"].values.sum() > 0

    # Real inventory populates many unique tree IDs and multiple species.
    tree_ids = np.unique(ds["tree_id"].values)
    tree_ids = tree_ids[tree_ids != -1]
    assert len(tree_ids) > 10, f"expected many trees, got {len(tree_ids)}"

    spcd_vals = np.unique(ds["spcd"].values)
    spcd_vals = spcd_vals[spcd_vals > 0]
    assert len(spcd_vals) >= 1


def test_voxelize_chm_inventory(treevox_runner):
    """Voxelize the Blue Mountain CHM inventory."""
    result = treevox_runner(
        static_inventory="static-test-blue-mtn-chm-inventory",
        bands=["volume_fraction", "bulk_density.foliage", "tree_id"],
    )

    ds = result.ds
    assert ds["volume_fraction"].values.sum() > 0
    tree_ids = np.unique(ds["tree_id"].values)
    tree_ids = tree_ids[tree_ids != -1]
    assert len(tree_ids) > 0


def test_inventory_subset_bands(treevox_runner):
    """Requesting only two bands produces exactly those zarr datasets."""
    result = treevox_runner(
        static_inventory="static-test-blue-mtn-pim-inventory",
        bands=["volume_fraction", "bulk_density.foliage"],
    )
    ds = result.ds
    all_tree_bands = {
        "volume_fraction",
        "bulk_density.foliage",
        "savr.foliage",
        "fuel_moisture.live",
        "spcd",
        "tree_id",
    }
    present = set(ds.data_vars) & all_tree_bands
    assert present == {"volume_fraction", "bulk_density.foliage"}


def test_inventory_with_uniform_moisture(treevox_runner):
    """fuel_moisture.live carries the moisture_model value on canopy cells."""
    result = treevox_runner(
        static_inventory="static-test-blue-mtn-pim-inventory",
        bands=["volume_fraction", "fuel_moisture.live"],
        moisture_model={"method": "uniform", "live": 75.0},
    )
    ds = result.ds
    fm = ds["fuel_moisture.live"].values
    vf = ds["volume_fraction"].values
    canopy = vf > 0
    assert canopy.any()
    np.testing.assert_allclose(fm[canopy], 75.0, atol=1e-5)
    assert (fm[~canopy] == 0.0).all()


def test_georeference_is_3d_and_chunk_shape_persisted(treevox_runner):
    """Grid doc carries 3D georeference and chunk_shape after completion."""
    result = treevox_runner(
        static_inventory="static-test-blue-mtn-pim-inventory",
        bands=["volume_fraction"],
    )
    geo = result.grid["georeference"]
    assert len(geo["shape"]) == 3
    assert geo["z_origin"] == 0.0
    assert geo["z_resolution"] > 0
    assert len(geo["transform"]) == 6

    chunk_shape = result.grid["chunk_shape"]
    assert chunk_shape is not None
    assert len(chunk_shape) == 3


# Semantics tests — need crafted inventories for precise control


def test_tallest_tree_wins_at_overlap(treevox_runner):
    """Two trees at identical (x,y) with different heights: the taller wins
    for overwrite-style bands (spcd, tree_id).
    """
    mid_x, mid_y = (720880.0, 5190204.0)  # rough center of blue_mtn
    trees = pd.DataFrame(
        {
            "x": [mid_x, mid_x],
            "y": [mid_y, mid_y],
            "fia_species_code": [122, 202],  # ponderosa, Douglas-fir
            "fia_status_code": [1, 1],
            "dbh": [20.0, 35.0],
            "height": [10.0, 20.0],
            "crown_ratio": [0.4, 0.5],
        }
    )

    result = treevox_runner(
        trees=trees,
        bands=["volume_fraction", "spcd", "tree_id"],
    )
    ds = result.ds
    vf = ds["volume_fraction"].values
    spcd = ds["spcd"].values
    tree_id = ds["tree_id"].values

    canopy = vf > 0
    assert canopy.any()

    # Taller tree is sorted last → its 202 / tree_id=1 wins at overlapping cells.
    populated_spcd = spcd[canopy]
    assert (populated_spcd == 202).any()
    populated_tree_id = tree_id[canopy]
    assert (populated_tree_id == 1).any()


def test_biomass_model_inventory_reads_column(treevox_runner):
    """biomass_model='inventory' reads crown_fuel_load from the given column."""
    mid_x, mid_y = (720880.0, 5190204.0)
    trees = pd.DataFrame(
        {
            "x": [mid_x, mid_x + 20.0],
            "y": [mid_y, mid_y + 20.0],
            "fia_species_code": [122, 122],
            "fia_status_code": [1, 1],
            "dbh": [20.0, 20.0],
            "height": [15.0, 15.0],
            "crown_ratio": [0.4, 0.4],
            "my_load": [5.0, 50.0],
        }
    )
    result = treevox_runner(
        trees=trees,
        bands=["volume_fraction", "bulk_density.foliage"],
        biomass_model="inventory",
        biomass_column="my_load",
    )
    bd = result.ds["bulk_density.foliage"].values
    # Sum ≈ 5 + 50 = 55 kg distributed across crown voxels.
    total = bd.sum()
    assert 50.0 < total < 60.0, f"expected total biomass ~55, got {total}"


def test_inventory_not_found_fails_gracefully(treevox_runner):
    """Bad source_inventory_id → grid status 'failed' with INVENTORY_NOT_FOUND."""
    result = treevox_runner(
        inventory_id_override="definitely-not-a-real-inventory-id",
        expect_failed=True,
    )
    assert result.grid["status"] == "failed"
    error = result.grid.get("error", {})
    assert error.get("code") == "INVENTORY_NOT_FOUND"
