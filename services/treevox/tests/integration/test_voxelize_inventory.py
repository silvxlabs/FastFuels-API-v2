"""Integration tests for the tree-inventory voxelization flow.

Deliberately thin. Most behavior is covered by fast unit tests; these
tests exist to exercise the pieces unit tests can't fake cheaply:

  - The full GCS parquet → multiprocessing pool → GCS zarr round-trip
    against a real domain large enough to span multiple chunks/batches.
  - The GCS 404 path for a missing source inventory, which is hard to
    simulate meaningfully with mocks.

Each test takes ~30s, so we keep the list short.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_voxelize_pim_inventory_all_bands(treevox_runner):
    """Full happy-path voxelization of the Blue Mountain PIM inventory.

    Exercises GCS parquet read, the persistent multiprocessing pool across
    multiple chunks/batches, per-band zarr writes, consolidated metadata,
    and the Firestore georeference/chunk_shape persistence. One
    end-to-end test is sufficient — the output shape/semantics of each
    band are covered by unit tests.
    """
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

    assert ds["volume_fraction"].values.sum() > 0
    assert ds["bulk_density.foliage"].values.sum() > 0

    tree_ids = np.unique(ds["tree_id"].values)
    tree_ids = tree_ids[tree_ids != -1]
    assert len(tree_ids) > 10, f"expected many trees, got {len(tree_ids)}"

    spcd_vals = np.unique(ds["spcd"].values)
    spcd_vals = spcd_vals[spcd_vals > 0]
    assert len(spcd_vals) >= 1

    # Firestore persistence — 3D georeference and 3-tuple chunk_shape.
    geo = result.grid["georeference"]
    assert len(geo["shape"]) == 3
    assert len(geo["transform"]) == 6
    chunk_shape = result.grid["chunk_shape"]
    assert chunk_shape is not None and len(chunk_shape) == 3


def test_inventory_not_found_fails_gracefully(treevox_runner):
    """Bad source_inventory_id → grid status 'failed' with INVENTORY_NOT_FOUND.

    Exercises the GCS 404 code path end-to-end. Hard to fake realistically
    at the unit level since gcsfs + lib.gcs both paper over the underlying
    googleapiclient NotFound; easier to just let the real stack raise.
    """
    result = treevox_runner(
        inventory_id_override="definitely-not-a-real-inventory-id",
        expect_failed=True,
    )
    assert result.grid["status"] == "failed"
    error = result.grid.get("error", {})
    assert error.get("code") == "INVENTORY_NOT_FOUND"
