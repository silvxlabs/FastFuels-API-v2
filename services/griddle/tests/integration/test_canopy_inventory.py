"""Integration test for the inventory canopy handler.

Deliberately thin. The translation and per-band semantics are covered by fast
unit tests (``tests/handlers/test_canopy_inventory.py``); this exercises the
piece unit tests fake away: the full GCS parquet read -> fastfuels-core canopy
metrics -> GCS zarr round trip against the real Blue Mountain PIM inventory on
the real domain lattice.

Reads the static ``static-test-blue-mtn-pim-inventory`` fixture directly (the
handler derives the parquet path from the id, and static fixtures are never
deleted by cleanup), so no per-test inventory staging is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_canopy_from_pim_inventory_all_bands(griddle_runner):
    """Full happy-path derivation of the four canopy bands from a real
    inventory: GCS parquet read, the canopy-profile computation, NaN->0
    non-forest fill, per-band zarr writes, and the Firestore
    georeference/chunks persistence."""
    result = griddle_runner("blue_mtn.json", "canopy_inventory.json")
    ds = result.ds

    for key in ("cbd", "cbh", "chm", "cc"):
        assert key in ds.data_vars, f"{key} missing from zarr"
        # Non-forest is filled to 0, so every cell is finite.
        assert np.isfinite(ds[key].values).all(), f"{key} has non-finite cells"

    assert ds["cbd"].dtype == np.float32
    # A forested inventory produces canopy somewhere in the domain.
    assert ds["cbd"].values.max() > 0.0
    assert ds["chm"].values.max() > 0.0
    # Canopy cover is a percentage.
    assert 0.0 <= ds["cc"].values.max() <= 100.0
    # cbh <= chm wherever there is canopy (both zero elsewhere).
    canopy = ds["chm"].values > 0
    assert (ds["cbh"].values[canopy] <= ds["chm"].values[canopy]).all()
