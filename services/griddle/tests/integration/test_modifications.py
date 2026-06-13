"""
Integration tests for grid modifications.

Exercise the full pipeline: domain + uniform grid + road Feature → griddle
applies the modification → zarr write → re-open and assert that masked cells
are zero and unmasked cells retain their uniform value.

Also covers the in-place pending-modifications path (#277): a completed grid
re-processed with ``pending_modifications`` loads its own zarr, applies only
the queued delta, merges it into the ``modifications`` ledger, and clears the
queue.

The fixture (``griddle_runner``) handles Firestore + GCS setup and teardown,
including the Feature document required for the worker's domain/status check.
"""

from uuid import uuid4

import numpy as np
from griddle.main import process_grid_request

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION
from lib.firestore.documents import get_document, update_document
from lib.zarr_utils import load_zarr

from .conftest import MockRequest


def test_feature_road_intersect_zeros_fuel_load(griddle_runner):
    """A road Feature should wipe `fuel_moisture.1hr` along the cells it touches.

    The road linestring runs east–west through the middle of the blackfoot
    domain. With ``target=cell, op=intersects, modifier=replace`` we expect
    cells the road crosses to land at 0 while cells well off the line keep
    their uniform value of 6.0.
    """
    feature_id = f"test-{uuid4().hex}"
    result = griddle_runner(
        "blackfoot.json",
        "uniform.json",
        feature_file="blackfoot_example_road.geojson",
        feature_id=feature_id,
        feature_doc=True,
        feature_type="road",
        modifications=[
            {
                "conditions": [
                    {
                        "source": "feature",
                        "operator": "intersects",
                        "target": "cell",
                        "feature_id": feature_id,
                    }
                ],
                "actions": [
                    {
                        "band": "fuel_moisture.1hr",
                        "modifier": "replace",
                        "value": 0.0,
                    }
                ],
            }
        ],
        timeout=180,
    )
    ds = result.ds
    arr = ds["fuel_moisture.1hr"].values

    # Some cells are masked (the row(s) the road crosses).
    masked = (arr == 0.0).sum()
    assert masked > 0, "no cells were masked by the road feature"

    # The grid is much larger than the road footprint — most cells retain
    # the uniform value.
    untouched = (arr == 6.0).sum()
    assert untouched > masked, (
        f"expected untouched cells ({untouched}) to outnumber masked cells ({masked})"
    )

    # Band 2 (fuel_moisture.10hr) was not targeted by the action.
    assert (ds["fuel_moisture.10hr"].values == 8.0).all()


def test_feature_road_buffer_widens_mask(griddle_runner):
    """buffer_m > 0 selects more cells than the unbuffered run."""
    # Unbuffered: zero out cells the road touches.
    feature_id_a = f"test-{uuid4().hex}"
    result_no_buffer = griddle_runner(
        "blackfoot.json",
        "uniform.json",
        feature_file="blackfoot_example_road.geojson",
        feature_id=feature_id_a,
        feature_doc=True,
        feature_type="road",
        modifications=[
            {
                "conditions": [
                    {
                        "source": "feature",
                        "operator": "intersects",
                        "target": "cell",
                        "feature_id": feature_id_a,
                    }
                ],
                "actions": [
                    {
                        "band": "fuel_moisture.1hr",
                        "modifier": "replace",
                        "value": 0.0,
                    }
                ],
            }
        ],
        timeout=180,
    )
    arr_no_buffer = result_no_buffer.ds["fuel_moisture.1hr"].values
    masked_no_buffer = (arr_no_buffer == 0.0).sum()

    # Buffered by 10 m — strictly more cells masked.
    feature_id_b = f"test-{uuid4().hex}"
    result_buffered = griddle_runner(
        "blackfoot.json",
        "uniform.json",
        feature_file="blackfoot_example_road.geojson",
        feature_id=feature_id_b,
        feature_doc=True,
        feature_type="road",
        modifications=[
            {
                "conditions": [
                    {
                        "source": "feature",
                        "operator": "intersects",
                        "target": "cell",
                        "feature_id": feature_id_b,
                        "buffer_m": 10.0,
                    }
                ],
                "actions": [
                    {
                        "band": "fuel_moisture.1hr",
                        "modifier": "replace",
                        "value": 0.0,
                    }
                ],
            }
        ],
        timeout=180,
    )
    arr_buffered = result_buffered.ds["fuel_moisture.1hr"].values
    masked_buffered = (arr_buffered == 0.0).sum()

    assert masked_buffered > masked_no_buffer, (
        f"buffered mask ({masked_buffered}) should be strictly larger than "
        f"unbuffered mask ({masked_no_buffer})"
    )


def test_pending_modifications_replace_gr1_with_gr2_in_place(griddle_runner):
    """The pending-modifications path (#277) loads the grid's own zarr,
    applies only the queued delta, merges it into the ledger, and clears the
    queue — demonstrated with the documented GR1→GR2 reclassification.

    The blue-mountain FBFM40 grid may not naturally contain GR1 (code 101),
    so the delta first reclassifies the domain's most common fuel model to
    GR1, then GR1 to GR2 (code 102) — rules apply in order within one run.
    The assertions are therefore deterministic regardless of the source data.
    """
    result = griddle_runner("blue_mtn.json", "landfire_fbfm40.json")
    grid_id = result.grid_id
    base = result.ds["fbfm"].values.copy()

    # Pick the most common valid fuel model code, excluding GR1/GR2 so the
    # expected final counts are simple sums.
    codes, counts = np.unique(base, return_counts=True)
    candidates = [
        (int(c), int(n))
        for c, n in zip(codes, counts)
        if 91 <= c <= 204 and c not in (101, 102)
    ]
    assert candidates, "fixture grid has no valid fuel model codes"
    target_code, target_count = max(candidates, key=lambda t: t[1])
    gr1_count = int((base == 101).sum())
    gr2_count = int((base == 102).sum())

    pending = [
        {
            "conditions": [{"band": "fbfm", "operator": "eq", "value": target_code}],
            "actions": [{"band": "fbfm", "modifier": "replace", "value": 101}],
        },
        {
            "conditions": [{"band": "fbfm", "operator": "eq", "value": 101}],
            "actions": [{"band": "fbfm", "modifier": "replace", "value": 102}],
        },
    ]
    update_document(
        GRIDS_COLLECTION,
        grid_id,
        {"pending_modifications": pending, "status": "pending"},
    )

    # Re-run griddle directly (a Cloud Tasks enqueue would reuse the create
    # task's tombstoned name in deployed mode, so call the local entry point).
    response, status_code = process_grid_request(MockRequest(data={"id": grid_id}))
    assert status_code == 200, response

    _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    grid = snapshot.to_dict()
    assert grid["status"] == "completed", grid.get("error")
    # The applied delta merged into the ledger atomically with completion.
    assert grid["pending_modifications"] == []
    assert grid["modifications"] == pending

    ds = load_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}")
    try:
        final = ds["fbfm"].values
        # No GR1 remains; GR2 absorbed the target code and any original GR1.
        assert int((final == 101).sum()) == 0
        assert int((final == 102).sum()) == target_count + gr1_count + gr2_count
        assert int((final == target_code).sum()) == 0
        # Every other code is untouched.
        for code, count in zip(codes, counts):
            if int(code) in (target_code, 101, 102):
                continue
            assert int((final == code).sum()) == int(count), f"code {code} changed"
        # The fbfm band summary was recomputed from the modified data: its
        # unique count matches the final array exactly (excluding nodata).
        fbfm_band = next(b for b in grid["bands"] if b["key"] == "fbfm")
        unique_vals = {int(c) for c in np.unique(final)}
        if fbfm_band.get("nodata") is not None:
            unique_vals.discard(int(fbfm_band["nodata"]))
        assert fbfm_band["summary"]["unique_count"] == len(unique_vals)
    finally:
        ds.close()
