"""
Integration tests for grid modifications.

Exercise the full pipeline: domain + uniform grid + road Feature → griddle
applies the modification → zarr write → re-open and assert that masked cells
are zero and unmasked cells retain their uniform value.

The fixture (``griddle_runner``) handles Firestore + GCS setup and teardown,
including the Feature document required for the worker's domain/status check.
"""

from uuid import uuid4


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
