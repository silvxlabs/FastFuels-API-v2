"""
Integration tests for the point-cloud CHM handler.

Runs the real handler against a real 3DEP cloud read from GCS, through the
whole griddle job — Firestore document, zarr write, band summaries and all.
The static cloud is `static-test-blackfoot-3dep`: 10,496,309 points over the
`blackfoot.json` domain, carrying ASPRS ground classification.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DOMAIN = "blackfoot.json"
GRID = "chm_point_cloud.json"
POINT_CLOUD = "static-test-blackfoot-3dep"


def _assert_valid_chm(ds):
    """The output contract every CHM consumer relies on."""
    assert "chm" in ds.data_vars
    assert ds["chm"].dims == ("y", "x")
    assert ds["chm"].dtype == np.float32
    assert ds.rio.crs is not None
    # Continuous grids carry NaN nodata, not a sentinel — standgen's ITD
    # neutralizes NaN and would be a no-op against an integer sentinel.
    assert np.isnan(ds["chm"].rio.nodata)
    values = ds["chm"].values
    assert np.isfinite(values).any(), "CHM is entirely nodata"
    assert np.nanmin(values) >= 0.0
    # The exporter contract: a completed grid must be writable as a raster.
    ds["chm"].rio.to_raster("/tmp/test_chm_point_cloud.tif")


class TestPointCloudChm:
    """Building a CHM from a stored point cloud, end to end."""

    def test_produces_a_valid_chm(self, griddle_runner):
        result = griddle_runner(DOMAIN, GRID, point_cloud=POINT_CLOUD)

        _assert_valid_chm(result.ds)

    def test_canopy_heights_are_plausible_for_a_conifer_forest(self, griddle_runner):
        """Guards against a ground-normalization failure.

        If ground were mis-derived the band would track terrain elevation —
        this cloud spans 1026-1274 m, so heights would land in the hundreds
        rather than the tens.
        """
        result = griddle_runner(DOMAIN, GRID, point_cloud=POINT_CLOUD)

        values = result.ds["chm"].values
        assert 20.0 < np.nanmax(values) < 60.0
        assert 1.0 < np.nanmean(values) < 15.0

    def test_records_how_ground_was_obtained(self, griddle_runner):
        """This cloud carries class 2, so ground is read, not inferred."""
        from lib.config import GRIDS_COLLECTION
        from lib.firestore.documents import get_document

        result = griddle_runner(DOMAIN, GRID, point_cloud=POINT_CLOUD)

        _, snapshot = get_document(GRIDS_COLLECTION, result.grid_id)
        ground = snapshot.to_dict()["source"]["ground"]
        assert ground["ground_source"] == "classification"
        assert ground["ground_coverage"] > 0.5
        assert ground["max_ground_distance_m"] >= 0.0

    def test_grid_covers_the_domain_at_the_requested_resolution(self, griddle_runner):
        result = griddle_runner(DOMAIN, GRID, point_cloud=POINT_CLOUD)

        transform = result.ds.rio.transform()
        assert transform.a == pytest.approx(1.0)
        assert abs(transform.e) == pytest.approx(1.0)

    def test_coarser_resolution_produces_a_smaller_grid(self, griddle_runner):
        """`alignment.resolution` drives the lattice, so 5 m is ~1/25 the cells."""
        fine = griddle_runner(DOMAIN, GRID, point_cloud=POINT_CLOUD)
        coarse = griddle_runner(
            DOMAIN,
            GRID,
            point_cloud=POINT_CLOUD,
            source_overrides={
                "alignment": {"target": "domain", "resolution": 5.0, "method": None}
            },
        )

        assert coarse.ds.rio.transform().a == pytest.approx(5.0)
        assert coarse.ds["chm"].size < fine.ds["chm"].size / 20
