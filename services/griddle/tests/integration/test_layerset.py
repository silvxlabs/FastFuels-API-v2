"""
Integration tests for layerset rasterization.

Exercises the full pipeline: domain + layerset GeoParquet in Firestore/GCS,
through ``handle_layerset`` → ``fastfuels_core.layersets.rasterize_layerset``
→ Zarr writeback → Firestore ``bands`` writeback. The shared example
fixture (``services/lib/tests/shared_data/features/blackfoot_example_layerset.geojson``)
declares 7 features spanning 3 fuel_types (``shrub``, ``herb``, ``litter``)
in EPSG:32612. Polygon shapes are derived from a Lubrecht site layerset,
translated into the Blackfoot example domain's bounds so the integration
tests pair cleanly with that domain.

Notes:
- ``rasterize_layerset`` uses random placement for the ``random_clusters``
  distribution and `seed` is not threaded through the API, so pixel-value
  assertions would be flaky. The tests assert on layout invariants only:
  variable set, dim ordering, band coord, CRS, and the Grid doc's bands
  field written back by the worker.
- ``alignment.resolution`` is pinned to 10.0 m in the grid template to keep
  test runtime tight while still exercising real rasterization.
"""

import numpy as np

from lib.config import GRIDS_COLLECTION
from lib.firestore.documents import get_document

# fastfuels_core.rasterize_layerset's 5-band per-variable output coord.
EXPECTED_BAND_COORD = [
    "loading",
    "height",
    "live_fuel_moisture",
    "dead_fuel_moisture",
    "heat_of_combustion",
]

# fuel_type values present in the Lubrecht layerset fixture.
EXPECTED_FUEL_TYPES = {"shrub", "herb", "litter"}


def test_blackfoot_example_layerset_native_rasterization(griddle_runner):
    """Native-alignment rasterize: one var per fuel_type, 5 bands each, EPSG:32612.

    Also verifies that ``handle_layerset`` writes the Grid doc's ``bands``
    field back to Firestore after rasterization (3 vars × 5 bands = 15 entries).

    Domain pick: ``blackfoot.json`` is in EPSG:32612 (UTM 12N, same as the
    layerset). With ``target="native"`` + a custom resolution,
    ``resolve_alignment_destination`` sets ``destination_crs`` to the
    domain's CRS — picking a matching-CRS domain keeps the output in
    EPSG:32612 (no CRS hop), which is the "native" semantics we want.
    """
    result = griddle_runner(
        "blackfoot.json",
        "rasterize_layerset.json",
        feature_file="blackfoot_example_layerset.geojson",
        timeout=180,
    )
    ds = result.ds

    # One xr.DataArray per unique fuel_type
    assert set(ds.data_vars) == EXPECTED_FUEL_TYPES

    for var_name in EXPECTED_FUEL_TYPES:
        da = ds[var_name]
        assert da.dims == ("band", "y", "x"), (
            f"{var_name} dims = {da.dims}, expected (band, y, x)"
        )
        assert da.dtype == np.float32, f"{var_name} dtype = {da.dtype}"
        assert list(da.coords["band"].values) == EXPECTED_BAND_COORD

    # Native alignment preserves the layerset's declared CRS.
    assert "32612" in str(ds.rio.crs)

    # Grid doc carries the worker-derived bands list (15 entries).
    _, snapshot = get_document(GRIDS_COLLECTION, result.grid_id)
    grid = snapshot.to_dict()
    bands = grid.get("bands") or []
    assert len(bands) == 15
    assert [b["index"] for b in bands] == list(range(15))

    # Every band entry is a continuous physical quantity; spot-check units.
    assert {b["type"] for b in bands} == {"continuous"}
    unit_for = {b["key"]: b["unit"] for b in bands}
    assert unit_for["shrub.loading"] == "kg/m**2"
    assert unit_for["herb.height"] == "m"
    assert unit_for["litter.live_fuel_moisture"] == "%"
    assert unit_for["shrub.heat_of_combustion"] == "kJ/kg"


def test_blackfoot_example_layerset_domain_alignment(griddle_runner):
    """Domain alignment reprojects the rasterized output to the domain's CRS.

    ``blue_mtn.json`` declares EPSG:32611 while the example layerset is
    EPSG:32612, so a successful run proves the per-variable post-process
    reprojection (mirroring ``resample.py``) wired in Task 8 works
    end-to-end across multi-variable, multi-band Datasets. Note: blue_mtn's
    extent does not spatially overlap the layerset polygons, so this test
    asserts only the reprojection plumbing — the data values land outside
    the destination extent and the Zarr is expected to be all-NaN.
    """
    result = griddle_runner(
        "blue_mtn.json",
        "rasterize_layerset.json",
        feature_file="blackfoot_example_layerset.geojson",
        source_overrides={
            "alignment": {
                "target": "domain",
                "resolution": 10.0,
                "method": None,
            }
        },
        timeout=180,
    )
    ds = result.ds

    # Output is reprojected to the domain's CRS (blue_mtn → EPSG:32611).
    assert "32611" in str(ds.rio.crs)

    # Variable set and band coord are unchanged by reprojection.
    assert set(ds.data_vars) == EXPECTED_FUEL_TYPES
    for var_name in EXPECTED_FUEL_TYPES:
        da = ds[var_name]
        assert da.dims == ("band", "y", "x")
        assert list(da.coords["band"].values) == EXPECTED_BAND_COORD

    # Horizontal extent matches the domain lattice (varies across variables
    # only in residual coord precision; size is identical per the
    # destination_shape pinned by resolve_alignment_destination).
    sizes = {tuple(ds[v].sizes[d] for d in ("y", "x")) for v in ds.data_vars}
    assert len(sizes) == 1, f"variables disagree on (y, x): {sizes}"
    ny, nx = next(iter(sizes))
    assert ny > 0 and nx > 0
