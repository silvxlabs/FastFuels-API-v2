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
  variable set (one 2D variable per band key), CRS, and the Grid doc's bands
  field written back by the worker.
- ``alignment.resolution`` is pinned to 10.0 m in the grid template to keep
  test runtime tight while still exercising real rasterization.
"""

import numpy as np

from lib.config import GRIDS_COLLECTION
from lib.firestore.documents import get_document

# fuel_type values present in the Lubrecht layerset fixture.
EXPECTED_FUEL_TYPES = {"shrub", "herb", "litter"}

# One stored variable per band, keyed `<quantity>.<fuel type>`.
EXPECTED_KEYS = {
    f"{prefix}.{fuel_type}"
    for prefix in (
        "fuel_load",
        "fuel_depth",
        "fuel_moisture.live",
        "fuel_moisture.dead",
        "heat_content",
    )
    for fuel_type in EXPECTED_FUEL_TYPES
}


def test_blackfoot_example_layerset_native_rasterization(griddle_runner):
    """Native-alignment rasterize: one 2D var per band (3 fuel types x 5), EPSG:32612.

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

    # One 2D variable per band, named by its key
    assert set(ds.data_vars) == EXPECTED_KEYS

    for key in EXPECTED_KEYS:
        da = ds[key]
        assert da.dims == ("y", "x"), f"{key} dims = {da.dims}, expected (y, x)"
        assert da.dtype == np.float32, f"{key} dtype = {da.dtype}"

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
    assert {b["key"] for b in bands} == EXPECTED_KEYS
    unit_for = {b["key"]: b["unit"] for b in bands}
    assert unit_for["fuel_load.shrub"] == "kg/m**2"
    assert unit_for["fuel_depth.herb"] == "m"
    assert unit_for["fuel_moisture.live.litter"] == "%"
    assert unit_for["heat_content.shrub"] == "kJ/kg"

    # Worker-derived bands carry human-readable name/description for the webapp.
    assert all(b.get("name") for b in bands)
    assert all(b.get("description") for b in bands)
    name_for = {b["key"]: b["name"] for b in bands}
    assert name_for["fuel_load.shrub"] == "Shrub Fuel Load"


def test_blackfoot_example_layerset_domain_alignment(griddle_runner):
    """Domain alignment reprojects the rasterized output to the domain's CRS.

    ``blue_mtn.json`` declares EPSG:32611 while the example layerset is
    EPSG:32612, so a successful run proves the per-variable post-process
    reprojection (mirroring ``resample.py``) wired in Task 8 works
    end-to-end across multi-variable, multi-band Datasets (reprojected
    before flattening). Note: blue_mtn's extent does not spatially overlap
    the layerset polygons, so this test asserts only the reprojection
    plumbing — the data values land outside the destination extent and the
    Zarr is expected to be all-NaN.
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

    # The flat band layout is unchanged by reprojection.
    assert set(ds.data_vars) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert ds[key].dims == ("y", "x")

    # Horizontal extent matches the domain lattice (varies across variables
    # only in residual coord precision; size is identical per the
    # destination_shape pinned by resolve_alignment_destination).
    sizes = {tuple(ds[v].sizes[d] for d in ("y", "x")) for v in ds.data_vars}
    assert len(sizes) == 1, f"variables disagree on (y, x): {sizes}"
    ny, nx = next(iter(sizes))
    assert ny > 0 and nx > 0
