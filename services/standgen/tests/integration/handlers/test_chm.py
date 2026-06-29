"""
Integration tests for CHM inventory extraction.

Tests the full standgen pipeline: Firestore setup -> process_inventory_request ->
verify GCS parquet output + Firestore georeference. Uses the Blue Mountain
domain (~1 sq km in Montana) with a static CHM grid.

These tests hit real CHM grids in GCS, run the stem isolation algorithms (LMF/VWF),
and write real parquet to GCS/Firestore.

Most tests share a single pipeline run via module-scoped fixtures to avoid
redundant processing. Additional tests verify algorithm parameter variation.
"""

from uuid import uuid4

import dask.dataframe as dd
import geopandas as gpd
import pytest
from shapely.geometry import box
from standgen.storage import load_grid

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists

from ..conftest import (
    DOMAINS_DIR,
    INVENTORIES_DIR,
    MockRequest,
    _poll_for_completion,
    _run_standgen,
    _stringify_coordinates,
    load_json,
)

# You will need to ensure this static grid exists in your test environment!
STATIC_CHM_GRID = "static-test-blackfoot-chm"

pytestmark = pytest.mark.parametrize(
    "module_chm_grid", [STATIC_CHM_GRID], indirect=True
)


@pytest.fixture(scope="module")
def shared_chm_inventory(module_chm_grid):
    """Run the CHM pipeline once and share the result across all tests in this module.

    Creates a domain + inventory in Firestore, runs standgen, and yields the
    completed inventory dict. Cleans up on teardown.
    """
    domain_data = load_json(DOMAINS_DIR / "blackfoot.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    # Assumes you have a chm_lmf.json template in your tests/data/inventories dir
    inventory_data = load_json(INVENTORIES_DIR / "chm_lmf.json")
    inventory_data["domain_id"] = domain_id
    inventory_data["source"]["source_chm_grid_id"] = module_chm_grid
    inventory_id = f"test-{uuid4().hex}"
    inventory_data["id"] = inventory_id
    set_document(INVENTORIES_COLLECTION, inventory_id, inventory_data)

    _run_standgen(inventory_id)

    if DEPLOYMENT_ENV != "local":
        inventory = _poll_for_completion(inventory_id)
    else:
        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()

    assert inventory["status"] == "completed", (
        f"Expected completed, got {inventory['status']}. "
        f"Error: {inventory.get('error')}"
    )
    assert inventory.get("georeference") is not None
    assert inventory.get("columns") is not None
    for col in inventory["columns"]:
        assert col["summary"] is not None
    assert inventory.get("forestry_metrics") is None

    yield inventory

    # Cleanup
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, inventory_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def shared_chm_df(shared_chm_inventory):
    """Read the parquet from the shared CHM inventory once, return as DataFrame."""
    path = f"gs://{INVENTORIES_BUCKET}/{shared_chm_inventory['id']}"
    return dd.read_parquet(path).compute()


# --- Tests using the shared pipeline run (Base Data Contracts) ---


def test_pipeline_completes(shared_chm_inventory):
    """CHM extraction completes successfully with georeference."""
    assert shared_chm_inventory["georeference"] is not None
    assert "crs" in shared_chm_inventory["georeference"]
    assert "bounds" in shared_chm_inventory["georeference"]


def test_parquet_has_correct_columns(shared_chm_df):
    """Output parquet should have exactly the ITD output columns."""
    assert sorted(shared_chm_df.columns.tolist()) == sorted(["x", "y", "height"])


def test_parquet_is_multipartition(shared_chm_inventory):
    """The chunked/parallel ITD path produces a multi-partition parquet end-to-end.

    The static CHM grid is chunked on disk (512x512 over ~1761x1629 px), so the
    chunked fastfuels-core filters yield one partition per CHM chunk. A single
    partition would mean the streaming path collapsed back to a whole-array compute.
    """
    path = f"gs://{INVENTORIES_BUCKET}/{shared_chm_inventory['id']}"
    assert dd.read_parquet(path).npartitions > 1


def test_parquet_values_are_valid(shared_chm_df):
    """CHM logic populates height and coordinates with valid values."""
    df = shared_chm_df

    if len(df) == 0:
        pytest.skip("No trees generated; skipping value validation")

    # Height should be valid and positive
    assert df["height"].min() > 0
    assert df["height"].max() < 150  # reasonable upper bound for trees


def test_tree_coordinates_are_valid_utm(shared_chm_df):
    """
    Trees are kept in their native grid bounds to preserve raster co-registration.
    We verify coordinates are valid UTM numbers, rather than strictly
    clipping them to the domain boundary.
    """
    df = shared_chm_df

    if len(df) == 0:
        pytest.skip("No trees generated; skipping coordinate validation")

    # Verify we aren't at Null Island (0,0) and coordinates are standard UTM scale
    assert df["x"].min() > 100000, "X coordinates do not look like valid UTM Eastings"
    assert df["y"].min() > 1000000, "Y coordinates do not look like valid UTM Northings"

    # Ensure there are no NaN values in our spatial columns
    assert not df["x"].isna().any()
    assert not df["y"].isna().any()


def test_tree_coordinates_within_source_grid_bounds(
    shared_chm_inventory, shared_chm_df, module_chm_grid
):
    """
    While we don't clip to the domain, trees physically cannot exist
    outside the spatial bounds of the source CHM raster itself.
    """
    df = shared_chm_df

    if len(df) == 0:
        pytest.skip("No trees generated; skipping coordinate validation")

    # Don't trust the mocked Firestore document!
    # Load the actual grid from GCS to get its true physical bounds.
    grid_ds = load_grid(module_chm_grid)
    chm_da = grid_ds["chm"]

    # Get bounds in the raster's native CRS
    minx, miny, maxx, maxy = chm_da.rio.bounds()
    source_crs = chm_da.rio.crs

    # The handler outputs trees in the Domain's CRS. We need to match that.
    target_crs = shared_chm_inventory["georeference"]["crs"]

    if str(source_crs) != str(target_crs):
        # Safely reproject the bounding box to match the tree coordinates
        grid_poly = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=source_crs)
        grid_poly = grid_poly.to_crs(target_crs)
        minx, miny, maxx, maxy = grid_poly.total_bounds

    # Add a tiny buffer to account for floating point edge-math
    buffer = 1.0

    assert df["x"].min() >= minx - buffer, "Trees found West of true CHM bounds"
    assert df["y"].min() >= miny - buffer, "Trees found South of true CHM bounds"
    assert df["x"].max() <= maxx + buffer, "Trees found East of true CHM bounds"
    assert df["y"].max() <= maxy + buffer, "Trees found North of true CHM bounds"


def test_georeference_structure(shared_chm_inventory):
    """Georeference should have CRS and bounds."""
    geo = shared_chm_inventory["georeference"]
    assert "crs" in geo
    assert "bounds" in geo
    assert len(geo["bounds"]) == 4


def test_georeference_crs_is_utm(shared_chm_inventory):
    """Blue Mountain domain should produce a UTM CRS."""
    crs = shared_chm_inventory["georeference"]["crs"]
    assert "utm" in crs.lower() or "326" in crs


def test_georeference_bounds_nonzero(shared_chm_inventory):
    """Bounds should have positive extent."""
    bounds = shared_chm_inventory["georeference"]["bounds"]
    x_extent = bounds[2] - bounds[0]
    y_extent = bounds[3] - bounds[1]
    assert x_extent > 100, f"X extent too small: {x_extent}"
    assert y_extent > 100, f"Y extent too small: {y_extent}"


# --- LMF Algorithm Parameter Variation Tests ---


def test_deterministic_lmf_extraction(
    shared_chm_inventory, standgen_runner, module_chm_grid
):
    """Running LMF twice with identical params produces the exact same tree count."""
    path_1 = f"gs://{INVENTORIES_BUCKET}/{shared_chm_inventory['id']}"
    count_1 = len(dd.read_parquet(path_1))

    inventory_2 = standgen_runner(
        "blackfoot.json",
        "chm_lmf.json",
        source_chm_grid_id=module_chm_grid,
    )
    path_2 = f"gs://{INVENTORIES_BUCKET}/{inventory_2['id']}"
    count_2 = len(dd.read_parquet(path_2))

    assert count_1 == count_2, (
        f"Expected deterministic tree count but got {count_1} vs {count_2}"
    )


def test_higher_min_height_reduces_tree_count(
    shared_chm_inventory, standgen_runner, module_chm_grid
):
    """Running LMF with a significantly higher min_height should find fewer trees."""
    path_1 = f"gs://{INVENTORIES_BUCKET}/{shared_chm_inventory['id']}"
    count_baseline = len(dd.read_parquet(path_1))

    if count_baseline == 0:
        pytest.skip("Baseline grid has 0 trees; cannot test reduction")

    # Override the algorithm config to use a very high minimum height (e.g., 20m)
    inventory_stricter = standgen_runner(
        "blackfoot.json",
        "chm_lmf.json",
        source_chm_grid_id=module_chm_grid,
        source_overrides={
            "algorithm": {"name": "lmf", "min_height": 20.0, "footprint_size": 3}
        },
    )
    path_stricter = f"gs://{INVENTORIES_BUCKET}/{inventory_stricter['id']}"
    count_stricter = len(dd.read_parquet(path_stricter))

    assert count_stricter < count_baseline, (
        f"Stricter min_height ({count_stricter} trees) did not reduce count compared "
        f"to baseline ({count_baseline} trees)"
    )


# --- VWF Algorithm Parameter Variation Tests ---


def test_vwf_extraction_completes_and_produces_trees(standgen_runner, module_chm_grid):
    """Running VWF end-to-end dynamically scales windows and produces a valid tree inventory."""
    inventory_vwf = standgen_runner(
        "blackfoot.json",
        "chm_vwf.json",
        source_chm_grid_id=module_chm_grid,
    )

    # Verify standard completion flags
    assert inventory_vwf["status"] == "completed"
    assert inventory_vwf.get("georeference") is not None

    # Verify the dask pipeline actually saved the data to GCS successfully
    path = f"gs://{INVENTORIES_BUCKET}/{inventory_vwf['id']}"
    df = dd.read_parquet(path).compute()

    assert len(df) > 0, "VWF algorithm failed to extract any trees from the grid."
    assert "height" in df.columns
    assert df["height"].min() >= 2.0


def test_higher_crown_ratio_reduces_tree_count(standgen_runner, module_chm_grid):
    """A larger crown_ratio in VWF creates wider search windows, swallowing adjacent peaks and reducing tree count."""

    # 1. Baseline VWF (Standard 10% ratio loaded natively from the new JSON)
    inv_base = standgen_runner(
        "blackfoot.json",
        "chm_vwf.json",
        source_chm_grid_id=module_chm_grid,
    )
    count_base = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inv_base['id']}"))

    if count_base == 0:
        pytest.skip("Baseline grid has 0 trees; cannot test reduction")

    # 2. Stricter VWF (Massive 50% ratio override)
    inv_wide = standgen_runner(
        "blackfoot.json",
        "chm_vwf.json",
        source_chm_grid_id=module_chm_grid,
        source_overrides={
            "algorithm": {
                "name": "vwf",
                "min_height": 2.0,
                "crown_ratio": 0.50,
                "crown_offset": 1.0,
            }
        },
    )
    count_wide = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inv_wide['id']}"))

    assert count_wide < count_base, (
        f"A larger crown ratio ({count_wide} trees) did not reduce the total count compared "
        f"to baseline VWF ({count_base} trees)."
    )


# --- Treatment Rejection ---


def test_treatment_bearing_chm_inventory_fails(module_chm_grid):
    """A CHM inventory carrying treatments fails fast (no diameter to thin against).

    The API rejects this at create time; this verifies standgen's defensive guard
    in case such a document still reaches the worker. The guard fires before the
    grid is loaded, so the grid contents are irrelevant.
    """
    domain_data = load_json(DOMAINS_DIR / "blackfoot.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    inv_id = f"test-{uuid4().hex}"
    inv_data = {
        "id": inv_id,
        "domain_id": domain_id,
        "name": "CHM with treatments",
        "status": "pending",
        "source": {
            "name": "chm",
            "source_chm_grid_id": module_chm_grid,
            "algorithm": {"name": "lmf"},
        },
        "treatments": [{"metric": "diameter", "method": "from_below", "value": 10.0}],
    }
    set_document(INVENTORIES_COLLECTION, inv_id, inv_data)

    try:
        process_inventory_request_local(inv_id)
        _, snapshot = get_document(INVENTORIES_COLLECTION, inv_id)
        inventory = snapshot.to_dict()
        assert inventory["status"] == "failed"
        assert inventory["error"]["code"] == "TREATMENTS_NOT_SUPPORTED_FOR_CHM"
    finally:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inv_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, inv_id)
        delete_document(DOMAINS_COLLECTION, domain_id)


def process_inventory_request_local(inventory_id: str) -> None:
    """Invoke standgen's request handler directly (status surfaced on the doc)."""
    from standgen.main import process_inventory_request

    process_inventory_request(MockRequest(data={"id": inventory_id}))


def test_column_summaries_reflect_data(shared_chm_inventory, shared_chm_df):
    """Column summaries reflect the actual parquet data."""
    if len(shared_chm_df) == 0:
        pytest.skip("No trees generated; skipping value validation")

    cols = {col["key"]: col["summary"] for col in shared_chm_inventory["columns"]}
    assert cols["height"]["count"] == len(shared_chm_df)
    assert (
        pytest.approx(cols["height"]["min"], rel=1e-4) == shared_chm_df["height"].min()
    )
    assert (
        pytest.approx(cols["height"]["max"], rel=1e-4) == shared_chm_df["height"].max()
    )
