"""
Integration tests for CHM inventory extraction.

Tests the full standgen pipeline: Firestore setup -> process_inventory_request ->
verify GCS parquet output + Firestore georeference. Uses the Blue Mountain
domain (~1 sq km in Montana) with a static CHM grid.

These tests hit real CHM grids in GCS, run the LMF algorithm, and write
real parquet to GCS/Firestore.

Most tests share a single pipeline run via module-scoped fixtures to avoid
redundant processing. Additional tests verify algorithm parameter variation.
"""

from uuid import uuid4

import dask.dataframe as dd
import pytest
from standgen.columns import BASE_COLUMNS

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


# --- Tests using the shared pipeline run ---


def test_pipeline_completes(shared_chm_inventory):
    """CHM extraction completes successfully with georeference."""
    assert shared_chm_inventory["georeference"] is not None
    assert "crs" in shared_chm_inventory["georeference"]
    assert "bounds" in shared_chm_inventory["georeference"]


def test_parquet_has_correct_columns(shared_chm_df):
    """Output parquet should have exactly the base columns."""
    assert sorted(shared_chm_df.columns.tolist()) == sorted(BASE_COLUMNS)


def test_parquet_values_reflect_chm_defaults(shared_chm_df):
    """CHM logic populates height/coords, but leaves other attributes null."""
    df = shared_chm_df

    if len(df) == 0:
        pytest.skip("No trees generated; skipping value validation")

    # Height should be valid and positive
    assert df["height"].min() > 0
    assert df["height"].max() < 150  # reasonable upper bound for trees

    # The fields we intentionally set to None in the handler should be null/NaN
    assert df["dbh"].isna().all()
    assert df["crown_ratio"].isna().all()
    assert df["fia_species_code"].isna().all()
    assert df["fia_status_code"].isna().all()


def test_tree_coordinates_within_domain(shared_chm_inventory, shared_chm_df):
    """All tree coordinates should be within or near the domain bounds."""
    df = shared_chm_df

    if len(df) == 0:
        pytest.skip("No trees generated; skipping coordinate validation")

    geo = shared_chm_inventory["georeference"]
    bounds = geo["bounds"]  # [minx, miny, maxx, maxy]

    buffer = 10.0  # CHM trees should be strictly within the grid bounds
    assert df["x"].min() >= bounds[0] - buffer
    assert df["y"].min() >= bounds[1] - buffer
    assert df["x"].max() <= bounds[2] + buffer
    assert df["y"].max() <= bounds[3] + buffer


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


# --- Tests that need their own pipeline runs ---


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
