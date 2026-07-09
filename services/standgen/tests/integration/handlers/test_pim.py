"""
Integration tests for PIM inventory expansion.

Tests the full standgen pipeline: Firestore setup -> process_inventory_request ->
verify GCS parquet output + Firestore georeference. Uses the Blue Mountain
domain (~1 sq km in Montana) with a static PIM TreeMap grid.

These tests hit real TreeMap tree tables in GCS, read real PIM grids, and write
real parquet to GCS/Firestore, so they require valid credentials.

Note: The static PIM grid is sparse (only a few valid plot pixels) because the
Blue Mountain domain sits on the boundary of UTM zones 11/12, so most TreeMap
pixels are NaN. Tests that need tree data use a minimum count guard. The pipeline
correctness (columns, structure, status transitions) is verified regardless.

Most tests share a single pipeline run via module-scoped fixtures to avoid
redundant processing. Only deterministic/seed tests run additional pipelines.
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

STATIC_PIM_GRID = "static-test-blue-mtn-pim-treemap"

pytestmark = pytest.mark.parametrize(
    "module_pim_grid", [STATIC_PIM_GRID], indirect=True
)


@pytest.fixture(scope="module")
def shared_pim_inventory(module_pim_grid):
    """Run the PIM pipeline once and share the result across all tests in this module.

    Creates a domain + inventory in Firestore, runs standgen, and yields the
    completed inventory dict. Cleans up on teardown.
    """
    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    inventory_data = load_json(INVENTORIES_DIR / "pim_treemap.json")
    inventory_data["domain_id"] = domain_id
    inventory_data["source"]["source_pim_grid_id"] = module_pim_grid
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

    yield inventory

    # Cleanup
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, inventory_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def shared_pim_df(shared_pim_inventory):
    """Read the parquet from the shared PIM inventory once, return as DataFrame."""
    path = f"gs://{INVENTORIES_BUCKET}/{shared_pim_inventory['id']}"
    return dd.read_parquet(path).compute()


# Tests using the shared pipeline run


def test_pipeline_completes(shared_pim_inventory):
    """PIM expansion completes successfully with georeference."""
    assert shared_pim_inventory["georeference"] is not None
    assert "crs" in shared_pim_inventory["georeference"]
    assert "bounds" in shared_pim_inventory["georeference"]


def test_parquet_has_correct_columns(shared_pim_df):
    """Output parquet should have exactly the base columns."""
    assert sorted(shared_pim_df.columns.tolist()) == sorted(BASE_COLUMNS)


def test_parquet_values_are_sensible(shared_pim_df):
    """Tree attribute values should be within physically reasonable ranges."""
    df = shared_pim_df

    if len(df) == 0:
        pytest.skip("No trees generated (sparse grid); skipping value validation")

    assert df["dbh"].min() > 0
    assert df["dbh"].max() < 300
    assert df["height"].min() > 0
    assert df["height"].max() < 100
    assert df["crown_ratio"].min() >= 0
    assert df["crown_ratio"].max() <= 1
    assert (df["fia_species_code"] > 0).all()
    assert df["fia_status_code"].isin([1, 2, 3]).all()
    assert not df.isna().any().any(), f"Found NaN values: {df.isna().sum()}"


def test_tree_coordinates_within_domain(shared_pim_inventory, shared_pim_df):
    """All tree coordinates should be within or near the domain bounds."""
    df = shared_pim_df

    if len(df) == 0:
        pytest.skip("No trees generated (sparse grid); skipping coordinate validation")

    geo = shared_pim_inventory["georeference"]
    bounds = geo["bounds"]  # [minx, miny, maxx, maxy]

    buffer = 30.0
    assert df["x"].min() >= bounds[0] - buffer
    assert df["y"].min() >= bounds[1] - buffer
    assert df["x"].max() <= bounds[2] + buffer
    assert df["y"].max() <= bounds[3] + buffer


def test_georeference_structure(shared_pim_inventory):
    """Georeference should have CRS and bounds."""
    geo = shared_pim_inventory["georeference"]
    assert "crs" in geo
    assert "bounds" in geo
    assert len(geo["bounds"]) == 4


def test_georeference_crs_is_utm(shared_pim_inventory):
    """Blue Mountain domain should produce a UTM CRS."""
    crs = shared_pim_inventory["georeference"]["crs"]
    assert "utm" in crs.lower() or "326" in crs


def test_georeference_bounds_nonzero(shared_pim_inventory):
    """Bounds should have positive extent."""
    bounds = shared_pim_inventory["georeference"]["bounds"]
    x_extent = bounds[2] - bounds[0]
    y_extent = bounds[3] - bounds[1]
    assert x_extent > 100, f"X extent too small: {x_extent}"
    assert y_extent > 100, f"Y extent too small: {y_extent}"


# Tests that need their own pipeline runs


def test_deterministic_tree_count(
    shared_pim_inventory, standgen_runner, module_pim_grid
):
    """Same seed produces the same tree count (deterministic via SeedSequence)."""
    path_1 = f"gs://{INVENTORIES_BUCKET}/{shared_pim_inventory['id']}"
    count_1 = len(dd.read_parquet(path_1))

    inventory_2 = standgen_runner(
        "blue_mtn.json",
        "pim_treemap.json",
        source_pim_grid_id=module_pim_grid,
    )
    path_2 = f"gs://{INVENTORIES_BUCKET}/{inventory_2['id']}"
    count_2 = len(dd.read_parquet(path_2))

    assert count_1 == count_2, (
        f"Expected deterministic tree count but got {count_1} vs {count_2}"
    )


def test_different_seed_different_count(
    shared_pim_inventory, standgen_runner, module_pim_grid
):
    """Different seeds produce different tree arrangements."""
    path_1 = f"gs://{INVENTORIES_BUCKET}/{shared_pim_inventory['id']}"
    count_1 = len(dd.read_parquet(path_1))

    inventory_2 = standgen_runner(
        "blue_mtn.json",
        "pim_treemap.json",
        source_pim_grid_id=module_pim_grid,
        source_overrides={"seed": 99},
    )
    path_2 = f"gs://{INVENTORIES_BUCKET}/{inventory_2['id']}"
    count_2 = len(dd.read_parquet(path_2))

    if count_1 == 0 and count_2 == 0:
        pytest.skip(
            "Grid too sparse to test seed variation (both runs produced 0 trees)"
        )

    assert count_1 != count_2


def test_column_summaries_reflect_data(shared_pim_inventory, shared_pim_df):
    """Column summaries reflect the actual parquet data."""
    if len(shared_pim_df) == 0:
        pytest.skip("No trees generated (sparse grid)")

    cols = {col["key"]: col["summary"] for col in shared_pim_inventory["columns"]}
    assert cols["dbh"]["count"] == len(shared_pim_df)
    assert pytest.approx(cols["dbh"]["min"], rel=1e-4) == shared_pim_df["dbh"].min()
    assert pytest.approx(cols["dbh"]["max"], rel=1e-4) == shared_pim_df["dbh"].max()
