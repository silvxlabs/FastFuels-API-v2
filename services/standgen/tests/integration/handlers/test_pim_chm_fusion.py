"""
Integration tests for PIM-CHM fusion inventory expansion (reimputation).

Runs the full standgen pipeline: Firestore setup -> process_inventory_request ->
GCS parquet output + Firestore georeference. Uses the Blue Mountain domain
(~1 sq km, Montana) with a static TreeMap PIM grid (30 m) and the static Blue
Mountain NAIP CHM grid (~0.6 m). Both live in EPSG:32611 on the same origin, so
the CHM conditions the PIM cleanly.

These hit real TreeMap tree tables and grids in GCS and write real parquet, so
they require valid credentials.

The Blue Mountain PIM is sparse (the domain straddles the UTM 11/12 boundary, so
most TreeMap pixels are nodata); tests guard on a minimum tree count and verify
pipeline correctness regardless.
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
STATIC_CHM_GRID = "static-test-blue-mtn-naip-chm"

pytestmark = [
    pytest.mark.parametrize("module_pim_grid", [STATIC_PIM_GRID], indirect=True),
    pytest.mark.parametrize("module_chm_grid", [STATIC_CHM_GRID], indirect=True),
]


@pytest.fixture(scope="module")
def shared_fusion_inventory(module_pim_grid, module_chm_grid):
    """Run the fusion pipeline once and share the result across the module.

    Creates a domain + fusion inventory in Firestore, runs standgen, yields the
    completed inventory dict, and cleans up on teardown.
    """
    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    inventory_data = load_json(INVENTORIES_DIR / "pim_chm_fusion.json")
    inventory_data["domain_id"] = domain_id
    inventory_data["source"]["source_pim_grid_id"] = module_pim_grid
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

    yield inventory

    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, inventory_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def shared_fusion_df(shared_fusion_inventory):
    """Read the fusion inventory's parquet once, as a pandas DataFrame."""
    path = f"gs://{INVENTORIES_BUCKET}/{shared_fusion_inventory['id']}"
    return dd.read_parquet(path).compute()


def test_pipeline_completes(shared_fusion_inventory):
    """Fusion expansion completes with a georeference."""
    geo = shared_fusion_inventory["georeference"]
    assert geo is not None
    assert "crs" in geo
    assert "bounds" in geo


def test_parquet_has_correct_columns(shared_fusion_df):
    """Output parquet has exactly the base tree columns."""
    assert sorted(shared_fusion_df.columns.tolist()) == sorted(BASE_COLUMNS)


def test_parquet_values_are_sensible(shared_fusion_df):
    """Tree attributes fall within physically reasonable ranges."""
    df = shared_fusion_df
    if len(df) == 0:
        pytest.skip("No trees after fusion (sparse grid); skipping value validation")

    assert df["dbh"].min() > 0
    assert df["dbh"].max() < 300
    assert df["height"].min() > 0
    assert df["height"].max() < 100
    assert df["crown_ratio"].min() >= 0
    assert df["crown_ratio"].max() <= 1
    assert (df["fia_species_code"] > 0).all()
    assert df["fia_status_code"].isin([1, 2, 3]).all()
    assert not df.isna().any().any(), f"Found NaN values: {df.isna().sum()}"


def test_trees_within_domain(shared_fusion_inventory, shared_fusion_df):
    """All tree coordinates fall within (a small buffer of) the domain bounds."""
    df = shared_fusion_df
    if len(df) == 0:
        pytest.skip("No trees after fusion (sparse grid)")

    bounds = shared_fusion_inventory["georeference"]["bounds"]
    buffer = 30.0
    assert df["x"].min() >= bounds[0] - buffer
    assert df["y"].min() >= bounds[1] - buffer
    assert df["x"].max() <= bounds[2] + buffer
    assert df["y"].max() <= bounds[3] + buffer


def _fusion_method(cover_threshold: float) -> dict:
    return {
        "name": "reimputation",
        "resolution": 7.5,
        "min_height": 2.0,
        "cover_threshold": cover_threshold,
    }


def test_higher_cover_threshold_yields_fewer_trees(
    standgen_runner, module_pim_grid, module_chm_grid
):
    """The CHM cover gate is real: a stricter cover_threshold keeps fewer plots.

    Reimputation is not purely subtractive against the PIM — it re-imputes plot
    IDs into the PIM's nodata gaps and keeps them wherever the CHM shows canopy,
    so it can produce more trees than a straight PIM run. The invariant that does
    hold is monotonic in the gate: raising cover_threshold can only shrink the
    retained (canopy-covered) area, hence the tree count. Two runs on the same
    grids and seed, far apart in threshold, make that unambiguous.
    """
    loose = standgen_runner(
        "blue_mtn.json",
        "pim_chm_fusion.json",
        source_pim_grid_id=module_pim_grid,
        source_chm_grid_id=module_chm_grid,
        source_overrides={"seed": 42, "method": _fusion_method(0.05)},
    )
    strict = standgen_runner(
        "blue_mtn.json",
        "pim_chm_fusion.json",
        source_pim_grid_id=module_pim_grid,
        source_chm_grid_id=module_chm_grid,
        source_overrides={"seed": 42, "method": _fusion_method(0.75)},
    )

    loose_count = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{loose['id']}"))
    strict_count = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{strict['id']}"))

    if loose_count == 0:
        pytest.skip("Loose-threshold fusion produced 0 trees (grid too sparse)")

    assert strict_count < loose_count, (
        f"Stricter cover_threshold kept {strict_count} trees vs the loose run's "
        f"{loose_count}; raising the gate must not increase retained canopy."
    )


def test_column_summaries_reflect_data(shared_fusion_inventory, shared_fusion_df):
    """Column summaries reflect the actual parquet data."""
    if len(shared_fusion_df) == 0:
        pytest.skip("No trees after fusion (sparse grid)")

    cols = {col["key"]: col["summary"] for col in shared_fusion_inventory["columns"]}
    assert cols["dbh"]["count"] == len(shared_fusion_df)
    assert pytest.approx(cols["dbh"]["min"], rel=1e-4) == shared_fusion_df["dbh"].min()
    assert pytest.approx(cols["dbh"]["max"], rel=1e-4) == shared_fusion_df["dbh"].max()
