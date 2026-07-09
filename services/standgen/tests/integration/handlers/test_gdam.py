"""
Integration tests for GDAM allometry imputation.

Tests the full standgen pipeline for the GDAM handler: stage a crafted
position+height source inventory parquet in GCS, create a gdam inventory that
references it, run standgen (which calls the live GDAM API), and verify the
output parquet has the missing morphology filled while existing values and
positions are preserved.

The source is crafted in-test (a few trees in the Blue Mountain domain) rather
than produced by another pipeline — GDAM only needs the source parquet, and a
static fixture can't live under services/lib/tests/shared_data (that path
redeploys the whole pipeline).

These tests hit real GCS + Firestore and the live GDAM service, and require
valid credentials. They run only in the integration step (`pytest
tests/integration/`).
"""

from uuid import uuid4

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from standgen import config as standgen_config

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.domain_utils import parse_domain_gdf
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists

from ..conftest import (
    DOMAINS_DIR,
    _poll_for_completion,
    _run_standgen,
    _stringify_coordinates,
    load_json,
)

# A known-present non-null dbh on the first tree, to prove it is preserved.
_EXISTING_DBH_CM = 25.0


@pytest.fixture(scope="module")
def gdam_source():
    """Stage a Blue Mountain domain and a crafted position+height source parquet.

    Yields (source_inventory_id, source_df, domain_id). The source has six trees
    with x/y in the domain CRS and height; the first tree also has an existing
    dbh (the rest are missing) so tests can prove fill-vs-preserve. Six trees plus
    a small batch size (set in ``completed_gdam_inventory``) splits the run across
    multiple dask partitions.
    """
    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    stored_domain = _stringify_coordinates(domain_data)
    stored_domain["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, stored_domain)

    # Pick interior points as fractions of the domain bounds so they are valid
    # in whatever CRS the domain uses (projected metres or geographic degrees).
    gdf = parse_domain_gdf(stored_domain)
    minx, miny, maxx, maxy = gdf.total_bounds
    fracs = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
    source_df = pd.DataFrame(
        {
            "x": [minx + (maxx - minx) * f for f in fracs],
            "y": [miny + (maxy - miny) * f for f in fracs],
            "height": [12.0, 18.0, 25.0, 14.0, 20.0, 16.0],
            # First tree keeps an existing dbh; the rest are imputed.
            "dbh": [_EXISTING_DBH_CM, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    source_id = f"test-{uuid4().hex}"
    source_path = f"gs://{INVENTORIES_BUCKET}/{source_id}"
    # Write the same way storage.save_parquet does, so load_inventory_parquet
    # reads it identically.
    dd.from_pandas(source_df, npartitions=1).to_parquet(
        source_path, write_metadata_file=True
    )

    yield source_id, source_df, domain_id

    if exists(source_path):
        delete_directory(source_path)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def completed_gdam_inventory(gdam_source):
    """Run the GDAM pipeline once and share the completed result across tests.

    Yields (completed_inventory_dict, source_df, source_id). Cleans up the
    result parquet and Firestore document on teardown.
    """
    source_id, source_df, domain_id = gdam_source

    inventory_id = f"test-{uuid4().hex}"
    inventory_data = {
        "id": inventory_id,
        "domain_id": domain_id,
        "type": "tree",
        "name": "GDAM integration inventory",
        "status": "pending",
        "source": {"name": "gdam", "source_tree_inventory_id": source_id},
        "modifications": [],
        "georeference": None,
        "columns": [
            {"key": "x", "type": "continuous", "unit": "m"},
            {"key": "y", "type": "continuous", "unit": "m"},
            {"key": "height", "type": "continuous", "unit": "m"},
            {"key": "dbh", "type": "continuous", "unit": "cm"},
            {"key": "crown_ratio", "type": "continuous"},
            {"key": "fia_species_code", "type": "categorical"},
        ],
    }
    set_document(INVENTORIES_COLLECTION, inventory_id, inventory_data)

    # Force a small batch so the 6-tree source splits into 3 dask partitions,
    # exercising the map_partitions path end-to-end. This only takes effect in
    # local (in-process) mode; in deployed mode the container uses the default
    # size and runs a single partition, which produces the same output.
    original_batch_size = standgen_config.GDAM_BATCH_SIZE
    standgen_config.GDAM_BATCH_SIZE = 2
    try:
        _run_standgen(inventory_id)
    finally:
        standgen_config.GDAM_BATCH_SIZE = original_batch_size

    if DEPLOYMENT_ENV != "local":
        inventory = _poll_for_completion(inventory_id)
    else:
        _, snapshot = get_document(INVENTORIES_COLLECTION, inventory_id)
        inventory = snapshot.to_dict()

    assert inventory["status"] == "completed", (
        f"GDAM inventory not completed: {inventory.get('error')}"
    )
    assert inventory.get("columns") is not None
    for col in inventory["columns"]:
        assert col["summary"] is not None

    yield inventory, source_df, source_id

    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, inventory_id)


def _result_df(inventory: dict) -> pd.DataFrame:
    """Load the completed inventory's parquet as a pandas DataFrame."""
    path = f"gs://{INVENTORIES_BUCKET}/{inventory['id']}"
    return dd.read_parquet(path).compute().reset_index(drop=True)


def test_column_summaries_populated(completed_gdam_inventory):
    """Column summaries are populated on completion."""
    inventory, _, _ = completed_gdam_inventory
    assert inventory.get("columns") is not None
    for col in inventory["columns"]:
        assert col["summary"] is not None


def test_pipeline_completes_with_georeference(completed_gdam_inventory):
    """The GDAM pipeline completes and populates the georeference."""
    inventory, _, _ = completed_gdam_inventory
    assert inventory["status"] == "completed"
    geo = inventory["georeference"]
    assert geo is not None
    assert "crs" in geo
    assert "bounds" in geo


def test_missing_morphology_is_filled(completed_gdam_inventory):
    """All originally-missing dbh / crown_ratio / species are populated.

    Also proves every partition's trees are reassembled — the result row count
    must equal the source (6) after the multi-partition run.
    """
    inventory, source_df, _ = completed_gdam_inventory
    result = _result_df(inventory)

    assert len(result) == len(source_df)
    for col in ("dbh", "crown_ratio", "fia_species_code"):
        assert col in result.columns
        assert result[col].notna().all(), f"{col} still has missing values"


def test_existing_dbh_is_preserved(completed_gdam_inventory):
    """The tree that already had a dbh keeps it (GDAM only fills the gaps)."""
    inventory, source_df, _ = completed_gdam_inventory
    result = _result_df(inventory)
    # Identify the tree by its (preserved) x coordinate rather than row position,
    # so the assertion is robust to partition ordering.
    first_x = source_df["x"].iloc[0]
    match = result.loc[result["x"] == first_x]
    assert len(match) == 1
    assert match["dbh"].iloc[0] == pytest.approx(_EXISTING_DBH_CM)


def test_position_and_height_unchanged(completed_gdam_inventory):
    """x / y / height are carried through untouched from the source."""
    inventory, source_df, _ = completed_gdam_inventory
    # Sort both by x so the comparison is independent of partition/row ordering.
    result = _result_df(inventory).sort_values("x").reset_index(drop=True)
    expected = source_df.sort_values("x").reset_index(drop=True)
    for col in ("x", "y", "height"):
        assert result[col].to_numpy() == pytest.approx(expected[col].to_numpy())


def test_filled_values_are_sensible(completed_gdam_inventory):
    """GDAM-predicted values land in physically plausible ranges (v2 units)."""
    inventory, _, _ = completed_gdam_inventory
    result = _result_df(inventory)

    assert result["dbh"].min() > 0
    assert result["dbh"].max() < 300  # cm
    assert result["crown_ratio"].min() >= 0
    assert result["crown_ratio"].max() <= 1  # fraction, not percent
    assert (result["fia_species_code"].astype(float) > 0).all()


def test_column_summaries_reflect_data(completed_gdam_inventory):
    """Column summaries reflect the actual parquet data."""
    inventory, _, _ = completed_gdam_inventory
    result = _result_df(inventory)
    cols = {col["key"]: col["summary"] for col in inventory["columns"]}
    assert cols["dbh"]["count"] == len(result)
    assert pytest.approx(cols["dbh"]["min"], rel=1e-4) == result["dbh"].min()
    assert pytest.approx(cols["dbh"]["max"], rel=1e-4) == result["dbh"].max()
