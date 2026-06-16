"""
Integration tests for the in-place inventory modifications pipeline.

Tests the full standgen pipeline for in-place modifications: a single PIM
inventory is created once (module-scoped), then each test copies that shared
source to a fresh inventory ID, queues a modification delta in
``pending_modifications``, runs standgen, and verifies the in-place result.
Copying keeps the shared PIM source pristine across tests.

These tests hit real GCS and Firestore and require valid credentials.
"""

import os
import tempfile
from uuid import uuid4

import dask.dataframe as dd
import geopandas as gpd
import pytest
from shapely.geometry import box

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.domain_utils import buffer_gdf
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import (
    delete_directory,
    delete_file,
    exists,
    gcsfs_client,
    upload_file,
)
from lib.testing import SHARED_TEST_INVENTORIES_DIR

from ..conftest import (
    DOMAINS_DIR,
    MockRequest,
    _poll_for_completion,
    _run_standgen,
    _stringify_coordinates,
    load_json,
)

STATIC_PIM_GRID = "static-test-blue-mtn-pim-treemap"
INVENTORIES_DIR = SHARED_TEST_INVENTORIES_DIR

pytestmark = pytest.mark.parametrize(
    "module_pim_grid", [STATIC_PIM_GRID], indirect=True
)


@pytest.fixture(scope="module")
def shared_pim_source(module_pim_grid):
    """Run the PIM pipeline once and share it as the source for all modification tests.

    Yields (pim_inventory_dict, pim_id, domain_id) so the modifications runner
    can reference the PIM output. Cleans up on teardown.
    """
    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    pim_data = load_json(INVENTORIES_DIR / "pim_treemap.json")
    pim_data["domain_id"] = domain_id
    pim_data["source"]["source_pim_grid_id"] = module_pim_grid
    pim_data["source"]["seed"] = 42
    pim_id = f"test-{uuid4().hex}"
    pim_data["id"] = pim_id
    set_document(INVENTORIES_COLLECTION, pim_id, pim_data)

    _run_standgen(pim_id)

    if DEPLOYMENT_ENV != "local":
        pim_inventory = _poll_for_completion(pim_id)
    else:
        _, snapshot = get_document(INVENTORIES_COLLECTION, pim_id)
        pim_inventory = snapshot.to_dict()

    assert pim_inventory["status"] == "completed", (
        f"PIM inventory not completed: {pim_inventory.get('error')}"
    )

    yield pim_inventory, pim_id, domain_id

    # Cleanup
    gcs_path = f"gs://{INVENTORIES_BUCKET}/{pim_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, pim_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def modifications_runner(shared_pim_source):
    """Run modifications against the shared PIM source.

    Each test gets its own modifications run, but they all share the single
    PIM pipeline output. Returns (pim_inventory, mod_inventory) tuple.
    """
    pim_inventory, pim_id, domain_id = shared_pim_source
    mod_ids = []

    def _run(modifications: list[dict]) -> tuple[dict, dict]:
        from standgen.main import process_inventory_request

        mod_id = f"test-{uuid4().hex}"
        # In-place modifications apply to the inventory's own data, so copy the
        # shared PIM parquet to a fresh ID and modify the copy — the shared
        # source stays pristine for other tests.
        gcsfs_client.copy(
            f"{INVENTORIES_BUCKET}/{pim_id}",
            f"{INVENTORIES_BUCKET}/{mod_id}",
            recursive=True,
        )
        mod_data = {
            "id": mod_id,
            "domain_id": domain_id,
            "name": "Modified Inventory",
            "status": "pending",
            "source": pim_inventory["source"],
            "georeference": pim_inventory["georeference"],
            "columns": pim_inventory.get("columns", []),
            "modifications": modifications,
            "pending_modifications": modifications,
        }
        set_document(INVENTORIES_COLLECTION, mod_id, mod_data)
        mod_ids.append(mod_id)

        request = MockRequest(data={"id": mod_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Modifications processing failed: {response}"

        _, mod_snapshot = get_document(INVENTORIES_COLLECTION, mod_id)
        mod_inventory = mod_snapshot.to_dict()
        assert mod_inventory["status"] == "completed", (
            f"Modifications inventory not completed: {mod_inventory.get('error')}"
        )
        assert mod_inventory.get("columns") is not None
        for col in mod_inventory["columns"]:
            assert col["summary"] is not None
        # The delta is applied; the work queue is cleared on completion.
        assert mod_inventory.get("pending_modifications") == []

        return pim_inventory, mod_inventory

    yield _run

    # Cleanup modification inventories only (PIM + domain handled by shared_pim_source)
    for mod_id in mod_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{mod_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, mod_id)


def test_pipeline_completes(modifications_runner):
    """Modifications pipeline completes with georeference."""
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "gt", "value": 0.0}],
            "actions": [{"attribute": "height", "modifier": "multiply", "value": 0.99}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    assert mod_inventory["status"] == "completed"
    assert mod_inventory["georeference"] is not None
    assert "crs" in mod_inventory["georeference"]
    assert "bounds" in mod_inventory["georeference"]


def test_remove_reduces_tree_count(modifications_runner):
    """Removing trees by dbh should produce fewer trees than the source."""
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 30.0}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    pim_inventory, mod_inventory = modifications_runner(modifications)

    pim_path = f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"
    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"

    pim_count = len(dd.read_parquet(pim_path))
    mod_count = len(dd.read_parquet(mod_path))

    if pim_count == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")

    assert mod_count <= pim_count


def test_remove_enforces_condition(modifications_runner):
    """After remove, no trees should violate the condition."""
    threshold = 10.0
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": threshold}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
    df = dd.read_parquet(mod_path).compute()

    if len(df) == 0:
        pytest.skip("No trees after modification (sparse grid)")

    assert (df["dbh"] >= threshold).all(), (
        f"Found trees with dbh < {threshold}: {df['dbh'].min()}"
    )


def test_modify_changes_values(modifications_runner):
    """Multiply modification should change tree attribute values."""
    factor = 0.5
    modifications = [
        {
            "conditions": [{"attribute": "height", "operator": "gt", "value": 0.0}],
            "actions": [
                {"attribute": "height", "modifier": "multiply", "value": factor}
            ],
        }
    ]
    pim_inventory, mod_inventory = modifications_runner(modifications)

    pim_path = f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"
    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"

    pim_df = dd.read_parquet(pim_path).compute()
    mod_df = dd.read_parquet(mod_path).compute()

    if len(pim_df) == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")

    assert len(mod_df) == len(pim_df)
    assert mod_df["height"].sum() == pytest.approx(
        pim_df["height"].sum() * factor, rel=1e-3
    )


def test_unit_conversion_in_condition(modifications_runner):
    """Unit conversion in conditions works end-to-end."""
    modifications = [
        {
            "conditions": [
                {"attribute": "dbh", "operator": "lt", "value": 4.0, "unit": "in"}
            ],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
    df = dd.read_parquet(mod_path).compute()

    if len(df) == 0:
        pytest.skip("No trees after modification (sparse grid)")

    # 4 inches = 10.16 cm
    assert (df["dbh"] >= 10.16 - 0.01).all()


def test_parquet_has_correct_columns(modifications_runner):
    """Modified parquet should have same columns as source."""
    from standgen.columns import BASE_COLUMNS

    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
    ddf = dd.read_parquet(mod_path)

    assert sorted(ddf.columns.tolist()) == sorted(BASE_COLUMNS)


def test_parquet_values_are_sensible(modifications_runner):
    """Modified tree values should be within reasonable ranges."""
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 2.54}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    mod_path = f"gs://{INVENTORIES_BUCKET}/{mod_inventory['id']}"
    df = dd.read_parquet(mod_path).compute()

    if len(df) == 0:
        pytest.skip("No trees after modification (sparse grid)")

    assert df["dbh"].min() >= 2.54
    assert df["dbh"].max() < 300
    assert df["height"].min() > 0
    assert df["height"].max() < 100
    assert df["crown_ratio"].min() >= 0
    assert df["crown_ratio"].max() <= 1
    assert (df["fia_species_code"] > 0).all()
    assert not df.isna().any().any()


def test_georeference_matches_source(modifications_runner):
    """Modified inventory georeference should match domain."""
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    pim_inventory, mod_inventory = modifications_runner(modifications)

    pim_geo = pim_inventory["georeference"]
    mod_geo = mod_inventory["georeference"]

    assert mod_geo["crs"] == pim_geo["crs"]
    assert mod_geo["bounds"] == pim_geo["bounds"]


def test_final_progress_is_100(modifications_runner):
    """After completion, progress should be 100%."""
    modifications = [
        {
            "conditions": [{"attribute": "dbh", "operator": "lt", "value": 5.0}],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = modifications_runner(modifications)

    assert mod_inventory["progress"]["percent"] == 100
    assert mod_inventory["progress"]["message"] == "Complete"


@pytest.fixture
def feature_modifications_runner(shared_pim_source):
    """Run a feature-based spatial modification against the shared PIM source.

    Uploads a Feature GeoParquet to the path the API would write
    (``{domain_id}/{feature_id}.parquet``) plus a completed Feature Firestore
    doc (the worker's domain/status check needs it), then runs a modifications
    inventory whose conditions reference that feature_id. Cleans up the feature
    blob, feature doc, and modifications inventory on teardown.

    The caller passes a geometry GeoDataFrame (in the domain CRS) and the
    modifications list; ``{feature_id}`` is substituted into the conditions.
    """
    pim_inventory, pim_id, domain_id = shared_pim_source
    mod_ids = []
    feature_blobs = []
    feature_doc_ids = []

    def _run(
        feature_gdf: gpd.GeoDataFrame, modifications: list[dict]
    ) -> tuple[dict, dict]:
        from standgen.main import process_inventory_request

        feature_id = f"test-{uuid4().hex}"
        feature_gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            feature_gdf.to_parquet(tmp_path, compression="zstd", row_group_size=1000)
            upload_file(tmp_path, feature_gcs_path)
        finally:
            os.unlink(tmp_path)
        feature_blobs.append(feature_gcs_path)

        set_document(
            FEATURES_COLLECTION,
            feature_id,
            {
                "id": feature_id,
                "domain_id": domain_id,
                "type": "water",
                "status": "completed",
                "source": {"product": "test"},
            },
        )
        feature_doc_ids.append(feature_id)

        # Substitute the generated feature_id into every feature condition.
        resolved_mods = []
        for mod in modifications:
            conditions = [
                {**c, "feature_id": feature_id} if c.get("source") == "feature" else c
                for c in mod["conditions"]
            ]
            resolved_mods.append({**mod, "conditions": conditions})

        mod_id = f"test-{uuid4().hex}"
        gcsfs_client.copy(
            f"{INVENTORIES_BUCKET}/{pim_id}",
            f"{INVENTORIES_BUCKET}/{mod_id}",
            recursive=True,
        )
        mod_data = {
            "id": mod_id,
            "domain_id": domain_id,
            "name": "Spatially Modified Inventory",
            "status": "pending",
            "source": pim_inventory["source"],
            "georeference": pim_inventory["georeference"],
            "columns": pim_inventory.get("columns", []),
            "modifications": resolved_mods,
            "pending_modifications": resolved_mods,
        }
        set_document(INVENTORIES_COLLECTION, mod_id, mod_data)
        mod_ids.append(mod_id)

        request = MockRequest(data={"id": mod_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Modifications processing failed: {response}"

        _, mod_snapshot = get_document(INVENTORIES_COLLECTION, mod_id)
        mod_inventory = mod_snapshot.to_dict()
        assert mod_inventory["status"] == "completed", (
            f"Modifications inventory not completed: {mod_inventory.get('error')}"
        )
        assert mod_inventory.get("columns") is not None
        for col in mod_inventory["columns"]:
            assert col["summary"] is not None
        return pim_inventory, mod_inventory

    yield _run

    for mod_id in mod_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{mod_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, mod_id)
    for feature_blob in feature_blobs:
        if exists(feature_blob):
            delete_file(feature_blob)
    for feature_doc_id in feature_doc_ids:
        delete_document(FEATURES_COLLECTION, feature_doc_id)


def _load_trees(inventory_id: str) -> "gpd.GeoDataFrame":
    """Load an inventory's trees as a GeoDataFrame of (x, y) points."""
    df = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}").compute()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))


def test_feature_remove_drops_trees_inside_buffered_geometry(
    feature_modifications_runner, shared_pim_source
):
    """RemoveAction with an InventoryFeatureSpatialCondition removes exactly the
    trees inside the (5 m buffered) feature, leaving the rest untouched."""
    pim_inventory, _, domain_id = shared_pim_source

    crs = pim_inventory["georeference"]["crs"]
    minx, miny, maxx, maxy = pim_inventory["georeference"]["bounds"]
    midx = (minx + maxx) / 2.0
    # Feature covers the western half of the domain working extent.
    base_geom = box(minx, miny, midx, maxy)
    feature_gdf = gpd.GeoDataFrame(geometry=[base_geom], crs=crs)

    buffer_m = 5.0

    # Reference geometry: exactly what the engine resolves (reproject is an
    # identity here; buffer in the projected domain CRS via the shared helper).
    resolved_geom = buffer_gdf(feature_gdf, buffer_m).geometry.union_all()

    source_trees = _load_trees(pim_inventory["id"])
    if len(source_trees) == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")
    inside = source_trees.within(resolved_geom)
    if inside.sum() == 0 or (~inside).sum() == 0:
        pytest.skip("Feature geometry does not split the tree set")

    modifications = [
        {
            "conditions": [
                {"source": "feature", "operator": "within", "buffer_m": buffer_m}
            ],
            "actions": [{"modifier": "remove"}],
        }
    ]
    _, mod_inventory = feature_modifications_runner(feature_gdf, modifications)

    mod_trees = _load_trees(mod_inventory["id"])

    # Every surviving tree is outside the buffered feature ...
    assert not mod_trees.within(resolved_geom).any(), (
        "found surviving trees inside the buffered feature"
    )
    # ... and the survivor count equals the originally-outside count.
    assert len(mod_trees) == int((~inside).sum())
