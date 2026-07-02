"""
Integration tests for the IN-PLACE inventory treatments pipeline.

Distinct from ``test_treatments.py``, which covers the create-time path
(treatments applied during PIM expansion). This file exercises the in-place
endpoint added in #302: a completed inventory's own Parquet is copied to a fresh
ID, a treatment delta is queued in ``pending_treatments``, standgen runs, and the
data is loaded → treated → written back under the same ID via the staging swap.

A single PIM inventory is built once (module-scoped) and shared as the source;
each test copies it so the shared source stays pristine.

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
    get_gcsfs_client,
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
    """Build the PIM pipeline once and share it as the source for all tests.

    Yields ``(pim_inventory, pim_id, domain_id)``. Cleans up on teardown.
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

    gcs_path = f"gs://{INVENTORIES_BUCKET}/{pim_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, pim_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def treatments_runner(shared_pim_source):
    """Apply an in-place treatment delta to a copy of the shared PIM source.

    Copies the shared PIM Parquet to a fresh ID, queues the treatments in
    ``pending_treatments`` while leaving the ``treatments`` ledger holding only
    ``prior_treatments`` (as the API does — the delta is merged into the ledger
    only on completion, not at queue time), runs standgen, and returns
    ``(pim_inventory, treated_inventory)``. Cleans up each copy on teardown.
    """
    pim_inventory, pim_id, domain_id = shared_pim_source
    treated_ids = []

    def _run(
        treatments: list[dict], prior_treatments: list[dict] | None = None
    ) -> tuple[dict, dict]:
        from standgen.main import process_inventory_request

        prior_treatments = prior_treatments or []
        treated_id = f"test-{uuid4().hex}"
        get_gcsfs_client().copy(
            f"{INVENTORIES_BUCKET}/{pim_id}",
            f"{INVENTORIES_BUCKET}/{treated_id}",
            recursive=True,
        )
        treated_data = {
            "id": treated_id,
            "domain_id": domain_id,
            "name": "Treated Inventory (in place)",
            "status": "pending",
            "source": pim_inventory["source"],
            "georeference": pim_inventory["georeference"],
            # The ledger holds only previously-applied treatments; the new delta
            # lives in pending_treatments until completion merges it in.
            "treatments": prior_treatments,
            "pending_treatments": treatments,
        }
        set_document(INVENTORIES_COLLECTION, treated_id, treated_data)
        treated_ids.append(treated_id)

        request = MockRequest(data={"id": treated_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Treatment processing failed: {response}"

        _, treated_snapshot = get_document(INVENTORIES_COLLECTION, treated_id)
        treated_inventory = treated_snapshot.to_dict()
        assert treated_inventory["status"] == "completed", (
            f"Treated inventory not completed: {treated_inventory.get('error')}"
        )
        # The delta is applied in place; on completion the work queue is cleared
        # and the delta is merged onto the end of the ledger (ledger == applied
        # data), never duplicated or dropped (#319).
        assert treated_inventory.get("pending_treatments") == []
        assert treated_inventory.get("treatments") == prior_treatments + treatments
        # The Parquet footprint is recorded on the in-place replace path too and
        # reflects the current dataset — a thinning treatment rewrites the whole
        # store, so it never accumulates onto the source footprint (#342).
        assert treated_inventory["size_bytes"] > 0
        assert treated_inventory["size_bytes"] < pim_inventory["size_bytes"] * 2

        return pim_inventory, treated_inventory

    yield _run

    for treated_id in treated_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{treated_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, treated_id)


def _load_df(inventory_id: str):
    return dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}").compute()


def test_pipeline_completes(treatments_runner):
    """In-place treatments pipeline completes with a georeference."""
    treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    _, treated = treatments_runner(treatments)

    assert treated["status"] == "completed"
    assert treated["georeference"] is not None
    assert "crs" in treated["georeference"]
    assert "bounds" in treated["georeference"]


def test_diameter_from_below_reduces_and_enforces_limit(treatments_runner):
    """from_below removes trees under the limit, in place; survivors are all
    >= the limit and no more numerous than the source."""
    threshold = 15.0
    treatments = [{"metric": "diameter", "method": "from_below", "value": threshold}]
    pim_inventory, treated = treatments_runner(treatments)

    pim_count = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"))
    df = _load_df(treated["id"])

    if pim_count == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")

    assert len(df) <= pim_count
    if len(df) == 0:
        pytest.skip("No trees survived the thinning (sparse grid)")
    assert (df["dbh"] >= threshold).all(), (
        f"Found trees with dbh < {threshold}: {df['dbh'].min()}"
    )


def test_diameter_from_above_enforces_limit(treatments_runner):
    """from_above removes trees over the limit; survivors are all below it."""
    threshold = 20.0
    treatments = [{"metric": "diameter", "method": "from_above", "value": threshold}]
    _, treated = treatments_runner(treatments)

    df = _load_df(treated["id"])
    if len(df) == 0:
        pytest.skip("No trees survived the thinning (sparse grid)")
    assert (df["dbh"] < threshold).all(), (
        f"Found trees with dbh >= {threshold}: {df['dbh'].max()}"
    )


def test_basal_area_thin_does_not_increase_count(treatments_runner):
    """An inventory-wide basal-area thin runs end-to-end and never adds trees."""
    treatments = [{"metric": "basal_area", "method": "from_below", "value": 10.0}]
    pim_inventory, treated = treatments_runner(treatments)

    pim_count = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{pim_inventory['id']}"))
    treated_count = len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{treated['id']}"))

    if pim_count == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")

    assert treated_count <= pim_count


def test_unit_conversion_in_treatment(treatments_runner):
    """Unit conversion on a diameter treatment works end-to-end (4 in = 10.16 cm)."""
    treatments = [
        {"metric": "diameter", "method": "from_below", "value": 4.0, "unit": "in"}
    ]
    _, treated = treatments_runner(treatments)

    df = _load_df(treated["id"])
    if len(df) == 0:
        pytest.skip("No trees survived the thinning (sparse grid)")
    assert (df["dbh"] >= 10.16 - 0.01).all()


def test_parquet_has_correct_columns(treatments_runner):
    """The in-place rewrite keeps the same schema columns as the source."""
    from standgen.columns import BASE_COLUMNS

    treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    _, treated = treatments_runner(treatments)

    ddf = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{treated['id']}")
    assert sorted(ddf.columns.tolist()) == sorted(BASE_COLUMNS)


def test_georeference_matches_source(treatments_runner):
    """Treated inventory georeference matches the source — data moves in place,
    the grid footprint never changes."""
    treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    pim_inventory, treated = treatments_runner(treatments)

    assert treated["georeference"]["crs"] == pim_inventory["georeference"]["crs"]
    assert treated["georeference"]["bounds"] == pim_inventory["georeference"]["bounds"]


def test_final_progress_is_100(treatments_runner):
    """After completion, progress should be 100%."""
    treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    _, treated = treatments_runner(treatments)

    assert treated["progress"]["percent"] == 100
    assert treated["progress"]["message"] == "Complete"


def test_pending_delta_appended_to_existing_ledger(treatments_runner):
    """A new treatment delta is appended to the existing ledger on completion,
    not replacing it: an inventory that already carries a completed treatment
    ends with both the prior treatment and the new delta, in order (#319)."""
    prior = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    delta = [{"metric": "diameter", "method": "from_above", "value": 30.0}]

    _, treated = treatments_runner(delta, prior_treatments=prior)

    assert treated.get("pending_treatments") == []
    assert treated.get("treatments") == prior + delta


@pytest.fixture
def feature_treatments_runner(shared_pim_source):
    """Apply an in-place feature-scoped treatment to a copy of the shared source.

    Uploads a Feature GeoParquet to the path the API writes
    (``{domain_id}/{feature_id}.parquet``) plus a completed Feature Firestore
    doc, then queues a treatment whose conditions reference that feature_id. This
    exercises ``resolve_spatial_conditions`` through the in-place handler. Cleans
    up the feature blob, feature doc, and treated inventory on teardown.
    """
    pim_inventory, pim_id, domain_id = shared_pim_source
    treated_ids = []
    feature_blobs = []
    feature_doc_ids = []

    def _run(feature_gdf: gpd.GeoDataFrame, treatments: list[dict]):
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
        resolved = []
        for treatment in treatments:
            conditions = [
                {**c, "feature_id": feature_id} if c.get("source") == "feature" else c
                for c in treatment.get("conditions", [])
            ]
            resolved.append({**treatment, "conditions": conditions})

        treated_id = f"test-{uuid4().hex}"
        get_gcsfs_client().copy(
            f"{INVENTORIES_BUCKET}/{pim_id}",
            f"{INVENTORIES_BUCKET}/{treated_id}",
            recursive=True,
        )
        treated_data = {
            "id": treated_id,
            "domain_id": domain_id,
            "name": "Spatially Treated Inventory (in place)",
            "status": "pending",
            "source": pim_inventory["source"],
            "georeference": pim_inventory["georeference"],
            # Ledger starts empty; the delta is queued and merged on completion.
            "treatments": [],
            "pending_treatments": resolved,
        }
        set_document(INVENTORIES_COLLECTION, treated_id, treated_data)
        treated_ids.append(treated_id)

        request = MockRequest(data={"id": treated_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Treatment processing failed: {response}"

        _, treated_snapshot = get_document(INVENTORIES_COLLECTION, treated_id)
        treated_inventory = treated_snapshot.to_dict()
        assert treated_inventory["status"] == "completed", (
            f"Treated inventory not completed: {treated_inventory.get('error')}"
        )
        # On completion the queued delta is merged into the ledger and the queue
        # cleared (#319).
        assert treated_inventory.get("pending_treatments") == []
        assert treated_inventory.get("treatments") == resolved
        return pim_inventory, treated_inventory

    yield _run

    for treated_id in treated_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{treated_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, treated_id)
    for feature_blob in feature_blobs:
        if exists(feature_blob):
            delete_file(feature_blob)
    for feature_doc_id in feature_doc_ids:
        delete_document(FEATURES_COLLECTION, feature_doc_id)


def _load_trees(inventory_id: str) -> "gpd.GeoDataFrame":
    """Load an inventory's trees as a GeoDataFrame of (x, y) points."""
    df = _load_df(inventory_id)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))


def test_feature_diameter_only_thins_trees_inside_feature(
    feature_treatments_runner, shared_pim_source
):
    """A diameter treatment scoped to a buffered feature thins only in-region
    trees: every surviving in-region tree clears the limit, while out-of-region
    trees are untouched (their count is preserved)."""
    pim_inventory, _, _ = shared_pim_source

    crs = pim_inventory["georeference"]["crs"]
    minx, miny, maxx, maxy = pim_inventory["georeference"]["bounds"]
    midx = (minx + maxx) / 2.0
    # Feature covers the western half of the domain working extent.
    feature_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, midx, maxy)], crs=crs)

    buffer_m = 5.0
    resolved_geom = buffer_gdf(feature_gdf, buffer_m).geometry.union_all()

    threshold = 20.0
    source_trees = _load_trees(pim_inventory["id"])
    if len(source_trees) == 0:
        pytest.skip("Source PIM produced 0 trees (sparse grid)")
    inside = source_trees.within(resolved_geom)
    outside_count = int((~inside).sum())
    inside_below = int(((source_trees["dbh"] < threshold) & inside).sum())
    inside_above = int(((source_trees["dbh"] >= threshold) & inside).sum())
    if inside_below == 0 or inside_above == 0 or outside_count == 0:
        pytest.skip("Feature geometry / threshold does not split the tree set")

    treatments = [
        {
            "metric": "diameter",
            "method": "from_below",
            "value": threshold,
            "conditions": [
                {"source": "feature", "operator": "within", "buffer_m": buffer_m}
            ],
        }
    ]
    _, treated = feature_treatments_runner(feature_gdf, treatments)

    treated_trees = _load_trees(treated["id"])
    survivors_inside = treated_trees.within(resolved_geom)

    # Every surviving in-region tree clears the limit ...
    assert (treated_trees[survivors_inside]["dbh"] >= threshold).all(), (
        "found surviving in-region trees below the diameter limit"
    )
    # ... and every out-of-region tree is untouched (count preserved).
    assert int((~survivors_inside).sum()) == outside_count
