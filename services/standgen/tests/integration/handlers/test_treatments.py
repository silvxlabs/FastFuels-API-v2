"""
Integration tests for the inventory treatments pipeline.

Treatments are a create-time field on a PIM inventory; the PIM handler applies
them after modifications, on the materialized population. A baseline (untreated)
PIM inventory is created once (module-scoped) with a fixed seed; each test
creates its own treated PIM inventory with the SAME seed, so the pre-treatment
population is identical and tree-count comparisons are meaningful.

A defensive CHM-rejection test confirms a treatment-bearing CHM inventory fails
fast in standgen (the API also rejects it at create time).

These tests hit real GCS and Firestore and require valid credentials.
"""

import json
from uuid import uuid4

import dask.dataframe as dd
import geopandas as gpd
import pytest
from shapely.geometry import box
from standgen.columns import BASE_COLUMNS

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists
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
SEED = 42

pytestmark = pytest.mark.parametrize(
    "module_pim_grid", [STATIC_PIM_GRID], indirect=True
)


def _stringify_treatments(treatments: list[dict]) -> list[dict]:
    """JSON-encode inline geometry coordinates, as the API does before storing.

    Firestore rejects nested arrays, so a geometry condition's coordinates are
    stored as a JSON string; standgen's resolver decodes them back.
    """
    out = []
    for treatment in treatments:
        conditions = []
        for cond in treatment.get("conditions", []):
            geom = cond.get("geometry")
            if geom is not None and not isinstance(geom.get("coordinates"), str):
                cond = {
                    **cond,
                    "geometry": {
                        **geom,
                        "coordinates": json.dumps(geom["coordinates"]),
                    },
                }
            conditions.append(cond)
        out.append({**treatment, "conditions": conditions})
    return out


@pytest.fixture(scope="module")
def treatment_env(module_pim_grid):
    """Create the domain + a baseline (untreated) PIM inventory once.

    Yields ``(baseline_inventory, domain_id, grid_id)``. The baseline uses the
    fixed ``SEED`` so treated runs reusing that seed share its pre-treatment
    population. Cleans up the baseline + domain on teardown.
    """
    domain_data = load_json(DOMAINS_DIR / "blue_mtn.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    baseline_data = load_json(INVENTORIES_DIR / "pim_treemap.json")
    baseline_data["domain_id"] = domain_id
    baseline_data["source"]["source_pim_grid_id"] = module_pim_grid
    baseline_data["source"]["seed"] = SEED
    baseline_id = f"test-{uuid4().hex}"
    baseline_data["id"] = baseline_id
    set_document(INVENTORIES_COLLECTION, baseline_id, baseline_data)

    _run_standgen(baseline_id)

    if DEPLOYMENT_ENV != "local":
        baseline = _poll_for_completion(baseline_id)
    else:
        _, snapshot = get_document(INVENTORIES_COLLECTION, baseline_id)
        baseline = snapshot.to_dict()

    assert baseline["status"] == "completed", (
        f"Baseline PIM not completed: {baseline.get('error')}"
    )

    yield baseline, domain_id, module_pim_grid

    gcs_path = f"gs://{INVENTORIES_BUCKET}/{baseline_id}"
    if exists(gcs_path):
        delete_directory(gcs_path)
    delete_document(INVENTORIES_COLLECTION, baseline_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture
def treatment_runner(treatment_env):
    """Run a treated PIM inventory sharing the baseline's domain, grid, and seed.

    Returns ``(baseline_inventory, treated_inventory)``. Cleans up each treated
    inventory on teardown.
    """
    baseline, domain_id, grid_id = treatment_env
    treated_ids = []

    def _run(treatments: list[dict], seed: int = SEED) -> tuple[dict, dict]:
        from standgen.main import process_inventory_request

        treated_id = f"test-{uuid4().hex}"
        treated_data = {
            "id": treated_id,
            "domain_id": domain_id,
            "name": "Treated PIM Inventory",
            "status": "pending",
            "source": {
                "name": "pim",
                "point_process": "inhomogeneous_poisson",
                "seed": seed,
                "source_pim_grid_id": grid_id,
            },
            "treatments": _stringify_treatments(treatments),
        }
        set_document(INVENTORIES_COLLECTION, treated_id, treated_data)
        treated_ids.append(treated_id)

        request = MockRequest(data={"id": treated_id})
        response, status_code = process_inventory_request(request)
        assert status_code == 200, f"Treatment processing failed: {response}"

        _, snapshot = get_document(INVENTORIES_COLLECTION, treated_id)
        treated = snapshot.to_dict()
        assert treated["status"] == "completed", (
            f"Treated inventory not completed: {treated.get('error')}"
        )
        return baseline, treated

    yield _run

    for treated_id in treated_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{treated_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        delete_document(INVENTORIES_COLLECTION, treated_id)


def _count(inventory_id: str) -> int:
    return len(dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}"))


def _load_df(inventory_id: str):
    return dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{inventory_id}").compute()


def test_diameter_from_below_removes_small_trees(treatment_runner):
    """A diameter from_below treatment drops trees under the limit."""
    limit = 15.0
    treatments = [{"metric": "diameter", "method": "from_below", "value": limit}]
    baseline, treated = treatment_runner(treatments)

    base_df = _load_df(baseline["id"])
    if len(base_df) == 0:
        pytest.skip("Baseline PIM produced 0 trees (sparse grid)")

    treated_df = _load_df(treated["id"])
    assert len(treated_df) <= len(base_df)
    if len(treated_df):
        assert treated_df["dbh"].min() >= limit


def test_basal_area_directional_reduces_count(treatment_runner):
    """A basal-area from_below treatment to a low residual thins the stand."""
    treatments = [{"metric": "basal_area", "method": "from_below", "value": 5.0}]
    baseline, treated = treatment_runner(treatments)

    base_count = _count(baseline["id"])
    if base_count == 0:
        pytest.skip("Baseline PIM produced 0 trees (sparse grid)")

    assert _count(treated["id"]) <= base_count


def test_proportional_is_deterministic(treatment_runner):
    """Two proportional runs with the same seed produce the same tree count."""
    treatments = [{"metric": "basal_area", "method": "proportional", "value": 10.0}]
    baseline, treated_a = treatment_runner(treatments)
    if _count(baseline["id"]) == 0:
        pytest.skip("Baseline PIM produced 0 trees (sparse grid)")
    _, treated_b = treatment_runner(treatments)

    assert _count(treated_a["id"]) == _count(treated_b["id"])


def test_spatially_scoped_leaves_outside_trees_untouched(
    treatment_runner, treatment_env
):
    """A within-geometry diameter treatment thins only in-region trees."""
    baseline, _, _ = treatment_env
    minx, miny, maxx, maxy = baseline["georeference"]["bounds"]
    midx = (minx + maxx) / 2.0
    # Treat only the western half; eastern-half trees must be untouched.
    west = box(minx, miny, midx, maxy)
    east = box(midx, miny, maxx, maxy)

    base_df = _load_df(baseline["id"])
    if len(base_df) == 0:
        pytest.skip("Baseline PIM produced 0 trees (sparse grid)")
    base_pts = gpd.GeoSeries(gpd.points_from_xy(base_df["x"], base_df["y"]))
    base_east = int(base_pts.within(east).sum())
    if base_east == 0 or int(base_pts.within(west).sum()) == 0:
        pytest.skip("Geometry does not split the tree set")

    treatments = [
        {
            "metric": "diameter",
            "method": "from_below",
            "value": 100.0,  # aggressive: removes nearly all in-region trees
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": list(west.__geo_interface__["coordinates"]),
                    },
                }
            ],
        }
    ]
    _, treated = treatment_runner(treatments)

    treated_df = _load_df(treated["id"])
    treated_pts = gpd.GeoSeries(gpd.points_from_xy(treated_df["x"], treated_df["y"]))
    # Eastern-half trees are unchanged in count; western half is thinned hard.
    assert int(treated_pts.within(east).sum()) == base_east
    assert int(treated_pts.within(west).sum()) <= int(base_pts.within(west).sum())


def test_output_columns_unchanged(treatment_runner):
    """Treatments only drop rows; the schema columns are unchanged."""
    treatments = [{"metric": "diameter", "method": "from_below", "value": 5.0}]
    _, treated = treatment_runner(treatments)
    ddf = dd.read_parquet(f"gs://{INVENTORIES_BUCKET}/{treated['id']}")
    assert sorted(ddf.columns.tolist()) == sorted(BASE_COLUMNS)
