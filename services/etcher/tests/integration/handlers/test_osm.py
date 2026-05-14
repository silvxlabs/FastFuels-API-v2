"""
Integration tests for OSM feature generation.

Tests the full feature pipeline: Firestore setup -> process_feature_request ->
verify GCS GeoJSON output + Firestore georeference. Uses a standard test domain
with live queries to OpenStreetMap via OSMnx.

These tests hit real OSM APIs and write real GeoJSON to GCS/Firestore, so
they require valid credentials and network access.

Most tests share a single pipeline run per feature type via module-scoped
fixtures to avoid redundant processing and rate-limiting from OSM.
"""

from uuid import uuid4

import geopandas as gpd
import pytest

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_file, exists

from ..conftest import (
    DOMAINS_DIR,
    FEATURES_DIR,
    _poll_for_completion,
    _run_feature_job,
    _stringify_coordinates,
    load_json,
)


@pytest.fixture(scope="module")
def shared_road_feature():
    """Run the OSM Road pipeline once and share the result."""
    # Setup Domain
    domain_data = load_json(DOMAINS_DIR / "naip_chm_2_tiles.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    # Setup Feature
    feature_data = load_json(FEATURES_DIR / "road_osm.json")
    feature_data["domain_id"] = domain_id
    feature_id = f"test-{uuid4().hex}"
    feature_data["id"] = feature_id
    set_document(FEATURES_COLLECTION, feature_id, feature_data)

    # Run Pipeline
    _run_feature_job(feature_id)

    # Poll/Verify Status
    if DEPLOYMENT_ENV != "local":
        feature = _poll_for_completion(feature_id)
    else:
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()

    assert feature["status"] == "completed", (
        f"Expected completed, got {feature['status']}. Error: {feature.get('error')}"
    )
    assert feature.get("georeference") is not None

    yield feature

    # Cleanup
    gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
    if exists(gcs_path):
        delete_file(gcs_path)
    delete_document(FEATURES_COLLECTION, feature_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def shared_water_feature():
    """Run the OSM Water pipeline once and share the result."""
    domain_data = load_json(DOMAINS_DIR / "naip_chm_2_tiles.json")
    domain_id = f"test-{uuid4().hex}"
    data = _stringify_coordinates(domain_data)
    data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, data)

    feature_data = load_json(FEATURES_DIR / "water_osm.json")
    feature_data["domain_id"] = domain_id
    feature_id = f"test-{uuid4().hex}"
    feature_data["id"] = feature_id
    set_document(FEATURES_COLLECTION, feature_id, feature_data)

    _run_feature_job(feature_id)

    if DEPLOYMENT_ENV != "local":
        feature = _poll_for_completion(feature_id)
    else:
        _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
        feature = snapshot.to_dict()

    assert feature["status"] == "completed", (
        f"Expected completed, got {feature['status']}. Error: {feature.get('error')}"
    )
    assert feature.get("georeference") is not None

    yield feature

    # Cleanup
    gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"
    if exists(gcs_path):
        delete_file(gcs_path)
    delete_document(FEATURES_COLLECTION, feature_id)
    delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def shared_road_gdf(shared_road_feature):
    """Read the GeoJSON from the shared road feature once."""
    path = f"gs://{FEATURES_BUCKET}/{shared_road_feature['domain_id']}/{shared_road_feature['id']}.geojson"
    return gpd.read_file(path)


@pytest.fixture(scope="module")
def shared_water_gdf(shared_water_feature):
    """Read the GeoJSON from the shared water feature once."""
    path = f"gs://{FEATURES_BUCKET}/{shared_water_feature['domain_id']}/{shared_water_feature['id']}.geojson"
    return gpd.read_file(path)


# --- ROAD TESTS ---


def test_road_pipeline_completes(shared_road_feature):
    """Road OSM expansion completes successfully with georeference."""
    assert shared_road_feature["georeference"] is not None
    assert "crs" in shared_road_feature["georeference"]
    assert "bounds" in shared_road_feature["georeference"]


def test_road_geojson_has_correct_columns(shared_road_gdf):
    """Output road GeoJSON should have exactly geometry, type, and name."""
    if len(shared_road_gdf) == 0:
        pytest.skip("No roads generated (sparse domain); skipping column validation")
    assert sorted(shared_road_gdf.columns.tolist()) == sorted(
        ["geometry", "type", "name"]
    )


def test_road_geometries_are_polygons(shared_road_gdf):
    """All road line strings should have been buffered into polygons."""
    if len(shared_road_gdf) == 0:
        pytest.skip("No roads generated; skipping geometry validation")

    geom_types = shared_road_gdf.geom_type.unique()
    for gtype in geom_types:
        assert gtype in ["Polygon", "MultiPolygon"], (
            f"Found unbuffered geometry: {gtype}"
        )


def test_road_coordinates_within_domain(shared_road_feature, shared_road_gdf):
    """All road coordinates should be clipped within the domain bounds."""
    if len(shared_road_gdf) == 0:
        pytest.skip("No roads generated; skipping coordinate validation")

    geo = shared_road_feature["georeference"]
    bounds = geo["bounds"]  # [minx, miny, maxx, maxy]

    # Get total bounds of the generated GeoDataFrame
    gdf_bounds = shared_road_gdf.total_bounds

    # Allow a minor float precision buffer
    buffer = 1.0
    assert gdf_bounds[0] >= bounds[0] - buffer
    assert gdf_bounds[1] >= bounds[1] - buffer
    assert gdf_bounds[2] <= bounds[2] + buffer
    assert gdf_bounds[3] <= bounds[3] + buffer


# --- WATER TESTS ---


def test_water_pipeline_completes(shared_water_feature):
    """Water OSM expansion completes successfully with georeference."""
    assert shared_water_feature["georeference"] is not None
    assert "crs" in shared_water_feature["georeference"]
    assert "bounds" in shared_water_feature["georeference"]


def test_water_geojson_has_correct_columns(shared_water_gdf):
    """Output water GeoJSON should have exactly geometry and name."""
    if len(shared_water_gdf) == 0:
        pytest.skip(
            "No water features generated (sparse domain); skipping column validation"
        )
    assert sorted(shared_water_gdf.columns.tolist()) == sorted(["geometry", "name"])


def test_water_geometries_are_polygons(shared_water_gdf):
    """All water features (lakes, buffered rivers) should be polygons."""
    if len(shared_water_gdf) == 0:
        pytest.skip("No water features generated; skipping geometry validation")

    geom_types = shared_water_gdf.geom_type.unique()
    for gtype in geom_types:
        assert gtype in ["Polygon", "MultiPolygon"], f"Found invalid geometry: {gtype}"


def test_water_coordinates_within_domain(shared_water_feature, shared_water_gdf):
    """All water coordinates should be clipped within the domain bounds."""
    if len(shared_water_gdf) == 0:
        pytest.skip("No water features generated; skipping coordinate validation")

    geo = shared_water_feature["georeference"]
    bounds = geo["bounds"]  # [minx, miny, maxx, maxy]

    gdf_bounds = shared_water_gdf.total_bounds
    buffer = 1.0
    assert gdf_bounds[0] >= bounds[0] - buffer
    assert gdf_bounds[1] >= bounds[1] - buffer
    assert gdf_bounds[2] <= bounds[2] + buffer
    assert gdf_bounds[3] <= bounds[3] + buffer
