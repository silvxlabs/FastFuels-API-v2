"""
Integration test for Overture building feature generation.

Runs the full pipeline on the Bigfork, MT docs domain: Firestore setup ->
process_feature_request -> GeoParquet on GCS + completed Firestore document.
Buildings are read live from Overture's public S3 bucket, so counts change
between releases and are asserted as ranges.
"""

import time
from uuid import uuid4

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, mapping

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_file, exists

from ..conftest import _poll_for_completion, _run_feature_job, _stringify_coordinates

# The Bigfork in-town domain from the FastFuels-Web "create OSM features" how-to.
BIGFORK_4326 = Polygon(
    [
        (-114.085846, 48.055021),
        (-114.065029, 48.054491),
        (-114.064148, 48.069978),
        (-114.084971, 48.070509),
    ]
)
BIGFORK_CRS = "EPSG:32611"

SOURCE = {
    "product": "overture",
    "description": "Overture Maps building footprints",
    "extent_buffer_m": 0,
    "release": None,
}


def bigfork_domain() -> dict:
    """The Bigfork domain as the API stores it: a UTM working-extent rectangle."""
    extent = gpd.GeoSeries([BIGFORK_4326], crs="EPSG:4326").to_crs(BIGFORK_CRS)
    rectangle = extent.envelope.iloc[0]
    return {
        "type": "FeatureCollection",
        "name": "bigfork-buildings",
        "crs": {"type": "name", "properties": {"name": BIGFORK_CRS}},
        "features": [
            {
                "id": "0",
                "type": "Feature",
                "properties": {"name": "domain"},
                "geometry": mapping(rectangle),
            }
        ],
    }


@pytest.fixture(scope="module")
def building_feature():
    """Run the Overture building pipeline once; yields (feature, seconds)."""
    domain_id = f"test-{uuid4().hex}"
    domain = _stringify_coordinates(bigfork_domain())
    domain["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, domain)

    feature_id = f"test-{uuid4().hex}"
    set_document(
        FEATURES_COLLECTION,
        feature_id,
        {
            "id": feature_id,
            "domain_id": domain_id,
            "name": "Test Building Feature",
            "description": "Integration test feature for Overture buildings",
            "type": "building",
            "status": "pending",
            "tags": ["test", "integration"],
            "source": SOURCE,
        },
    )

    try:
        start = time.monotonic()
        _run_feature_job(feature_id)
        if DEPLOYMENT_ENV != "local":
            feature = _poll_for_completion(feature_id)
        else:
            _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
            feature = snapshot.to_dict()
        seconds = time.monotonic() - start

        assert feature["status"] == "completed", (
            f"Expected completed, got {feature['status']}. "
            f"Error: {feature.get('error')}"
        )
        yield feature, seconds
    finally:
        gcs_path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
        if exists(gcs_path):
            delete_file(gcs_path)
        delete_document(FEATURES_COLLECTION, feature_id)
        delete_document(DOMAINS_COLLECTION, domain_id)


@pytest.fixture(scope="module")
def building_gdf(building_feature):
    feature, _ = building_feature
    path = f"gs://{FEATURES_BUCKET}/{feature['domain_id']}/{feature['id']}.parquet"
    return gpd.read_parquet(path)


def test_building_count(building_feature, building_gdf):
    _, seconds = building_feature
    print(f"buildings={len(building_gdf)} seconds={seconds:.1f}")
    assert 450 <= len(building_gdf) <= 650


def test_building_columns(building_gdf):
    assert list(building_gdf.columns) == [
        "geometry",
        "height",
        "height_source",
        "source_dataset",
    ]
    assert building_gdf.crs.to_epsg() == 32611
    assert set(building_gdf.geom_type) <= {"Polygon", "MultiPolygon"}
    assert building_gdf["source_dataset"].notna().all()
    has_height = building_gdf["height"].notna()
    assert has_height.mean() > 0.5
    assert building_gdf.loc[has_height, "height_source"].notna().all()
    assert building_gdf.loc[~has_height, "height_source"].isna().all()


def test_building_within_domain(building_feature, building_gdf):
    feature, _ = building_feature
    minx, miny, maxx, maxy = feature["georeference"]["bounds"]
    gminx, gminy, gmaxx, gmaxy = building_gdf.total_bounds
    assert gminx >= minx - 1.0
    assert gminy >= miny - 1.0
    assert gmaxx <= maxx + 1.0
    assert gmaxy <= maxy + 1.0


def test_building_attributions(building_feature):
    feature, _ = building_feature
    datasets = {a["dataset"] for a in feature["attributions"]}
    assert {
        "Overture Maps Foundation",
        "OpenStreetMap",
        "Microsoft ML Buildings",
    } <= datasets
    assert all(a["license_name"] and a["accessed_on"] for a in feature["attributions"])


def test_building_release_recorded(building_feature):
    """source.release is set without clobbering the other source fields."""
    feature, _ = building_feature
    source = feature["source"]
    assert source["release"]
    assert {k: v for k, v in source.items() if k != "release"} == {
        k: v for k, v in SOURCE.items() if k != "release"
    }
