"""
Integration tests for CHM crown feature generation.

Seeds crowns from the static Blue Mountain CHM inventory on the static NAIP CHM
grid it was detected from (both read-only), runs the full pipeline, and checks
the GeoParquet written to GCS.
"""

from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from etcher.handlers.crown import load_chm
from rasterio.enums import MergeAlg
from rasterio.features import rasterize

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    FEATURES_BUCKET,
    FEATURES_COLLECTION,
    INVENTORIES_BUCKET,
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

STATIC_INVENTORY = "static-test-blue-mtn-chm-inventory"
STATIC_CHM_GRID = "static-test-blue-mtn-naip-chm"


@pytest.fixture(scope="module")
def crown_run():
    """Run the crown pipeline once and share the feature and its crowns."""
    domain_id = f"test-{uuid4().hex}"
    domain = _stringify_coordinates(load_json(DOMAINS_DIR / "blue_mtn.json"))
    domain["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, domain)

    feature_id = f"test-{uuid4().hex}"
    feature = load_json(FEATURES_DIR / "crown_chm.json")
    feature["id"] = feature_id
    feature["domain_id"] = domain_id
    set_document(FEATURES_COLLECTION, feature_id, feature)

    path = f"gs://{FEATURES_BUCKET}/{domain_id}/{feature_id}.parquet"
    try:
        _run_feature_job(feature_id)
        if DEPLOYMENT_ENV != "local":
            feature = _poll_for_completion(feature_id, timeout=600)
        else:
            _, snapshot = get_document(FEATURES_COLLECTION, feature_id)
            feature = snapshot.to_dict()
        assert feature["status"] == "completed", feature.get("error")
        yield feature, gpd.read_parquet(path)
    finally:
        if exists(path):
            delete_file(path)
        delete_document(FEATURES_COLLECTION, feature_id)
        delete_document(DOMAINS_COLLECTION, domain_id)


def test_feature_completed_with_georeference(crown_run):
    feature, _ = crown_run
    assert feature["georeference"]["crs"] == "EPSG:32611"
    assert feature["size_bytes"] > 0


def test_one_polygon_per_tree(crown_run):
    _, crowns = crown_run
    trees = pd.read_parquet(
        f"gs://{INVENTORIES_BUCKET}/{STATIC_INVENTORY}", columns=["tree_id"]
    )
    assert list(crowns.columns) == ["tree_id", "geometry"]
    assert crowns["tree_id"].dtype == np.int32
    assert crowns["tree_id"].is_unique
    assert crowns["tree_id"].is_monotonic_increasing
    # CHM detection puts every tree on its own valid cell, so every tree seeds.
    assert set(crowns["tree_id"]) == set(trees["tree_id"])
    assert (crowns.geom_type == "Polygon").all()
    assert crowns.is_valid.all()
    assert crowns.crs.to_epsg() == 32611


def test_crowns_cover_whole_cells_without_overlap(crown_run):
    _, crowns = crown_run
    chm = load_chm(STATIC_CHM_GRID)
    transform = chm.rio.transform()
    counts = rasterize(
        ((g, 1) for g in crowns.geometry),
        out_shape=chm.shape,
        transform=transform,
        merge_alg=MergeAlg.add,
        dtype=np.int32,
    )
    assert counts.max() == 1
    cell_area = abs(transform.a * transform.e)
    np.testing.assert_allclose(counts.sum() * cell_area, crowns.area.sum())
