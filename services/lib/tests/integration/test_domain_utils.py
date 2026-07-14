"""
Integration tests for lib.domain_utils with real Firestore data.

Tests the full round-trip: write domain to Firestore (with stringified
coordinates) → load from Firestore → parse with parse_domain_gdf → verify
GeoDataFrame. This mirrors how _load_domain works in griddle/standgen main.py.

Requires GCP credentials and Firestore access.
"""

import copy
import json
from uuid import uuid4

import geopandas as gpd
import pytest

from lib.config import DOMAINS_COLLECTION
from lib.domain_utils import EmptyDomainError, parse_domain_gdf
from lib.firestore.documents import (
    DocumentNotFoundError,
    delete_document,
    get_document,
    set_document,
)
from lib.testing import SHARED_TEST_DOMAINS_DIR, load_json

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


def _load_domain_json(filename: str) -> dict:
    """Load a domain JSON file from griddle test data."""
    return load_json(DOMAINS_DIR / filename)


def _stringify_coordinates(domain_data: dict) -> dict:
    """Stringify nested coordinate arrays for Firestore compatibility."""
    data = copy.deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


@pytest.fixture
def firestore_domain():
    """Create a domain document in Firestore and clean up on teardown.

    Returns a callable that accepts a domain JSON filename, writes it to
    Firestore with stringified coordinates, and returns (domain_id, domain_data).
    """
    created_ids = []

    def _create(domain_file: str) -> tuple[str, dict]:
        domain_data = _load_domain_json(domain_file)
        data = _stringify_coordinates(domain_data)
        domain_id = f"test-{uuid4().hex}"
        data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, data)
        created_ids.append(domain_id)
        return domain_id, data

    yield _create

    for domain_id in created_ids:
        delete_document(DOMAINS_COLLECTION, domain_id)


class TestFirestoreRoundTrip:
    """Full round-trip: Firestore write → read → parse_domain_gdf."""

    def test_blue_mountain_round_trip(self, firestore_domain):
        """Blue Mountain domain survives Firestore serialization."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        gdf = parse_domain_gdf(snapshot.to_dict())

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 32611
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_blue_mountain_bounds(self, firestore_domain):
        """Blue Mountain bounds are preserved through Firestore."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        gdf = parse_domain_gdf(snapshot.to_dict())

        bounds = gdf.total_bounds
        assert 720000 < bounds[0] < 722000
        assert 5189000 < bounds[1] < 5191000

    def test_tile_boundary_round_trip(self, firestore_domain):
        """Tile boundary domain (different UTM zone) survives Firestore."""
        domain_id, _ = firestore_domain("meta_chm_2_tiles.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        gdf = parse_domain_gdf(snapshot.to_dict())

        assert gdf.crs.to_epsg() == 32612
        bounds = gdf.total_bounds
        x_extent = bounds[2] - bounds[0]
        y_extent = bounds[3] - bounds[1]
        assert pytest.approx(x_extent, rel=0.01) == 1000.0
        assert pytest.approx(y_extent, rel=0.01) == 1000.0

    def test_meta_chm_4_tiles_round_trip(self, firestore_domain):
        """Four-tile corner domain survives Firestore."""
        domain_id, _ = firestore_domain("meta_chm_4_tiles.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        gdf = parse_domain_gdf(snapshot.to_dict())

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 32612
        assert len(gdf) == 1


class TestFirestoreCoordinateSerialization:
    """Verify that Firestore's coordinate stringification is handled correctly."""

    def test_coordinates_stored_as_strings(self, firestore_domain):
        """After writing to Firestore, coordinates should be JSON strings."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        data = snapshot.to_dict()

        coords = data["features"][0]["geometry"]["coordinates"]
        assert isinstance(coords, str), (
            f"Expected stringified coordinates in Firestore, got {type(coords)}"
        )

    def test_parse_handles_string_coordinates(self, firestore_domain):
        """parse_domain_gdf correctly deserializes stringified coordinates."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        gdf = parse_domain_gdf(snapshot.to_dict())

        # Geometry should be valid despite string serialization
        assert gdf.geometry.iloc[0].is_valid
        assert gdf.geometry.iloc[0].area > 0


class TestFirestoreCrsHandling:
    """Verify CRS handling through Firestore round-trip."""

    def test_geojson_crs_object_preserved(self, firestore_domain):
        """CRS stored as GeoJSON object is preserved in Firestore."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        data = snapshot.to_dict()

        # CRS should be stored as a dict (GeoJSON CRS object)
        crs = data["crs"]
        assert isinstance(crs, dict)
        assert crs["properties"]["name"] == "EPSG:32611"


class TestFirestoreDocumentNotFound:
    """Verify behavior when domain document doesn't exist."""

    def test_missing_document_raises(self):
        """get_document raises DocumentNotFoundError for missing domains."""
        with pytest.raises(DocumentNotFoundError):
            get_document(DOMAINS_COLLECTION, f"nonexistent-{uuid4().hex}")

    def test_deleted_document_raises(self, firestore_domain):
        """After deletion, get_document raises DocumentNotFoundError."""
        domain_id, _ = firestore_domain("blue_mtn.json")

        # Verify it exists
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        assert snapshot.exists

        # Delete it
        delete_document(DOMAINS_COLLECTION, domain_id)

        # Now it should raise
        with pytest.raises(DocumentNotFoundError):
            get_document(DOMAINS_COLLECTION, domain_id)


class TestFirestoreEmptyDomain:
    """Verify behavior with empty domain data in Firestore."""

    def test_empty_features_from_firestore(self):
        """parse_domain_gdf raises EmptyDomainError for empty features."""
        domain_id = f"test-{uuid4().hex}"
        data = {"id": domain_id, "crs": "EPSG:4326", "features": []}
        set_document(DOMAINS_COLLECTION, domain_id, data)

        try:
            _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
            with pytest.raises(EmptyDomainError):
                parse_domain_gdf(snapshot.to_dict())
        finally:
            delete_document(DOMAINS_COLLECTION, domain_id)
