"""
Shared factory functions for v2 API tests.

These functions create test data matching Firestore document structure.
Import them in test files to avoid duplication:

    from tests.fixtures import make_domain_data, make_grid_data
"""

import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta

# Set by conftest.py from the pre-seeded Firestore key document.
DEFAULT_OWNER_ID: str | None = None


def make_domain_data(
    owner_id: str | None = None,
    name: str = "Test Domain",
    description: str = "Test domain created by fixture",
    tags: list | None = None,
) -> dict:
    """Factory function to create domain data as stored in Firestore.

    Produces the two-feature format: a "domain" feature (working extent /
    bounding box) and an "input" feature (the original user polygon). For a
    rectangular input, both features have the same geometry.
    """
    polygon_coords = [
        [500000.0, 5200000.0],
        [501000.0, 5200000.0],
        [501000.0, 5201000.0],
        [500000.0, 5201000.0],
        [500000.0, 5200000.0],
    ]
    return {
        "type": "FeatureCollection",
        "id": f"test-{uuid.uuid4().hex}",
        "name": name,
        "description": description,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "tags": tags or [],
        "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "domain"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": json.dumps([polygon_coords]),
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "input"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": json.dumps([polygon_coords]),
                },
            },
        ],
        "bbox": [500000.0, 5200000.0, 501000.0, 5201000.0],
        "pad_to_resolution": None,
    }


def make_grid_data(
    domain_id: str,
    owner_id: str | None = None,
    name: str = "Test Grid",
    description: str = "Test grid created by fixture",
    status: str = "pending",
    tags: list | None = None,
    source: dict | None = None,
    bands: list | None = None,
    georeference: dict | None = None,
    chunks: dict | None = None,
) -> dict:
    """Factory function to create grid data as stored in Firestore."""
    data = {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "name": name,
        "description": description,
        "status": status,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "source": source
        or {
            "name": "landfire",
            "product": "fbfm40",
            "version": "2022",
            "description": "Scott-Burgan 40 fire behavior fuel models",
        },
        "modifications": [],
        "bands": bands
        or [
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
            {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m²", "index": 1},
        ],
        "georeference": (
            georeference
            if georeference is not None
            else {
                "crs": "EPSG:32611",
                "transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 5201000.0),
                "shape": (34, 34),
            }
        ),
        "tags": tags or [],
    }
    if chunks is not None:
        data["chunks"] = chunks
    return data


def make_layerset_feature_data(
    domain_id: str,
    owner_id: str | None = None,
    name: str = "Test Layerset",
    description: str = "Test layerset created by fixture",
    tags: list | None = None,
) -> dict:
    """Factory function for a layerset Feature document.

    Mirrors the doc shape written by
    ``services/api/api/resources/features/layerset/router.py``.
    Use to seed a layerset that the rasterize endpoint can reference by
    ``layerset_id``. No GCS upload is performed — router-level tests only
    need the Firestore doc.
    """
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "type": "layerset",
        "name": name,
        "description": description,
        "status": "completed",
        "progress": None,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "source": {"product": "Upload", "description": "User-uploaded layerset"},
        "georeference": {
            "crs": "EPSG:4326",
            "bounds": (-114.12, 46.82, -114.09, 46.84),
        },
        "error": None,
        "tags": tags or [],
    }


def make_inventory_data(
    domain_id: str,
    owner_id: str | None = None,
    name: str = "Test Inventory",
    description: str = "Test inventory created by fixture",
    status: str = "pending",
    tags: list | None = None,
    source: dict | None = None,
    inventory_type: str = "tree",
    georeference: dict | None = None,
) -> dict:
    """Factory function to create inventory data as stored in Firestore."""
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "type": inventory_type,
        "name": name,
        "description": description,
        "status": status,
        "progress": None,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "source": source
        or {
            "name": "pim",
            "source_pim_grid_id": f"test-{uuid.uuid4().hex}",
            "point_process": "inhomogeneous_poisson",
            "seed": 42,
        },
        "modifications": [],
        "columns": [
            {"key": "x", "type": "continuous", "unit": "m"},
            {"key": "y", "type": "continuous", "unit": "m"},
            {"key": "fia_species_code", "type": "categorical", "unit": None},
            {"key": "fia_status_code", "type": "categorical", "unit": None},
            {"key": "dbh", "type": "continuous", "unit": "cm"},
            {"key": "height", "type": "continuous", "unit": "m"},
            {"key": "crown_ratio", "type": "continuous", "unit": None},
        ],
        "georeference": georeference,
        "error": None,
        "tags": tags or [],
    }


def make_application_data(
    owner_id: str | None = None,
    name: str = "Test Application",
    description: str | None = "Test application created by fixture",
) -> dict:
    """Factory function to create application data as stored in Firestore."""
    now = datetime.now(UTC)
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "name": name,
        "description": description,
        "created_on": now,
        "modified_on": now,
    }


def make_export_data(
    domain_id: str,
    owner_id: str | None = None,
    name: str = "Test Export",
    description: str = "Test export created by fixture",
    status: str = "pending",
    tags: list | None = None,
    source: dict | None = None,
    signed_url: str | None = None,
    expires_on: datetime | None = None,
) -> dict:
    """Factory function to create export data as stored in Firestore."""
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "name": name,
        "description": description,
        "status": status,
        "progress": None,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id or DEFAULT_OWNER_ID,
        "source": source
        or {
            "name": "geotiff",
            "grid_id": f"test-{uuid.uuid4().hex}",
            "bands": None,
        },
        "signed_url": signed_url,
        "expires_on": expires_on,
        "error": None,
        "tags": tags or [],
    }


def make_key_data(
    owner_id: str | None = None,
    creator_id: str | None = None,
    name: str = "Test Key",
    description: str | None = "Test key created by fixture",
    scopes: list[str] | None = None,
    access: str = "personal",
    application_id: str | None = None,
    valid_days: int = 30,
) -> dict:
    """Factory function to create key data as stored in Firestore.

    Generates a real secret/hash pair. The raw secret is stored in
    ``_test_secret`` for test convenience (not part of the Firestore doc).
    """
    resolved = owner_id or DEFAULT_OWNER_ID
    raw_secret = secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_secret.encode()).hexdigest()
    now = datetime.now(UTC)

    data = {
        "id": key_hash,
        "owner_id": resolved,
        "creator_id": creator_id or resolved,
        "name": name,
        "description": description,
        "scopes": scopes or ["read"],
        "access": access,
        "application_id": application_id,
        "valid_days": valid_days,
        "created_on": now,
        "expires_on": now + timedelta(days=valid_days),
        "_test_secret": raw_secret,
    }
    return data
