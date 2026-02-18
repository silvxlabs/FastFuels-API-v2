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

from lib.config import DEV_OWNER_ID


def make_domain_data(
    owner_id: str = DEV_OWNER_ID,
    name: str = "Test Domain",
    description: str = "Test domain created by fixture",
    tags: list | None = None,
) -> dict:
    """Factory function to create domain data as stored in Firestore.

    Args:
        owner_id: Owner ID for access control. Default: DEV_OWNER_ID
        name: Domain name.
        description: Domain description.
        tags: List of tags.

    Returns:
        Dict matching Firestore domain document structure.
    """
    return {
        "type": "FeatureCollection",
        "id": f"test-{uuid.uuid4().hex}",
        "name": name,
        "description": description,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id,
        "tags": tags or [],
        "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": json.dumps(
                        [
                            [
                                [500000.0, 5200000.0],
                                [501000.0, 5200000.0],
                                [501000.0, 5201000.0],
                                [500000.0, 5201000.0],
                                [500000.0, 5200000.0],
                            ]
                        ]
                    ),
                },
            }
        ],
    }


def make_grid_data(
    domain_id: str,
    owner_id: str = DEV_OWNER_ID,
    name: str = "Test Grid",
    description: str = "Test grid created by fixture",
    status: str = "pending",
    tags: list | None = None,
    source: dict | None = None,
    bands: list | None = None,
    georeference: dict | None = None,
) -> dict:
    """Factory function to create grid data as stored in Firestore.

    Args:
        domain_id: Required domain ID this grid belongs to.
        owner_id: Owner ID for access control. Default: DEV_OWNER_ID
        name: Grid name.
        description: Grid description.
        status: Grid status. Default: "pending".
        tags: List of tags.
        source: Source specification dict. Defaults to LANDFIRE FBFM40.
        bands: Band list. Defaults to fbfm + fuel_load.1hr.
        georeference: Georeference dict. Defaults to a 34x34 grid at 30m.

    Returns:
        Dict matching Firestore grid document structure.
    """
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "name": name,
        "description": description,
        "status": status,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id,
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


def make_application_data(
    owner_id: str = DEV_OWNER_ID,
    name: str = "Test Application",
    description: str | None = "Test application created by fixture",
) -> dict:
    """Factory function to create application data as stored in Firestore.

    Args:
        owner_id: Owner ID for access control. Default: DEV_OWNER_ID
        name: Application name.
        description: Application description.

    Returns:
        Dict matching Firestore application document structure.
    """
    now = datetime.now(UTC)
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "owner_id": owner_id,
        "name": name,
        "description": description,
        "created_on": now,
        "modified_on": now,
    }


def make_export_data(
    domain_id: str,
    owner_id: str = DEV_OWNER_ID,
    name: str = "Test Export",
    description: str = "Test export created by fixture",
    status: str = "pending",
    tags: list | None = None,
    source: dict | None = None,
    signed_url: str | None = None,
    curl: str | None = None,
    expires_on: datetime | None = None,
) -> dict:
    """Factory function to create export data as stored in Firestore.

    Args:
        domain_id: Required domain ID this export came from.
        owner_id: Owner ID for access control. Default: DEV_OWNER_ID
        name: Export name.
        description: Export description.
        status: Export status. Default: "pending".
        tags: List of tags.
        source: Source specification dict. Defaults to geotiff export.
        signed_url: Signed download URL (populated when completed).
        curl: curl command for downloading (populated when completed).
        expires_on: When the signed URL expires.

    Returns:
        Dict matching Firestore export document structure.
    """
    return {
        "id": f"test-{uuid.uuid4().hex}",
        "domain_id": domain_id,
        "name": name,
        "description": description,
        "status": status,
        "progress": None,
        "created_on": datetime.now(),
        "modified_on": datetime.now(),
        "owner_id": owner_id,
        "source": source
        or {
            "name": "geotiff",
            "grid_id": f"test-{uuid.uuid4().hex}",
            "bands": None,
        },
        "signed_url": signed_url,
        "curl": curl,
        "expires_on": expires_on,
        "error": None,
        "tags": tags or [],
    }


def make_key_data(
    owner_id: str = DEV_OWNER_ID,
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

    Args:
        owner_id: Owner ID for access control. Default: DEV_OWNER_ID
        creator_id: Human user who created the key. Default: same as owner_id.
        name: Key name.
        description: Key description.
        scopes: List of scopes. Default: ["read"].
        access: Access type ("personal" or "application").
        application_id: Application ID (required when access="application").
        valid_days: Number of days the key is valid.

    Returns:
        Dict matching Firestore key document structure, with an extra
        ``_test_secret`` field containing the raw secret.
    """
    raw_secret = secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_secret.encode()).hexdigest()
    now = datetime.now(UTC)

    data = {
        "id": key_hash,
        "owner_id": owner_id,
        "creator_id": creator_id or owner_id,
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
