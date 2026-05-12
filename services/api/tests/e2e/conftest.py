"""
Fixtures for generating static test data via the full API pipeline.

The ``create_static_fixture`` fixture drives the end-to-end flow:
POST grid -> poll Firestore -> copy zarr to static path -> save JSON template.

For chained fixtures, the caller passes ``dependencies={"grids": [...],
"inventories": [...]}`` listing every static fixture the API will look up
during source validation; each is temporarily registered as a Firestore doc
for the duration of the request. Use ``@pytest.mark.dependency`` to control
test ordering so the parent fixture's GCS data + JSON template exist before
the child runs.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import gcsfs
import pytest
from api.resources.domains.examples import EXAMPLE_WGS84_DEFAULT
from google.cloud import firestore

from lib.config import (
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    INVENTORIES_BUCKET,
    INVENTORIES_COLLECTION,
)
from lib.testing import SHARED_TEST_GRIDS_DIR, SHARED_TEST_INVENTORIES_DIR

logger = logging.getLogger(__name__)

STATIC_GRIDS_DIR = SHARED_TEST_GRIDS_DIR
STATIC_INVENTORIES_DIR = SHARED_TEST_INVENTORIES_DIR


@dataclass(frozen=True)
class StaticResourceType:
    collection: str
    template_dir: Path


STATIC_RESOURCE_TYPES = {
    "grids": StaticResourceType(
        collection=GRIDS_COLLECTION,
        template_dir=STATIC_GRIDS_DIR,
    ),
    "inventories": StaticResourceType(
        collection=INVENTORIES_COLLECTION,
        template_dir=STATIC_INVENTORIES_DIR,
    ),
}

# Fields to strip from grid documents when saving JSON templates.
# These are runtime-specific and get set dynamically in tests.
STRIP_FIELDS = {"id", "domain_id", "owner_id", "created_on", "modified_on"}


def _poll_for_completion(
    client,
    domain_id: str,
    resource_type: str,
    resource_id: str,
    timeout: int = 120,
    interval: float = 1,
) -> dict:
    """Poll the API until the resource reaches a terminal status."""
    url = f"/domains/{domain_id}/{resource_type}/{resource_id}"
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(
                f"{resource_type}/{resource_id} did not complete within {timeout}s"
            )

        response = client.get(url)
        if response.status_code != 200:
            pytest.fail(f"GET {url} returned {response.status_code}: {response.text}")

        resource = response.json()
        status = resource.get("status")
        progress = resource.get("progress")

        logger.info(
            f"{resource_type}/{resource_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return resource
        if status == "failed":
            error = resource.get("error") or {}
            pytest.fail(
                f"{resource_type}/{resource_id} failed: "
                f"{error.get('code')} - {error.get('message')}"
            )

        time.sleep(interval)


def _save_json_file(path: Path, template: dict) -> None:
    """Write a static JSON template with stable alphabetical key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(template, f, indent=2, default=str, sort_keys=True)
        f.write("\n")


def _save_json_template(grid: dict, static_name: str) -> None:
    """Save a completed grid document as a JSON template for griddle tests.

    Always overwrites so the template stays consistent with the regenerated
    zarr in GCS. Runtime-specific fields are stripped (see STRIP_FIELDS), so
    repeated runs against unchanged griddle output produce identical files.
    """
    out_path = STATIC_GRIDS_DIR / f"{static_name}.json"
    template = {k: v for k, v in grid.items() if k not in STRIP_FIELDS}
    _save_json_file(out_path, template)
    logger.info(f"Saved JSON template to {out_path}")


def _load_static_resource_template(resource_type: str, static_name: str) -> dict:
    """Load a previously-generated static resource JSON template."""
    config = STATIC_RESOURCE_TYPES[resource_type]
    path = config.template_dir / f"{static_name}.json"
    if not path.exists():
        pytest.fail(
            f"Static {resource_type} template {path} not found. "
            f"Run the base fixture test first to generate it."
        )
    with open(path) as f:
        return json.load(f)


def _register_static_resource(
    fs_client: firestore.Client,
    resource_type: str,
    static_name: str,
    owner_id: str,
    domain_id: str,
) -> str:
    """Temporarily register a static fixture as a Firestore resource.

    Creates a Firestore document that points to existing static data in GCS,
    with the correct owner_id and domain_id so the API's source validation
    passes. The document ID is the static_name itself (deterministic, no
    collision with uuid-based IDs).

    Returns the resource ID (== static_name) for use in API calls.
    """
    config = STATIC_RESOURCE_TYPES[resource_type]
    template = _load_static_resource_template(resource_type, static_name)
    template["id"] = static_name
    template["owner_id"] = owner_id
    template["domain_id"] = domain_id

    fs_client.collection(config.collection).document(static_name).set(template)
    logger.info(
        f"Registered static fixture {static_name} as Firestore {resource_type} doc"
    )
    return static_name


def _unregister_static_resource(
    fs_client: firestore.Client, resource_type: str, static_name: str
) -> None:
    """Remove a temporarily-registered static resource from Firestore."""
    config = STATIC_RESOURCE_TYPES[resource_type]
    fs_client.collection(config.collection).document(static_name).delete()
    logger.info(
        f"Unregistered static fixture {static_name} from Firestore {resource_type}"
    )


def _register_dependencies(
    fs_client: firestore.Client,
    dependencies: dict[str, list[str]] | None,
    owner_id: str,
    domain_id: str,
) -> list[tuple[str, str]]:
    """Register every dependency in Firestore. Returns (resource_type, ref) tuples for cleanup."""
    registered: list[tuple[str, str]] = []
    for resource_type, refs in (dependencies or {}).items():
        if resource_type not in STATIC_RESOURCE_TYPES:
            supported = ", ".join(sorted(STATIC_RESOURCE_TYPES))
            raise ValueError(
                f"Unsupported static resource dependency type {resource_type!r}. "
                f"Supported types: {supported}."
            )
        for ref in refs:
            _register_static_resource(
                fs_client, resource_type, ref, owner_id, domain_id
            )
            registered.append((resource_type, ref))
    return registered


@pytest.fixture
def create_static_fixture(firestore_client, test_owner_id):
    """Factory fixture that creates a static grid fixture in GCS.

    Creates a grid via the API, polls for completion, copies the zarr
    to a static path, saves a JSON template, then cleans up the
    temporary grid.

    Pass ``dependencies={"grids": [...], "inventories": [...]}`` listing every
    static fixture the API will look up during source validation. Each ref is
    temporarily registered as a Firestore doc (with the test user's owner_id
    and domain_id) and unregistered after the grid is created, regardless of
    success or failure.

    Use ``@pytest.mark.dependency`` on the test functions to control
    execution order for chained fixtures.
    """

    def _create(
        client,
        domain_id,
        endpoint,
        body,
        static_name,
        dependencies: dict[str, list[str]] | None = None,
    ):
        registered = _register_dependencies(
            firestore_client, dependencies, test_owner_id, domain_id
        )

        try:
            # Create the grid via the API
            url = f"/domains/{domain_id}{endpoint}"
            response = client.post(url, json=body, timeout=30.0)
            assert response.status_code == 201, (
                f"POST {url} returned {response.status_code}: {response.text}"
            )
            grid = response.json()
            grid_id = grid["id"]
            logger.info(f"Created grid {grid_id} via {url}")

            # Poll until completed
            completed_grid = _poll_for_completion(client, domain_id, "grids", grid_id)

            # Copy zarr to static path
            fs = gcsfs.GCSFileSystem()
            src = f"{GRIDS_BUCKET}/{grid_id}"
            dst = f"{GRIDS_BUCKET}/{static_name}"

            # Remove existing static data if present
            if fs.exists(dst):
                fs.rm(dst, recursive=True)

            fs.cp(src, dst, recursive=True)
            logger.info(f"Copied zarr gs://{src} -> gs://{dst}")

            # Save JSON template
            _save_json_template(completed_grid, static_name)

            # Clean up the temporary grid via the API
            delete_url = f"/domains/{domain_id}/grids/{grid_id}"
            del_response = client.delete(delete_url, timeout=30.0)
            logger.info(f"Deleted grid {grid_id}: {del_response.status_code}")

        finally:
            # Always clean up dependency registrations
            for resource_type, ref in registered:
                _unregister_static_resource(firestore_client, resource_type, ref)

    yield _create


def _save_inventory_json_template(inventory: dict, static_name: str) -> None:
    """Save a completed inventory document as a JSON template.

    Always overwrites so the template stays consistent with the regenerated
    parquet in GCS. Runtime-specific fields are stripped (see STRIP_FIELDS).
    """
    out_path = STATIC_INVENTORIES_DIR / f"{static_name}.json"
    template = {k: v for k, v in inventory.items() if k not in STRIP_FIELDS}
    _save_json_file(out_path, template)
    logger.info(f"Saved inventory JSON template to {out_path}")


@pytest.fixture
def create_static_inventory_fixture(firestore_client, test_owner_id):
    """Factory fixture that creates a static inventory fixture in GCS.

    Creates an inventory via the API, polls for completion, copies the
    parquet to a static path, saves a JSON template, then cleans up the
    temporary inventory.

    Pass ``dependencies={"grids": [...], "inventories": [...]}`` listing every
    static fixture the API will look up during source validation. Each ref is
    temporarily registered as a Firestore doc and unregistered after, regardless
    of success or failure.
    """

    def _create(
        client,
        domain_id,
        endpoint,
        body,
        static_name,
        dependencies: dict[str, list[str]] | None = None,
    ):
        registered = _register_dependencies(
            firestore_client, dependencies, test_owner_id, domain_id
        )

        try:
            # Create the inventory via the API
            url = f"/domains/{domain_id}{endpoint}"
            response = client.post(url, json=body, timeout=30.0)
            assert response.status_code == 201, (
                f"POST {url} returned {response.status_code}: {response.text}"
            )
            inventory = response.json()
            inventory_id = inventory["id"]
            logger.info(f"Created inventory {inventory_id} via {url}")

            # Poll until completed
            completed_inventory = _poll_for_completion(
                client, domain_id, "inventories", inventory_id
            )

            # Copy parquet to static path
            fs = gcsfs.GCSFileSystem()
            src = f"{INVENTORIES_BUCKET}/{inventory_id}"
            dst = f"{INVENTORIES_BUCKET}/{static_name}"

            # Remove existing static data if present
            if fs.exists(dst):
                fs.rm(dst, recursive=True)

            fs.cp(src, dst, recursive=True)
            logger.info(f"Copied parquet gs://{src} -> gs://{dst}")

            # Save JSON template
            _save_inventory_json_template(completed_inventory, static_name)

            # Clean up the temporary inventory via the API
            delete_url = f"/domains/{domain_id}/inventories/{inventory_id}"
            del_response = client.delete(delete_url, timeout=30.0)
            logger.info(f"Deleted inventory {inventory_id}: {del_response.status_code}")

        finally:
            # Always clean up dependency registrations
            for resource_type, ref in registered:
                _unregister_static_resource(firestore_client, resource_type, ref)

    yield _create


@pytest.fixture(scope="session")
def blue_mountain_domain(client):
    """Create the Blue Mountain domain via the API, clean up after.

    Uses EXAMPLE_WGS84_DEFAULT — the canonical Blue Mountain Recreation Area
    (~1 sq km near Missoula, Montana). The API auto-projects to UTM zone 11N.
    """
    response = client.post("/domains", json=EXAMPLE_WGS84_DEFAULT, timeout=30.0)
    assert response.status_code == 201, (
        f"POST /domains returned {response.status_code}: {response.text}"
    )
    domain = response.json()
    logger.info(f"Created Blue Mountain domain {domain['id']}")

    yield domain

    # Clean up
    del_response = client.delete(
        f"/domains/{domain['id']}", params={"force": True}, timeout=30.0
    )
    logger.info(
        f"Deleted Blue Mountain domain {domain['id']}: {del_response.status_code}"
    )


@pytest.fixture(scope="session")
def blue_mountain_padded_domain(client):
    """Same Blue Mountain extent, but the bbox is padded to a clean 2 m lattice.

    Use this for fixtures the QUIC-Fire export consumes — every 2 m
    Domain-anchored fetch lands on the same lattice as the existing 3D
    canopy voxels (which the voxelizer already snapped to integer-2 m
    coordinates).
    """
    payload = {**EXAMPLE_WGS84_DEFAULT, "pad_to_resolution": 2}
    response = client.post("/domains", json=payload, timeout=30.0)
    assert response.status_code == 201, (
        f"POST /domains returned {response.status_code}: {response.text}"
    )
    domain = response.json()
    logger.info(f"Created Blue Mountain padded (2 m) domain {domain['id']}")

    yield domain

    del_response = client.delete(
        f"/domains/{domain['id']}", params={"force": True}, timeout=30.0
    )
    logger.info(
        f"Deleted Blue Mountain padded domain {domain['id']}: "
        f"{del_response.status_code}"
    )
