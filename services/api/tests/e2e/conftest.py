"""
Fixtures for generating static test data via the full API pipeline.

The ``create_static_fixture`` fixture drives the end-to-end flow:
POST grid -> poll Firestore -> copy zarr to static path -> save JSON template.

For chained fixtures (e.g., resample depends on lookup which depends on
LANDFIRE), any ``static-test-*`` values found in the request body are
automatically registered as temporary Firestore grid docs so the API's
source validation passes. Use ``@pytest.mark.dependency`` to control
test ordering.
"""

import json
import logging
import time
from pathlib import Path

import gcsfs
import pytest
from google.cloud import firestore

from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION

logger = logging.getLogger(__name__)

GRIDDLE_TEST_DATA = (
    Path(__file__).resolve().parents[3] / "griddle" / "tests" / "data" / "grids"
)

STATIC_PREFIX = "static-test-"

# Fields to strip from grid documents when saving JSON templates.
# These are runtime-specific and get set dynamically in tests.
STRIP_FIELDS = {"id", "domain_id", "owner_id", "created_on", "modified_on"}


def _find_static_refs(obj) -> list[str]:
    """Find all static-test-* string values in a nested dict/list."""
    refs = []
    if isinstance(obj, str) and obj.startswith(STATIC_PREFIX):
        refs.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            refs.extend(_find_static_refs(v))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(_find_static_refs(item))
    return refs


def _poll_for_completion(
    fs_client: firestore.Client, grid_id: str, timeout: int = 300
) -> dict:
    """Poll Firestore until the grid reaches a terminal status.

    Uses exponential backoff starting at 2s, maxing at 10s.
    """
    start = time.time()
    interval = 1.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Grid {grid_id} did not complete within {timeout}s")

        doc = fs_client.collection(GRIDS_COLLECTION).document(grid_id).get()
        if not doc.exists:
            pytest.fail(f"Grid document {grid_id} does not exist")

        grid = doc.to_dict()
        status = grid.get("status")
        progress = grid.get("progress")

        logger.info(
            f"Grid {grid_id}: status={status}, progress={progress}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return grid
        if status == "failed":
            error = grid.get("error", {})
            pytest.fail(
                f"Grid {grid_id} failed: {error.get('code')} - {error.get('message')}"
            )

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)


def _save_json_template(grid: dict, static_name: str) -> None:
    """Save a completed grid document as a JSON template for griddle tests."""
    template = {k: v for k, v in grid.items() if k not in STRIP_FIELDS}
    out_path = GRIDDLE_TEST_DATA / f"{static_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(template, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Saved JSON template to {out_path}")


def _load_static_template(static_name: str) -> dict:
    """Load a previously-generated static fixture JSON template."""
    path = GRIDDLE_TEST_DATA / f"{static_name}.json"
    if not path.exists():
        pytest.fail(
            f"Static template {path} not found. "
            f"Run the base fixture test first to generate it."
        )
    with open(path) as f:
        return json.load(f)


def _register_static_as_grid(
    fs_client: firestore.Client,
    static_name: str,
    owner_id: str,
    domain_id: str,
) -> str:
    """Temporarily register a static fixture as a completed grid in Firestore.

    Creates a Firestore document that points to the existing static zarr in
    GCS, with the correct owner_id and domain_id so the API's source grid
    validation passes. The document ID is the static_name itself (deterministic,
    no collision with uuid-based IDs).

    Returns the grid_id (== static_name) for use as source_grid_id in API calls.
    """
    template = _load_static_template(static_name)
    template["id"] = static_name
    template["owner_id"] = owner_id
    template["domain_id"] = domain_id

    fs_client.collection(GRIDS_COLLECTION).document(static_name).set(template)
    logger.info(f"Registered static fixture {static_name} as Firestore grid doc")
    return static_name


def _unregister_static_grid(fs_client: firestore.Client, grid_id: str) -> None:
    """Remove a temporarily-registered static grid from Firestore."""
    fs_client.collection(GRIDS_COLLECTION).document(grid_id).delete()
    logger.info(f"Unregistered static fixture {grid_id} from Firestore")


@pytest.fixture
def create_static_fixture(firestore_client, test_owner_id):
    """Factory fixture that creates a static grid fixture in GCS.

    Creates a grid via the API, polls for completion, copies the zarr
    to a static path, saves a JSON template, then cleans up the
    temporary grid.

    Any ``static-test-*`` values found in ``body`` are automatically
    registered as temporary Firestore grid docs (with the test user's
    owner_id and domain_id) so the API can validate them as source grids.
    They are cleaned up after the grid is created, regardless of success
    or failure.

    Use ``@pytest.mark.dependency`` on the test functions to control
    execution order for chained fixtures.
    """

    def _create(client, domain_id, endpoint, body, static_name):
        # Auto-detect static-test-* references in the request body
        static_refs = _find_static_refs(body)

        # Register dependencies as temporary Firestore grid docs
        registered = []
        for ref in static_refs:
            _register_static_as_grid(firestore_client, ref, test_owner_id, domain_id)
            registered.append(ref)

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
            completed_grid = _poll_for_completion(firestore_client, grid_id)

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
            for ref in registered:
                _unregister_static_grid(firestore_client, ref)

    yield _create


@pytest.fixture(scope="session")
def blue_mountain_domain(client):
    """Create the Blue Mountain domain via the API, clean up after.

    Blue Mountain Recreation Area (~1 sq km near Missoula, Montana).
    Geographic coordinates — the API auto-projects to UTM zone 11N.
    """
    body = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-113.98819, 46.87174],
                            [-114.00274, 46.87159],
                            [-114.00304, 46.86407],
                            [-113.98849, 46.86421],
                            [-113.98819, 46.87174],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
        "name": "Blue Mountain",
        "description": "Blue Mountain Recreation Area near Missoula, Montana.",
    }
    response = client.post("/domains", json=body, timeout=30.0)
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
