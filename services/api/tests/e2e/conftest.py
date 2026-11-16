"""
Fixtures for the API end-to-end suite.

Each test drives the full API -> Cloud Tasks -> worker pipeline: POST a
resource, poll until it completes, and return the completed resource. Chained
tests (PIM grid -> PIM inventory -> voxels; FBFM40 -> lookup) build on the
resources created earlier in the same session, so the suite never depends on
or writes to shared state.

By default nothing leaves the run: every resource lives in a session-scoped
domain that is force-deleted at session end. With ``E2E_PUBLISH_FIXTURES=1``
the suite also publishes each output to its ``static-test-*`` path in GCS and
writes the JSON template under ``services/lib/tests/shared_data``, which is how
the static fixtures other services' integration tests read are regenerated.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import gcsfs
import httpx
import pytest
from api.resources.domains.examples import EXAMPLE_WGS84_DEFAULT

from lib.config import (
    FEATURES_BUCKET,
    GRIDS_BUCKET,
    INVENTORIES_BUCKET,
    POINT_CLOUDS_BUCKET,
)
from lib.testing import (
    SHARED_TEST_DOMAINS_DIR,
    SHARED_TEST_FEATURES_DIR,
    SHARED_TEST_GRIDS_DIR,
    SHARED_TEST_INVENTORIES_DIR,
    SHARED_TEST_POINT_CLOUDS_DIR,
)
from tests.conftest import TEST_API_KEY, TEST_URL

logger = logging.getLogger(__name__)

STATIC_GRIDS_DIR = SHARED_TEST_GRIDS_DIR
STATIC_INVENTORIES_DIR = SHARED_TEST_INVENTORIES_DIR
STATIC_FEATURES_DIR = SHARED_TEST_FEATURES_DIR
STATIC_POINT_CLOUDS_DIR = SHARED_TEST_POINT_CLOUDS_DIR
E2E_CREATE_TIMEOUT_SECONDS = 240.0
# Per-request read timeout for completion-poll GETs. The client default (30s)
# is too tight when the service under test is cold-starting after a fresh
# deploy, so each poll GET gets a generous budget and transient transport
# errors are retried (see _poll_for_completion).
POLL_REQUEST_TIMEOUT_SECONDS = 60.0
# Per-request read timeout for fixture-cleanup DELETEs, and the overall
# budget for retrying one that hits a transient transport error (see
# _delete_with_retry).
DELETE_REQUEST_TIMEOUT_SECONDS = 30.0
DELETE_RETRY_TIMEOUT_SECONDS = 90.0


# Publishing overwrites the shared static fixtures that other services'
# integration tests read, so it only happens when asked for explicitly (#796).
PUBLISH_FIXTURES = os.environ.get("E2E_PUBLISH_FIXTURES") == "1"


@dataclass(frozen=True)
class ResourceType:
    path: str  # API path segment, e.g. /domains/{id}/{path}/{resource_id}
    bucket: str
    template_dir: Path
    # Features are one GeoParquet blob under {bucket}/{domain_id}/; the other
    # types are a directory under {bucket}/{resource_id}/.
    single_blob: bool = False


RESOURCE_TYPES = {
    "grids": ResourceType("grids", GRIDS_BUCKET, STATIC_GRIDS_DIR),
    "inventories": ResourceType(
        "inventories", INVENTORIES_BUCKET, STATIC_INVENTORIES_DIR
    ),
    "pointclouds": ResourceType(
        "pointclouds", POINT_CLOUDS_BUCKET, STATIC_POINT_CLOUDS_DIR
    ),
    "features": ResourceType(
        "features", FEATURES_BUCKET, STATIC_FEATURES_DIR, single_blob=True
    ),
}

# Fields to strip from resource documents when saving JSON templates.
# These are runtime-specific and get set dynamically in tests.
STRIP_FIELDS = {"id", "domain_id", "owner_id", "created_on", "modified_on", "checksum"}


def _poll_for_completion(
    client,
    domain_id: str,
    resource_type: str,
    resource_id: str,
    timeout: int = 300,
    interval: float = 1,
) -> dict:
    """Poll the API until the resource reaches a terminal status.

    Each poll GET uses a generous per-request read timeout, and transient
    transport errors (read timeouts, dropped connections) are logged and
    retried within the overall ``timeout`` budget rather than failing the test.
    This tolerates the slower cold-start responses that follow a fresh deploy
    of the service under test.
    """
    url = f"/domains/{domain_id}/{resource_type}/{resource_id}"
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(
                f"{resource_type}/{resource_id} did not complete within {timeout}s"
            )

        try:
            response = client.get(url, timeout=POLL_REQUEST_TIMEOUT_SECONDS)
        except httpx.TransportError as exc:
            logger.warning(
                f"Transient error polling {resource_type}/{resource_id} "
                f"(elapsed={elapsed:.0f}s); retrying: {type(exc).__name__}: {exc}"
            )
            time.sleep(interval)
            continue

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


def _delete_with_retry(
    client,
    url: str,
    params: dict | None = None,
    timeout: float = DELETE_RETRY_TIMEOUT_SECONDS,
    interval: float = 1,
):
    """DELETE a resource, retrying transient transport errors.

    Mirrors _poll_for_completion: each attempt uses a generous per-request
    read timeout, and transient transport errors (read timeouts, dropped
    connections) are logged and retried within the overall ``timeout``
    budget rather than failing the test.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"DELETE {url} did not succeed within {timeout}s")

        try:
            return client.delete(
                url, params=params, timeout=DELETE_REQUEST_TIMEOUT_SECONDS
            )
        except httpx.TransportError as exc:
            logger.warning(
                f"Transient error deleting {url} (elapsed={elapsed:.0f}s); "
                f"retrying: {type(exc).__name__}: {exc}"
            )
            time.sleep(interval)


def _save_json_file(path: Path, template: dict) -> None:
    """Write a static JSON template with stable alphabetical key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(template, f, indent=2, default=str, sort_keys=True)
        f.write("\n")


def _replace_ids(value, mapping: dict[str, str]):
    """Return ``value`` with every string equal to a key of ``mapping`` replaced.

    Walks dicts and lists, so it rewrites source IDs anywhere in a request
    body or a resource document. Matching is exact, never by substring.
    """
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, dict):
        return {k: _replace_ids(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_ids(v, mapping) for v in value]
    return value


def _resolve_dependencies(
    dependencies: dict[str, list[str]] | None,
    run_resources: dict[str, tuple[str, str]],
) -> dict[str, str]:
    """Map each dependency's static name to the resource this session created.

    Chained tests name their sources by static name. The upstream test records
    the resource it created under that name, and the downstream request uses
    that resource instead of the shared static fixture.
    """
    resolved: dict[str, str] = {}
    for resource_type, refs in (dependencies or {}).items():
        if resource_type not in RESOURCE_TYPES:
            supported = ", ".join(sorted(RESOURCE_TYPES))
            raise ValueError(
                f"Unsupported dependency type {resource_type!r}. "
                f"Supported types: {supported}."
            )
        for ref in refs:
            created_type, resource_id = run_resources.get(ref, (None, None))
            if created_type is None:
                pytest.fail(
                    f"{ref} was not created earlier in this session. Run the "
                    f"test that creates it in the same pytest session."
                )
            if created_type != resource_type:
                raise ValueError(
                    f"{ref} is a {created_type} resource, not {resource_type}."
                )
            resolved[ref] = resource_id
    return resolved


def _overwrite_static_dir(fs: gcsfs.GCSFileSystem, src: str, dst: str) -> None:
    """Overwrite a static GCS directory in place, without deleting it first.

    Sibling pipelines (griddle, exporter, treevox) read static fixtures while
    the e2e suite regenerates them, so the static path must never disappear.
    A recursive delete before the copy leaves the path missing for the whole
    rm+cp window (#311). Instead, copy each object over its existing
    counterpart (GCS object copies overwrite atomically), then remove only
    leftover objects that are not part of the new fixture.
    """
    src_files = sorted(fs.find(src))
    assert src_files, f"no objects found under gs://{src}"
    dst_files = [f"{dst}{f.removeprefix(src)}" for f in src_files]
    stale = sorted(set(fs.find(dst)) - set(dst_files))
    fs.cp(src_files, dst_files)
    if stale:
        fs.rm(stale)


def _publish(
    resource_type: str,
    domain_id: str,
    resource: dict,
    static_name: str,
    run_resources: dict[str, tuple[str, str]],
) -> None:
    """Copy a completed resource to its static path and write its template.

    Source IDs in the template that point at resources from this session are
    mapped back to their static names, so the committed template references
    the published fixtures rather than resources that no longer exist.
    """
    config = RESOURCE_TYPES[resource_type]
    fs = gcsfs.GCSFileSystem()
    resource_id = resource["id"]
    if config.single_blob:
        # Production stores features at {bucket}/{domain_id}/{feature_id}.parquet,
        # but the run's domain is deleted at session end, so the static blob
        # lives at the bucket root; the integration-test conftest copies it into
        # its test domain. A GCS object copy overwrites atomically, so readers
        # never see the blob missing (#311).
        src = f"{config.bucket}/{domain_id}/{resource_id}.parquet"
        dst = f"{config.bucket}/{static_name}.parquet"
        fs.cp(src, dst)
    else:
        src = f"{config.bucket}/{resource_id}"
        dst = f"{config.bucket}/{static_name}"
        _overwrite_static_dir(fs, src, dst)
    logger.info(f"Published gs://{src} -> gs://{dst}")

    static_names = {rid: name for name, (_, rid) in run_resources.items()}
    template = {k: v for k, v in resource.items() if k not in STRIP_FIELDS}
    out_path = config.template_dir / f"{static_name}.json"
    _save_json_file(out_path, _replace_ids(template, static_names))
    logger.info(f"Saved JSON template to {out_path}")


@pytest.fixture(scope="session")
def run_resources() -> dict[str, tuple[str, str]]:
    """Resources created in this session: static name -> (type, resource ID)."""
    return {}


def _resource_factory(
    resource_type: str,
    run_resources: dict[str, tuple[str, str]],
    default_timeout: int = 300,
):
    """Build a factory that creates one resource through the full pipeline.

    The factory POSTs ``body`` to ``endpoint``, polls until the resource
    completes, records it under ``static_name`` for later tests in the
    session, and returns the completed resource. It publishes the resource to
    ``static_name`` only when ``E2E_PUBLISH_FIXTURES=1``.

    ``dependencies={"grids": [...], "inventories": [...]}`` lists the static
    names the body uses as sources. Each is replaced with the resource the
    session created under that name, so a chained test must run after the test
    that creates its source (see ``@pytest.mark.dependency``).

    The resource is not deleted here: later tests may build on it. The
    session's domain is force-deleted at teardown, which removes it.
    """

    def _create(
        client,
        domain_id,
        endpoint,
        body,
        static_name,
        dependencies: dict[str, list[str]] | None = None,
        timeout: int = default_timeout,
    ):
        source_ids = _resolve_dependencies(dependencies, run_resources)
        url = f"/domains/{domain_id}{endpoint}"
        response = client.post(
            url,
            json=_replace_ids(body, source_ids),
            timeout=E2E_CREATE_TIMEOUT_SECONDS,
        )
        assert response.status_code == 201, (
            f"POST {url} returned {response.status_code}: {response.text}"
        )
        resource_id = response.json()["id"]
        logger.info(f"Created {resource_type} {resource_id} via {url}")

        completed = _poll_for_completion(
            client,
            domain_id,
            RESOURCE_TYPES[resource_type].path,
            resource_id,
            timeout=timeout,
        )
        run_resources[static_name] = (resource_type, resource_id)

        if PUBLISH_FIXTURES:
            _publish(resource_type, domain_id, completed, static_name, run_resources)

        return completed

    return _create


@pytest.fixture
def create_static_fixture(run_resources):
    """Factory that creates a grid through the full pipeline (see
    ``_resource_factory``)."""
    return _resource_factory("grids", run_resources)


@pytest.fixture
def create_static_inventory_fixture(run_resources):
    """Factory that creates an inventory through the full pipeline (see
    ``_resource_factory``)."""
    return _resource_factory("inventories", run_resources)


@pytest.fixture
def create_static_point_cloud_fixture(run_resources):
    """Factory that creates a point cloud through the full pipeline (see
    ``_resource_factory``).

    A 3DEP fetch is slower than a raster job (it streams octree nodes from the
    USGS archive), so the completion poll gets a longer budget than the default.
    """
    return _resource_factory("pointclouds", run_resources, default_timeout=900)


@pytest.fixture
def create_static_feature_fixture(run_resources):
    """Factory that creates a feature through the full pipeline (see
    ``_resource_factory``)."""
    return _resource_factory("features", run_resources)


@pytest.fixture(scope="session")
def client(test_owner_id):
    """Overrides the shared ``client`` without the skip-dispatch header: these
    tests need the worker jobs to actually run."""
    headers = {"API-KEY": TEST_API_KEY}
    with httpx.Client(base_url=TEST_URL, headers=headers, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def blackfoot_domain(client):
    """Create the Blackfoot domain via the API, clean up after.

    The payload is built from ``shared_data/domains/blackfoot.json`` — the same
    document downstream services seed — so a fixture generated here sits on
    exactly the geometry the existing ``static-test-blackfoot-*`` fixtures use.
    Authoring the coordinates separately here would let the two drift apart
    silently, and a point cloud that no longer registers with the blackfoot CHM
    is useless as an input to it.

    Unlike Blue Mountain this domain is a slightly rotated quadrilateral rather
    than an axis-aligned box, which is what makes it worth keeping: it exercises
    the point-in-polygon clip in the worker instead of the bounding-box fast
    path an axis-aligned domain takes.
    """
    # Read directly rather than through lib.testing.load_json: that helper
    # stamps Firestore timestamp fields, which have no place in a request body.
    with open(SHARED_TEST_DOMAINS_DIR / "blackfoot.json") as f:
        source = json.load(f)
    payload = {
        "type": "FeatureCollection",
        "features": source["features"],
        "crs": source["crs"],
        "name": "Blackfoot static test domain",
        "description": "3DEP point cloud fixture source (#329, #330).",
    }
    response = client.post("/domains", json=payload, timeout=E2E_CREATE_TIMEOUT_SECONDS)
    assert response.status_code == 201, (
        f"POST /domains returned {response.status_code}: {response.text}"
    )
    domain = response.json()
    logger.info(f"Created Blackfoot domain {domain['id']}")

    yield domain

    delete_url = f"/domains/{domain['id']}"
    del_response = _delete_with_retry(client, delete_url, params={"force": True})
    logger.info(f"Deleted Blackfoot domain {domain['id']}: {del_response.status_code}")


@pytest.fixture(scope="session")
def blue_mountain_domain(client):
    """Create the Blue Mountain domain via the API, clean up after.

    Uses EXAMPLE_WGS84_DEFAULT with `pad_to_resolution=2` — the canonical
    ~1 sq km Blue Mountain Recreation Area near Missoula, MT, with the
    UTM bbox snapped to a clean 2 m lattice so every 2 m Domain-anchored
    grid created from this Domain lands on a uniform integer-2 m corner.

    Session-scoped so chained tests can use resources created by earlier
    tests (a source must be in the same domain). The force delete at teardown
    removes every resource the session created in it.
    """
    payload = {**EXAMPLE_WGS84_DEFAULT, "pad_to_resolution": 2}
    response = client.post("/domains", json=payload, timeout=E2E_CREATE_TIMEOUT_SECONDS)
    assert response.status_code == 201, (
        f"POST /domains returned {response.status_code}: {response.text}"
    )
    domain = response.json()
    logger.info(f"Created Blue Mountain domain {domain['id']}")

    yield domain

    delete_url = f"/domains/{domain['id']}"
    del_response = _delete_with_retry(client, delete_url, params={"force": True})
    logger.info(
        f"Deleted Blue Mountain domain {domain['id']}: {del_response.status_code}"
    )
