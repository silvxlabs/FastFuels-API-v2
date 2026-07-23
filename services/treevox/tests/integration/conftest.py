"""Shared fixtures for treevox integration tests.

Supports two execution modes:
- local: Directly calls process_grid_request() with a MockRequest.
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion.

Which mode runs matters more than it looks. Only ``deployed`` puts the work in
the built image on Cloud Run; ``local`` runs the handler in whatever process
pytest is in. For DUET that is the difference between testing the artifact we
ship — binary, libgfortran5, amd64, the real memory and timeout — and testing
the developer's laptop. CI sets ``DEPLOYMENT_ENV`` to the deploy environment, so
CI is always ``deployed``.

Two runner fixtures, one per handler:

- ``treevox_runner`` — stages a tree inventory parquet (either copied from a
  static fixture in GCS or uploaded from a caller-supplied DataFrame), creates
  domain/grid documents, runs treevox, polls Firestore, opens the 3D zarr, and
  cleans up on teardown.
- ``duet_runner`` — takes a completed 3D tree grid from ``treevox_runner`` and
  runs DUET against it, returning the 2D surface grid.

Real integration tests should use ``static_inventory="static-test-blue-mtn-..."``
to point at fixture data produced by ``services/api/tests/e2e``. The ``trees=``
DataFrame path remains for tests that need crafted inventories to exercise
specific semantics (overlap behavior, inventory-column biomass, etc.).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import NamedTuple
from uuid import uuid4

import gcsfs
import pandas as pd
import pytest
import xarray as xr

from lib.config import (
    DEPLOYMENT_ENV,
    DOMAINS_COLLECTION,
    GRIDS_BUCKET,
    GRIDS_COLLECTION,
    INVENTORIES_BUCKET,
)
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists, upload_file
from lib.testing import SHARED_TEST_DOMAINS_DIR, load_json


class TreevoxResult(NamedTuple):
    ds: xr.Dataset
    grid_id: str
    grid: dict


logger = logging.getLogger(__name__)

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


class MockRequest:
    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


def _stringify_coordinates(domain_data: dict) -> dict:
    """Firestore doesn't support nested arrays — mirror the API's coord stringify."""
    data = copy.deepcopy(domain_data)
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


def _run_treevox(grid_id: str) -> None:
    if DEPLOYMENT_ENV == "local":
        _run_local(grid_id)
    else:
        _run_deployed(grid_id)


def _run_local(grid_id: str) -> None:
    from treevox.main import process_grid_request

    request = MockRequest(data={"id": grid_id})
    response, status_code = process_grid_request(request)
    if status_code != 200:
        pytest.fail(f"process_grid_request returned {status_code}: {response}")


def _run_deployed(grid_id: str) -> None:
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import run_v2, tasks_v2
    from google.cloud.tasks_v2 import HttpMethod

    from lib.config import GCP_PROJECT, GCP_REGION, TREEVOX_QUEUE, TREEVOX_SERVICE

    async def _enqueue():
        run_client = run_v2.ServicesAsyncClient()
        service_name = (
            f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{TREEVOX_SERVICE}"
        )
        svc = await run_client.get_service(name=service_name)
        url = svc.uri

        tasks_client = tasks_v2.CloudTasksAsyncClient()
        parent = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, TREEVOX_QUEUE)
        task = tasks_v2.Task(
            name=tasks_client.task_path(
                GCP_PROJECT, GCP_REGION, TREEVOX_QUEUE, grid_id
            ),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": grid_id}).encode(),
            },
        )
        try:
            await tasks_client.create_task(parent=parent, task=task)
        except AlreadyExists:
            pass

    asyncio.run(_enqueue())


def _poll_for_completion(grid_id: str, timeout: int = 600) -> dict:
    start = time.time()
    interval = 2.0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            pytest.fail(f"Grid {grid_id} did not complete within {timeout}s")

        _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
        grid = snapshot.to_dict()
        status = grid.get("status")
        progress = grid.get("progress")

        logger.info(
            f"Grid {grid_id}: status={status}, progress={progress}, elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            return grid
        if status == "failed":
            return grid  # tests may want to inspect the failure

        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)


def _upload_parquet(df: pd.DataFrame, gcs_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path)
        upload_file(tmp_path, gcs_path)
    finally:
        os.unlink(tmp_path)


def _copy_static_inventory(static_name: str, inventory_id: str) -> None:
    """Copy a static inventory parquet to a test-specific path in GCS.

    Mirrors griddle's `source_grid` fixture pattern — static fixtures are
    generated once by `services/api/tests/e2e` and copied per-test so each
    run gets an isolated path that can be cleaned up.
    """
    fs = gcsfs.GCSFileSystem()
    src = f"{INVENTORIES_BUCKET}/{static_name}"
    dst = f"{INVENTORIES_BUCKET}/{inventory_id}"
    if not fs.exists(src):
        pytest.fail(
            f"Static inventory {src} not found. "
            f"Run services/api/tests/e2e to generate it."
        )
    fs.cp(src, dst, recursive=True)


@pytest.fixture
def treevox_runner():
    """Run treevox for a (domain, grid) pair and return the output dataset.

    One of ``static_inventory`` or ``trees`` must be supplied:
      - ``static_inventory="static-test-blue-mtn-pim-inventory"`` — copies
        the real fixture parquet (generated by services/api/tests/e2e) to a
        test path. This is the preferred path for integration tests.
      - ``trees=DataFrame`` — uploads an ad-hoc parquet. Use this only when
        a test needs to control tree placement precisely (overlap behavior,
        inventory-column biomass).

    Usage::

        def test_happy_path(treevox_runner):
            result = treevox_runner(
                static_inventory="static-test-blue-mtn-pim-inventory",
                bands=["volume_fraction", "bulk_density.foliage.live"],
            )
            assert "volume_fraction" in result.ds.data_vars
    """
    grid_ids: list[str] = []
    domain_ids: list[str] = []
    inventory_ids: list[str] = []
    datasets: list[xr.Dataset] = []

    def _run(
        static_inventory: str | None = None,
        trees: pd.DataFrame | None = None,
        domain_file: str = "blue_mtn.json",
        bands: list[str] | None = None,
        resolution: dict | None = None,
        crown_profile_model: str = "purves",
        biomass_source: dict | None = None,
        moisture_model: dict | None = None,
        inventory_id_override: str | None = None,
        expect_failed: bool = False,
        timeout: int = 600,
    ) -> TreevoxResult:
        bands = bands or ["volume_fraction", "bulk_density.foliage.live"]
        resolution = resolution or {"horizontal": 2.0, "vertical": 1.0}
        biomass_source = biomass_source or {
            "type": "allometry",
            "equations": "nsvb",
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        }

        # Decide how to stage the inventory parquet.
        if inventory_id_override is not None:
            inventory_id = inventory_id_override
        elif static_inventory is not None:
            inventory_id = f"test-{uuid4().hex}"
            _copy_static_inventory(static_inventory, inventory_id)
            inventory_ids.append(inventory_id)
        elif trees is not None:
            inventory_id = f"test-{uuid4().hex}"
            _upload_parquet(trees, f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
            inventory_ids.append(inventory_id)
        else:
            raise ValueError(
                "treevox_runner requires one of: static_inventory, trees, "
                "or inventory_id_override."
            )

        # Create domain document.
        domain_data = _stringify_coordinates(load_json(DOMAINS_DIR / domain_file))
        domain_id = f"test-{uuid4().hex}"
        domain_data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, domain_data)
        domain_ids.append(domain_id)

        # Create grid document mirroring what the API writes.
        grid_id = f"test-{uuid4().hex}"
        _BAND_DEFS = {
            "bulk_density.foliage.live": {"type": "continuous", "unit": "kg/m**3"},
            "bulk_density.foliage.dead": {"type": "continuous", "unit": "kg/m**3"},
            "bulk_density.branchwood.live": {"type": "continuous", "unit": "kg/m**3"},
            "bulk_density.branchwood.dead": {"type": "continuous", "unit": "kg/m**3"},
            "bulk_density.fine.live": {"type": "continuous", "unit": "kg/m**3"},
            "bulk_density.fine.dead": {"type": "continuous", "unit": "kg/m**3"},
            "fuel_moisture.live": {"type": "continuous", "unit": "%"},
            "fuel_moisture.dead": {"type": "continuous", "unit": "%"},
            "savr.foliage": {"type": "continuous", "unit": "1/m"},
            "spcd": {"type": "categorical", "unit": None},
            "tree_id": {"type": "categorical", "unit": None},
            "volume_fraction": {"type": "continuous", "unit": None},
        }
        band_defs = [
            {
                "key": b,
                "type": _BAND_DEFS[b]["type"],
                "unit": _BAND_DEFS[b]["unit"],
                "index": i,
            }
            for i, b in enumerate(bands)
        ]

        grid_data = {
            "id": grid_id,
            "domain_id": domain_id,
            "name": "",
            "description": "",
            "status": "pending",
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": {
                "operation": "voxelize",
                "input": "inventory",
                "entity": "tree",
                "source_inventory_id": inventory_id,
                "resolution": resolution,
                "bands": bands,
                "crown_profile_model": crown_profile_model,
                "biomass_source": biomass_source,
                "moisture_model": moisture_model,
            },
            "bands": band_defs,
            "modifications": [],
            "georeference": None,
            "chunks": None,
            "tags": [],
        }
        set_document(GRIDS_COLLECTION, grid_id, grid_data)
        grid_ids.append(grid_id)

        _run_treevox(grid_id)

        if DEPLOYMENT_ENV != "local":
            grid = _poll_for_completion(grid_id, timeout=timeout)
        else:
            _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
            grid = snapshot.to_dict()

        if expect_failed:
            assert grid["status"] == "failed", (
                f"Expected failed, got {grid['status']}: {grid.get('error')}"
            )
            return TreevoxResult(ds=xr.Dataset(), grid_id=grid_id, grid=grid)

        assert grid["status"] == "completed", (
            f"Expected completed, got {grid['status']}: {grid.get('error')}"
        )
        geo = grid["georeference"]
        assert geo is not None
        assert len(geo["shape"]) == 3  # (z, y, x)
        assert all(s > 0 for s in geo["shape"])

        ds = xr.open_zarr(
            f"gs://{GRIDS_BUCKET}/{grid_id}", consolidated=True, decode_coords="all"
        )
        datasets.append(ds)
        return TreevoxResult(ds=ds, grid_id=grid_id, grid=grid)

    yield _run

    for ds in datasets:
        ds.close()

    for grid_id in grid_ids:
        gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        try:
            delete_document(GRIDS_COLLECTION, grid_id)
        except Exception:
            pass

    for inventory_id in inventory_ids:
        gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)

    for domain_id in domain_ids:
        try:
            delete_document(DOMAINS_COLLECTION, domain_id)
        except Exception:
            pass


# DUET band metadata, mirroring what the API's duet schema writes onto the
# grid document. Duplicated here for the same reason `_BAND_DEFS` above is:
# treevox cannot import the API, and the fixture's job is to imitate the
# document the API would have written.
_DUET_UNITS = {"fuel_load": "kg/m**2", "fuel_depth": "m", "fuel_moisture": "%"}


def _duet_band_defs(bands: list[str]) -> list[dict]:
    defs = []
    for i, key in enumerate(bands):
        parameter = next(p for p in _DUET_UNITS if key.startswith(f"{p}."))
        defs.append(
            {
                "key": key,
                "type": "continuous",
                "unit": _DUET_UNITS[parameter],
                "index": i,
            }
        )
    return defs


@pytest.fixture
def duet_runner():
    """Run DUET against a completed 3D tree grid and return the 2D surface grid.

    The source grid must carry `bulk_density.foliage.live`, `spcd`, and
    `fuel_moisture.live` — build it with ``treevox_runner`` requesting those
    bands.

    Usage::

        def test_duet(treevox_runner, duet_runner):
            source = treevox_runner(
                static_inventory="static-test-blue-mtn-pim-inventory",
                bands=["bulk_density.foliage.live", "spcd", "fuel_moisture.live"],
                moisture_model={"live": {"method": "uniform", "value": 100.0}},
            )
            result = duet_runner(
                source_grid_id=source.grid_id,
                domain_id=source.grid["domain_id"],
                years_since_burn=25,
            )
    """
    grid_ids: list[str] = []
    datasets: list[xr.Dataset] = []

    def _run(
        source_grid_id: str,
        domain_id: str,
        bands: list[str] | None = None,
        years_since_burn: int = 25,
        wind_direction: int = 270,
        wind_variability: int = 30,
        calibration: dict | None = None,
        expect_failed: bool = False,
        timeout: int = 900,
    ) -> TreevoxResult:
        bands = bands or ["fuel_load.grass", "fuel_load.litter"]

        grid_id = f"test-{uuid4().hex}"
        source = {
            "operation": "duet",
            "input": "grid",
            "entity": "tree",
            "source_grid_id": source_grid_id,
            "years_since_burn": years_since_burn,
            "wind_direction": wind_direction,
            "wind_variability": wind_variability,
            "bands": bands,
        }
        if calibration is not None:
            source["calibration"] = calibration

        grid_data = {
            "id": grid_id,
            "domain_id": domain_id,
            "name": "",
            "description": "",
            "status": "pending",
            "created_on": datetime.now(),
            "modified_on": datetime.now(),
            "source": source,
            "bands": _duet_band_defs(bands),
            "modifications": [],
            "georeference": None,
            "chunks": None,
            "tags": [],
        }
        set_document(GRIDS_COLLECTION, grid_id, grid_data)
        grid_ids.append(grid_id)

        _run_treevox(grid_id)

        if DEPLOYMENT_ENV != "local":
            grid = _poll_for_completion(grid_id, timeout=timeout)
        else:
            _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
            grid = snapshot.to_dict()

        if expect_failed:
            assert grid["status"] == "failed", (
                f"Expected failed, got {grid['status']}: {grid.get('error')}"
            )
            return TreevoxResult(ds=xr.Dataset(), grid_id=grid_id, grid=grid)

        assert grid["status"] == "completed", (
            f"Expected completed, got {grid['status']}: {grid.get('error')}"
        )
        geo = grid["georeference"]
        assert geo is not None
        # DUET reads a 3D canopy and writes a 2D surface, so unlike every other
        # treevox output this georeference has no z axis.
        assert len(geo["shape"]) == 2  # (y, x)
        assert all(s > 0 for s in geo["shape"])

        ds = xr.open_zarr(
            f"gs://{GRIDS_BUCKET}/{grid_id}", consolidated=True, decode_coords="all"
        )
        datasets.append(ds)
        return TreevoxResult(ds=ds, grid_id=grid_id, grid=grid)

    yield _run

    for ds in datasets:
        ds.close()

    for grid_id in grid_ids:
        gcs_path = f"gs://{GRIDS_BUCKET}/{grid_id}"
        if exists(gcs_path):
            delete_directory(gcs_path)
        try:
            delete_document(GRIDS_COLLECTION, grid_id)
        except Exception:
            pass


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    """Cleanly shut down gcsfs sessions after all tests complete.

    Prevents RuntimeError during atexit when fsspec IO thread tries to close
    aiohttp session bound to a different loop.
    """
    yield

    import fsspec.asyn as fasyn
    import gcsfs

    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=5)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None

    gcsfs.GCSFileSystem.clear_instance_cache()
