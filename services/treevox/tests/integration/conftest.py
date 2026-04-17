"""Shared fixtures for treevox integration tests.

Supports two execution modes:
- local: Directly calls process_grid_request() with a MockRequest.
- deployed: Enqueues via Cloud Tasks, polls Firestore for completion.

The main fixture is ``treevox_runner``, which handles the full lifecycle:
uploads a tree inventory parquet, creates domain/grid documents, runs
treevox, polls, opens the 3D zarr, and cleans up on teardown.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

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
from lib.testing import SHARED_TEST_DOMAINS_DIR


class TreevoxResult(NamedTuple):
    ds: xr.Dataset
    grid_id: str
    grid: dict


logger = logging.getLogger(__name__)

DOMAINS_DIR = SHARED_TEST_DOMAINS_DIR


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


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


def _poll_for_completion(grid_id: str, timeout: int = 300) -> dict:
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


def _make_tiny_inventory_df(
    domain_gdf_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Build a 5-tree DataFrame within the given bounds.

    All trees are live (fia_status_code == 1) ponderosa pine (SPCD 122) with
    reasonable DBH/height/CR values so fastfuels-core allometry works.
    """
    minx, miny, maxx, maxy = domain_gdf_bounds
    mid_x = (minx + maxx) / 2
    mid_y = (miny + maxy) / 2
    # Scatter 5 trees around the center, far enough from edges that their
    # crowns render fully inside the padded grid.
    offsets = [(-50, -50), (-50, 50), (50, -50), (50, 50), (0, 0)]
    return pd.DataFrame(
        {
            "x": [mid_x + ox for ox, _ in offsets],
            "y": [mid_y + oy for _, oy in offsets],
            "fia_species_code": [122] * 5,
            "fia_status_code": [1] * 5,
            "dbh": [25.0, 30.0, 22.0, 28.0, 26.0],  # cm
            "height": [15.0, 18.0, 13.0, 17.0, 14.0],  # m
            "crown_ratio": [0.4, 0.45, 0.35, 0.5, 0.4],
        }
    )


def _upload_parquet(df: pd.DataFrame, gcs_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path)
        upload_file(tmp_path, gcs_path)
    finally:
        os.unlink(tmp_path)


def _domain_bounds_from_features(
    domain_data: dict,
) -> tuple[float, float, float, float]:
    """Extract total bounds from a GeoJSON FeatureCollection (pre-stringify)."""
    xs: list[float] = []
    ys: list[float] = []
    for feature in domain_data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        # coords is [[ring], ...] for a Polygon.
        for ring in coords:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


@pytest.fixture
def treevox_runner():
    """Run treevox for a (domain, grid) pair and return the output dataset.

    Handles the full lifecycle: uploads a tree inventory parquet, creates
    the domain and grid documents, runs treevox (local or deployed), polls
    Firestore, and opens the result zarr. Cleans up Firestore documents and
    all GCS data on teardown.

    Usage::

        def test_something(treevox_runner):
            result = treevox_runner(
                domain_file="blue_mtn.json",
                bands=["volume_fraction", "bulk_density.foliage"],
                resolution=(2.0, 2.0, 1.0),
            )
            assert "volume_fraction" in result.ds.data_vars
    """
    grid_ids: list[str] = []
    domain_ids: list[str] = []
    inventory_ids: list[str] = []
    datasets: list[xr.Dataset] = []

    def _run(
        domain_file: str = "blue_mtn.json",
        bands: list[str] | None = None,
        resolution: tuple[float, float, float] = (2.0, 2.0, 1.0),
        crown_profile_model: str = "purves",
        biomass_model: str = "nsvb",
        biomass_column: str | None = None,
        moisture_model: dict | None = None,
        trees: pd.DataFrame | None = None,
        inventory_id_override: str | None = None,
        expect_failed: bool = False,
        timeout: int = 300,
    ) -> TreevoxResult:
        bands = bands or ["volume_fraction", "bulk_density.foliage"]

        # 1. Upload tree inventory parquet (unless caller wants a bad inventory ID).
        inventory_id = inventory_id_override or f"test-{uuid4().hex}"
        if inventory_id_override is None:
            domain_data_raw = load_json(DOMAINS_DIR / domain_file)
            bounds = _domain_bounds_from_features(domain_data_raw)
            trees_df = trees if trees is not None else _make_tiny_inventory_df(bounds)
            _upload_parquet(trees_df, f"gs://{INVENTORIES_BUCKET}/{inventory_id}")
            inventory_ids.append(inventory_id)

        # 2. Create domain document.
        domain_data = _stringify_coordinates(load_json(DOMAINS_DIR / domain_file))
        domain_id = f"test-{uuid4().hex}"
        domain_data["id"] = domain_id
        set_document(DOMAINS_COLLECTION, domain_id, domain_data)
        domain_ids.append(domain_id)

        # 3. Create grid document mirroring what the API writes.
        grid_id = f"test-{uuid4().hex}"
        # Build band metadata inline (matches api/resources/grids/tree/schema.py).
        _BAND_DEFS = {
            "bulk_density.foliage": {"type": "continuous", "unit": "kg/m³"},
            "fuel_moisture.live": {"type": "continuous", "unit": "%"},
            "savr.foliage": {"type": "continuous", "unit": "m⁻¹"},
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
            "source": {
                "name": "inventory",
                "product": "tree",
                "description": "3D tree fuel grid from tree inventory voxelization",
                "source_inventory_id": inventory_id,
                "resolution": list(resolution),
                "bands": bands,
                "crown_profile_model": crown_profile_model,
                "biomass_model": biomass_model,
                "biomass_column": biomass_column,
                "moisture_model": moisture_model,
            },
            "bands": band_defs,
            "modifications": [],
            "georeference": None,
            "chunk_shape": None,
            "tags": [],
        }
        set_document(GRIDS_COLLECTION, grid_id, grid_data)
        grid_ids.append(grid_id)

        # 4. Run treevox.
        _run_treevox(grid_id)

        # 5. Read final grid state.
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

        # 6. Open the output zarr.
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
