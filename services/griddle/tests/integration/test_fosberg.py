"""
Integration test for the Fosberg 1-hr dead fuel moisture handler.

Builds two completed source grids on GCS — a topography grid (slope + aspect)
and a 2D leaflux surface irradiance grid — then runs the full griddle pipeline
(``process_grid_request``) over a pending Fosberg grid that references both, and
checks the output against a direct call to the core model.
"""

from __future__ import annotations

import copy
import json
from uuid import uuid4

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from fastfuels_core.fuel_moisture.fosberg import calculate_1hr_fuel_moisture
from griddle.main import process_grid_request

from lib.config import DOMAINS_COLLECTION, GRIDS_BUCKET, GRIDS_COLLECTION
from lib.firestore.documents import delete_document, get_document, set_document
from lib.gcs.blobs import delete_directory, exists
from lib.testing import SHARED_TEST_DOMAINS_DIR, load_json
from lib.zarr_utils import load_zarr, save_zarr

pytestmark = pytest.mark.integration

CRS = "EPSG:32611"
SURFACE_KEY = "irradiance.surface.relative"
OUTPUT_KEY = "fuel_moisture.dead.1hr"
OWNER = "test-owner"


class MockRequest:
    def __init__(self, data: dict):
        self._json = data
        self.headers = {}

    def get_json(self, silent: bool = False):
        return self._json


def _coords(height, width, res=10.0, x0=500000.0, y0=5000000.0):
    y = y0 - (np.arange(height) + 0.5) * res
    x = x0 + (np.arange(width) + 0.5) * res
    return y, x


def _make_domain() -> str:
    """Create a Blue Mountain domain doc (geometry is unused by the handler,
    but main.py loads and parses it)."""
    domain_data = copy.deepcopy(load_json(SHARED_TEST_DOMAINS_DIR / "blue_mtn.json"))
    for feature in domain_data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is not None and not isinstance(coords, str):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    domain_id = f"test-{uuid4().hex}"
    domain_data["id"] = domain_id
    set_document(DOMAINS_COLLECTION, domain_id, domain_data)
    return domain_id


def _save_source_grid(domain_id, bands, data_vars, shape, checksum) -> str:
    grid_id = f"test-{uuid4().hex}"
    height, width = shape
    y, x = _coords(height, width)
    ds = xr.Dataset(
        {k: (("y", "x"), v) for k, v in data_vars.items()},
        coords={"y": y, "x": x},
    ).rio.write_crs(CRS)
    save_zarr(f"gs://{GRIDS_BUCKET}/{grid_id}", ds, chunk_shape=(512, 512))
    set_document(
        GRIDS_COLLECTION,
        grid_id,
        {
            "id": grid_id,
            "domain_id": domain_id,
            "owner_id": OWNER,
            "status": "completed",
            "checksum": checksum,
            "source": {"name": "test-fixture"},
            "bands": [
                {"key": k, "type": "continuous", "unit": None, "index": i}
                for i, k in enumerate(bands)
            ],
        },
    )
    return grid_id


def test_fosberg_pipeline_produces_dead_1hr_surface():
    height, width = 24, 24
    rng = np.random.default_rng(0)
    slope = rng.uniform(0, 60, size=(height, width)).astype(np.float32)
    aspect = rng.uniform(0, 360, size=(height, width)).astype(np.float32)
    surface = rng.uniform(0, 1, size=(height, width)).astype(np.float32)

    created_grids: list[str] = []
    domain_id = _make_domain()
    try:
        topo_id = _save_source_grid(
            domain_id,
            ["slope", "aspect"],
            {"slope": slope, "aspect": aspect},
            (height, width),
            "topo-c",
        )
        created_grids.append(topo_id)
        irr_id = _save_source_grid(
            domain_id,
            [SURFACE_KEY],
            {SURFACE_KEY: surface},
            (height, width),
            "irr-c",
        )
        created_grids.append(irr_id)

        fosberg_id = f"test-{uuid4().hex}"
        created_grids.append(fosberg_id)
        set_document(
            GRIDS_COLLECTION,
            fosberg_id,
            {
                "id": fosberg_id,
                "domain_id": domain_id,
                "owner_id": OWNER,
                "status": "pending",
                "checksum": uuid4().hex,
                "source": {
                    "name": "fosberg",
                    "source_topography_grid_id": topo_id,
                    "source_topography_grid_checksum": "topo-c",
                    "source_irradiance_grid_id": irr_id,
                    "source_irradiance_grid_checksum": "irr-c",
                    "dry_bulb_temp": 75,
                    "relative_humidity": 30,
                    "time": 1200,
                    "month": "June",
                    "elevation": "near",
                },
                "modifications": [],
                "bands": [
                    {
                        "key": OUTPUT_KEY,
                        "type": "continuous",
                        "unit": "%",
                        "index": 0,
                    }
                ],
                "georeference": None,
                "chunks": {"shape": [512, 512], "count": None, "count_by_axis": None},
            },
        )

        response, status_code = process_grid_request(MockRequest({"id": fosberg_id}))
        assert status_code == 200, response

        _, snapshot = get_document(GRIDS_COLLECTION, fosberg_id)
        grid = snapshot.to_dict()
        assert grid["status"] == "completed", grid.get("error")
        geo = grid["georeference"]
        assert len(geo["shape"]) == 2
        assert tuple(geo["shape"]) == (height, width)
        assert CRS.split(":")[1] in geo["crs"]
        assert grid["bands"][0]["summary"] is not None

        ds = load_zarr(f"gs://{GRIDS_BUCKET}/{fosberg_id}")
        try:
            assert list(ds.data_vars) == [OUTPUT_KEY]
            assert ds[OUTPUT_KEY].dims == ("y", "x")
            # The handler classifies slope by percent grade; the source band is
            # degrees, so the reference converts it before calling the core.
            expected = calculate_1hr_fuel_moisture(
                dry_bulb_temp=75,
                relative_humidity=30,
                aspect=aspect.astype(float),
                slope=np.tan(np.radians(slope.astype(float))) * 100,
                time=1200,
                month="June",
                elevation=1,
                shading=1.0 - surface.astype(float),
            )
            np.testing.assert_allclose(
                ds[OUTPUT_KEY].values, expected, rtol=0, atol=1e-4
            )
        finally:
            ds.close()
    finally:
        for grid_id in created_grids:
            path = f"gs://{GRIDS_BUCKET}/{grid_id}"
            if exists(path):
                delete_directory(path)
            delete_document(GRIDS_COLLECTION, grid_id)
        delete_document(DOMAINS_COLLECTION, domain_id)
