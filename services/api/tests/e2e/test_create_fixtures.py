"""
End-to-end tests that generate static test data for backend services.

Each test creates a grid or inventory through the full API → Cloud Tasks →
backend service pipeline, copies the output to a well-known static path in
GCS, and saves a JSON template for use in integration tests.

Run manually when fixture data needs (re)generating:
    cd services/api && uv run pytest tests/e2e/ -v --log-cli-level=INFO

Tests use @pytest.mark.dependency to enforce execution order for chained
fixtures (PIM treemap → PIM inventory → tree voxels; FBFM40 → lookup).

`blue_mountain_domain` is `pad_to_resolution=2`, so every 2 m
Domain-anchored fetch lands on origin `(720226, 5190646)` — the same
lattice the tree-inventory voxelizer snaps to.
"""

import logging
import tempfile

import pandas as pd
import pytest
import requests

from lib.config import INVENTORIES_BUCKET, UPLOADS_BUCKET
from lib.gcs import delete_directory, exists
from tests.e2e.conftest import (
    E2E_CREATE_TIMEOUT_SECONDS,
    _poll_for_completion,
    _save_inventory_json_template,
)

logger = logging.getLogger(__name__)

# Tree coordinates within the Blue Mountain domain (EPSG:32611, UTM zone 11N)
# Domain bounds: x=[720228, 721534], y=[5189763, 5190645]
_UPLOAD_SAMPLE_X = [720500.0, 720700.0, 720900.0]
_UPLOAD_SAMPLE_Y = [5190000.0, 5190100.0, 5190200.0]
_UPLOAD_SAMPLE_HEIGHT = [10.0, 15.0, 20.0]


@pytest.mark.dependency()
def test_create_blue_mtn_upload_inventory_csv(client, blue_mountain_domain):
    """Create static upload inventory fixture on Blue Mountain domain.

    Full flow: POST → PUT CSV to signed URL → assert file in UPLOADS_BUCKET →
    Eventarc → uploader → assert Parquet matches uploaded data →
    copy parquet to static path → save JSON template.
    """
    import dask.dataframe as dd
    import gcsfs

    domain_id = blue_mountain_domain["id"]
    inventory_id = None

    try:
        response = client.post(
            f"/domains/{domain_id}/inventories/tree/upload",
            json={"format": "csv"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        inventory_id = data["inventory"]["id"]
        object_name = data["inventory"]["source"]["object_name"]
        signed_url = data["upload"]["url"]
        content_type = data["upload"]["content_type"]
        max_size_bytes = data["upload"]["max_size_bytes"]
        logger.info(f"Created upload inventory {inventory_id}")

        df = pd.DataFrame(
            {
                "x": _UPLOAD_SAMPLE_X,
                "y": _UPLOAD_SAMPLE_Y,
                "height": _UPLOAD_SAMPLE_HEIGHT,
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        with open(tmp_path, "rb") as f:
            put_response = requests.put(
                signed_url,
                data=f,
                headers={
                    "Content-Type": content_type,
                    "x-goog-content-length-range": f"0,{max_size_bytes}",
                },
                timeout=30,
            )
        assert put_response.status_code == 200, (
            f"PUT to signed URL failed: {put_response.status_code} {put_response.text}"
        )
        logger.info(f"PUT CSV to signed URL: {put_response.status_code}")

        # File must be in UPLOADS_BUCKET before Eventarc picks it up
        assert exists(f"gs://{UPLOADS_BUCKET}/{object_name}"), (
            f"Uploaded file not found at gs://{UPLOADS_BUCKET}/{object_name}"
        )

        completed = _poll_for_completion(
            client,
            domain_id,
            "inventories",
            inventory_id,
            timeout=E2E_CREATE_TIMEOUT_SECONDS,
        )

        # Uploader deletes the staged file after processing
        assert not exists(f"gs://{UPLOADS_BUCKET}/{object_name}"), (
            "Staged file should have been deleted from UPLOADS_BUCKET after processing"
        )

        # Parquet content must match what we uploaded
        parquet_df = dd.read_parquet(
            f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
        ).compute()
        assert len(parquet_df) == len(_UPLOAD_SAMPLE_X)
        assert list(parquet_df["x"]) == _UPLOAD_SAMPLE_X
        assert list(parquet_df["y"]) == _UPLOAD_SAMPLE_Y
        assert list(parquet_df["height"]) == _UPLOAD_SAMPLE_HEIGHT

        fs = gcsfs.GCSFileSystem()
        src = f"{INVENTORIES_BUCKET}/{inventory_id}"
        dst = f"{INVENTORIES_BUCKET}/static-test-blue-mtn-upload-inventory"
        if fs.exists(dst):
            fs.rm(dst, recursive=True)
        fs.cp(src, dst, recursive=True)
        logger.info(f"Copied parquet gs://{src} -> gs://{dst}")

        _save_inventory_json_template(
            completed, "static-test-blue-mtn-upload-inventory"
        )

        del_response = client.delete(
            f"/domains/{domain_id}/inventories/{inventory_id}", timeout=30
        )
        logger.info(f"Deleted inventory {inventory_id}: {del_response.status_code}")

    except Exception:
        if inventory_id:
            gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
            if exists(gcs_path):
                delete_directory(gcs_path)
            client.delete(
                f"/domains/{domain_id}/inventories/{inventory_id}", timeout=30
            )
        raise


@pytest.mark.dependency()
def test_create_blue_mtn_pim_treemap(
    create_static_fixture, client, blue_mountain_domain
):
    """Create static TreeMap PIM fixture on Blue Mountain domain."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/pim/treemap",
        body={},
        static_name="static-test-blue-mtn-pim-treemap",
    )


@pytest.mark.dependency(depends=["test_create_blue_mtn_pim_treemap"])
def test_create_blue_mtn_pim_inventory(
    create_static_inventory_fixture, client, blue_mountain_domain
):
    """Create static PIM inventory fixture on Blue Mountain domain."""
    create_static_inventory_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/inventories/tree/pim",
        body={
            "source_pim_grid_id": "static-test-blue-mtn-pim-treemap",
            "seed": 898870608,
        },
        static_name="static-test-blue-mtn-pim-inventory",
        dependencies={"grids": ["static-test-blue-mtn-pim-treemap"]},
    )


@pytest.mark.dependency(depends=["test_create_blue_mtn_pim_inventory"])
def test_create_blue_mtn_tree_inventory_voxels(
    create_static_fixture, client, blue_mountain_domain
):
    """Create static 3D tree voxel grid fixture from Blue Mountain PIM inventory.

    Bands cover both general 3D-grid testing (volume_fraction,
    bulk_density.foliage.live) and the canopy roles QUIC-Fire export needs
    (fuel_moisture.live, savr.foliage).
    """
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/tree/inventory",
        body={
            "source_inventory_id": "static-test-blue-mtn-pim-inventory",
            "resolution": {"horizontal": 2, "vertical": 1},
            "bands": [
                "volume_fraction",
                "bulk_density.foliage.live",
                "fuel_moisture.live",
                "savr.foliage",
            ],
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "seed": 42,
        },
        static_name="static-test-blue-mtn-tree-inventory-voxels",
        dependencies={"inventories": ["static-test-blue-mtn-pim-inventory"]},
    )


@pytest.mark.dependency()
def test_create_blue_mtn_naip_chm(create_static_fixture, client, blue_mountain_domain):
    """Create static NAIP CHM fixture on Blue Mountain domain."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/canopy/naip",
        body={},
        static_name="static-test-blue-mtn-naip-chm",
    )


@pytest.mark.dependency(depends=["test_create_blue_mtn_naip_chm"])
def test_create_blue_mtn_chm_inventory(
    create_static_inventory_fixture, client, blue_mountain_domain
):
    """Create static CHM inventory fixture on Blue Mountain domain."""
    create_static_inventory_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/inventories/tree/chm",
        body={"source_chm_grid_id": "static-test-blue-mtn-naip-chm"},
        static_name="static-test-blue-mtn-chm-inventory",
        dependencies={"grids": ["static-test-blue-mtn-naip-chm"]},
    )


@pytest.mark.dependency()
def test_create_blue_mtn_landfire_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    """Create LANDFIRE FBFM40 fixture at 2 m, Domain-anchored.

    Exercises the inline `alignment.target="domain", resolution=2` path on
    the LANDFIRE FBFM40 handler — no separate resample step.
    """
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/fbfm40/landfire",
        body={
            "alignment": {"target": "domain", "resolution": 2, "method": "nearest"},
        },
        static_name="static-test-blue-mtn-landfire-fbfm40",
    )


@pytest.mark.dependency(depends=["test_create_blue_mtn_landfire_fbfm40"])
def test_create_blue_mtn_lookup_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    """Create FBFM40 lookup grid at 2 m with the three surface roles
    QUIC-Fire needs (fuel_load.1hr, fuel_depth, savr.1hr)."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/lookup/fbfm40",
        body={
            "source_grid_id": "static-test-blue-mtn-landfire-fbfm40",
            "bands": ["fuel_load.1hr", "fuel_depth", "savr.1hr"],
        },
        static_name="static-test-blue-mtn-lookup-fbfm40",
        dependencies={"grids": ["static-test-blue-mtn-landfire-fbfm40"]},
    )


@pytest.mark.dependency()
def test_create_blue_mtn_landfire_topography(
    create_static_fixture, client, blue_mountain_domain
):
    """Create LANDFIRE topography fixture at 2 m with all three bands
    (elevation, slope, aspect). QUIC-Fire only consumes `elevation`, but
    the multi-band shape is also exercised by exporter integration tests."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/topography/landfire",
        body={
            "alignment": {"target": "domain", "resolution": 2},
            "bands": ["elevation", "slope", "aspect"],
        },
        static_name="static-test-blue-mtn-landfire-topography",
    )


@pytest.mark.dependency()
def test_create_blue_mtn_uniform_moisture(
    create_static_fixture, client, blue_mountain_domain
):
    """Create uniform surface-moisture fixture at 2 m with
    `fuel_moisture.1hr = 6.0 %`."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/uniform",
        body={
            "resolution": 2,
            "bands": [{"key": "fuel_moisture.1hr", "value": 6.0}],
        },
        static_name="static-test-blue-mtn-uniform-moisture",
    )
