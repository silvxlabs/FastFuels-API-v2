"""
End-to-end tests that generate static test data for backend services.

Each test creates a grid through the full API -> Cloud Tasks -> backend
service pipeline, then copies the output zarr to a well-known static path in
GCS and saves a JSON template for use in integration tests.

Run manually when fixture data needs (re)generating:
    cd services/api && uv run pytest tests/e2e/ -v --log-cli-level=INFO

Tests use @pytest.mark.dependency to enforce execution order for chained
fixtures (e.g., resample depends on LANDFIRE existing first).
"""

import pytest

# Blue Mountain domain fixtures


@pytest.mark.dependency()
def test_create_blue_mtn_landfire_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    """Create static LANDFIRE FBFM40 fixture on Blue Mountain domain."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/fbfm40/landfire",
        body={},
        static_name="static-test-blue-mtn-landfire-fbfm40",
    )


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
    """Create static 3D tree voxel grid fixture from Blue Mountain PIM inventory."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/tree/inventory",
        body={
            "source_inventory_id": "static-test-blue-mtn-pim-inventory",
            "resolution": [20.0, 20.0, 5.0],
            "bands": [
                "volume_fraction",
                "bulk_density.foliage.live",
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
        endpoint="/grids/chm/naip",
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
        body={
            "source_chm_grid_id": "static-test-blue-mtn-naip-chm",
        },
        static_name="static-test-blue-mtn-chm-inventory",
        dependencies={"grids": ["static-test-blue-mtn-naip-chm"]},
    )


@pytest.mark.dependency()
def test_create_blue_mtn_landfire_topography(
    create_static_fixture, client, blue_mountain_domain
):
    """Create static LANDFIRE topography fixture on Blue Mountain domain."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/topography/landfire",
        body={},
        static_name="static-test-blue-mtn-landfire-topography",
    )


# TODO: Test is failing. Come back to this after fixing the underlying issue (in griddle resample handler).
@pytest.mark.skip
@pytest.mark.dependency(depends=["test_create_blue_mtn_landfire_fbfm40"])
def test_create_blue_mtn_fbfm40_2m(create_static_fixture, client, blue_mountain_domain):
    """Create static 2m resampled FBFM40 fixture on Blue Mountain domain."""
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/resample",
        body={
            "source_grid_id": "static-test-blue-mtn-landfire-fbfm40",
            "resolution": 2,
            "method": "nearest",
        },
        static_name="static-test-blue-mtn-fbfm40-2m",
        dependencies={"grids": ["static-test-blue-mtn-landfire-fbfm40"]},
    )
