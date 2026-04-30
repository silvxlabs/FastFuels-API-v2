"""
Integration tests for api/v2/resources/grids/resample/router.py

Tests the resample endpoint (POST /domains/{domain_id}/grids/resample).
These tests make real HTTP requests to the API and interact with Firestore.

The resample endpoint requires a source grid that:
1. Exists and is owned by the requesting user
2. Has status "completed"
3. Has a georeference
4. Belongs to the same domain as the URL path
"""

import pytest
from api.resources.grids.resample.examples import ALL_RESAMPLE_EXAMPLE_VALUES

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_domain_data, make_grid_data


@pytest.fixture(scope="session")
def complete_grid(firestore_client, domain_for_testing):
    """A complete grid with bands and georeference for use as a resample source."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Source grid for resample tests",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def pending_grid(firestore_client, domain_for_testing):
    """A grid with status "pending" (not yet complete)."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Pending grid for resample tests",
        status="pending",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def complete_grid_no_georeference(firestore_client, domain_for_testing):
    """A complete grid without a georeference."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Complete grid without georeference",
        status="completed",
    )
    grid_data["georeference"] = None
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def second_domain(firestore_client):
    """A second domain owned by test-owner for cross-domain tests."""
    domain_data = make_domain_data(name="Second Test Domain")
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def grid_in_different_domain(firestore_client, second_domain):
    """A complete grid with georeference in a different domain."""
    grid_data = make_grid_data(
        domain_id=second_domain["id"],
        name="Grid in different domain for resample tests",
        status="completed",
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


@pytest.fixture(scope="session")
def complete_3d_grid(firestore_client, domain_for_testing):
    """A complete 3D grid (tree/inventory-style) — cannot be resampled."""
    grid_data = make_grid_data(
        domain_id=domain_for_testing["id"],
        name="Complete 3D grid for resample guard tests",
        status="completed",
        source={
            "name": "inventory",
            "product": "tree",
            "description": "3D tree fuel grid from tree inventory voxelization",
            "source_inventory_id": "test-source-inv",
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
            "crown_profile_model": "purves",
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
        },
        bands=[
            {
                "key": "bulk_density.foliage.live",
                "type": "continuous",
                "unit": "kg/m³",
                "index": 0,
            },
        ],
        georeference={
            "crs": "EPSG:32611",
            "transform": (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
            "shape": (40, 500, 500),
            "z_resolution": 1.0,
            "z_origin": 0.0,
        },
    )
    doc_ref = firestore_client.collection(GRIDS_COLLECTION).document(grid_data["id"])
    doc_ref.set(grid_data)
    yield grid_data
    doc_ref.delete()


class TestCreateResample:
    """Test the POST /domains/{domain_id}/grids/resample endpoint."""

    # Happy path tests

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, complete_grid
    ):
        """Minimal request with required fields creates a resampled grid."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Check source
        assert data["source"]["name"] == "resample"
        assert data["source"]["source_grid_id"] == complete_grid["id"]
        assert data["source"]["target_resolution"] == 2.0
        assert data["source"]["method"] == "bilinear"

        # Check bands are present
        assert len(data["bands"]) > 0

    def test_request_with_metadata(self, client, domain_for_testing, complete_grid):
        """Request with name, description, and tags."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
            "name": "Resampled fuels at 2m",
            "description": "30m LANDFIRE resampled to 2m",
            "tags": ["resampled", "2m"],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Resampled fuels at 2m"
        assert data["description"] == "30m LANDFIRE resampled to 2m"
        assert data["tags"] == ["resampled", "2m"]

    def test_bands_propagated_from_source(
        self, client, domain_for_testing, complete_grid
    ):
        """Output band keys/types/units match source grid's bands exactly."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        source_bands = complete_grid["bands"]
        assert len(data["bands"]) == len(source_bands)
        for output_band, source_band in zip(data["bands"], source_bands):
            assert output_band["key"] == source_band["key"]
            assert output_band["type"] == source_band["type"]
            assert output_band["unit"] == source_band["unit"]

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, complete_grid
    ):
        """Georeference is None on creation (backend sets it after resampling)."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["georeference"] is None

    def test_method_overrides_in_source(
        self, client, domain_for_testing, complete_grid
    ):
        """Method overrides appear in source metadata."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
            "method_overrides": {"fbfm": "nearest"},
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201

        data = response.json()
        assert data["source"]["method_overrides"] == {"fbfm": "nearest"}

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, complete_grid
    ):
        """Response should not expose the owner_id field."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)
        assert response.status_code == 201

        data = response.json()
        assert "owner_id" not in data

    # Validation tests

    def test_nonexistent_source_grid_returns_404(self, client, domain_for_testing):
        """Non-existent source_grid_id returns 404."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": "00000000000000000000000000000000",
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_source_grid_not_complete_returns_422(
        self, client, domain_for_testing, pending_grid
    ):
        """Source grid with status != 'completed' returns 422."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": pending_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 422
        assert "status" in response.json()["detail"].lower()

    def test_source_grid_missing_georeference_returns_422(
        self, client, domain_for_testing, complete_grid_no_georeference
    ):
        """Source grid without a georeference returns 422."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid_no_georeference["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 422
        assert "georeference" in response.json()["detail"].lower()

    def test_source_grid_in_different_domain_returns_404(
        self, client, domain_for_testing, grid_in_different_domain
    ):
        """Source grid belonging to a different domain returns 404."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": grid_in_different_domain["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 404

    def test_invalid_method_override_key_returns_422(
        self, client, domain_for_testing, complete_grid
    ):
        """Override key not in source bands returns 422."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_grid["id"],
            "resolution": 2.0,
            "method_overrides": {"nonexistent_band": "nearest"},
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 422
        assert "nonexistent_band" in response.json()["detail"].lower()

    def test_3d_source_grid_returns_422(
        self, client, domain_for_testing, complete_3d_grid
    ):
        """Resampling a 3D grid is not supported — must 422 before enqueue."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            "source_grid_id": complete_3d_grid["id"],
            "resolution": 2.0,
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 422
        assert "3d" in response.json()["detail"].lower()

    def test_invalid_resolution_returns_422(
        self, client, domain_for_testing, complete_grid
    ):
        """Resolution below 1m returns 422."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        for bad_resolution in [-1.0, 0, 0.5]:
            request_body = {
                "source_grid_id": complete_grid["id"],
                "resolution": bad_resolution,
            }

            response = client.post(route, json=request_body)

            assert response.status_code == 422, (
                f"resolution={bad_resolution} should be rejected"
            )

    # Example tests

    @pytest.mark.parametrize("example_name,example_value", ALL_RESAMPLE_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        complete_grid,
        example_name,
        example_value,
    ):
        """Each documented resample example should successfully create a grid."""
        route = f"/domains/{domain_for_testing['id']}/grids/resample"
        request_body = {
            **example_value,
            "source_grid_id": complete_grid["id"],
        }

        response = client.post(route, json=request_body)

        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )

        data = response.json()
        assert data["source"]["name"] == "resample"
        assert data["source"]["source_grid_id"] == complete_grid["id"]
