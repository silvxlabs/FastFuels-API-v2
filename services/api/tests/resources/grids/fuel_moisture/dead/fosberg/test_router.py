"""
Integration tests for the Fosberg 1-hr dead fuel moisture router.

Tests POST /domains/{domain_id}/grids/fuel-moisture/dead/fosberg against a
live local API server, interacting with Firestore and Cloud Tasks.
"""

import pytest
from api.resources.grids.fuel_moisture.dead.fosberg.examples import (
    ALL_FOSBERG_EXAMPLE_VALUES,
)

from lib.config import GRIDS_COLLECTION
from tests.fixtures import make_grid_data

SURFACE = "irradiance.surface.relative"


@pytest.fixture
def grid_factory(firestore_client):
    """Create source-grid documents in Firestore and clean them up after."""
    created = []

    def _make(
        domain_id,
        *,
        status="completed",
        bands=("slope", "aspect"),
        shape=(40, 40),
        checksum="src-checksum",
    ):
        data = make_grid_data(domain_id=domain_id, name="source grid", status=status)
        data["bands"] = [{"key": key} for key in bands]
        data["georeference"] = {"shape": list(shape)}
        data["checksum"] = checksum
        firestore_client.collection(GRIDS_COLLECTION).document(data["id"]).set(data)
        created.append(data["id"])
        return data

    yield _make

    for grid_id in created:
        firestore_client.collection(GRIDS_COLLECTION).document(grid_id).delete()


@pytest.fixture
def topography_grid(grid_factory, domain_for_testing):
    """A completed 2D grid with slope + aspect bands."""
    return grid_factory(
        domain_for_testing["id"],
        bands=("elevation", "slope", "aspect"),
        shape=(40, 40),
        checksum="topo-checksum",
    )


@pytest.fixture
def irradiance_grid(grid_factory, domain_for_testing):
    """A completed 2D leaflux surface irradiance grid."""
    return grid_factory(
        domain_for_testing["id"],
        bands=(SURFACE,),
        shape=(40, 40),
        checksum="irr-checksum",
    )


class TestCreateFosbergFuelMoistureGrid:
    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/fuel-moisture/dead/fosberg"

    def _body(self, topo, irr, **overrides):
        body = {
            "source_topography_grid_id": topo["id"],
            "source_irradiance_grid_id": irr["id"],
            "dry_bulb_temp": 75,
            "relative_humidity": 30,
            "time": 1200,
            "month": "June",
        }
        body.update(overrides)
        return body

    # --- Success paths ---

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=self._body(topography_grid, irradiance_grid),
        )
        assert response.status_code == 201, response.json()

        data = response.json()
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["georeference"] is None
        assert data["modifications"] == []

        source = data["source"]
        assert source["name"] == "fosberg"
        assert source["source_topography_grid_id"] == topography_grid["id"]
        assert source["source_irradiance_grid_id"] == irradiance_grid["id"]
        assert source["source_topography_grid_checksum"] == topography_grid["checksum"]
        assert source["source_irradiance_grid_checksum"] == irradiance_grid["checksum"]
        assert source["dry_bulb_temp"] == 75
        assert source["relative_humidity"] == 30
        assert source["time"] == 1200
        assert source["month"] == "June"
        assert source["elevation"] == "near"

        assert len(data["bands"]) == 1
        assert data["bands"][0]["key"] == "fuel_moisture.dead.1hr"
        assert data["bands"][0]["type"] == "continuous"
        assert data["bands"][0]["unit"] == "%"
        assert data["bands"][0]["index"] == 0

    def test_full_request_with_elevation(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        body = self._body(
            topography_grid,
            irradiance_grid,
            name="Peak burn 1-hr DFMC",
            tags=["fuel-moisture"],
            dry_bulb_temp=82,
            relative_humidity=20,
            time=1400,
            month="August",
            elevation="above",
        )
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.json()
        data = response.json()
        assert data["name"] == "Peak burn 1-hr DFMC"
        assert data["tags"] == ["fuel-moisture"]
        assert data["source"]["elevation"] == "above"
        assert data["source"]["month"] == "August"

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json=self._body(topography_grid, irradiance_grid),
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    # --- Domain validation ---

    def test_invalid_domain_returns_404(self, client, topography_grid, irradiance_grid):
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json=self._body(topography_grid, irradiance_grid),
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, topography_grid, irradiance_grid
    ):
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json=self._body(topography_grid, irradiance_grid),
        )
        assert response.status_code == 404

    # --- Topography grid validation ---

    def test_nonexistent_topography_returns_404(
        self, client, domain_for_testing, irradiance_grid
    ):
        body = self._body({"id": "00000000000000000000000000000000"}, irradiance_grid)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_topography_not_completed_returns_422(
        self, client, domain_for_testing, grid_factory, irradiance_grid
    ):
        pending = grid_factory(
            domain_for_testing["id"],
            status="pending",
            bands=("slope", "aspect"),
        )
        body = self._body(pending, irradiance_grid)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_topography_missing_slope_returns_422(
        self, client, domain_for_testing, grid_factory, irradiance_grid
    ):
        no_slope = grid_factory(domain_for_testing["id"], bands=("aspect",))
        body = self._body(no_slope, irradiance_grid)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert "slope" in response.json()["detail"]

    def test_topography_not_2d_returns_422(
        self, client, domain_for_testing, grid_factory, irradiance_grid
    ):
        three_d = grid_factory(
            domain_for_testing["id"], bands=("slope", "aspect"), shape=(6, 40, 40)
        )
        body = self._body(three_d, irradiance_grid)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    # --- Irradiance grid validation ---

    def test_nonexistent_irradiance_returns_404(
        self, client, domain_for_testing, topography_grid
    ):
        body = self._body(topography_grid, {"id": "00000000000000000000000000000000"})
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 404

    def test_irradiance_not_completed_returns_422(
        self, client, domain_for_testing, topography_grid, grid_factory
    ):
        pending = grid_factory(
            domain_for_testing["id"], status="pending", bands=(SURFACE,)
        )
        body = self._body(topography_grid, pending)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_irradiance_missing_surface_band_returns_422(
        self, client, domain_for_testing, topography_grid, grid_factory
    ):
        no_surface = grid_factory(
            domain_for_testing["id"], bands=("irradiance.canopy.relative",)
        )
        body = self._body(topography_grid, no_surface)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert SURFACE in response.json()["detail"]

    # --- Request body validation ---

    @pytest.mark.parametrize(
        "field",
        [
            "source_topography_grid_id",
            "source_irradiance_grid_id",
            "dry_bulb_temp",
            "relative_humidity",
            "time",
            "month",
        ],
    )
    def test_missing_required_field_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid, field
    ):
        body = self._body(topography_grid, irradiance_grid)
        del body[field]
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_temp_below_min_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        body = self._body(topography_grid, irradiance_grid, dry_bulb_temp=9)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    @pytest.mark.parametrize("rh", [-1, 101])
    def test_relative_humidity_out_of_range_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid, rh
    ):
        body = self._body(topography_grid, irradiance_grid, relative_humidity=rh)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    @pytest.mark.parametrize("time", [700, 799, 1960, 2100])
    def test_time_out_of_range_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid, time
    ):
        body = self._body(topography_grid, irradiance_grid, time=time)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_month_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        body = self._body(topography_grid, irradiance_grid, month="Smarch")
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_invalid_elevation_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        body = self._body(topography_grid, irradiance_grid, elevation="sea-level")
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_unknown_field_returns_422(
        self, client, domain_for_testing, topography_grid, irradiance_grid
    ):
        body = self._body(topography_grid, irradiance_grid, resolution=10)
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    # --- Documented examples ---

    @pytest.mark.parametrize("example_name,example_value", ALL_FOSBERG_EXAMPLE_VALUES)
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        topography_grid,
        irradiance_grid,
        example_name,
        example_value,
    ):
        body = {**example_value}
        if body.get("source_topography_grid_id") == "TOPOGRAPHY_GRID_ID":
            body["source_topography_grid_id"] = topography_grid["id"]
        if body.get("source_irradiance_grid_id") == "IRRADIANCE_GRID_ID":
            body["source_irradiance_grid_id"] = irradiance_grid["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status "
            f"{response.status_code}: {response.json()}"
        )
        assert response.json()["source"]["name"] == "fosberg"
