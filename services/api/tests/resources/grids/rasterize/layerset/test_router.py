"""
Integration tests for api/v2/resources/grids/rasterize/layerset/router.py

Tests the layerset rasterize endpoint. These tests make real HTTP requests
to the local API and seed/clean real Firestore documents. They require:
- A running API server on http://127.0.0.1:8080
- ``INFRA_ENV=prod`` in `.env`
- ``TEST_API_KEY`` pre-seeded in the real Firestore ``keys-v2`` collection.

Per the project convention in services/api/tests/README.md, layerset/feature
ownership mismatches return 404 (not 403) to avoid leaking document existence.
"""

import pytest
from api.resources.grids.rasterize.layerset.examples import (
    ALL_LAYERSET_RASTERIZE_EXAMPLE_VALUES,
)

from lib.config import FEATURES_COLLECTION
from tests.fixtures import make_layerset_feature_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_feature(firestore_client, feature_data: dict) -> dict:
    doc_ref = firestore_client.collection(FEATURES_COLLECTION).document(
        feature_data["id"]
    )
    doc_ref.set(feature_data)
    return feature_data


@pytest.fixture(scope="session")
def caller_layerset(firestore_client, domain_for_testing):
    """A completed layerset Feature in the caller's domain, owned by the caller."""
    data = make_layerset_feature_data(domain_id=domain_for_testing["id"])
    _seed_feature(firestore_client, data)
    yield data
    firestore_client.collection(FEATURES_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def layerset_in_different_domain(firestore_client, second_domain):
    """A layerset in ``second_domain`` (same caller, different domain).

    Used to verify that calling the rasterize endpoint for
    ``domain_for_testing`` with a layerset_id from ``second_domain`` is
    rejected with 404 (domain mismatch).
    """
    data = make_layerset_feature_data(domain_id=second_domain["id"])
    _seed_feature(firestore_client, data)
    yield data
    firestore_client.collection(FEATURES_COLLECTION).document(data["id"]).delete()


@pytest.fixture(scope="session")
def layerset_owned_by_other_user(firestore_client, domain_with_different_owner):
    """A layerset owned by a different user, in their domain.

    Used to verify owner-mismatch returns 404.
    """
    data = make_layerset_feature_data(
        owner_id="different-owner",
        domain_id=domain_with_different_owner["id"],
    )
    _seed_feature(firestore_client, data)
    yield data
    firestore_client.collection(FEATURES_COLLECTION).document(data["id"]).delete()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateLayersetRasterize:
    """Test the POST /domains/{domain_id}/grids/rasterize/layerset endpoint."""

    def route(self, domain_id: str) -> str:
        return f"/domains/{domain_id}/grids/rasterize/layerset"

    # --- Happy path -------------------------------------------------------

    def test_minimal_request_creates_grid(
        self, client, domain_for_testing, caller_layerset
    ):
        """Minimal request creates a pending Grid with default overlap_method=mean."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"]},
        )

        assert response.status_code == 201, response.json()
        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32
        assert data["domain_id"] == domain_for_testing["id"]
        assert data["status"] == "pending"
        assert data["name"] == ""
        assert data["description"] == ""
        assert data["tags"] == []

        # Source discriminator + overlap method default
        assert data["source"]["name"] == "layerset"
        assert data["source"]["product"] == "layerset"
        assert data["source"]["layerset_id"] == caller_layerset["id"]
        assert data["source"]["overlap_method"] == "mean"
        assert data["source"]["extent_buffer_cells"] == 0

        # Bands are populated by the griddle worker after rasterization
        # (one entry per fuel_type × physical band). At create time the
        # list is empty because the API can't see the uploaded GeoJSON.
        assert data["bands"] == []

    def test_request_with_metadata(self, client, domain_for_testing, caller_layerset):
        body = {
            "layerset_id": caller_layerset["id"],
            "name": "Custom surface fuels",
            "description": "A custom layerset rasterize",
            "tags": ["layerset", "custom"],
        }
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Custom surface fuels"
        assert data["description"] == "A custom layerset rasterize"
        assert data["tags"] == ["layerset", "custom"]

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"]},
        )
        assert response.status_code == 201
        assert response.json()["georeference"] is None

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"]},
        )
        assert response.status_code == 201
        assert "owner_id" not in response.json()

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_LAYERSET_RASTERIZE_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        caller_layerset,
        example_name,
        example_value,
    ):
        """Each documented example creates a grid (with placeholder layerset_id swapped)."""
        body = {**example_value, "layerset_id": caller_layerset["id"]}
        response = client.post(self.route(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, (
            f"Example '{example_name}' failed with status {response.status_code}: "
            f"{response.json()}"
        )
        assert response.json()["source"]["name"] == "layerset"

    # --- Overlap method ---------------------------------------------------

    @pytest.mark.parametrize("method", ["mean", "max", "min"])
    def test_each_overlap_method_accepted(
        self, client, domain_for_testing, caller_layerset, method
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"], "overlap_method": method},
        )
        assert response.status_code == 201
        assert response.json()["source"]["overlap_method"] == method

    def test_invalid_overlap_method_returns_422(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"], "overlap_method": "bogus"},
        )
        assert response.status_code == 422

    # --- extent_buffer_cells ---------------------------------------------

    def test_extent_buffer_cells_defaults_to_zero(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": caller_layerset["id"]},
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == 0

    @pytest.mark.parametrize("buffer", [0, 10])
    def test_extent_buffer_cells_explicit_value_persisted(
        self, client, domain_for_testing, caller_layerset, buffer
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "layerset_id": caller_layerset["id"],
                "extent_buffer_cells": buffer,
            },
        )
        assert response.status_code == 201
        assert response.json()["source"]["extent_buffer_cells"] == buffer

    def test_extent_buffer_cells_negative_rejected(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "layerset_id": caller_layerset["id"],
                "extent_buffer_cells": -1,
            },
        )
        assert response.status_code == 422

    def test_extent_buffer_cells_above_maximum_rejected(
        self, client, domain_for_testing, caller_layerset
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "layerset_id": caller_layerset["id"],
                "extent_buffer_cells": 11,
            },
        )
        assert response.status_code == 422

    # --- Required fields --------------------------------------------------

    def test_missing_layerset_id_returns_422(self, client, domain_for_testing):
        response = client.post(self.route(domain_for_testing["id"]), json={})
        assert response.status_code == 422

    # --- Alignment --------------------------------------------------------

    def test_alignment_target_grid_missing_grid_id_rejected(
        self, client, domain_for_testing, caller_layerset
    ):
        """alignment.target='grid' without a grid_id is rejected by validation."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "layerset_id": caller_layerset["id"],
                "alignment": {"target": "grid"},
            },
        )
        assert response.status_code == 422

    # --- Ownership / domain mismatches (all 404, never 403) --------------

    def test_unknown_layerset_id_returns_404(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": "00000000000000000000000000000000"},
        )
        assert response.status_code == 404

    def test_layerset_owned_by_other_user_returns_404(
        self, client, domain_for_testing, layerset_owned_by_other_user
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": layerset_owned_by_other_user["id"]},
        )
        assert response.status_code == 404

    def test_layerset_in_different_domain_returns_404(
        self, client, domain_for_testing, layerset_in_different_domain
    ):
        """A layerset_id from another domain (same owner) is treated as not-found."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"layerset_id": layerset_in_different_domain["id"]},
        )
        assert response.status_code == 404

    def test_invalid_domain_returns_404(self, client, caller_layerset):
        response = client.post(
            self.route("00000000000000000000000000000000"),
            json={"layerset_id": caller_layerset["id"]},
        )
        assert response.status_code == 404

    def test_wrong_owner_domain_returns_404(
        self, client, domain_with_different_owner, caller_layerset
    ):
        response = client.post(
            self.route(domain_with_different_owner["id"]),
            json={"layerset_id": caller_layerset["id"]},
        )
        assert response.status_code == 404
