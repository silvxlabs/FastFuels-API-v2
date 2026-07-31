"""
Integration tests for api/v2/resources/point_clouds/threedep/router.py

POST   /domains/{domain_id}/pointclouds/3dep
GET    /domains/{domain_id}/pointclouds/3dep/coverage

These make real HTTP requests to the API and real reads of the USGS 3DEP
acquisition catalog. Two domain fixtures with known, opposite coverage are used:

- Bondurant, WY (EPSG:32612): fully covered by one acquisition,
  ``WY_Southwest_1_2020``, at roughly 1.7M points.
- The shared test domain (EPSG:32611, Idaho panhandle): no 3DEP lidar at all.

Creation only enqueues a job, so these assert the pending resource and its
recorded provenance; the fetch itself is covered by lakitu's integration tests.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from api.resources.point_clouds.threedep.examples import ALL_3DEP_EXAMPLE_VALUES

from lib.config import DOMAINS_COLLECTION, POINT_CLOUDS_COLLECTION
from lib.testing import SHARED_TEST_DOMAINS_DIR

# The acquisition covering Bondurant. Pinning tests use it because a pinned
# name must both exist and overlap the domain.
BONDURANT_DATASET = "WY_Southwest_1_2020"


def _load_domain(path: Path, owner_id: str) -> dict:
    """Load a shared domain JSON and prepare it for Firestore."""
    with open(path) as f:
        data = json.load(f)
    data["id"] = f"test-{uuid.uuid4().hex}"
    data["owner_id"] = owner_id
    data["created_on"] = datetime.now()
    data["modified_on"] = datetime.now()
    # Stringify coordinates for Firestore
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if isinstance(coords, list):
            feature["geometry"]["coordinates"] = json.dumps(coords)
    return data


@pytest.fixture(scope="session")
def covered_domain(firestore_client, test_owner_id):
    """Bondurant, WY — fully covered by a single 3DEP acquisition."""
    domain_data = _load_domain(
        SHARED_TEST_DOMAINS_DIR / "bondurant.json", test_owner_id
    )
    doc_ref = firestore_client.collection(DOMAINS_COLLECTION).document(
        domain_data["id"]
    )
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()


@pytest.fixture
def cleanup_point_clouds(firestore_client):
    """Delete any point cloud created during a test."""
    created: list[str] = []
    yield created
    for point_cloud_id in created:
        firestore_client.collection(POINT_CLOUDS_COLLECTION).document(
            point_cloud_id
        ).delete()


def _create_route(domain_id: str) -> str:
    return f"/domains/{domain_id}/pointclouds/3dep"


def _coverage_route(domain_id: str) -> str:
    return f"/domains/{domain_id}/pointclouds/3dep/coverage"


class TestCoverageEndpoint:
    """Tests for the pre-flight coverage check."""

    def test_reports_coverage_for_a_covered_domain(self, client, covered_domain):
        response = client.get(_coverage_route(covered_domain["id"]))
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["available"] is True
        assert data["coverage_fraction"] == pytest.approx(1.0, abs=1e-3)
        assert data["estimated_point_count"] > 0
        assert data["exceeds_point_budget"] is False
        assert data["point_budget"] > 0

        names = [d["name"] for d in data["datasets"]]
        assert BONDURANT_DATASET in names

    def test_dataset_contributions_are_disjoint(self, client, covered_domain):
        """Contributions must sum to the coverage, never past it."""
        data = client.get(_coverage_route(covered_domain["id"])).json()
        total = sum(d["contribution_fraction"] for d in data["datasets"])
        assert total == pytest.approx(data["coverage_fraction"], abs=1e-6)
        assert total <= 1.0 + 1e-6

    def test_reports_no_coverage_for_an_uncovered_domain(
        self, client, domain_for_testing
    ):
        response = client.get(_coverage_route(domain_for_testing["id"]))
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["available"] is False
        assert data["coverage_fraction"] == 0.0
        assert data["datasets"] == []
        assert data["estimated_point_count"] == 0

    def test_requires_an_owned_domain(self, client, domain_with_different_owner):
        response = client.get(_coverage_route(domain_with_different_owner["id"]))
        assert response.status_code == 404

    def test_unknown_domain_returns_404(self, client):
        response = client.get(_coverage_route("does-not-exist"))
        assert response.status_code == 404


class TestCreate3DepPointCloud:
    """Tests for creating a point cloud from 3DEP."""

    def test_creates_pending_als_point_cloud(
        self, client, firestore_client, covered_domain, cleanup_point_clouds
    ):
        body = {"name": "Bondurant ALS", "tags": ["bondurant"]}
        response = client.post(_create_route(covered_domain["id"]), json=body)
        assert response.status_code == 201, response.text

        data = response.json()
        cleanup_point_clouds.append(data["id"])

        # 3DEP is airborne, so the type is set by the server, not the request.
        assert data["type"] == "als"
        assert data["status"] == "pending"
        assert data["name"] == "Bondurant ALS"
        assert data["domain_id"] == covered_domain["id"]
        assert data["georeference"] is None
        assert data["summary"] is None
        assert data["checksum"]

        doc = (
            firestore_client.collection(POINT_CLOUDS_COLLECTION)
            .document(data["id"])
            .get()
        )
        assert doc.exists
        assert doc.to_dict()["status"] == "pending"

    def test_records_resolved_provenance(
        self, client, covered_domain, cleanup_point_clouds
    ):
        """The chosen acquisitions are recorded so the fetch is reproducible."""
        response = client.post(_create_route(covered_domain["id"]), json={})
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_point_clouds.append(data["id"])

        source = data["source"]
        assert source["name"] == "3dep"
        assert source["datasets"] == [BONDURANT_DATASET]
        assert source["requested_datasets"] is None
        assert source["coverage_fraction"] == pytest.approx(1.0, abs=1e-3)

    def test_pinned_datasets_are_recorded(
        self, client, covered_domain, cleanup_point_clouds
    ):
        body = {"datasets": [BONDURANT_DATASET]}
        response = client.post(_create_route(covered_domain["id"]), json=body)
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_point_clouds.append(data["id"])

        assert data["source"]["requested_datasets"] == [BONDURANT_DATASET]
        assert data["source"]["datasets"] == [BONDURANT_DATASET]

    @pytest.mark.parametrize("example_name,body", ALL_3DEP_EXAMPLE_VALUES)
    def test_documented_examples_are_accepted(
        self, client, covered_domain, cleanup_point_clouds, example_name, body
    ):
        response = client.post(_create_route(covered_domain["id"]), json=body)
        assert response.status_code == 201, f"{example_name}: {response.text}"
        cleanup_point_clouds.append(response.json()["id"])

    def test_body_is_optional(self, client, covered_domain, cleanup_point_clouds):
        """Every field has a default, so an empty body is a valid request."""
        response = client.post(_create_route(covered_domain["id"]), json={})
        assert response.status_code == 201, response.text
        cleanup_point_clouds.append(response.json()["id"])


class TestCreateValidation:
    """Tests for requests the API refuses before creating anything."""

    def test_uncovered_domain_is_rejected(
        self, client, firestore_client, domain_for_testing
    ):
        response = client.post(_create_route(domain_for_testing["id"]), json={})
        assert response.status_code == 422, response.text
        assert "no usgs 3dep lidar" in response.json()["detail"].lower()

        # A rejected request must not leave a resource behind.
        docs = (
            firestore_client.collection(POINT_CLOUDS_COLLECTION)
            .where("domain_id", "==", domain_for_testing["id"])
            .get()
        )
        assert [d.id for d in docs] == []

    def test_unknown_pinned_dataset_is_rejected(self, client, covered_domain):
        response = client.post(
            _create_route(covered_domain["id"]),
            json={"datasets": ["NOT_A_REAL_ACQUISITION_2020"]},
        )
        assert response.status_code == 422, response.text
        assert "not a usgs 3dep" in response.json()["detail"].lower()

    def test_non_overlapping_pinned_dataset_is_rejected(self, client, covered_domain):
        """A real acquisition somewhere else is still unusable here."""
        response = client.post(
            _create_route(covered_domain["id"]),
            json={"datasets": ["MT_Helena_2012"]},
        )
        assert response.status_code == 422, response.text
        assert "does not overlap" in response.json()["detail"].lower()

    def test_requires_an_owned_domain(self, client, domain_with_different_owner):
        response = client.post(
            _create_route(domain_with_different_owner["id"]), json={}
        )
        assert response.status_code == 404

    def test_unknown_domain_returns_404(self, client):
        response = client.post(_create_route("does-not-exist"), json={})
        assert response.status_code == 404

    def test_type_is_not_accepted_in_the_body(
        self, client, covered_domain, cleanup_point_clouds
    ):
        """3DEP is airborne by definition, so there is no type to choose.

        The field is ignored rather than rejected, and the result is always
        als — this pins the contract so a stray `type` can never produce a tls
        point cloud.
        """
        response = client.post(
            _create_route(covered_domain["id"]), json={"type": "tls"}
        )
        assert response.status_code == 201, response.text
        data = response.json()
        cleanup_point_clouds.append(data["id"])
        assert data["type"] == "als"
