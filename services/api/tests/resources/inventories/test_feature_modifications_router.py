"""
Create-time validation of inventory ``modifications[].conditions[].feature_id``.

Issue #282 (the inventory-side mirror of grid #279): a feature-source spatial
condition must reference a Feature that exists in the same domain and is in
``completed`` status, and that has to be checked at create time — not left to
fail in the worker.

Both live inventory endpoints that accept user-supplied ``modifications`` —
``tree/pim`` and ``tree/chm`` — delegate to the same
``validate_feature_modifications`` helper, so the ``endpoint`` fixture runs the
whole case sweep against each to prove both are wired in.

(The in-place ``{inventory_id}/modifications`` endpoint gets the same validation
in its router; it is exercised over HTTP in ``test_modifications_router.py``.)
"""

import pytest

from lib.config import FEATURES_COLLECTION, GRIDS_COLLECTION
from tests.fixtures import make_feature_data, make_grid_data

UNKNOWN_ID = "00000000000000000000000000000000"


# Source-grid fixtures: pim needs a completed PIM grid, chm a completed canopy
# grid. Both are just the source the endpoint requires; the feature is what
# varies across tests.


@pytest.fixture(scope="session")
def pim_grid(firestore_client, domain_for_testing):
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        status="completed",
        source={"name": "pim", "product": "treemap", "version": "2022"},
        bands=[{"key": "tm_id", "type": "categorical", "unit": None, "index": 0}],
    )
    ref = firestore_client.collection(GRIDS_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(scope="session")
def canopy_grid(firestore_client, domain_for_testing):
    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        status="completed",
        source={"name": "canopy", "product": "meta"},
        bands=[{"key": "chm", "type": "continuous", "unit": "m", "index": 0}],
    )
    ref = firestore_client.collection(GRIDS_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(params=["pim", "chm"])
def endpoint(request, domain_for_testing, pim_grid, canopy_grid):
    """The two live endpoints, each as ``(url, base_body)`` where base_body is
    the minimal valid request before ``modifications`` is added."""
    domain_id = domain_for_testing["id"]
    if request.param == "pim":
        url = f"/domains/{domain_id}/inventories/tree/pim"
        return url, {"source_pim_grid_id": pim_grid["id"], "seed": 42}
    url = f"/domains/{domain_id}/inventories/tree/chm"
    return url, {"source_chm_grid_id": canopy_grid["id"]}


# Feature fixtures: one per validator failure branch, plus the happy path.


@pytest.fixture(scope="session")
def completed_feature(firestore_client, domain_for_testing):
    data = make_feature_data(domain_id=domain_for_testing["id"], status="completed")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(scope="session")
def pending_feature(firestore_client, domain_for_testing):
    data = make_feature_data(domain_id=domain_for_testing["id"], status="pending")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


@pytest.fixture(scope="session")
def feature_in_different_domain(firestore_client, second_domain):
    data = make_feature_data(domain_id=second_domain["id"], status="completed")
    ref = firestore_client.collection(FEATURES_COLLECTION).document(data["id"])
    ref.set(data)
    yield data
    ref.delete()


def _request(base_body: dict, feature_id: str) -> dict:
    """Add a single feature-source modification to a base request body."""
    return {
        **base_body,
        "modifications": [
            {
                "conditions": [
                    {
                        "source": "feature",
                        "operator": "intersects",
                        "feature_id": feature_id,
                    }
                ],
                "actions": [{"modifier": "remove"}],
            }
        ],
    }


def test_unknown_feature_id_returns_422(client, endpoint):
    url, base = endpoint
    response = client.post(url, json=_request(base, UNKNOWN_ID))
    assert response.status_code == 422, response.json()
    assert UNKNOWN_ID in response.json()["detail"]


def test_feature_in_different_domain_returns_422(
    client, endpoint, feature_in_different_domain
):
    url, base = endpoint
    feature_id = feature_in_different_domain["id"]
    response = client.post(url, json=_request(base, feature_id))
    assert response.status_code == 422
    assert feature_id in response.json()["detail"]


def test_pending_feature_returns_422(client, endpoint, pending_feature):
    url, base = endpoint
    feature_id = pending_feature["id"]
    response = client.post(url, json=_request(base, feature_id))
    assert response.status_code == 422
    assert feature_id in response.json()["detail"]


def test_completed_feature_succeeds(client, endpoint, completed_feature):
    url, base = endpoint
    feature_id = completed_feature["id"]
    response = client.post(url, json=_request(base, feature_id))
    assert response.status_code == 201, response.json()
    condition = response.json()["modifications"][0]["conditions"][0]
    assert condition["feature_id"] == feature_id
