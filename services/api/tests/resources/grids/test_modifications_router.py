"""
Cross-router validation tests for ``modifications[].conditions[].feature_id``.

Every router that accepts a ``modifications`` list with feature-source spatial
conditions must reject features that are missing, owned by another user, in
another domain, or not yet completed. The same case sweep is parametrized
over every endpoint so a regression on any single router is caught here
rather than in per-product test files.

Endpoints under test (every grid POST that accepts user-supplied
``modifications``):

- ``POST /domains/{id}/grids/fbfm40/landfire``
- ``POST /domains/{id}/grids/fccs/landfire``
- ``POST /domains/{id}/grids/topography/landfire``
- ``POST /domains/{id}/grids/topography/3dep``
- ``POST /domains/{id}/grids/pim/treemap``
- ``POST /domains/{id}/grids/canopy/meta``
- ``POST /domains/{id}/grids/canopy/naip``
- ``POST /domains/{id}/grids/canopy/landfire``
- ``POST /domains/{id}/grids/lookup/fbfm40``
- ``POST /domains/{id}/grids/rasterize/layerset``
- ``POST /domains/{id}/grids/resample``
- ``POST /domains/{id}/grids/uniform``

3D voxelize endpoints and upload endpoints do not accept user-supplied
``modifications`` and are excluded.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from lib.config import FEATURES_COLLECTION
from tests.fixtures import (
    make_feature_data,
    make_grid_data,
    make_layerset_feature_data,
)

# Endpoint registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointSpec:
    """Defines a single endpoint that accepts user-supplied ``modifications``."""

    name: str
    path_suffix: str
    extra_body: Callable[[dict], dict]

    def url(self, domain_id: str) -> str:
        return f"/domains/{domain_id}/grids/{self.path_suffix}"


def _empty_extra(_helpers: dict) -> dict:
    return {}


def _resample_extra(helpers: dict) -> dict:
    return {
        "source_grid_id": helpers["source_grid_id"],
        "alignment": {"target": "domain", "resolution": 2.0},
    }


def _layerset_extra(helpers: dict) -> dict:
    return {"layerset_id": helpers["layerset_id"]}


def _uniform_extra(_helpers: dict) -> dict:
    return {
        "resolution": 2.0,
        "bands": [{"key": "fuel_moisture.1hr", "value": 6.0}],
    }


def _lookup_extra(helpers: dict) -> dict:
    return {
        "source_grid_id": helpers["fbfm40_grid_id"],
        "bands": ["fuel_load.1hr"],
    }


ENDPOINTS = [
    EndpointSpec("fbfm40_landfire", "fbfm40/landfire", _empty_extra),
    EndpointSpec("fccs_landfire", "fccs/landfire", _empty_extra),
    EndpointSpec("topography_landfire", "topography/landfire", _empty_extra),
    EndpointSpec("topography_3dep", "topography/3dep", _empty_extra),
    EndpointSpec("pim_treemap", "pim/treemap", _empty_extra),
    EndpointSpec("chm_meta", "canopy/meta", _empty_extra),
    EndpointSpec("chm_naip", "canopy/naip", _empty_extra),
    EndpointSpec("canopy_landfire", "canopy/landfire", _empty_extra),
    EndpointSpec("lookup_fbfm40", "lookup/fbfm40", _lookup_extra),
    EndpointSpec("rasterize_layerset", "rasterize/layerset", _layerset_extra),
    EndpointSpec("resample", "resample", _resample_extra),
    EndpointSpec("uniform", "uniform", _uniform_extra),
]


# Feature fixtures
# ---------------------------------------------------------------------------


def _seed(firestore_client, collection: str, data: dict) -> dict:
    firestore_client.collection(collection).document(data["id"]).set(data)
    return data


def _delete(firestore_client, collection: str, doc_id: str) -> None:
    firestore_client.collection(collection).document(doc_id).delete()


@pytest.fixture(scope="session")
def completed_feature(firestore_client, domain_for_testing):
    """Feature in the caller's domain, owned by the caller, status=completed."""
    data = make_feature_data(domain_id=domain_for_testing["id"], status="completed")
    _seed(firestore_client, FEATURES_COLLECTION, data)
    yield data
    _delete(firestore_client, FEATURES_COLLECTION, data["id"])


@pytest.fixture(scope="session")
def pending_feature(firestore_client, domain_for_testing):
    data = make_feature_data(domain_id=domain_for_testing["id"], status="pending")
    _seed(firestore_client, FEATURES_COLLECTION, data)
    yield data
    _delete(firestore_client, FEATURES_COLLECTION, data["id"])


@pytest.fixture(scope="session")
def feature_in_different_domain(firestore_client, second_domain):
    data = make_feature_data(domain_id=second_domain["id"], status="completed")
    _seed(firestore_client, FEATURES_COLLECTION, data)
    yield data
    _delete(firestore_client, FEATURES_COLLECTION, data["id"])


@pytest.fixture(scope="session")
def feature_owned_by_other_user(firestore_client, domain_with_different_owner):
    data = make_feature_data(
        owner_id="different-owner",
        domain_id=domain_with_different_owner["id"],
        status="completed",
    )
    _seed(firestore_client, FEATURES_COLLECTION, data)
    yield data
    _delete(firestore_client, FEATURES_COLLECTION, data["id"])


# Endpoint-specific support fixtures (layerset for rasterize, source grid for
# resample / lookup). Other endpoints take an empty body.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def layerset_feature(firestore_client, domain_for_testing):
    data = make_layerset_feature_data(domain_id=domain_for_testing["id"])
    _seed(firestore_client, FEATURES_COLLECTION, data)
    yield data
    _delete(firestore_client, FEATURES_COLLECTION, data["id"])


@pytest.fixture(scope="session")
def fbfm40_source_grid(firestore_client, domain_for_testing):
    """A completed FBFM40 grid usable as the source for /lookup/fbfm40."""
    from lib.config import GRIDS_COLLECTION

    data = make_grid_data(
        domain_id=domain_for_testing["id"],
        status="completed",
        bands=[{"key": "fbfm", "type": "categorical", "unit": None, "index": 0}],
        source={"name": "landfire", "product": "fbfm40", "version": "2024"},
    )
    _seed(firestore_client, GRIDS_COLLECTION, data)
    yield data
    _delete(firestore_client, GRIDS_COLLECTION, data["id"])


@pytest.fixture
def endpoint_helpers(complete_grid, layerset_feature, fbfm40_source_grid):
    """Per-test bundle of ids the endpoint-specific ``extra_body`` callers
    need. Function-scoped so that each parametrized test gets a fresh dict
    even though the underlying fixtures are session-scoped."""
    return {
        "source_grid_id": complete_grid["id"],
        "layerset_id": layerset_feature["id"],
        "fbfm40_grid_id": fbfm40_source_grid["id"],
    }


# Request body helpers
# ---------------------------------------------------------------------------


def _feature_modification(feature_id: str) -> dict:
    """Build a feature-source modification. The action's band is arbitrary —
    the API only validates conditions[].feature_id at create time; downstream
    band-vs-grid compatibility is enforced by griddle."""
    return {
        "conditions": [
            {
                "source": "feature",
                "operator": "intersects",
                "feature_id": feature_id,
            }
        ],
        "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
    }


def _body(spec: EndpointSpec, feature_id: str, helpers: dict) -> dict:
    return {
        **spec.extra_body(helpers),
        "modifications": [_feature_modification(feature_id)],
    }


# Inline-geometry coordinates are a nested array, which Firestore cannot store.
# This polygon exercises the write-side stringification + read-side parse.
INLINE_POLYGON_COORDS = [
    [
        [-120.0, 38.0],
        [-119.5, 38.0],
        [-119.5, 38.5],
        [-120.0, 38.5],
        [-120.0, 38.0],
    ]
]


def _geometry_modification() -> dict:
    """Build an inline-geometry (source="geometry") modification with a
    Polygon. No `crs` → defaults to the domain CRS; the create path does not
    validate inline geometry against the domain, so an arbitrary polygon is
    accepted at create time."""
    return {
        "conditions": [
            {
                "source": "geometry",
                "operator": "within",
                "geometry": {"type": "Polygon", "coordinates": INLINE_POLYGON_COORDS},
            }
        ],
        "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
    }


def _geometry_body(spec: EndpointSpec, helpers: dict) -> dict:
    return {
        **spec.extra_body(helpers),
        "modifications": [_geometry_modification()],
    }


def _first_geometry_coords(grid: dict):
    """Return the coordinates of the first geometry-source condition in a grid
    response, or fail if none is present."""
    for modification in grid["modifications"]:
        for condition in modification["conditions"]:
            if condition.get("source") == "geometry":
                return condition["geometry"]["coordinates"]
    raise AssertionError("response had no geometry-source condition")


# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ENDPOINTS, ids=lambda s: s.name)
class TestFeatureModificationValidation:
    """The shared feature-modification validator must reject every invalid
    case on every endpoint that accepts user-supplied ``modifications``.

    The grid is rejected at create-time (422) before the document is
    persisted to Firestore or the griddle task is enqueued — see issue
    #279.
    """

    def test_unknown_feature_id_returns_422(
        self, client, domain_for_testing, endpoint_helpers, spec
    ):
        body = _body(spec, "00000000000000000000000000000000", endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)
        assert response.status_code == 422, response.json()
        assert "00000000000000000000000000000000" in response.json()["detail"]

    def test_feature_in_different_domain_returns_422(
        self,
        client,
        domain_for_testing,
        endpoint_helpers,
        feature_in_different_domain,
        spec,
    ):
        body = _body(spec, feature_in_different_domain["id"], endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert feature_in_different_domain["id"] in response.json()["detail"]

    def test_feature_owned_by_other_user_returns_422(
        self,
        client,
        domain_for_testing,
        endpoint_helpers,
        feature_owned_by_other_user,
        spec,
    ):
        body = _body(spec, feature_owned_by_other_user["id"], endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)
        assert response.status_code == 422

    def test_pending_feature_returns_422(
        self,
        client,
        domain_for_testing,
        endpoint_helpers,
        pending_feature,
        spec,
    ):
        body = _body(spec, pending_feature["id"], endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)
        assert response.status_code == 422
        assert pending_feature["id"] in response.json()["detail"]

    def test_valid_feature_succeeds(
        self,
        client,
        domain_for_testing,
        endpoint_helpers,
        completed_feature,
        spec,
    ):
        body = _body(spec, completed_feature["id"], endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)
        assert response.status_code == 201, response.json()
        data = response.json()
        feature_conditions = [
            c
            for m in data["modifications"]
            for c in m["conditions"]
            if c.get("source") == "feature"
        ]
        assert any(
            c["feature_id"] == completed_feature["id"] for c in feature_conditions
        )


@pytest.mark.parametrize("spec", ENDPOINTS, ids=lambda s: s.name)
class TestInlineGeometryModification:
    """Inline-geometry (source="geometry") conditions carry nested coordinate
    arrays. Every create router must JSON-encode those coordinates before the
    Firestore write (Firestore rejects nested arrays) and the Grid read-back
    must decode them again.

    Running this across every endpoint is the guard against a new create
    router forgetting the write-side stringification: without the fix the POST
    is a prod-only 500 that feature-source tests never exercise (#280).
    """

    def test_inline_geometry_succeeds_and_round_trips(
        self, client, domain_for_testing, endpoint_helpers, spec
    ):
        body = _geometry_body(spec, endpoint_helpers)
        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        # No Firestore 500 — the nested coordinates were stringified on write.
        assert response.status_code == 201, response.json()

        # POST response decodes the stored string back to nested-list GeoJSON.
        created = response.json()
        assert _first_geometry_coords(created) == INLINE_POLYGON_COORDS

        # And a fresh GET round-trips through real Firestore: coordinates come
        # back as a nested list, not the JSON string they are stored as.
        get_response = client.get(
            f"/domains/{domain_for_testing['id']}/grids/{created['id']}"
        )
        assert get_response.status_code == 200, get_response.json()
        assert _first_geometry_coords(get_response.json()) == INLINE_POLYGON_COORDS
