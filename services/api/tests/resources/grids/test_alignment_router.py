"""
Cross-router validation tests for ``alignment.target='grid'``.

Every router that accepts a ``GridAlignmentSpecification`` must reject
target grids that are missing, owned by another user, in another
domain, not yet completed, or missing a georeference. The same
six-case sweep is parametrized over every endpoint so a regression on
any single router is caught here rather than in per-product test files.

Endpoints under test:

- ``POST /domains/{id}/grids/fbfm40/landfire``
- ``POST /domains/{id}/grids/fccs/landfire``
- ``POST /domains/{id}/grids/canopy/meta``
- ``POST /domains/{id}/grids/canopy/naip``
- ``POST /domains/{id}/grids/pim/treemap``
- ``POST /domains/{id}/grids/topography/landfire``
- ``POST /domains/{id}/grids/topography/3dep``
- ``POST /domains/{id}/grids/resample`` (also covered by the
  resample-specific test file; included here for symmetry).
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class EndpointSpec:
    """Defines a single endpoint that accepts ``alignment.target='grid'``."""

    name: str
    path_suffix: str
    extra_body: Callable[[dict], dict]

    def url(self, domain_id: str) -> str:
        return f"/domains/{domain_id}/grids/{self.path_suffix}"


def _empty_extra(_complete_grid: dict) -> dict:
    return {}


def _resample_extra(complete_grid: dict) -> dict:
    return {"source_grid_id": complete_grid["id"]}


ENDPOINTS = [
    EndpointSpec("fbfm40_landfire", "fbfm40/landfire", _empty_extra),
    EndpointSpec("fccs_landfire", "fccs/landfire", _empty_extra),
    EndpointSpec("chm_meta", "canopy/meta", _empty_extra),
    EndpointSpec("chm_naip", "canopy/naip", _empty_extra),
    EndpointSpec("pim_treemap", "pim/treemap", _empty_extra),
    EndpointSpec("topography_landfire", "topography/landfire", _empty_extra),
    EndpointSpec("topography_3dep", "topography/3dep", _empty_extra),
    EndpointSpec("resample", "resample", _resample_extra),
]


def _alignment_body(spec: EndpointSpec, grid_id: str, complete_grid: dict) -> dict:
    return {
        **spec.extra_body(complete_grid),
        "alignment": {"target": "grid", "grid_id": grid_id},
    }


@pytest.mark.parametrize("spec", ENDPOINTS, ids=lambda s: s.name)
class TestAlignmentTargetGridValidation:
    """The shared target-grid validator must reject every invalid case
    on every endpoint that accepts ``alignment.target='grid'``."""

    def test_nonexistent_target_grid_returns_404(
        self, client, domain_for_testing, complete_grid, spec
    ):
        body = _alignment_body(spec, "00000000000000000000000000000000", complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 404

    def test_target_grid_owned_by_other_user_returns_404(
        self,
        client,
        domain_for_testing,
        complete_grid,
        grid_owned_by_other_user,
        spec,
    ):
        body = _alignment_body(spec, grid_owned_by_other_user["id"], complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 404

    def test_target_grid_in_other_domain_returns_404(
        self,
        client,
        domain_for_testing,
        complete_grid,
        grid_in_different_domain,
        spec,
    ):
        body = _alignment_body(spec, grid_in_different_domain["id"], complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 404

    def test_pending_target_grid_returns_422(
        self, client, domain_for_testing, complete_grid, pending_grid, spec
    ):
        body = _alignment_body(spec, pending_grid["id"], complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    def test_target_grid_without_georeference_returns_422(
        self,
        client,
        domain_for_testing,
        complete_grid,
        complete_grid_no_georeference,
        spec,
    ):
        body = _alignment_body(spec, complete_grid_no_georeference["id"], complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 422

    def test_valid_target_grid_succeeds(
        self, client, domain_for_testing, complete_grid, spec
    ):
        body = _alignment_body(spec, complete_grid["id"], complete_grid)

        response = client.post(spec.url(domain_for_testing["id"]), json=body)

        assert response.status_code == 201
        data = response.json()
        assert data["source"]["alignment"]["target"] == "grid"
        assert data["source"]["alignment"]["grid_id"] == complete_grid["id"]
