"""
Tests for POST /domains/{domain_id}/grids/canopy/point_cloud.

The endpoint's job is to validate the source point cloud and the resolved
lattice before a grid document exists, so a request that cannot succeed never
becomes a failed grid.
"""

import pytest
from api.resources.grids.canopy.examples import POINT_CLOUD_CHM_EXAMPLE_VALUES


class TestCreatePointCloudChm:
    """Creating a canopy height model grid from a point cloud."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/point_cloud"

    def test_creates_grid_with_chm_band(
        self, client, domain_for_testing, als_point_cloud
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert response.status_code == 201
        grid = response.json()
        assert grid["status"] == "pending"
        assert grid["source"]["name"] == "canopy"
        assert grid["source"]["product"] == "point_cloud"
        assert grid["source"]["source_point_cloud_id"] == als_point_cloud["id"]
        assert [b["key"] for b in grid["bands"]] == ["chm"]
        assert grid["bands"][0]["unit"] == "m"
        assert grid["bands"][0]["type"] == "continuous"

    def test_captures_source_checksum_for_staleness(
        self, client, domain_for_testing, als_point_cloud
    ):
        """The grid records the cloud version it was built from."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert response.status_code == 201
        source = response.json()["source"]
        assert source["source_point_cloud_checksum"] == als_point_cloud["checksum"]

    def test_georeference_is_null_on_creation(
        self, client, domain_for_testing, als_point_cloud
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert response.json()["georeference"] is None

    def test_ground_provenance_absent_until_processed(
        self, client, domain_for_testing, als_point_cloud
    ):
        """`source.ground` is populated by the worker, not at creation."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert response.json()["source"]["ground"] is None

    def test_response_excludes_owner_id(
        self, client, domain_for_testing, als_point_cloud
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert "owner_id" not in response.json()

    @pytest.mark.parametrize(
        "example_name,example_value", POINT_CLOUD_CHM_EXAMPLE_VALUES
    )
    def test_documented_example_creates_grid(
        self,
        client,
        domain_for_testing,
        als_point_cloud,
        complete_grid,
        example_name,
        example_value,
    ):
        """Every documented OpenAPI example is a request that actually works."""
        body = dict(example_value)
        body["source_point_cloud_id"] = als_point_cloud["id"]
        if body.get("alignment", {}).get("target") == "grid":
            body["alignment"] = dict(body["alignment"])
            body["alignment"]["grid_id"] = complete_grid["id"]

        response = client.post(self.route(domain_for_testing["id"]), json=body)

        assert response.status_code == 201, response.text


class TestSourcePointCloudValidation:
    """Rejections that keep a doomed request from becoming a failed grid."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/point_cloud"

    def test_missing_point_cloud_returns_404(self, client, domain_for_testing):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": "does-not-exist"},
        )

        assert response.status_code == 404

    def test_terrestrial_point_cloud_returns_422(
        self, client, domain_for_testing, tls_point_cloud
    ):
        """A terrestrial scan has no landscape canopy surface."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": tls_point_cloud["id"]},
        )

        assert response.status_code == 422
        assert "als" in response.json()["detail"]

    def test_pending_point_cloud_returns_422(
        self, client, domain_for_testing, pending_point_cloud
    ):
        """A cloud still being produced has no LAZ to read."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": pending_point_cloud["id"]},
        )

        assert response.status_code == 422

    def test_point_cloud_from_another_domain_returns_404(
        self, client, domain_for_testing, point_cloud_in_other_domain
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": point_cloud_in_other_domain["id"]},
        )

        assert response.status_code == 404

    def test_point_cloud_of_another_owner_returns_404(
        self, client, domain_for_testing, point_cloud_of_other_owner
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": point_cloud_of_other_owner["id"]},
        )

        assert response.status_code == 404

    def test_missing_source_point_cloud_id_returns_422(
        self, client, domain_for_testing
    ):
        response = client.post(self.route(domain_for_testing["id"]), json={})

        assert response.status_code == 422


class TestAlignment:
    """Lattice controls specific to a source with no native pixel size."""

    def route(self, domain_id):
        return f"/domains/{domain_id}/grids/canopy/point_cloud"

    def test_native_alignment_returns_422(
        self, client, domain_for_testing, als_point_cloud
    ):
        """There is no source raster whose pixel anchor could be preserved."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_point_cloud_id": als_point_cloud["id"],
                "alignment": {"target": "native"},
            },
        )

        assert response.status_code == 422
        assert "native" in response.json()["detail"]

    def test_omitted_resolution_is_resolved_and_persisted(
        self, client, domain_for_testing, als_point_cloud
    ):
        """`null` has nothing to defer to here, so the default is stored.

        The worker reads this value directly; leaving it null would mean two
        copies of the default that have to agree.
        """
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={"source_point_cloud_id": als_point_cloud["id"]},
        )

        assert response.status_code == 201
        assert response.json()["source"]["alignment"]["resolution"] == 1.0

    def test_requested_resolution_is_persisted(
        self, client, domain_for_testing, als_point_cloud
    ):
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_point_cloud_id": als_point_cloud["id"],
                "alignment": {"target": "domain", "resolution": 5.0},
            },
        )

        assert response.status_code == 201
        assert response.json()["source"]["alignment"]["resolution"] == 5.0

    def test_sub_metre_resolution_rejected(
        self, client, domain_for_testing, als_point_cloud
    ):
        """The shared alignment schema floors resolution at 1 m."""
        response = client.post(
            self.route(domain_for_testing["id"]),
            json={
                "source_point_cloud_id": als_point_cloud["id"],
                "alignment": {"target": "domain", "resolution": 0.25},
            },
        )

        assert response.status_code == 422


class TestCellCountCap:
    """The lattice-size guard.

    Exercised directly rather than over HTTP: `client` drives a separate server
    process, so a monkeypatched cap in this process would not reach it, and the
    test domain is far too small to exceed the real cap at the 1 m floor.
    """

    def test_lattice_within_the_cap_is_accepted(self, domain_for_testing):
        from api.resources.grids.canopy.router import _validate_cell_count

        _validate_cell_count(domain_for_testing, 1.0)

    def test_oversized_lattice_raises_422(self, domain_for_testing, monkeypatch):
        from api.resources.grids.canopy import router as canopy_router
        from fastapi import HTTPException

        monkeypatch.setattr(canopy_router, "MAX_CHM_CELLS", 100)

        with pytest.raises(HTTPException) as excinfo:
            canopy_router._validate_cell_count(domain_for_testing, 1.0)

        assert excinfo.value.status_code == 422
        assert "cells" in excinfo.value.detail
        assert "resolution" in excinfo.value.detail

    def test_coarser_resolution_shrinks_the_lattice(
        self, domain_for_testing, monkeypatch
    ):
        """The cap is on cells, so a coarser request can fit where 1 m cannot."""
        from api.resources.grids.canopy import router as canopy_router
        from fastapi import HTTPException

        minx, miny, maxx, maxy = _domain_bounds(domain_for_testing)
        cells_at_1m = ((maxx - minx) / 1.0) * ((maxy - miny) / 1.0)
        monkeypatch.setattr(canopy_router, "MAX_CHM_CELLS", int(cells_at_1m / 2))

        with pytest.raises(HTTPException):
            canopy_router._validate_cell_count(domain_for_testing, 1.0)

        canopy_router._validate_cell_count(domain_for_testing, 10.0)


def _domain_bounds(domain):
    from lib.domain_utils import parse_domain_gdf

    return parse_domain_gdf(domain).total_bounds
