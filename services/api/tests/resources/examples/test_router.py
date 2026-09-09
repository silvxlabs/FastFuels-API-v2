"""
Unit tests for the /examples discovery endpoint (#582).

``list_examples`` is called directly with the Firestore query mocked, so no
server or Firestore is required. The endpoint must group example grids under
their example domain, drop grids that are unfinished or belong to a
non-example domain, and return an empty list when nothing is seeded.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.resources.examples.router import list_examples

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION

pytestmark = pytest.mark.anyio


def _snap(doc_id: str, data: dict):
    snap = MagicMock()
    snap.id = doc_id
    snap.to_dict.return_value = data
    return snap


def _patch_queries(domain_snaps: list, grid_snaps: list):
    """Patch _example_query so each collection returns its snapshots via get()."""
    results = {
        DOMAINS_COLLECTION: domain_snaps,
        GRIDS_COLLECTION: grid_snaps,
    }

    def _query(collection: str):
        q = MagicMock()
        q.get = AsyncMock(return_value=results[collection])
        return q

    return patch("api.resources.examples.router._example_query", side_effect=_query)


class TestListExamples:
    async def test_empty_when_nothing_seeded(self):
        with _patch_queries([], []):
            resp = await list_examples()
        assert resp.examples == []

    async def test_groups_completed_grids_under_domain(self):
        domains = [_snap("dom1", {"name": "Blue Mountain"})]
        grids = [
            _snap(
                "grid1",
                {"name": "NAIP", "domain_id": "dom1", "status": "completed"},
            )
        ]
        with _patch_queries(domains, grids):
            resp = await list_examples()

        assert len(resp.examples) == 1
        ex = resp.examples[0]
        assert ex.domain_id == "dom1"
        assert ex.name == "Blue Mountain"
        assert [g.id for g in ex.grids] == ["grid1"]
        assert ex.grids[0].domain_id == "dom1"

    async def test_unfinished_grid_is_not_advertised(self):
        domains = [_snap("dom1", {"name": "Blue Mountain"})]
        grids = [_snap("g", {"domain_id": "dom1", "status": "pending"})]
        with _patch_queries(domains, grids):
            resp = await list_examples()
        assert resp.examples[0].grids == []

    async def test_grid_for_unlisted_domain_is_skipped(self):
        # An example grid whose domain is not itself a listed example must never
        # be advertised (it would be an unreadable deep-link).
        domains = [_snap("dom1", {"name": "Blue Mountain"})]
        grids = [
            _snap("g1", {"domain_id": "dom1", "status": "completed"}),
            _snap("g2", {"domain_id": "orphan", "status": "completed"}),
        ]
        with _patch_queries(domains, grids):
            resp = await list_examples()
        assert [g.id for g in resp.examples[0].grids] == ["g1"]
