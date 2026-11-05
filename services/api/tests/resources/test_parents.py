"""
Unit tests for api.resources.parents (#775).

Firestore is replaced by an in-memory fake that records every document a
batched ``get_all`` reads, so these tests prove each parent is read exactly
once and that missing parents are kept, as ``None``, rather than dropped.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.resources.grids.resample.router import create_resample
from api.resources.grids.resample.schema import CreateResampleRequest
from api.resources.parents import Parents, load_parents, validate_parent
from fastapi import HTTPException

from lib.config import FEATURES_COLLECTION, GRIDS_COLLECTION, INVENTORIES_COLLECTION
from tests.fixtures import make_feature_data, make_grid_data, make_osm_attribution

OWNER_ID = "owner"
DOMAIN_ID = "domain"


class FakeFirestore:
    """Serves documents from ``store`` and records each ``get_all`` batch."""

    def __init__(self, store: dict[tuple[str, str], dict]):
        self.store = store
        self.batches: list[list[tuple[str, str]]] = []

    def collection(self, collection: str):
        return SimpleNamespace(
            document=lambda document_id: (collection, document_id),
        )

    async def get_all(self, refs):
        refs = list(refs)
        self.batches.append(refs)
        for collection, document_id in refs:
            data = self.store.get((collection, document_id))
            yield SimpleNamespace(
                id=document_id,
                exists=data is not None,
                reference=SimpleNamespace(parent=SimpleNamespace(id=collection)),
                to_dict=lambda data=data: data,
            )


class TestLoadParents:
    @pytest.mark.asyncio
    async def test_one_entry_per_requested_id_with_missing_marked(self):
        grid = {"id": "g1"}
        feature = {"id": "f1"}
        fake = FakeFirestore(
            {(GRIDS_COLLECTION, "g1"): grid, (FEATURES_COLLECTION, "f1"): feature}
        )
        with patch("api.resources.parents.firestore_client", fake):
            parents = await load_parents(
                grids=["g1", "missing-grid", "g1", None],
                features=["f1", "missing-feature"],
                inventories=["missing-inventory"],
            )

        assert parents.grids == {"g1": grid, "missing-grid": None}
        assert parents.features == {"f1": feature, "missing-feature": None}
        assert parents.inventories == {"missing-inventory": None}
        assert parents.point_clouds == {}
        assert list(parents) == [grid, feature]
        # One batch; the duplicate and None IDs are not read.
        assert fake.batches == [
            [
                (GRIDS_COLLECTION, "g1"),
                (GRIDS_COLLECTION, "missing-grid"),
                (FEATURES_COLLECTION, "f1"),
                (FEATURES_COLLECTION, "missing-feature"),
                (INVENTORIES_COLLECTION, "missing-inventory"),
            ]
        ]

    @pytest.mark.asyncio
    async def test_no_ids_reads_nothing(self):
        fake = FakeFirestore({})
        with patch("api.resources.parents.firestore_client", fake):
            parents = await load_parents(grids=[None])
        assert parents == Parents()
        assert fake.batches == []


class TestValidateParent:
    def _parents(self, grid: dict | None) -> Parents:
        return Parents(grids={"g1": grid})

    def test_missing_parent_is_404(self):
        with pytest.raises(HTTPException) as exc:
            validate_parent(
                self._parents(None), GRIDS_COLLECTION, "g1", OWNER_ID, DOMAIN_ID
            )
        assert exc.value.status_code == 404
        assert exc.value.detail == "Document not found: grids/g1"

    def test_other_owner_is_404(self):
        grid = {"owner_id": "someone", "domain_id": DOMAIN_ID, "status": "completed"}
        with pytest.raises(HTTPException) as exc:
            validate_parent(
                self._parents(grid), GRIDS_COLLECTION, "g1", OWNER_ID, DOMAIN_ID
            )
        assert exc.value.status_code == 404
        assert exc.value.detail == "Document not found: grids/g1"

    def test_other_domain_is_404(self):
        grid = {"owner_id": OWNER_ID, "domain_id": "other", "status": "completed"}
        with pytest.raises(HTTPException) as exc:
            validate_parent(
                self._parents(grid), GRIDS_COLLECTION, "g1", OWNER_ID, DOMAIN_ID
            )
        assert exc.value.status_code == 404

    def test_not_completed_is_422(self):
        grid = {"owner_id": OWNER_ID, "domain_id": DOMAIN_ID, "status": "pending"}
        with pytest.raises(HTTPException) as exc:
            validate_parent(
                self._parents(grid), GRIDS_COLLECTION, "g1", OWNER_ID, DOMAIN_ID
            )
        assert exc.value.status_code == 422
        assert exc.value.detail == (
            "grids/g1 status is 'pending', expected 'completed'."
        )

    def test_completed_parent_passes(self):
        grid = {"owner_id": OWNER_ID, "domain_id": DOMAIN_ID, "status": "completed"}
        validate_parent(
            self._parents(grid), GRIDS_COLLECTION, "g1", OWNER_ID, DOMAIN_ID
        )


class TestResampleReadsEachParentOnce:
    """Resample with a source grid, an alignment target grid, and a feature
    referenced by two modifications: one batched read, one read per parent,
    and the same documents feed validation and attributions."""

    def _body(self, source_id: str, target_id: str, feature_id: str):
        modification = {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": feature_id,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
        }
        return CreateResampleRequest.model_validate(
            {
                "source_grid_id": source_id,
                "alignment": {"target": "grid", "grid_id": target_id},
                "modifications": [modification, modification],
            }
        )

    async def _create(
        self, fake: FakeFirestore, body: CreateResampleRequest, set_document
    ):
        router = "api.resources.grids.resample.router"
        with (
            patch("api.resources.parents.firestore_client", fake),
            patch(f"{router}.enforce_create_quotas", AsyncMock()),
            patch(f"{router}.set_document_async", set_document),
            patch(f"{router}.create_http_task_async", AsyncMock()),
            patch(f"{router}.register_dispatch", MagicMock()),
        ):
            return await create_resample(
                request=SimpleNamespace(state=SimpleNamespace(id=OWNER_ID)),
                response=MagicMock(),
                background_tasks=MagicMock(),
                domain={"id": DOMAIN_ID},
                body=body,
            )

    def _parents(self):
        meta = {
            "dataset": "Meta CHM",
            "license_name": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Meta and World Resources Institute",
            "access_url": "https://registry.opendata.aws/dataforgood-fb-forests/",
            "accessed_on": "2026-10-01",
        }
        osm = make_osm_attribution("2026-11-05")
        naip = {**meta, "dataset": "NAIP", "license_name": "CC0-1.0"}
        source = make_grid_data(DOMAIN_ID, owner_id=OWNER_ID, status="completed")
        source["attributions"] = [meta]
        target = make_grid_data(DOMAIN_ID, owner_id=OWNER_ID, status="completed")
        target["attributions"] = [naip]
        feature = make_feature_data(DOMAIN_ID, owner_id=OWNER_ID)
        feature["attributions"] = [osm]
        return source, target, feature, [meta, naip, osm]

    @pytest.mark.asyncio
    async def test_each_parent_read_once(self):
        source, target, feature, expected = self._parents()
        fake = FakeFirestore(
            {
                (GRIDS_COLLECTION, source["id"]): source,
                (GRIDS_COLLECTION, target["id"]): target,
                (FEATURES_COLLECTION, feature["id"]): feature,
            }
        )
        set_document = AsyncMock()
        grid = await self._create(
            fake, self._body(source["id"], target["id"], feature["id"]), set_document
        )

        assert fake.batches == [
            [
                (GRIDS_COLLECTION, source["id"]),
                (GRIDS_COLLECTION, target["id"]),
                (FEATURES_COLLECTION, feature["id"]),
            ]
        ]
        stored = set_document.await_args.args[2]
        assert stored["attributions"] == expected
        assert [a.model_dump() for a in grid.attributions] == expected

    @pytest.mark.asyncio
    async def test_missing_target_grid_is_404_before_any_write(self):
        source, target, feature, _ = self._parents()
        fake = FakeFirestore(
            {
                (GRIDS_COLLECTION, source["id"]): source,
                (FEATURES_COLLECTION, feature["id"]): feature,
            }
        )
        set_document = AsyncMock()
        with pytest.raises(HTTPException) as exc:
            await self._create(
                fake,
                self._body(source["id"], target["id"], feature["id"]),
                set_document,
            )
        assert exc.value.status_code == 404
        assert exc.value.detail == f"Document not found: grids/{target['id']}"
        assert len(fake.batches) == 1
        set_document.assert_not_awaited()
