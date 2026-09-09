"""
Unit tests for api.access — the shared-example read path (#582).

The security contract: a non-owner may read a resource ONLY when it is both
owned by the example owner AND flagged ``is_example``. These tests prove a
guest CAN read the flagged example and CANNOT read anything else — a normal
owner's resource, an example-owner resource that is not flagged, or a resource
someone else has tried to self-flag as public.

Pure unit tests: ``api.access.get_document_async`` is mocked, so no server or
Firestore is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.access import get_readable_document_async, is_example_resource
from fastapi import HTTPException

from lib.config import EXAMPLE_OWNER_ID, GRIDS_COLLECTION

pytestmark = pytest.mark.anyio

VIEWER = "guest-uid-123"
OTHER = "another-real-owner"


def _snapshot(data: dict):
    snap = MagicMock()
    snap.to_dict.return_value = data
    return snap


def _patch_get(data: dict) -> AsyncMock:
    """Patch get_document_async to return a snapshot carrying ``data``."""
    mock = AsyncMock(return_value=(MagicMock(), _snapshot(data)))
    return patch("api.access.get_document_async", mock)


class TestIsExampleResource:
    """The predicate requires BOTH the example owner AND the is_example flag."""

    def test_flagged_example_owner_is_example(self):
        assert is_example_resource({"owner_id": EXAMPLE_OWNER_ID, "is_example": True})

    def test_example_owner_without_flag_is_not_example(self):
        assert not is_example_resource({"owner_id": EXAMPLE_OWNER_ID})
        assert not is_example_resource(
            {"owner_id": EXAMPLE_OWNER_ID, "is_example": False}
        )

    def test_flag_without_example_owner_is_not_example(self):
        # A normal owner cannot self-declare their resource public.
        assert not is_example_resource({"owner_id": OTHER, "is_example": True})

    def test_truthy_but_non_true_flag_is_not_example(self):
        # `is True` is deliberate: only a real boolean True opens access.
        assert not is_example_resource(
            {"owner_id": EXAMPLE_OWNER_ID, "is_example": "yes"}
        )
        assert not is_example_resource({"owner_id": EXAMPLE_OWNER_ID, "is_example": 1})


class TestReadOwnResource:
    async def test_owner_reads_own_resource(self):
        data = {"owner_id": VIEWER, "id": "d1"}
        with _patch_get(data):
            _, snap = await get_readable_document_async("domains-v2", "d1", VIEWER)
        assert snap.to_dict()["owner_id"] == VIEWER

    async def test_get_document_called_without_owner_filter(self):
        # Authorization happens in access.py, NOT via get_document_async's owner
        # filter — otherwise the example (owned by someone else) would 404 first.
        data = {"owner_id": VIEWER, "id": "d1"}
        with _patch_get(data) as mock:
            await get_readable_document_async("grids-v2", "g1", VIEWER, domain_id="d1")
        _, kwargs = mock.call_args
        assert "owner_id" not in kwargs
        assert kwargs.get("domain_id") == "d1"


class TestReadExample:
    async def test_guest_reads_flagged_example(self):
        data = {"owner_id": EXAMPLE_OWNER_ID, "is_example": True, "id": "ex"}
        with _patch_get(data):
            _, snap = await get_readable_document_async("domains-v2", "ex", VIEWER)
        assert snap.to_dict()["id"] == "ex"

    async def test_guest_reads_flagged_example_grid_data_status(self):
        data = {
            "owner_id": EXAMPLE_OWNER_ID,
            "is_example": True,
            "status": "completed",
        }
        with _patch_get(data):
            _, snap = await get_readable_document_async(
                GRIDS_COLLECTION,
                "g1",
                VIEWER,
                domain_id="ex",
                document_status="completed",
            )
        assert snap.to_dict()["status"] == "completed"


class TestCrossOwnerDenial:
    """The core security property: no reading across owners."""

    async def test_guest_cannot_read_other_owners_resource(self):
        data = {"owner_id": OTHER, "id": "private"}
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async("domains-v2", "private", VIEWER)
        assert exc.value.status_code == 404

    async def test_guest_cannot_read_unflagged_example_owner_resource(self):
        # Owned by the example owner but NOT flagged: still private.
        data = {"owner_id": EXAMPLE_OWNER_ID, "id": "draft"}
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async("domains-v2", "draft", VIEWER)
        assert exc.value.status_code == 404

    async def test_guest_cannot_read_self_flagged_foreign_resource(self):
        # An attacker sets is_example=True on their OWN doc. It must NOT become
        # publicly readable: the owner gate blocks it.
        data = {"owner_id": OTHER, "is_example": True, "id": "spoof"}
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async("domains-v2", "spoof", VIEWER)
        assert exc.value.status_code == 404

    async def test_denial_does_not_leak_via_status_422(self):
        # A non-owned, non-example resource must 404 on the authorization check
        # BEFORE the status check — a 422 would confirm the resource exists.
        data = {"owner_id": OTHER, "status": "pending"}
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async(
                    GRIDS_COLLECTION,
                    "g1",
                    VIEWER,
                    document_status="completed",
                )
        assert exc.value.status_code == 404


class TestStatusCheck:
    async def test_example_wrong_status_raises_422(self):
        # The caller IS authorized (flagged example) but the grid isn't finished.
        data = {
            "owner_id": EXAMPLE_OWNER_ID,
            "is_example": True,
            "status": "pending",
        }
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async(
                    GRIDS_COLLECTION,
                    "g1",
                    VIEWER,
                    document_status="completed",
                )
        assert exc.value.status_code == 422

    async def test_owner_wrong_status_raises_422(self):
        data = {"owner_id": VIEWER, "status": "pending"}
        with _patch_get(data):
            with pytest.raises(HTTPException) as exc:
                await get_readable_document_async(
                    GRIDS_COLLECTION,
                    "g1",
                    VIEWER,
                    document_status="completed",
                )
        assert exc.value.status_code == 422
