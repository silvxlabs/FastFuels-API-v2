"""
Unit tests for api.quota.

Covers the quota vocabulary defaults, the tier presets, the phase-1 resolver,
and the 429 error-detail shape. No server or Firestore required.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.quota import (
    _RESOURCE_QUOTAS,
    RETRY_AFTER_SECONDS,
    TIER_PRESETS,
    QuotaExceededDetail,
    Quotas,
    _raise_quota_exceeded,
    resolve_quotas,
)
from api.resources.keys.schema import Access
from fastapi import HTTPException

from lib.config import APPLICATIONS_COLLECTION, DOMAINS_COLLECTION, USERS_COLLECTION


class TestQuotasDefaults:
    def test_active_job_defaults(self):
        q = Quotas()
        assert q.max_active_grids == 25
        assert q.max_active_exports == 10
        assert q.max_active_inventories == 10
        assert q.max_active_features == 10
        assert q.max_active_pointclouds == 5

    def test_count_and_storage_defaults(self):
        q = Quotas()
        assert q.max_grids == 1_000
        assert q.max_api_keys == 50
        assert q.max_grid_storage_bytes == 50 * 2**30
        assert q.max_feature_storage_bytes == 1 * 2**30

    def test_lifecycle_defaults(self):
        q = Quotas()
        assert q.resource_ttl_days == 180
        assert q.failed_resource_ttl_days == 14


class TestTierPresets:
    def test_standard_is_defaults(self):
        assert Quotas(**TIER_PRESETS["standard"]) == Quotas()

    def test_application_raises_selected_limits(self):
        q = Quotas(**TIER_PRESETS["application"])
        assert q.max_active_grids == 100
        assert q.max_active_exports == 50
        assert q.max_grids == 10_000
        assert q.max_grid_storage_bytes == 500 * 2**30
        assert q.resource_ttl_days is None
        # Fields the preset does not mention keep the standard defaults.
        assert q.max_active_inventories == 10

    def test_partner_is_placeholder_defaults(self):
        assert Quotas(**TIER_PRESETS["partner"]) == Quotas()

    def test_suspended_zeroes_every_create_limit(self):
        q = Quotas(**TIER_PRESETS["suspended"])
        zeroed = [f for f in Quotas.model_fields if f.startswith("max_")]
        assert zeroed  # guard against the comprehension silently matching nothing
        for field in zeroed:
            assert getattr(q, field) == 0, field
        # Suspension blocks creates but does not change retention.
        assert q.resource_ttl_days == 180
        assert q.failed_resource_ttl_days == 14


def _fake_firestore(*, exists: bool, data: dict | None = None) -> MagicMock:
    """A firestore_client stand-in whose single-document read returns a snapshot."""
    snapshot = MagicMock()
    snapshot.exists = exists
    snapshot.to_dict.return_value = data or {}
    client = MagicMock()
    client.collection.return_value.document.return_value.get = AsyncMock(
        return_value=snapshot
    )
    return client


class TestResolveQuotas:
    pytestmark = pytest.mark.anyio

    # Each test uses a unique owner id because resolve_quotas is @lru-cached.

    async def test_missing_doc_resolves_to_defaults(self):
        client = _fake_firestore(exists=False)
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-missing", Access.PERSONAL)
        assert result == Quotas()
        client.collection.assert_called_once_with(USERS_COLLECTION)

    async def test_application_access_reads_applications_collection(self):
        client = _fake_firestore(exists=False)
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-app-missing", Access.APPLICATION)
        assert result == Quotas()
        client.collection.assert_called_once_with(APPLICATIONS_COLLECTION)

    async def test_tier_preset_is_applied(self):
        client = _fake_firestore(exists=True, data={"tier": "application"})
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-tier", Access.APPLICATION)
        assert result == Quotas(**TIER_PRESETS["application"])
        assert result.max_active_grids == 100

    async def test_overrides_beat_tier_and_defaults(self):
        data = {"tier": "standard", "quota_overrides": {"max_active_grids": 3}}
        client = _fake_firestore(exists=True, data=data)
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-override", Access.PERSONAL)
        assert result.max_active_grids == 3
        assert result.max_active_features == Quotas().max_active_features

    async def test_suspended_tier_zeroes_create_limits(self):
        client = _fake_firestore(exists=True, data={"tier": "suspended"})
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-suspended", Access.PERSONAL)
        assert result.max_active_grids == 0
        assert result.max_grids == 0

    async def test_unknown_tier_falls_back_to_defaults(self):
        client = _fake_firestore(exists=True, data={"tier": "platinum"})
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-bad-tier", Access.PERSONAL)
        assert result == Quotas()

    async def test_unknown_override_key_is_ignored(self):
        data = {"quota_overrides": {"not_a_quota": 5, "max_active_features": 2}}
        client = _fake_firestore(exists=True, data=data)
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-bad-key", Access.PERSONAL)
        assert result.max_active_features == 2

    async def test_malformed_override_value_falls_back_to_defaults(self, caplog):
        data = {"quota_overrides": {"max_active_features": "not-an-int"}}
        client = _fake_firestore(exists=True, data=data)
        with caplog.at_level(logging.ERROR):
            with patch("api.quota.firestore_client", client):
                result = await resolve_quotas("owner-bad-value", Access.PERSONAL)
        assert result == Quotas()
        assert "Malformed quota config" in caplog.text

    async def test_non_dict_overrides_falls_back_to_defaults(self):
        client = _fake_firestore(exists=True, data={"quota_overrides": "nope"})
        with patch("api.quota.firestore_client", client):
            result = await resolve_quotas("owner-bad-overrides", Access.PERSONAL)
        assert result == Quotas()


class TestQuotaExceededDetail:
    def test_shape_and_default_reason(self):
        detail = QuotaExceededDetail(
            quota="max_active_grids", message="msg", current=25, limit=25
        )
        assert detail.reason == "QUOTA_EXCEEDED"
        assert detail.model_dump() == {
            "reason": "QUOTA_EXCEEDED",
            "quota": "max_active_grids",
            "message": "msg",
            "current": 25,
            "limit": 25,
        }


class TestResourceQuotas:
    """The table driving enforce_create_quotas must name real Quotas fields."""

    def test_every_field_exists_on_quotas(self):
        for collection, spec in _RESOURCE_QUOTAS.items():
            assert spec.count_field in Quotas.model_fields, collection
            if spec.active_field is not None:
                assert spec.active_field in Quotas.model_fields, collection
            if spec.storage_field is not None:
                assert spec.storage_field in Quotas.model_fields, collection

    def test_storage_tracked_types_have_active_and_storage(self):
        # The five async job types carry all three limits.
        storage_types = [
            s for s in _RESOURCE_QUOTAS.values() if s.storage_field is not None
        ]
        assert len(storage_types) == 5
        for spec in storage_types:
            assert spec.active_field is not None

    def test_domains_are_count_only(self):
        spec = _RESOURCE_QUOTAS[DOMAINS_COLLECTION]
        assert spec.active_field is None
        assert spec.storage_field is None
        assert spec.count_field == "max_domains"


class TestRaiseQuotaExceeded:
    """Retry-After is present only for limits that clear by waiting."""

    def test_retry_after_true_sets_header(self):
        with pytest.raises(HTTPException) as exc:
            _raise_quota_exceeded(
                quota="max_active_grids",
                message="msg",
                current=25,
                limit=25,
                retry_after=True,
            )
        assert exc.value.status_code == 429
        assert exc.value.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
        assert exc.value.detail == {
            "reason": "QUOTA_EXCEEDED",
            "quota": "max_active_grids",
            "message": "msg",
            "current": 25,
            "limit": 25,
        }

    def test_retry_after_false_omits_header(self):
        with pytest.raises(HTTPException) as exc:
            _raise_quota_exceeded(
                quota="max_grids",
                message="msg",
                current=1000,
                limit=1000,
                retry_after=False,
            )
        assert exc.value.status_code == 429
        assert exc.value.headers is None
        assert exc.value.detail["quota"] == "max_grids"
