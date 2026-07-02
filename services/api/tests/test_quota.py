"""
Unit tests for api.quota.

Covers the quota vocabulary defaults, the tier presets, the phase-1 resolver,
and the 429 error-detail shape. No server or Firestore required.
"""

import pytest
from api.quota import TIER_PRESETS, QuotaExceededDetail, Quotas, resolve_quotas
from api.resources.keys.schema import Access


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


class TestResolveQuotas:
    pytestmark = pytest.mark.anyio

    async def test_personal_resolves_to_standard(self):
        assert await resolve_quotas("owner-1", Access.PERSONAL) == Quotas()

    async def test_application_resolves_to_standard(self):
        # Phase 1: applications also default to standard; tier grants are phase 2.
        assert await resolve_quotas("app-1", Access.APPLICATION) == Quotas()


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
