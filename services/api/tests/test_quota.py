"""
Unit tests for api.quota.

Covers the quota vocabulary defaults, the tier presets, the phase-1 resolver,
the 429 error-detail shape, and the weekly dispatch budget (#431). No server
or Firestore required.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.quota import (
    _RESOURCE_QUOTAS,
    RETRY_AFTER_SECONDS,
    TIER_PRESETS,
    OwnerQuotaConfig,
    QuotaExceededDetail,
    Quotas,
    _increment_budget,
    _raise_quota_exceeded,
    _WeeklyBudget,
    enforce_create_quotas,
    iso_week_id,
    next_week_start,
    register_dispatch,
    resolve_owner_config,
    resolve_quotas,
)
from api.resources.keys.schema import Access
from fastapi import BackgroundTasks, HTTPException, Response
from google.cloud.firestore import Increment

from lib.config import (
    APPLICATIONS_COLLECTION,
    CREATE_BUDGETS_COLLECTION,
    DOMAINS_COLLECTION,
    GRIDS_COLLECTION,
    USERS_COLLECTION,
)


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

    def test_weekly_dispatch_defaults(self):
        q = Quotas()
        assert q.max_weekly_grid_dispatches == 500
        assert q.max_weekly_export_dispatches == 250
        assert q.max_weekly_inventory_dispatches == 250
        assert q.max_weekly_feature_dispatches == 250
        assert q.max_weekly_pointcloud_dispatches == 50

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
        assert q.max_weekly_grid_dispatches == 5_000
        assert q.max_weekly_pointcloud_dispatches == 250
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
        # The max_ glob picks up the weekly budgets too — load-bearing naming.
        assert q.max_weekly_grid_dispatches == 0
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


class TestResolveOwnerConfig:
    pytestmark = pytest.mark.anyio

    # resolve_owner_config surfaces the effective tier alongside the quotas.
    # Each test uses a unique owner id because it is @lru-cached.

    async def test_reports_effective_tier_and_quotas(self):
        client = _fake_firestore(exists=True, data={"tier": "application"})
        with patch("api.quota.firestore_client", client):
            cfg = await resolve_owner_config("cfg-app", Access.APPLICATION)
        assert isinstance(cfg, OwnerQuotaConfig)
        assert cfg.tier == "application"
        assert cfg.quotas.max_active_grids == 100

    async def test_missing_doc_reports_standard(self):
        client = _fake_firestore(exists=False)
        with patch("api.quota.firestore_client", client):
            cfg = await resolve_owner_config("cfg-missing", Access.PERSONAL)
        assert cfg.tier == "standard"
        assert cfg.quotas == Quotas()

    async def test_unknown_tier_normalizes_to_standard(self):
        # An unrecognized stored tier reports the tier actually in effect, so
        # tier and quotas never disagree.
        client = _fake_firestore(exists=True, data={"tier": "platinum"})
        with patch("api.quota.firestore_client", client):
            cfg = await resolve_owner_config("cfg-bad-tier", Access.PERSONAL)
        assert cfg.tier == "standard"
        assert cfg.quotas == Quotas()

    async def test_malformed_overrides_keep_tier_use_defaults(self):
        data = {"tier": "standard", "quota_overrides": {"max_active_features": "x"}}
        client = _fake_firestore(exists=True, data=data)
        with patch("api.quota.firestore_client", client):
            cfg = await resolve_owner_config("cfg-bad-value", Access.PERSONAL)
        assert cfg.tier == "standard"
        assert cfg.quotas == Quotas()

    async def test_resolve_quotas_wraps_owner_config(self):
        data = {"quota_overrides": {"max_active_grids": 3}}
        client = _fake_firestore(exists=True, data=data)
        with patch("api.quota.firestore_client", client):
            quotas = await resolve_quotas("cfg-wrap", Access.PERSONAL)
        assert isinstance(quotas, Quotas)
        assert quotas.max_active_grids == 3


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
            "window_reset_on": None,
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
            if spec.weekly_field is not None:
                assert spec.weekly_field in Quotas.model_fields, collection

    def test_dispatching_types_have_weekly_budgets(self):
        # The five async job types carry a weekly dispatch budget with a
        # matching counter field; domains and applications dispatch no jobs.
        for spec in _RESOURCE_QUOTAS.values():
            if spec.storage_field is not None:
                assert spec.weekly_field is not None, spec.label
                assert spec.counter_field is not None, spec.label
            else:
                assert spec.weekly_field is None, spec.label
                assert spec.counter_field is None, spec.label

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
        # Non-windowed rejections don't carry the reset field at all.
        assert "window_reset_on" not in exc.value.detail

    def test_window_reset_on_serializes_to_json(self):
        # The detail must survive FastAPI's json.dumps of the 429 body.
        with pytest.raises(HTTPException) as exc:
            _raise_quota_exceeded(
                quota="max_weekly_grid_dispatches",
                message="msg",
                current=500,
                limit=500,
                retry_after=False,
                window_reset_on=datetime(2026, 7, 20, tzinfo=UTC),
            )
        assert exc.value.detail["window_reset_on"] == "2026-07-20T00:00:00Z"
        json.dumps(exc.value.detail)


class TestWeekMath:
    def test_iso_week_id_mid_week(self):
        assert iso_week_id(datetime(2026, 7, 15, 12, 0, tzinfo=UTC)) == "2026-W29"

    def test_iso_week_id_uses_iso_year_at_boundary(self):
        # 2025-12-29 is a Monday belonging to ISO week 1 of 2026.
        assert iso_week_id(datetime(2025, 12, 29, tzinfo=UTC)) == "2026-W01"
        # 2027-01-01 is a Friday belonging to ISO week 53 of 2026.
        assert iso_week_id(datetime(2027, 1, 1, tzinfo=UTC)) == "2026-W53"

    def test_iso_week_id_zero_pads(self):
        assert iso_week_id(datetime(2026, 1, 7, tzinfo=UTC)) == "2026-W02"

    def test_next_week_start_is_upcoming_monday(self):
        # Wednesday -> next Monday.
        got = next_week_start(datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
        assert got == datetime(2026, 7, 20, tzinfo=UTC)

    def test_next_week_start_from_sunday_night(self):
        got = next_week_start(datetime(2026, 7, 19, 23, 59, 59, tzinfo=UTC))
        assert got == datetime(2026, 7, 20, tzinfo=UTC)

    def test_next_week_start_from_monday_is_a_full_week_out(self):
        got = next_week_start(datetime(2026, 7, 20, 0, 0, tzinfo=UTC))
        assert got == datetime(2026, 7, 27, tzinfo=UTC)


def _agg(value):
    """An aggregation-query stand-in whose get() returns [[obj(value=...)]]"""
    obj = MagicMock()
    obj.get = AsyncMock(return_value=[[SimpleNamespace(value=value)]])
    return obj


def _fake_enforce_client(
    *,
    count: int = 0,
    active: int = 0,
    total_bytes: int = 0,
    budget_data: dict | None = None,
    budget_error: Exception | None = None,
) -> MagicMock:
    """A firestore_client stand-in for enforce_create_quotas.

    Serves the three aggregation queries for the resource collection and the
    weekly budget doc read (``budget_data=None`` means the doc doesn't exist;
    ``budget_error`` makes the read raise).
    """
    base = MagicMock()
    base.count.return_value = _agg(count)
    base.where.return_value.count.return_value = _agg(active)
    base.sum.return_value = _agg(total_bytes)

    snapshot = MagicMock()
    snapshot.exists = budget_data is not None
    snapshot.to_dict.return_value = budget_data
    budget_doc = MagicMock()
    if budget_error is not None:
        budget_doc.get = AsyncMock(side_effect=budget_error)
    else:
        budget_doc.get = AsyncMock(return_value=snapshot)

    client = MagicMock()

    def collection(name):
        col = MagicMock()
        if name == CREATE_BUDGETS_COLLECTION:
            col.document.return_value.collection.return_value.document.return_value = (
                budget_doc
            )
        else:
            col.where.return_value = base
        return col

    client.collection.side_effect = collection
    client._budget_doc = budget_doc
    return client


def _fake_request(owner_id: str = "owner-weekly") -> MagicMock:
    request = MagicMock()
    request.state = SimpleNamespace(id=owner_id, access=Access.PERSONAL)
    return request


class TestEnforceWeeklyBudget:
    pytestmark = pytest.mark.anyio

    # resolve_quotas is patched directly so the @lru cache never interferes.

    async def _enforce(self, client, request, collection=GRIDS_COLLECTION, **kwargs):
        with (
            patch("api.quota.firestore_client", client),
            patch("api.quota.resolve_quotas", AsyncMock(return_value=Quotas())),
        ):
            await enforce_create_quotas(collection, request, **kwargs)

    async def test_under_limit_passes_and_stashes_state(self):
        client = _fake_enforce_client(budget_data={"grid_dispatches": 10})
        request = _fake_request()
        await self._enforce(client, request)
        budget = request.state.weekly_budget
        assert budget.quota_field == "max_weekly_grid_dispatches"
        assert budget.counter_field == "grid_dispatches"
        assert budget.used == 10
        assert budget.limit == Quotas().max_weekly_grid_dispatches
        assert budget.reset_at == next_week_start(datetime.now(UTC))

    async def test_missing_doc_counts_as_zero(self):
        client = _fake_enforce_client(budget_data=None)
        request = _fake_request()
        await self._enforce(client, request)
        assert request.state.weekly_budget.used == 0

    async def test_at_limit_raises_budget_shaped_429(self):
        limit = Quotas().max_weekly_grid_dispatches
        client = _fake_enforce_client(budget_data={"grid_dispatches": limit})
        request = _fake_request()
        with pytest.raises(HTTPException) as exc:
            await self._enforce(client, request)
        assert exc.value.status_code == 429
        assert exc.value.detail["quota"] == "max_weekly_grid_dispatches"
        assert exc.value.detail["current"] == limit
        # Budget 429s carry the window reset, never Retry-After: the week-out
        # reset would be a retry footgun.
        assert exc.value.headers is None
        assert exc.value.detail["window_reset_on"]
        assert "delete" not in exc.value.detail["message"].lower()

    async def test_read_error_fails_open_with_warning(self, caplog):
        client = _fake_enforce_client(budget_error=RuntimeError("firestore down"))
        request = _fake_request()
        with caplog.at_level(logging.WARNING):
            await self._enforce(client, request)
        assert "failing open" in caplog.text
        # The stash records the unknown state so headers are skipped later,
        # but the increment still runs.
        assert request.state.weekly_budget.used is None

    async def test_dispatch_false_skips_budget_entirely(self):
        client = _fake_enforce_client(budget_error=RuntimeError("must not be read"))
        request = _fake_request()
        await self._enforce(client, request, dispatch=False)
        assert getattr(request.state, "weekly_budget", None) is None
        client._budget_doc.get.assert_not_awaited()

    async def test_domains_have_no_weekly_budget(self):
        client = _fake_enforce_client(budget_error=RuntimeError("must not be read"))
        request = _fake_request()
        await self._enforce(client, request, collection=DOMAINS_COLLECTION)
        assert getattr(request.state, "weekly_budget", None) is None
        client._budget_doc.get.assert_not_awaited()


class TestRegisterDispatch:
    def _budget(self, *, used, limit=500):
        return _WeeklyBudget(
            owner_id="owner-rd",
            quota_field="max_weekly_grid_dispatches",
            counter_field="grid_dispatches",
            limit=limit,
            used=used,
            reset_at=datetime.now(UTC) + timedelta(days=3),
        )

    def test_sets_headers_and_schedules_increment(self):
        request = _fake_request()
        request.state.weekly_budget = self._budget(used=10)
        response = Response()
        background_tasks = BackgroundTasks()
        register_dispatch(request, response, background_tasks)
        # Remaining accounts for the dispatch just made (10 used + this one).
        assert response.headers["RateLimit"].startswith(
            '"max_weekly_grid_dispatches";r=489;t='
        )
        seconds = int(response.headers["RateLimit"].rsplit("t=", 1)[1])
        assert 0 < seconds <= 3 * 24 * 3600
        assert (
            response.headers["RateLimit-Policy"]
            == '"max_weekly_grid_dispatches";q=500;w=604800'
        )
        assert len(background_tasks.tasks) == 1
        task = background_tasks.tasks[0]
        assert task.func is _increment_budget
        assert task.args == ("owner-rd", "grid_dispatches")

    def test_remaining_floors_at_zero(self):
        request = _fake_request()
        request.state.weekly_budget = self._budget(used=499, limit=500)
        response = Response()
        register_dispatch(request, response, BackgroundTasks())
        assert ";r=0;" in response.headers["RateLimit"]

    def test_failed_open_read_skips_headers_but_still_increments(self):
        request = _fake_request()
        request.state.weekly_budget = self._budget(used=None)
        response = Response()
        background_tasks = BackgroundTasks()
        register_dispatch(request, response, background_tasks)
        assert "RateLimit" not in response.headers
        assert len(background_tasks.tasks) == 1

    def test_noop_without_stashed_budget(self):
        request = _fake_request()
        response = Response()
        background_tasks = BackgroundTasks()
        register_dispatch(request, response, background_tasks)
        assert "RateLimit" not in response.headers
        assert background_tasks.tasks == []


class TestIncrementBudget:
    pytestmark = pytest.mark.anyio

    def _client_with_set(self, *, error: Exception | None = None):
        budget_doc = MagicMock()
        budget_doc.set = AsyncMock(side_effect=error)
        client = MagicMock()
        client.collection.return_value.document.return_value.collection.return_value.document.return_value = budget_doc
        client._budget_doc = budget_doc
        return client

    async def test_merges_increment_with_week_metadata(self):
        client = self._client_with_set()
        with patch("api.quota.firestore_client", client):
            await _increment_budget("owner-inc", "grid_dispatches")
        client.collection.assert_called_once_with(CREATE_BUDGETS_COLLECTION)
        client.collection.return_value.document.assert_called_once_with("owner-inc")
        week_doc = client.collection.return_value.document.return_value
        week_doc.collection.assert_called_once_with("weeks")
        now = datetime.now(UTC)
        week_doc.collection.return_value.document.assert_called_once_with(
            iso_week_id(now)
        )
        (payload,), kwargs = client._budget_doc.set.await_args
        assert kwargs == {"merge": True}
        assert isinstance(payload["grid_dispatches"], Increment)
        assert payload["owner_id"] == "owner-inc"
        assert payload["iso_week"] == iso_week_id(now)
        # Week docs are retained indefinitely as a per-owner usage ledger.
        assert "expire_at" not in payload

    async def test_write_error_is_swallowed_with_warning(self, caplog):
        client = self._client_with_set(error=RuntimeError("firestore down"))
        with caplog.at_level(logging.WARNING):
            with patch("api.quota.firestore_client", client):
                await _increment_budget("owner-inc-err", "grid_dispatches")
        assert "failing open" in caplog.text
