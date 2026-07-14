"""
api/quota.py

Per-owner usage quotas (epic #340).

Each resource-creating endpoint calls ``enforce_create_quotas()`` before
writing; if the owner is over an active-jobs limit, a structured 429 is raised.
The limits come from ``resolve_quotas()``, which layers an owner's tier and overrides on top of the defaults.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from fastapi import BackgroundTasks, HTTPException, Request, Response, status
from google.cloud.firestore import AsyncDocumentReference, FieldFilter, Increment
from pydantic import BaseModel, Field, ValidationError
from ring import lru

from api.db.documents import firestore_client
from api.resources.keys.schema import Access
from api.schema import JobStatus
from lib.config import (
    APPLICATIONS_COLLECTION,
    CREATE_BUDGETS_COLLECTION,
    DOMAINS_COLLECTION,
    EXPORTS_COLLECTION,
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    KEYS_COLLECTION,
    POINT_CLOUDS_COLLECTION,
    SUPPORT_EMAIL,
    USERS_COLLECTION,
)

logger = logging.getLogger(__name__)

GiB = 2**30  # bytes; storage quotas are expressed in GiB


class Quotas(BaseModel):
    """Usage limits for an owner. Field defaults are the standard tier."""

    # Concurrent jobs (status in {pending, running}), per resource type.
    max_active_grids: int = 25
    max_active_exports: int = 10
    max_active_inventories: int = 10
    max_active_features: int = 10
    max_active_pointclouds: int = 5

    # Total resource counts (enforced in phase 3).
    max_domains: int = 50
    max_grids: int = 1_000
    max_exports: int = 500
    max_inventories: int = 500
    max_features: int = 500
    max_pointclouds: int = 50
    max_api_keys: int = 50
    max_applications: int = 5

    # Storage: GCS artifact bytes per resource type (enforced in phase 3).
    max_grid_storage_bytes: int = 50 * GiB
    max_export_storage_bytes: int = 25 * GiB
    max_inventory_storage_bytes: int = 10 * GiB
    max_feature_storage_bytes: int = 1 * GiB
    max_pointcloud_storage_bytes: int = 50 * GiB

    # Weekly worker-job dispatch budgets (#431); reset at ISO-week boundaries.
    max_weekly_grid_dispatches: int = 500
    max_weekly_export_dispatches: int = 250
    max_weekly_inventory_dispatches: int = 250
    max_weekly_feature_dispatches: int = 250
    max_weekly_pointcloud_dispatches: int = 50

    # Lifecycle (enforced by the sweeper in phase 5); None = never expires.
    resource_ttl_days: int | None = 180
    failed_resource_ttl_days: int | None = 14


# Named limit presets applied per owner in phase 2. "standard" is the schema
# defaults; "suspended" is the abuse-response kill switch (zero every create
# limit while reads and deletes keep working). Phase 1 resolves every owner to
# the default tier.
TIER_PRESETS: dict[str, dict] = {
    "standard": {},
    "application": {
        "max_active_grids": 100,
        "max_active_exports": 50,
        "max_grids": 10_000,
        "max_exports": 5_000,
        "max_grid_storage_bytes": 500 * GiB,
        "max_export_storage_bytes": 250 * GiB,
        "max_pointcloud_storage_bytes": 500 * GiB,
        "max_weekly_grid_dispatches": 5_000,
        "max_weekly_export_dispatches": 2_500,
        "max_weekly_inventory_dispatches": 2_500,
        "max_weekly_feature_dispatches": 1_000,
        "max_weekly_pointcloud_dispatches": 250,
        "resource_ttl_days": None,
    },
    "partner": {},  # negotiated per partner; defaults until one is onboarded
    "suspended": {
        field: 0 for field in Quotas.model_fields if field.startswith("max_")
    },
}

_DEFAULT_TIER = "standard"


@dataclass(frozen=True)
class OwnerQuotaConfig:
    """An owner's resolved quota configuration: the effective tier and limits."""

    tier: str  # always a known TIER_PRESETS key (the tier actually in effect)
    quotas: Quotas


@lru(force_asyncio=True, expire=300)
async def resolve_owner_config(owner_id: str, access: Access) -> OwnerQuotaConfig:
    """Resolve an owner's tier and quotas from its owner document.

    Applies the owner's tier preset and any ``quota_overrides`` on top of the
    defaults. A missing or malformed document resolves to the default tier. The
    reported ``tier`` is always a known preset — an unrecognized stored value
    reports (and applies) the default tier, so ``tier`` and ``quotas`` never
    disagree.
    """
    collection = (
        USERS_COLLECTION if access == Access.PERSONAL else APPLICATIONS_COLLECTION
    )
    doc = await firestore_client.collection(collection).document(owner_id).get()
    data = doc.to_dict() if doc.exists else {}

    tier = data.get("tier") or _DEFAULT_TIER
    if tier not in TIER_PRESETS:
        tier = _DEFAULT_TIER
    preset = TIER_PRESETS[tier]
    overrides = data.get("quota_overrides") or {}
    try:
        quotas = Quotas(**{**preset, **overrides})
    except (TypeError, ValidationError):
        logger.error(
            "Malformed quota config for owner %s; using tier defaults", owner_id
        )
        quotas = Quotas(**preset)
    return OwnerQuotaConfig(tier=tier, quotas=quotas)


async def resolve_quotas(owner_id: str, access: Access) -> Quotas:
    """Resolve an owner's usage limits (see :func:`resolve_owner_config`)."""
    return (await resolve_owner_config(owner_id, access)).quotas


@dataclass(frozen=True)
class _ResourceQuota:
    """The quota fields checked when creating a resource of one type.

    ``active_field`` and ``storage_field`` are ``None`` for types without that
    limit (a domain has no in-flight job and no stored artifact).
    ``weekly_field`` / ``counter_field`` are ``None`` for types whose creation
    never dispatches a worker job (domains, applications).
    """

    label: str  # singular noun for user messages ("grid", "point cloud")
    active_field: str | None  # max_active_<type>: concurrent pending/running jobs
    count_field: str  # max_<type> / max_domains: total resources owned
    storage_field: str | None  # max_<type>_storage_bytes: summed artifact bytes
    weekly_field: str | None = None  # max_weekly_<type>_dispatches
    counter_field: str | None = None  # budget-doc count field, e.g. "grid_dispatches"


# Collection -> the quota fields enforced on create. A collection absent here is
# not quota-checked (API keys are pending a metering-identity decision).
_RESOURCE_QUOTAS: dict[str, _ResourceQuota] = {
    GRIDS_COLLECTION: _ResourceQuota(
        "grid",
        "max_active_grids",
        "max_grids",
        "max_grid_storage_bytes",
        "max_weekly_grid_dispatches",
        "grid_dispatches",
    ),
    EXPORTS_COLLECTION: _ResourceQuota(
        "export",
        "max_active_exports",
        "max_exports",
        "max_export_storage_bytes",
        "max_weekly_export_dispatches",
        "export_dispatches",
    ),
    INVENTORIES_COLLECTION: _ResourceQuota(
        "inventory",
        "max_active_inventories",
        "max_inventories",
        "max_inventory_storage_bytes",
        "max_weekly_inventory_dispatches",
        "inventory_dispatches",
    ),
    FEATURES_COLLECTION: _ResourceQuota(
        "feature",
        "max_active_features",
        "max_features",
        "max_feature_storage_bytes",
        "max_weekly_feature_dispatches",
        "feature_dispatches",
    ),
    POINT_CLOUDS_COLLECTION: _ResourceQuota(
        "point cloud",
        "max_active_pointclouds",
        "max_pointclouds",
        "max_pointcloud_storage_bytes",
        "max_weekly_pointcloud_dispatches",
        "pointcloud_dispatches",
    ),
    DOMAINS_COLLECTION: _ResourceQuota("domain", None, "max_domains", None),
    APPLICATIONS_COLLECTION: _ResourceQuota(
        "application", None, "max_applications", None
    ),
}

# A job is "in flight" while pending or running; both count so the limit caps
# total concurrent work. completed / failed never count.
_ACTIVE_STATUSES = [JobStatus.pending.value, JobStatus.running.value]

RETRY_AFTER_SECONDS = 60


class QuotaExceededDetail(BaseModel):
    """Structured ``detail`` for a 429 quota rejection.

    The flat ``{reason, quota, message, current, limit}`` shape is the template
    for future structured error details: a machine-readable ``reason`` code plus
    flat, typed context fields.
    """

    reason: str = Field("QUOTA_EXCEEDED", description="Machine-readable error code.")
    quota: str = Field(..., description="The Quotas field that was exceeded.")
    message: str = Field(..., description="Human-readable explanation and next steps.")
    current: int = Field(..., description="The owner's current usage for this quota.")
    limit: int = Field(..., description="The limit that was reached.")
    window_reset_on: datetime | None = Field(
        None,
        description=(
            "When a windowed (weekly) quota resets; absent for non-windowed quotas."
        ),
    )


# Declared by every resource-creating route so the 429 is part of the OpenAPI
# surface (the user-facing documentation).
QUOTA_429_RESPONSE: dict = {
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": QuotaExceededDetail,
        "description": (
            "Quota exceeded. The `detail` names the exact `quota`; active-job "
            "rejections also include a `Retry-After` header, and weekly-budget "
            "rejections include `window_reset_on` plus the IETF `RateLimit` / "
            "`RateLimit-Policy` headers with `r=0`."
        ),
    }
}


def _raise_quota_exceeded(
    *,
    quota: str,
    message: str,
    current: int,
    limit: int,
    retry_after: bool,
    window_reset_on: datetime | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Raise a structured 429. ``retry_after`` adds the header for limits that
    clear by waiting (active jobs) and omits it for those that need deletion
    (count, storage) or a window reset (weekly budgets, which carry
    ``window_reset_on`` in the detail and the ``RateLimit`` headers instead)."""
    all_headers = dict(headers or {})
    if retry_after:
        all_headers["Retry-After"] = str(RETRY_AFTER_SECONDS)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=QuotaExceededDetail(
            quota=quota,
            message=message,
            current=current,
            limit=limit,
            window_reset_on=window_reset_on,
        ).model_dump(mode="json", exclude_none=True),
        headers=all_headers or None,
    )


# Weekly dispatch budgets (#431). One Firestore doc per owner per ISO week at
# create-budgets-v2/{owner_id}/weeks/{iso_week} holds per-type dispatch
# counters. A new week is a new doc id, so the reset is free. One doc per
# owner-week is deliberate: Firestore sustains ~1 write/s/doc, far above any
# owner's dispatch rate; sharding is the documented escalation if that ever
# changes. The budget fails OPEN — an outage in the limiter must never block a
# create — and every fail-open is logged at WARNING so it stays visible.
# Week docs are retained indefinitely: they are the only durable record of
# per-owner weekly activity (resource docs vanish on delete).

_WEEK_SECONDS = 7 * 24 * 3600


def iso_week_id(now: datetime) -> str:
    """The ISO week id for ``now``, e.g. ``"2026-W28"``.

    Uses the ISO year, which differs from the calendar year around January 1
    (2025-12-29 falls in 2026-W01).
    """
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def next_week_start(now: datetime) -> datetime:
    """The next ISO week boundary: the upcoming Monday at 00:00 UTC.

    ``now`` must be timezone-aware UTC.
    """
    monday = now.date() - timedelta(days=now.weekday())
    return datetime.combine(monday + timedelta(days=7), time.min, tzinfo=UTC)


@dataclass(frozen=True)
class _WeeklyBudget:
    """Weekly-budget state stashed on ``request.state.weekly_budget`` by
    :func:`enforce_create_quotas` so :func:`register_dispatch` can set the
    RateLimit headers and schedule the increment without a second read."""

    owner_id: str
    quota_field: str  # max_weekly_<type>_dispatches
    counter_field: str  # budget-doc count field, e.g. "grid_dispatches"
    limit: int
    used: int | None  # None: the read failed open
    reset_at: datetime


def _ratelimit_headers(
    quota_field: str, remaining: int, limit: int, reset_at: datetime
) -> dict[str, str]:
    """The IETF ``RateLimit`` / ``RateLimit-Policy`` header pair for a weekly
    budget — sent on successful dispatches (gauge) and on the budget 429
    (``r=0``), so generic client throttle logic works on both."""
    reset_seconds = max(int((reset_at - datetime.now(UTC)).total_seconds()), 0)
    return {
        "RateLimit": f'"{quota_field}";r={remaining};t={reset_seconds}',
        "RateLimit-Policy": f'"{quota_field}";q={limit};w={_WEEK_SECONDS}',
    }


def _budget_ref(owner_id: str, week_id: str) -> AsyncDocumentReference:
    return (
        firestore_client.collection(CREATE_BUDGETS_COLLECTION)
        .document(owner_id)
        .collection("weeks")
        .document(week_id)
    )


async def _read_budget_used(
    owner_id: str, week_id: str, counter_field: str
) -> int | None:
    """Read the owner's dispatch count for the week; never raises.

    A missing doc means nothing dispatched yet (0). ``None`` means the read
    failed and the caller must fail open.
    """
    try:
        doc = await _budget_ref(owner_id, week_id).get()
        return int((doc.to_dict() or {}).get(counter_field, 0) or 0)
    except Exception:
        logger.warning(
            "Weekly budget read failed for owner %s; failing open",
            owner_id,
            exc_info=True,
        )
        return None


async def _increment_budget(owner_id: str, counter_field: str) -> None:
    """Increment the owner's weekly dispatch counter; never raises.

    Runs as a fire-and-forget background task after the response is sent.
    """
    try:
        week_id = iso_week_id(datetime.now(UTC))
        await _budget_ref(owner_id, week_id).set(
            {
                counter_field: Increment(1),
                "owner_id": owner_id,
                "iso_week": week_id,
            },
            merge=True,
        )
    except Exception:
        logger.warning(
            "Weekly budget increment (%s) failed for owner %s; failing open",
            counter_field,
            owner_id,
            exc_info=True,
        )


def register_dispatch(
    request: Request, response: Response, background_tasks: BackgroundTasks
) -> None:
    """Record a worker-job dispatch against the owner's weekly budget.

    Called once, immediately after a job is dispatched — never before, so
    requests that fail validation don't consume budget. Sets the IETF
    ``RateLimit`` / ``RateLimit-Policy`` headers on the response (skipped when
    the budget read failed open) and schedules the counter increment. No-op if
    :func:`enforce_create_quotas` did not run a weekly check for this request.
    """
    budget: _WeeklyBudget | None = getattr(request.state, "weekly_budget", None)
    if budget is None:
        return
    if budget.used is not None:
        remaining = max(budget.limit - budget.used - 1, 0)
        response.headers.update(
            _ratelimit_headers(
                budget.quota_field, remaining, budget.limit, budget.reset_at
            )
        )
    background_tasks.add_task(_increment_budget, budget.owner_id, budget.counter_field)


async def enforce_create_quotas(
    collection: str, request: Request, *, dispatch: bool = True
) -> None:
    """Check create-time quotas for the authenticated owner; raise 429 if over.

    Called at the top of every resource-creating endpoint, before any validation
    or Firestore write. Runs up to three aggregation queries concurrently over
    ``owner_id ==``: the active-jobs count (also filtered ``status in {pending,
    running}``), the total-resource count, and the summed ``size_bytes``. The sum
    is a separate query because its ``(owner_id, size_bytes)`` index is sparse —
    docs without ``size_bytes`` are absent from it — so folding it into the count
    would undercount in-flight resources. Checks are evaluated active-jobs ->
    count -> storage -> weekly budget; the first over its limit raises.
    Collections absent from ``_RESOURCE_QUOTAS`` are a no-op.

    The weekly dispatch budget is read in the same gather and checked only when
    ``dispatch`` is true (the default) and the type has a budget. Endpoints that
    create a resource without commissioning a worker job (layerset features)
    pass ``dispatch=False``. On pass, the budget state is stashed on
    ``request.state.weekly_budget`` for :func:`register_dispatch`.
    """
    spec = _RESOURCE_QUOTAS.get(collection)
    if spec is None:
        return

    owner_id = request.state.id
    quotas = await resolve_quotas(owner_id, request.state.access)
    base = firestore_client.collection(collection).where(
        filter=FieldFilter("owner_id", "==", owner_id)
    )

    # Each check is its own aggregation query, run concurrently. count() over
    # owner_id== counts every doc the owner has (non-sparse single-field index).
    # The storage sum needs the (owner_id, size_bytes) composite index, which is
    # sparse — it omits docs that lack size_bytes (still-in-flight jobs) — so it
    # can NOT be folded into count() without undercounting; it stays separate.
    aggregations = {"count": base.count(alias="count")}
    if spec.active_field:
        aggregations["active"] = base.where(
            filter=FieldFilter("status", "in", _ACTIVE_STATUSES)
        ).count(alias="active")
    if spec.storage_field:
        aggregations["bytes"] = base.sum("size_bytes", alias="bytes")

    check_weekly = dispatch and spec.weekly_field is not None
    now = datetime.now(UTC)

    names = list(aggregations)
    coros = [aggregations[n].get() for n in names]
    if check_weekly:
        # _read_budget_used never raises (fails open to None), so appending it
        # to the gather cannot break the aggregation checks.
        coros.append(_read_budget_used(owner_id, iso_week_id(now), spec.counter_field))
    results = await asyncio.gather(*coros)
    values = {n: res[0][0].value for n, res in zip(names, results)}

    count = values["count"]
    active = values.get("active")
    total_bytes = values.get("bytes") or 0

    if spec.active_field:
        limit = getattr(quotas, spec.active_field)
        if active >= limit:
            _raise_quota_exceeded(
                quota=spec.active_field,
                current=active,
                limit=limit,
                retry_after=True,
                message=(
                    f"You have {active} {spec.label} jobs in progress (limit "
                    f"{limit}). Wait for jobs to complete or delete unneeded "
                    f"{spec.label}s, then retry. To request a higher limit, "
                    f"contact {SUPPORT_EMAIL}."
                ),
            )

    limit = getattr(quotas, spec.count_field)
    if count >= limit:
        _raise_quota_exceeded(
            quota=spec.count_field,
            current=count,
            limit=limit,
            retry_after=False,
            message=(
                f"You have {count} {spec.label}s (limit {limit}). Delete unneeded "
                f"{spec.label}s, then retry. To request a higher limit, contact "
                f"{SUPPORT_EMAIL}."
            ),
        )

    if spec.storage_field:
        limit = getattr(quotas, spec.storage_field)
        if total_bytes >= limit:
            _raise_quota_exceeded(
                quota=spec.storage_field,
                current=total_bytes,
                limit=limit,
                retry_after=False,
                message=(
                    f"Your {spec.label}s use {total_bytes / GiB:.1f} GiB of storage "
                    f"(limit {limit / GiB:.0f} GiB). Delete unneeded {spec.label}s "
                    f"to free space, then retry. To request a higher limit, contact "
                    f"{SUPPORT_EMAIL}."
                ),
            )

    if check_weekly:
        used = results[-1]
        limit = getattr(quotas, spec.weekly_field)
        reset_at = next_week_start(now)
        # Deleting resources refunds nothing: the budget counts dispatches, so
        # the only ways forward are the window reset or a raised limit.
        if used is not None and used >= limit:
            _raise_quota_exceeded(
                quota=spec.weekly_field,
                current=used,
                limit=limit,
                retry_after=False,
                window_reset_on=reset_at,
                headers=_ratelimit_headers(spec.weekly_field, 0, limit, reset_at),
                message=(
                    f"Your weekly {spec.label} budget is spent ({used}/{limit} "
                    f"jobs this week). It resets {reset_at:%Y-%m-%d} (UTC). To "
                    f"request a higher limit, contact {SUPPORT_EMAIL}."
                ),
            )
        request.state.weekly_budget = _WeeklyBudget(
            owner_id=owner_id,
            quota_field=spec.weekly_field,
            counter_field=spec.counter_field,
            limit=limit,
            used=used,
            reset_at=reset_at,
        )


# Output-key -> collection for GET /users/me/usage. The keys are the JSON field
# names the read API exposes (e.g. "pointclouds", not the "point cloud" message
# label in _RESOURCE_QUOTAS); order sets the response order. API keys are handled
# separately because their metering identity depends on the caller (see below).
_USAGE_COLLECTIONS: dict[str, str] = {
    "grids": GRIDS_COLLECTION,
    "exports": EXPORTS_COLLECTION,
    "inventories": INVENTORIES_COLLECTION,
    "features": FEATURES_COLLECTION,
    "pointclouds": POINT_CLOUDS_COLLECTION,
    "domains": DOMAINS_COLLECTION,
    "applications": APPLICATIONS_COLLECTION,
}


async def get_usage(owner_id: str, access: Access) -> dict:
    """Compute an owner's current usage against their resolved limits.

    Backs ``GET /users/me/usage``. Runs the same aggregations enforcement uses —
    a total ``count()`` per resource type, an active-jobs ``count()`` and a
    ``sum(size_bytes)`` where those limits apply, and an API-key ``count()`` —
    all concurrently in one ``asyncio.gather``. Not on any hot path. Returns the
    §9 usage shape; ``lifecycle.next_expiry_on`` is ``None`` until the retention
    sweeper ships (phase 5).
    """
    quotas = (await resolve_owner_config(owner_id, access)).quotas

    # Build every aggregation up front, keyed by (output-key, metric), so they
    # can be gathered together and reassembled by key.
    jobs: dict[tuple[str, str], object] = {}
    for out_key, collection in _USAGE_COLLECTIONS.items():
        spec = _RESOURCE_QUOTAS[collection]
        base = firestore_client.collection(collection).where(
            filter=FieldFilter("owner_id", "==", owner_id)
        )
        jobs[(out_key, "total")] = base.count(alias="v").get()
        if spec.active_field:
            jobs[(out_key, "active")] = (
                base.where(filter=FieldFilter("status", "in", _ACTIVE_STATUSES))
                .count(alias="v")
                .get()
            )
        if spec.storage_field:
            jobs[(out_key, "bytes")] = base.sum("size_bytes", alias="v").get()

    # API keys: mirror the keys list endpoint's identity — personal callers own
    # keys by creator_id (they may have created application keys too), while an
    # application owns keys by owner_id.
    key_field = "creator_id" if access == Access.PERSONAL else "owner_id"
    jobs[("api_keys", "total")] = (
        firestore_client.collection(KEYS_COLLECTION)
        .where(filter=FieldFilter(key_field, "==", owner_id))
        .count(alias="v")
        .get()
    )

    keys = list(jobs)
    results = await asyncio.gather(*(jobs[k] for k in keys))
    values = {k: res[0][0].value for k, res in zip(keys, results)}

    def _count(usage: object, limit_field: str) -> dict:
        return {"usage": int(usage or 0), "limit": getattr(quotas, limit_field)}

    usage: dict = {}
    for out_key, collection in _USAGE_COLLECTIONS.items():
        spec = _RESOURCE_QUOTAS[collection]
        entry = {"total": _count(values[(out_key, "total")], spec.count_field)}
        if spec.active_field:
            entry["active"] = _count(values[(out_key, "active")], spec.active_field)
        if spec.storage_field:
            entry["storage"] = {
                "usage_bytes": int(values[(out_key, "bytes")] or 0),
                "limit_bytes": getattr(quotas, spec.storage_field),
            }
        usage[out_key] = entry

    usage["api_keys"] = {"total": _count(values[("api_keys", "total")], "max_api_keys")}
    usage["lifecycle"] = {
        "resource_ttl_days": quotas.resource_ttl_days,
        "failed_resource_ttl_days": quotas.failed_resource_ttl_days,
        "next_expiry_on": None,
    }
    return usage
