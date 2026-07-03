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

from fastapi import HTTPException, Request, status
from google.cloud.firestore import FieldFilter
from pydantic import BaseModel, Field, ValidationError
from ring import lru

from api.db.documents import firestore_client
from api.resources.keys.schema import Access
from api.schema import JobStatus
from lib.config import (
    APPLICATIONS_COLLECTION,
    DOMAINS_COLLECTION,
    EXPORTS_COLLECTION,
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
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

    # Storage: GCS artifact bytes per resource type (enforced in phase 3).
    max_grid_storage_bytes: int = 50 * GiB
    max_export_storage_bytes: int = 25 * GiB
    max_inventory_storage_bytes: int = 10 * GiB
    max_feature_storage_bytes: int = 1 * GiB
    max_pointcloud_storage_bytes: int = 50 * GiB

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
        "resource_ttl_days": None,
    },
    "partner": {},  # negotiated per partner; defaults until one is onboarded
    "suspended": {
        field: 0 for field in Quotas.model_fields if field.startswith("max_")
    },
}

_DEFAULT_TIER = "standard"


@lru(force_asyncio=True, expire=300)
async def resolve_quotas(owner_id: str, access: Access) -> Quotas:
    """Resolve an owner's quotas from its owner document.

    Applies the owner's tier preset and any ``quota_overrides`` on top of the
    defaults. A missing or malformed document resolves to the defaults.
    """
    collection = (
        USERS_COLLECTION if access == Access.PERSONAL else APPLICATIONS_COLLECTION
    )
    doc = await firestore_client.collection(collection).document(owner_id).get()
    data = doc.to_dict() if doc.exists else {}

    preset = TIER_PRESETS.get(data.get("tier") or _DEFAULT_TIER, {})
    overrides = data.get("quota_overrides") or {}
    try:
        return Quotas(**{**preset, **overrides})
    except (TypeError, ValidationError):
        logger.error(
            "Malformed quota config for owner %s; using tier defaults", owner_id
        )
        return Quotas(**preset)


@dataclass(frozen=True)
class _ResourceQuota:
    """The quota fields checked when creating a resource of one type.

    ``active_field`` and ``storage_field`` are ``None`` for types without that
    limit (a domain has no in-flight job and no stored artifact).
    """

    label: str  # singular noun for user messages ("grid", "point cloud")
    active_field: str | None  # max_active_<type>: concurrent pending/running jobs
    count_field: str  # max_<type> / max_domains: total resources owned
    storage_field: str | None  # max_<type>_storage_bytes: summed artifact bytes


# Collection -> the quota fields enforced on create. A collection absent here is
# not quota-checked (API keys are pending a metering-identity decision).
_RESOURCE_QUOTAS: dict[str, _ResourceQuota] = {
    GRIDS_COLLECTION: _ResourceQuota(
        "grid", "max_active_grids", "max_grids", "max_grid_storage_bytes"
    ),
    EXPORTS_COLLECTION: _ResourceQuota(
        "export", "max_active_exports", "max_exports", "max_export_storage_bytes"
    ),
    INVENTORIES_COLLECTION: _ResourceQuota(
        "inventory",
        "max_active_inventories",
        "max_inventories",
        "max_inventory_storage_bytes",
    ),
    FEATURES_COLLECTION: _ResourceQuota(
        "feature", "max_active_features", "max_features", "max_feature_storage_bytes"
    ),
    POINT_CLOUDS_COLLECTION: _ResourceQuota(
        "point cloud",
        "max_active_pointclouds",
        "max_pointclouds",
        "max_pointcloud_storage_bytes",
    ),
    DOMAINS_COLLECTION: _ResourceQuota("domain", None, "max_domains", None),
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


# Declared by every resource-creating route so the 429 is part of the OpenAPI
# surface (the user-facing documentation).
QUOTA_429_RESPONSE: dict = {
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": QuotaExceededDetail,
        "description": (
            "Quota exceeded. The `detail` names the exact `quota`; active-job "
            "rejections also include a `Retry-After` header."
        ),
    }
}


def _raise_quota_exceeded(
    *, quota: str, message: str, current: int, limit: int, retry_after: bool
) -> None:
    """Raise a structured 429. ``retry_after`` adds the header for limits that
    clear by waiting (active jobs) and omits it for those that need deletion
    (count, storage)."""
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=QuotaExceededDetail(
            quota=quota, message=message, current=current, limit=limit
        ).model_dump(),
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)} if retry_after else None,
    )


async def enforce_create_quotas(collection: str, request: Request) -> None:
    """Check create-time quotas for the authenticated owner; raise 429 if over.

    Called at the top of every resource-creating endpoint, before any validation
    or Firestore write. Runs up to three aggregation queries concurrently over
    ``owner_id ==``: the active-jobs count (also filtered ``status in {pending,
    running}``), the total-resource count, and the summed ``size_bytes``. The sum
    is a separate query because its ``(owner_id, size_bytes)`` index is sparse —
    docs without ``size_bytes`` are absent from it — so folding it into the count
    would undercount in-flight resources. Checks are evaluated active-jobs ->
    count -> storage; the first over its limit raises. Collections absent from
    ``_RESOURCE_QUOTAS`` are a no-op.
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

    names = list(aggregations)
    results = await asyncio.gather(*(aggregations[n].get() for n in names))
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
