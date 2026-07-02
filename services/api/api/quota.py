"""
api/quota.py

Per-owner usage quotas (epic #340, phase 1).

Default limits apply to every owner. Each resource-creating endpoint calls
``enforce_create_quotas()`` before writing; if the owner already has too many
jobs in flight, a 429 with a structured, human-readable detail is raised.

Later phases extend this module without touching the call sites: phase 2 makes
``resolve_quotas()`` read a per-owner ``tier`` / ``quota_overrides`` from the
owner document; phase 3 adds total-count and per-type storage checks to
``enforce_create_quotas()``.
"""

import logging

from fastapi import HTTPException, Request, status
from google.cloud.firestore import FieldFilter
from pydantic import BaseModel, Field

from api.db.documents import firestore_client
from api.resources.keys.schema import Access
from api.schema import JobStatus
from lib.config import (
    EXPORTS_COLLECTION,
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    POINT_CLOUDS_COLLECTION,
    SUPPORT_EMAIL,
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


async def resolve_quotas(owner_id: str, access: Access) -> Quotas:
    """Resolve the quota set for an owner.

    Phase 1: every owner resolves to the default tier — no Firestore read, no
    per-owner overrides. The signature is final; phase 2 fills in the body to
    read ``tier`` / ``quota_overrides`` from the owner document (keyed on
    ``access``) and cache the result.
    """
    return Quotas(**TIER_PRESETS[_DEFAULT_TIER])


# Collection -> (active-jobs quota field, singular resource label). Collections
# absent here have no active-jobs limit (domains and API keys are count-limited
# only, which arrives in phase 3).
_ACTIVE_JOB_QUOTAS: dict[str, tuple[str, str]] = {
    GRIDS_COLLECTION: ("max_active_grids", "grid"),
    EXPORTS_COLLECTION: ("max_active_exports", "export"),
    INVENTORIES_COLLECTION: ("max_active_inventories", "inventory"),
    FEATURES_COLLECTION: ("max_active_features", "feature"),
    POINT_CLOUDS_COLLECTION: ("max_active_pointclouds", "point cloud"),
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


async def enforce_create_quotas(collection: str, request: Request) -> None:
    """Check create-time quotas for the authenticated owner; raise 429 if over.

    Called at the top of every resource-creating endpoint, before any validation
    or Firestore write. Phase 1 enforces the active-jobs limit only:
    ``count(owner_id ==, status in {pending, running}) >= max_active_<type>``.
    Collections without an active-jobs limit (domains, API keys) are a no-op.
    """
    quota = _ACTIVE_JOB_QUOTAS.get(collection)
    if quota is None:
        return
    field, label = quota

    owner_id = request.state.id
    quotas = await resolve_quotas(owner_id, request.state.access)
    limit = getattr(quotas, field)

    query = (
        firestore_client.collection(collection)
        .where(filter=FieldFilter("owner_id", "==", owner_id))
        .where(filter=FieldFilter("status", "in", _ACTIVE_STATUSES))
    )
    current = (await query.count().get())[0][0].value

    if current >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=QuotaExceededDetail(
                quota=field,
                message=(
                    f"You have {current} {label} jobs in progress (limit {limit}). "
                    f"Wait for jobs to complete or delete unneeded {label}s, then "
                    f"retry. To request a higher limit, contact {SUPPORT_EMAIL}."
                ),
                current=current,
                limit=limit,
            ).model_dump(),
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
