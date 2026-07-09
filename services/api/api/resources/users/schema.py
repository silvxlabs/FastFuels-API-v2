"""
api/resources/users/schema.py

Response models for the read API of the quota system (GET /users/me and
GET /users/me/usage). "me" is the authenticated owner — a user or an
application — with ``kind`` disambiguating the two.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from api.quota import Quotas


class UserMeResponse(BaseModel):
    """The authenticated owner's identity and resolved quota configuration."""

    id: str = Field(..., description="The authenticated owner's unique ID.")
    kind: Literal["user", "application"] = Field(
        ...,
        description="Whether the credential authenticated a user or an application.",
    )
    tier: str = Field(..., description="The quota tier in effect for this owner.")
    quotas: Quotas = Field(
        ...,
        description="The owner's resolved usage limits (defaults, tier, and overrides).",
    )


class UsageCount(BaseModel):
    """A count-based usage/limit pair (resources or concurrent jobs)."""

    usage: int = Field(..., description="Current count.")
    limit: int = Field(..., description="The limit this count is measured against.")


class UsageStorage(BaseModel):
    """A storage usage/limit pair, in bytes."""

    usage_bytes: int = Field(..., description="Summed GCS artifact bytes in use.")
    limit_bytes: int = Field(..., description="The storage limit, in bytes.")


class JobResourceUsage(BaseModel):
    """Usage for a resource type that produces jobs and stores artifacts."""

    active: UsageCount = Field(
        ...,
        description="Concurrent in-flight (pending/running) jobs vs. the active limit.",
    )
    total: UsageCount = Field(
        ..., description="Total resources of this type vs. the count limit."
    )
    storage: UsageStorage = Field(
        ..., description="Summed stored-artifact bytes vs. the per-type storage limit."
    )


class CountUsage(BaseModel):
    """Usage for a count-only resource type (domains, applications, API keys)."""

    total: UsageCount = Field(
        ..., description="Total resources of this type vs. the count limit."
    )


class UsageLifecycle(BaseModel):
    """Retention policy in effect for the owner's resources."""

    resource_ttl_days: int | None = Field(
        ...,
        description="Days a resource is retained after last modification; null never expires.",
    )
    failed_resource_ttl_days: int | None = Field(
        ..., description="Shorter retention for failed resources; null never expires."
    )
    next_expiry_on: datetime | None = Field(
        None,
        description="When the owner's next resource is scheduled to expire. "
        "Populated once retention enforcement ships.",
    )


class Usage(BaseModel):
    """The authenticated owner's current usage against their resolved limits."""

    grids: JobResourceUsage
    exports: JobResourceUsage
    inventories: JobResourceUsage
    features: JobResourceUsage
    pointclouds: JobResourceUsage
    domains: CountUsage
    applications: CountUsage
    api_keys: CountUsage
    lifecycle: UsageLifecycle
