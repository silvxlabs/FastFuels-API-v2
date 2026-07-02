"""
api/v2/resources/applications/schema.py

Pydantic models for application resources.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from api.schema import PaginatedResponse

# Reject unknown fields so an attempt to set an admin-only field (tier,
# quota_overrides) is a 422, not a silent no-op.
_REQUEST_CONFIG = ConfigDict(extra="forbid")


class CreateApplicationRequest(BaseModel):
    """Request body for creating an application."""

    model_config = _REQUEST_CONFIG

    name: str = Field(..., max_length=255, description="Name of the application.")
    description: str | None = Field(
        None, max_length=2000, description="Description of the application."
    )


class UpdateApplicationRequest(BaseModel):
    """Request body for updating an application."""

    model_config = _REQUEST_CONFIG

    name: str | None = Field(
        None, max_length=255, description="New name for the application."
    )
    description: str | None = Field(
        None, max_length=2000, description="New description for the application."
    )


class Application(BaseModel):
    """Represents an application that can own API keys."""

    id: str = Field(..., description="Unique identifier for the application.")
    owner_id: str = Field(
        ...,
        description="The unique ID of the user who owns the application.",
    )
    name: str = Field(..., description="Name of the application.")
    description: str | None = Field(None, description="Description of the application.")
    created_on: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the application was created.",
    )
    modified_on: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the application was last modified.",
    )
    tier: str | None = Field(
        None, description="Quota tier for the application. Set by the FastFuels team."
    )
    quota_overrides: dict | None = Field(
        None,
        description="Per-application quota overrides. Set by the FastFuels team.",
    )


class ListApplicationsResponse(PaginatedResponse):
    """Paginated response for listing applications."""

    applications: list[Application] = Field(..., description="A list of applications.")
