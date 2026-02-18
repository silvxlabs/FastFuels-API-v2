"""
api/v2/resources/applications/schema.py

Pydantic models for application resources.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from api.schema import PaginatedResponse


class CreateApplicationRequest(BaseModel):
    """Request body for creating an application."""

    name: str = Field(..., max_length=255, description="Name of the application.")
    description: str | None = Field(
        None, max_length=2000, description="Description of the application."
    )


class UpdateApplicationRequest(BaseModel):
    """Request body for updating an application."""

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


class ListApplicationsResponse(PaginatedResponse):
    """Paginated response for listing applications."""

    applications: list[Application] = Field(..., description="A list of applications.")
