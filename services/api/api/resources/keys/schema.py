"""
api/v2/resources/keys/schema.py

Pydantic models for API key resources.
"""

from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from api.resources.input_requirements import application, input_requirements
from api.schema import PaginatedResponse


class Scope(StrEnum):
    """Available permissions for an API key."""

    READ = auto()
    WRITE = auto()


class Access(StrEnum):
    """Access types for an API key."""

    PERSONAL = auto()
    APPLICATION = auto()


class CreateKeyRequestBase(BaseModel):
    """Fields shared by every key creation request."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="A name to semantically identify the key.",
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description="An optional description of the key's purpose.",
    )
    valid_days: int = Field(
        30,
        ge=1,
        description="Number of days for which this key will be valid.",
    )
    scopes: list[Scope] = Field(
        default=[Scope.READ],
        description="A list of scopes available to the key.",
    )


class CreatePersonalKeyRequest(CreateKeyRequestBase):
    """Request body for a key that acts as you."""

    access: Literal[Access.PERSONAL] = Field(
        default=Access.PERSONAL,
        description="Access type for the API key.",
    )


class CreateApplicationKeyRequest(CreateKeyRequestBase):
    """Request body for a key that acts as one of your applications."""

    model_config = ConfigDict(
        json_schema_extra=input_requirements(application("application_id")),
    )

    access: Literal[Access.APPLICATION] = Field(
        description="Access type for the API key."
    )
    application_id: str = Field(
        min_length=1, description="Application ID accessed by the API key."
    )


def _default_access(value):
    if isinstance(value, dict) and "access" not in value:
        return {**value, "access": Access.PERSONAL}
    return value


CreateKeyRequest = Annotated[
    Annotated[
        CreatePersonalKeyRequest | CreateApplicationKeyRequest,
        Field(discriminator="access"),
    ],
    BeforeValidator(_default_access),
]


class Key(BaseModel):
    """Represents an API key for authenticating programmatic API access."""

    id: str = Field(
        ...,
        description="Unique identifier for the key (SHA-256 hash of the secret).",
    )
    owner_id: str = Field(
        ...,
        description="The unique ID of the user or application who owns the key.",
    )
    creator_id: str = Field(
        ...,
        description="The unique ID of the human user who created the key.",
    )
    name: str = Field(..., description="A name to semantically identify the key.")
    description: str | None = Field(
        None, description="An optional description of the key's purpose."
    )
    valid_days: int = Field(
        30,
        description="Number of days for which this key will be valid.",
    )
    scopes: list[Scope] = Field(
        default=[Scope.READ],
        description="A list of scopes available to the key.",
    )
    access: Access = Field(
        default=Access.PERSONAL,
        description="Access type for the API key.",
    )
    application_id: str | None = Field(
        None,
        description="Application ID accessed by the API key.",
    )
    created_on: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="The date and time the key was created.",
    )
    expires_on: datetime = Field(
        default=None,
        description="The date at which this key is no longer valid.",
    )

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        return self.expires_on < datetime.now(UTC)

    def has_permission(self, method: str) -> bool:
        """Check if the key has appropriate permissions for the HTTP method."""
        if method != "GET" and Scope.WRITE not in self.scopes:
            return False
        return True


class CreateKeyResponse(Key):
    """Response for key creation. Includes the secret, which is shown only once."""

    secret: str = Field(
        ...,
        description="The API key secret. Store this securely — it cannot be retrieved again.",
    )


class ListKeysResponse(PaginatedResponse):
    """Paginated response for listing API keys."""

    keys: list[Key] = Field(..., description="A list of API keys.")
