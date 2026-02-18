"""
api/v2/schema.py

Shared schema models for the v2 API.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Status of an async job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobProgress(BaseModel):
    """Progress information for running async jobs.

    Provides real-time feedback during long-running operations.

    Attributes:
        percent: Completion percentage (0-100). Null for indeterminate operations
            like "Connecting to LANDFIRE..." where progress can't be quantified.
        message: Human-readable status message describing what's happening.
    """

    percent: int | None = Field(
        None,
        ge=0,
        le=100,
        description="Completion percentage (0-100), null if indeterminate",
    )
    message: str = Field(
        ...,
        description="Human-readable status message, e.g. 'Fetching LANDFIRE data...'",
    )


class JobError(BaseModel):
    """Error information for failed async jobs.

    Provides structured error information with user-facing messages and
    developer debugging information. The traceback is stored in Firestore
    but excluded from API responses.

    Attributes:
        code: Machine-readable error code for programmatic handling.
            Examples: "LANDFIRE_COVERAGE_ERROR", "SOURCE_GRID_NOT_FOUND"
        message: User-friendly explanation of what went wrong.
        suggestion: Optional actionable advice for resolving the error.
        traceback: Full Python stack trace for debugging. Stored in Firestore
            but not included in API responses.
    """

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="User-friendly error message")
    suggestion: str | None = Field(
        None, description="Actionable suggestion for the user"
    )
    traceback: str | None = Field(
        None,
        exclude=True,
        description="Full stack trace (stored but not exposed in API responses)",
    )


class SortOrder(StrEnum):
    """Sort order for list results."""

    ascending = "ascending"
    descending = "descending"


class PaginatedResponse(BaseModel):
    """Base model for paginated list responses.

    Subclasses should add a field for the actual data items, e.g.:

        class ListDomainsResponse(PaginatedResponse):
            domains: list[Domain]
    """

    current_page: int = Field(
        ..., description="The current page number (zero-indexed)."
    )
    page_size: int = Field(..., description="The number of items per page.")
    total_items: int = Field(
        ..., description="The total number of items across all pages."
    )
