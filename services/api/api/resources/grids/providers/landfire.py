"""
api/v2/resources/grids/providers/landfire.py

Shared base model for LANDFIRE data products.

LANDFIRE provides raster products at 30m resolution. Product-specific
subclasses (FBFM40, Topography, etc.) live in their respective product
directories and inherit from LandfireSource.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)
from lib.landfire import CoverageStatus, LandfireRelease


class LandfireSource(BaseModel):
    """Base source specification for LANDFIRE data products."""

    name: Literal["landfire"] = "landfire"
    product: str
    version: str
    description: str = ""
    extent_buffer_cells: int = Field(0, ge=0, le=10)
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget
    )


class NonBurnableFuelModel(StrEnum):
    """Non-burnable LANDFIRE fuel model codes, shared across FBFM40/FBFM13."""

    NB1 = "NB1"  # Urban/developed (91)
    NB2 = "NB2"  # Snow/ice (92)
    NB3 = "NB3"  # Agriculture (93)
    NB8 = "NB8"  # Water (98)
    NB9 = "NB9"  # Bare ground (99)


def check_no_duplicate_non_burnable(v):
    """Shared validator for remove_non_burnable fields."""
    if v is not None and len(v) != len(set(v)):
        raise ValueError("Duplicate non-burnable fuel model codes are not allowed")
    return v


class LandfireCoverage(StrEnum):
    """How much of a domain a LANDFIRE release covers."""

    full = "full"
    partial = "partial"
    none = "none"
    unpublished = "unpublished"


_COVERAGE_BY_STATUS = {
    CoverageStatus.FULL: LandfireCoverage.full,
    CoverageStatus.PARTIAL: LandfireCoverage.partial,
    CoverageStatus.NONE: LandfireCoverage.none,
    CoverageStatus.NO_SUCH_PRODUCT: LandfireCoverage.unpublished,
}


class LandfireCreateLink(BaseModel):
    """The create request that fetches a release for this domain."""

    href: str = Field(
        description="Path of the create endpoint, relative to the API base URL."
    )
    method: Literal["POST"] = "POST"
    body: dict = Field(description="Request body selecting this release.")


class LandfireReleaseLinks(BaseModel):
    """Actions available for a release on this domain."""

    create: LandfireCreateLink | None = Field(
        description=(
            "Request that creates a grid from this release. Null when the "
            "release doesn't cover the domain, so the create would be rejected."
        )
    )


class LandfireReleaseCoverage(BaseModel):
    """One release of a LANDFIRE product and its coverage of the domain."""

    version: str = Field(description="LANDFIRE landscape vintage year.")
    season: str | None = Field(
        description="LANDFIRE Seasonal Fuels window, or null for the annual product."
    )
    year: int | None = Field(
        description=(
            "Calendar year the data represents: the vintage for annual "
            "releases, the projected season year for seasonal ones. Null for "
            "a season LANDFIRE hasn't published yet."
        )
    )
    coverage: LandfireCoverage = Field(
        description=(
            "`full` or `partial` releases can be created for this domain; "
            "`none` is served elsewhere but not here; `unpublished` isn't "
            "served anywhere yet."
        )
    )
    links: LandfireReleaseLinks


class LandfireCoverageResponse(BaseModel):
    """Response model for the LANDFIRE release coverage pre-flight check."""

    product: str
    latest: LandfireReleaseCoverage | None = Field(
        description=(
            "The release representing the most recent point in time that "
            "fully covers the domain. Null when none does."
        )
    )
    releases: list[LandfireReleaseCoverage] = Field(
        description=(
            "Every release the API serves, newest first by the time the data "
            "represents. Seasons LANDFIRE hasn't published yet are listed where "
            "they will land once published."
        )
    )


def build_landfire_coverage_response(
    product: str, releases: list[LandfireRelease], create_href: str
) -> LandfireCoverageResponse:
    """Build the coverage response, attaching a create link to each creatable release."""
    items = []
    for release in releases:
        coverage = _COVERAGE_BY_STATUS[release.coverage]
        create = None
        if coverage in (LandfireCoverage.full, LandfireCoverage.partial):
            body = {"version": release.version}
            if release.season is not None:
                body["season"] = release.season
            create = LandfireCreateLink(href=create_href, body=body)
        items.append(
            LandfireReleaseCoverage(
                version=release.version,
                season=release.season,
                year=release.year,
                coverage=coverage,
                links=LandfireReleaseLinks(create=create),
            )
        )

    latest = next((r for r in items if r.coverage is LandfireCoverage.full), None)
    return LandfireCoverageResponse(product=product, latest=latest, releases=items)
