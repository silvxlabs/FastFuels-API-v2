"""
api/v2/resources/grids/resample/schema.py

Schema models for grid resampling operations.

Resample endpoints change a grid's spatial resolution and/or anchor while
preserving its bands. The new shape is controlled by the shared
``alignment`` discriminated union; see ``api.resources.grids.alignment``.
"""

from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
    ResamplingMethod,
)
from api.resources.grids.modification_models import GridModification


class ResampleSource(BaseModel):
    """Source metadata for grids created via resampling."""

    name: Literal["resample"] = "resample"
    source_grid_id: str = Field(..., description="Grid to resample")
    source_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The source grid's `checksum` at the time this grid was created from "
            "it. Compare it against the source grid's current `checksum` to tell "
            "whether the source has changed since."
        ),
    )
    alignment: GridAlignmentSpecification = Field(
        ...,
        description="Output alignment target.",
    )
    method_overrides: dict[str, ResamplingMethod] = Field(
        default_factory=dict,
        description="Per-band resampling method overrides keyed by band key",
    )


class CreateResampleRequest(BaseModel):
    """Request to create a grid by resampling an existing grid.

    Unlike entry-point grid creation requests, ``domain_id`` is not required
    because derived grids carry the same domain reference as their source.

    The ``alignment`` field controls the output lattice. ``alignment.resolution``
    is required for ``target="domain"`` and ``target="native"``; for
    ``target="grid"`` it is optional (defaults to the target grid's exact
    transform/shape; if supplied, keeps the target's CRS and origin and
    recomputes shape at the new cell size).
    """

    source_grid_id: str = Field(..., description="Grid to resample")
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget,
        description=(
            'Output alignment target. Default `target="domain"` anchors the '
            "resampled grid to the domain origin."
        ),
    )
    method_overrides: dict[str, ResamplingMethod] = Field(
        default_factory=dict,
        description=(
            "Per-band resampling method overrides keyed by band key. "
            "Wins over ``alignment.method`` for the listed bands. Useful "
            "for using nearest-neighbor on categorical bands while using "
            "bilinear on continuous bands."
        ),
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)
