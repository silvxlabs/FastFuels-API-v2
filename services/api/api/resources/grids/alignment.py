"""Grid alignment schema.

Discriminated union of alignment targets for raster-backed grid creation.
The default (`target="domain"`) lands output cells on the domain-origin
lattice so cross-source composition works by construction. Users can opt
into source-pixel preservation (`target="native"`) or exact alignment to
an existing grid (`target="grid"`).
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ResamplingMethod(StrEnum):
    nearest = "nearest"
    bilinear = "bilinear"
    cubic = "cubic"
    cubic_spline = "cubic_spline"
    lanczos = "lanczos"
    average = "average"
    mode = "mode"
    max = "max"
    min = "min"
    median = "median"
    first_quartile = "first_quartile"
    third_quartile = "third_quartile"
    sum = "sum"
    root_mean_square = "root_mean_square"


class GridAlignmentDomainTarget(BaseModel):
    """Anchor output to the domain origin.

    `resolution=None` uses the source's native cell size. Output cells tile
    the domain bounding box (already snapped at domain creation if
    `pad_to_resolution` was set).
    """

    target: Literal["domain"] = "domain"
    resolution: float | None = Field(default=None, ge=1.0)
    method: ResamplingMethod | None = None


class GridAlignmentNativeTarget(BaseModel):
    """Preserve the source raster's pixel anchor.

    `resolution=None` is exactly today's behavior — no anchor or resolution
    change beyond the standard ROI-CRS reprojection.
    """

    target: Literal["native"]
    resolution: float | None = Field(default=None, ge=1.0)
    method: ResamplingMethod | None = None


class GridAlignmentGridTarget(BaseModel):
    """Align to an existing grid by id.

    `resolution=None` produces an exact lattice match (CRS, transform, and
    shape from the target grid). With an explicit `resolution`, the output
    keeps the target's CRS and origin but uses the new cell size; shape is
    recomputed from the target grid's bounds.
    """

    target: Literal["grid"]
    grid_id: str
    resolution: float | None = Field(default=None, ge=1.0)
    method: ResamplingMethod | None = None


GridAlignmentSpecification = Annotated[
    GridAlignmentDomainTarget | GridAlignmentNativeTarget | GridAlignmentGridTarget,
    Field(discriminator="target"),
]
