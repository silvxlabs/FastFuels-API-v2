"""
api/v2/resources/grids/resample/schema.py

Schema models for grid resampling operations.

Resample endpoints change a grid's spatial resolution while preserving its
bands. For example, resampling a 30m LANDFIRE grid to 2m for QUIC-Fire input.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.modifications import GridModification


class ResamplingMethod(StrEnum):
    """Resampling methods supported by the resample endpoint.

    Includes both interpolation methods (for upsampling) and aggregation
    methods (for downsampling), though any method can be used in either
    direction.
    """

    # Interpolation methods
    nearest = "nearest"
    bilinear = "bilinear"
    cubic = "cubic"
    cubic_spline = "cubic_spline"
    lanczos = "lanczos"

    # Aggregation methods
    average = "average"
    mode = "mode"
    max = "max"
    min = "min"
    median = "median"
    first_quartile = "first_quartile"
    third_quartile = "third_quartile"
    sum = "sum"
    root_mean_square = "root_mean_square"


class ResampleSource(BaseModel):
    """Source metadata for grids created via resampling."""

    name: Literal["resample"] = "resample"
    source_grid_id: str = Field(..., description="Grid to resample")
    target_resolution: float = Field(
        ..., description="Target resolution after resampling (meters)"
    )
    method: ResamplingMethod = Field(..., description="Default resampling method")
    method_overrides: dict[str, ResamplingMethod] = Field(
        default_factory=dict,
        description="Per-band resampling method overrides keyed by band key",
    )


class CreateResampleRequest(BaseModel):
    """Request to create a grid by resampling an existing grid.

    Unlike entry-point grid creation requests, domain_id is not required
    because derived grids carry the same domain reference as their source.
    """

    source_grid_id: str = Field(..., description="Grid to resample")
    resolution: float = Field(
        ...,
        ge=1,
        description=(
            "Target resolution in meters. Minimum 1m. "
            "Contact the developers if you need sub-meter resolution."
        ),
    )
    method: ResamplingMethod = Field(
        default=ResamplingMethod.bilinear,
        description="Default resampling method",
    )
    method_overrides: dict[str, ResamplingMethod] = Field(
        default_factory=dict,
        description=(
            "Per-band resampling method overrides keyed by band key. "
            "For example, use nearest-neighbor for categorical bands "
            "while using bilinear for continuous bands."
        ),
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)
