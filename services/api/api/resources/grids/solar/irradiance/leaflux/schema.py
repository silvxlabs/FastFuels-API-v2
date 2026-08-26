"""
api/v2/resources/grids/solar/irradiance/leaflux/schema.py

Schema models for producing a LeafLux irradiance grid from a 3D voxelized
fuel grid (the source grid's `leaf_area_density` band).

Includes the band vocabulary, the request schema, and the persisted-source
schema stored on the resulting Grid document.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class LeafluxBand(StrEnum):
    """Available output bands for LeafLux irradiance grids.

    Each value is the band's dot-notation key as it appears in the stored
    Grid document. Band definitions (type, unit) live in LEAFLUX_BAND_DEFS.
    """

    irradiance_canopy_relative = "irradiance.canopy.relative"
    irradiance_surface_relative = "irradiance.surface.relative"


LEAFLUX_BAND_DEFS: dict[LeafluxBand, dict] = {
    LeafluxBand.irradiance_canopy_relative: {
        "key": "irradiance.canopy.relative",
        "name": "Relative Irradiance in Canopy",
        "description": "Per-voxel relative irradiance within the canopy [0-1], "
        "from Beer-Lambert attenuation through leaf area density.",
        "type": BandType.continuous,
        "unit": None,
    },
    LeafluxBand.irradiance_surface_relative: {
        "key": "irradiance.surface.relative",
        "name": "Relative Irradiance on Surface",
        "description": "Relative irradiance on the terrain surface [0-1] "
        "beneath the canopy.",
        "type": BandType.continuous,
        "unit": None,
    },
}


def build_leaflux_bands(requested: list[LeafluxBand]) -> list[Band]:
    """Build Band objects for requested LeafLux bands with indices in request order."""
    return [
        Band(index=i, **LEAFLUX_BAND_DEFS[band]) for i, band in enumerate(requested)
    ]


# Surface is default band
def _default_bands() -> list[LeafluxBand]:
    return [LeafluxBand.irradiance_surface_relative]


class IrradianceLeafluxSource(BaseModel):
    """Source metadata stored on the Grid document for reproducibility.

    Records the source grid and every resolved model choice so the
    irradiance grid can be exactly reproduced.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal["irradiance"] = "irradiance"
    input: Literal["grid"] = "grid"
    entity: Literal["solar"] = "solar"

    source_lad_grid_id: str
    source_lad_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The leaf-area-density grid's `checksum` at the time this grid was "
            "created from it. Compare it against that grid's current `checksum` "
            "to tell whether the source has changed since."
        ),
    )
    source_terrain_grid_id: str | None = None
    bands: list[LeafluxBand]
    date_time: datetime
    extinction_coefficient: float


class CreateLeafluxIrradianceRequest(BaseModel):
    """Request body for creating a LeafLux irradiance grid from a 3D fuel grid.

    Does not extend CreateGridRequestBase because 3D grids do not support
    modifications. This is a grid -> grid derivation aligned to the source
    grid's geometry, so there is no resolution input.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_lad_grid_id: str = Field(
        description=(
            "ID of the completed 3D fuel grid whose `leaf_area_density` (LAD) "
            "band drives the Beer-Lambert light attenuation. This is the "
            "primary input the irradiance field is computed from. Named for "
            "the band it consumes so it reads unambiguously alongside "
            "`source_terrain_grid_id`."
        ),
    )

    source_terrain_grid_id: str | None = Field(
        default=None,
        description=(
            "(optional) ID of a completed 2D terrain grid (with an `elevation` "
            "band) in the same domain, used to drape the surface irradiance "
            "band over real terrain instead of a flat plane."
        ),
    )
    bands: list[LeafluxBand] = Field(
        default_factory=_default_bands,
        min_length=1,
        description=(
            "Which output bands to produce. Defaults to `irradiance.surface.relative`."
        ),
    )
    # TODO: For now, requires in UTC, like leaflux. This could be changed if we think it's confusing
    date_time: datetime = Field(
        description="UTC instant at which to compute irradiance.",
    )

    extinction_coefficient: float = Field(
        default=0.5,
        gt=0.0,
        description="Beer-Lambert extinction coefficient (leaflux `extn`).",
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[LeafluxBand]) -> list[LeafluxBand]:
        return validate_no_duplicates(v)
