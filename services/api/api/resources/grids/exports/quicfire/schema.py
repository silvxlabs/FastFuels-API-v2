"""
api/v2/resources/grids/exports/quicfire/schema.py

Schema models for the QUIC-Fire combined export endpoint.

A QUIC-Fire export bundles surface fuel + canopy fuel + (optional) topography
grids into a zip archive of `trees*.dat` files for QUIC-Fire input. Every
physical quantity QUIC-Fire needs is expressed as a `{grid_id, band}` role.

Used at: POST /v2/domains/{domain_id}/grids/exports/quicfire
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FieldSource(BaseModel):
    """A single physical quantity drawn from one band on one grid.

    Every per-role input to the QUIC-Fire export uses this shape so the schema
    is uniform across roles. The forward path for `nfuel>1` (when QUIC-Fire's
    multi-fuel-type capability becomes relevant) is to allow each per-fuel-type
    role to accept `FieldSource | dict[FuelType, FieldSource]`; today's scalar
    requests keep working unchanged when that lands.
    """

    grid_id: str = Field(..., description="Grid containing the source band.")
    band: str = Field(..., description="Band key on that grid (e.g. 'fuel_load.1hr').")


class QuicfireExportRequest(BaseModel):
    """Request body for creating a QUIC-Fire combined export.

    Five required roles produce `treesrhof.dat`, `treesmoist.dat`, and
    `treesfueldepth.dat`. `topography` (optional) produces `topo.dat`. The
    SAVR pair (optional, both-or-neither) produces `treesss.dat`.
    """

    canopy_bulk_density: FieldSource = Field(
        ...,
        description=(
            "3D canopy bulk density (kg/m³). Must reference a band on a 3D "
            "tree grid, e.g. 'bulk_density.foliage.live'."
        ),
    )
    canopy_moisture: FieldSource = Field(
        ...,
        description=(
            "3D canopy fuel moisture (%). Typically a tree-grid band, e.g. "
            "'fuel_moisture.live'."
        ),
    )
    canopy_savr: FieldSource | None = Field(
        default=None,
        description=(
            "3D canopy SAVR (m⁻¹), optional. When provided, must be paired "
            "with `surface_savr`; together they produce `treesss.dat` in the "
            "output zip."
        ),
    )
    surface_fuel_load: FieldSource = Field(
        ...,
        description=(
            "2D surface fuel load (kg/m²). Typically a lookup-grid band, e.g. "
            "'fuel_load.1hr'."
        ),
    )
    surface_fuel_depth: FieldSource = Field(
        ...,
        description=(
            "2D surface fuel bed depth (m). Typically a lookup-grid band, "
            "e.g. 'fuel_depth'."
        ),
    )
    surface_moisture: FieldSource = Field(
        ...,
        description=(
            "2D surface fuel moisture (%). Often a uniform-grid band, e.g. "
            "'fuel_moisture.1hr'."
        ),
    )
    surface_savr: FieldSource | None = Field(
        default=None,
        description=("2D surface SAVR (m⁻¹), optional. Pairs with `canopy_savr`."),
    )
    topography: FieldSource | None = Field(
        default=None,
        description=("2D elevation (m), optional. When provided, produces `topo.dat`."),
    )

    expiration_days: int = Field(
        default=7,
        ge=1,
        le=7,
        description="Days until the signed download URL expires (max 7).",
    )
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def savr_pair_or_none(self) -> "QuicfireExportRequest":
        if (self.canopy_savr is None) != (self.surface_savr is None):
            raise ValueError(
                "canopy_savr and surface_savr must both be provided or both "
                "omitted; treesss.dat needs both layers."
            )
        return self


class QuicfireExportSource(BaseModel):
    """Stored source metadata for a QUIC-Fire export, recorded in `Export.source`.

    `resolved` snapshots CRS/transform/shape/units per role at request time so
    the exporter is a pure consumer and the export is reproducible even if a
    source grid is later modified or deleted.
    """

    name: Literal["quicfire"] = "quicfire"
    domain_id: str

    canopy_bulk_density: FieldSource
    canopy_moisture: FieldSource
    canopy_savr: FieldSource | None = None
    surface_fuel_load: FieldSource
    surface_fuel_depth: FieldSource
    surface_moisture: FieldSource
    surface_savr: FieldSource | None = None
    topography: FieldSource | None = None

    resolved: dict = Field(
        ...,
        description=(
            "Snapshot of CRS, transform, shape, and band units per role at "
            "request time. Used by the exporter to consume pre-validated data."
        ),
    )
