"""
api/v2/resources/grids/exports/quicfire/schema.py

Schema models for the QUIC-Fire combined export endpoint.

A QUIC-Fire export bundles surface fuel + canopy fuel + (optional) topography
grids into a zip archive of `trees*.dat` files for QUIC-Fire input. Every
physical quantity QUIC-Fire needs is expressed as a `{grid_id, band}` role.

Used at: POST /v2/domains/{domain_id}/grids/exports/quicfire
"""

from math import isclose
from typing import Annotated, Literal

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


class QUICFireExportAlignmentDomainTarget(BaseModel):
    """Anchor the fire grid to the Domain bounding box.

    Output cells tile the Domain bbox at the given `dx` / `dy`, padded
    outward if the bbox isn't already a whole multiple. `dz` sets the
    uniform vertical cell size; the exporter always writes `aa1=1` so
    fuel layers map 1:1 to QUIC-Fire cells. Defaults are QUIC-Fire's
    recommended values (2 m horizontal, 1 m vertical).
    """

    target: Literal["domain"] = "domain"
    dx: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Horizontal fire-grid cell size in x (UTM east-west), in meters. "
            "QUIC-Fire recommends 2 m."
        ),
    )
    dy: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Horizontal fire-grid cell size in y (UTM north-south), in meters. "
            "Must equal `dx` — non-square fire-grid cells are not supported."
        ),
    )
    dz: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Vertical fire-grid cell size, in meters. QUIC-Fire recommends 1 m."
        ),
    )

    @model_validator(mode="after")
    def _require_square_cells(self) -> "QUICFireExportAlignmentDomainTarget":
        if not isclose(self.dx, self.dy, abs_tol=1e-9):
            raise ValueError(
                f"dx ({self.dx}) and dy ({self.dy}) must be equal — "
                "non-square fire-grid cells are not supported."
            )
        return self


class QUICFireExportAlignmentGridTarget(BaseModel):
    """Anchor the fire grid to an existing grid's lattice.

    Useful when role grids share a non-Domain-anchored lattice (e.g. all
    chained off a `target="native"` master grid). The referenced grid's
    CRS, transform, and shape become the fire grid's horizontal lattice;
    vertical cell size and layer count are taken from the canopy grid.
    """

    target: Literal["grid"]
    grid_id: str = Field(
        ...,
        description=(
            "Existing grid whose horizontal lattice (CRS, transform, shape) "
            "the fire grid should match exactly."
        ),
    )


QUICFireExportAlignmentSpec = Annotated[
    QUICFireExportAlignmentDomainTarget | QUICFireExportAlignmentGridTarget,
    Field(discriminator="target"),
]


class QuicfireExportRequest(BaseModel):
    """Request body for creating a QUIC-Fire combined export.

    Five required roles produce `treesrhof.dat`, `treesmoist.dat`, and
    `treesfueldepth.dat`. `topography` (optional) produces `topo.dat`. The
    SAVR pair (optional, both-or-neither) produces `treesss.dat`.

    The fire grid is defined by the `alignment` field — either the Domain
    bounding box padded to `(dx, dy)` (with `dz` vertical), or the lattice
    of an existing grid. Every role grid must be lattice-aligned to this
    fire grid and cover its full extent; otherwise the request is rejected.
    The exporter only crops oversized roles by integer slicing — it never
    resamples or reprojects.
    """

    alignment: QUICFireExportAlignmentSpec = Field(
        default_factory=QUICFireExportAlignmentDomainTarget,
        description=(
            "How the fire grid lattice is defined. Discriminated by `target`: "
            "`'domain'` (default) pads the Domain bbox to `(dx, dy)`; `'grid'` "
            "matches an existing grid's lattice exactly. Omit for the default "
            "Domain-anchored 2 m / 1 m fire grid."
        ),
    )

    canopy_bulk_density: FieldSource = Field(
        ...,
        description=(
            "3D canopy bulk density (kg/m**3). Must reference a band on a 3D "
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
            "3D canopy SAVR (1/m), optional. When provided, must be paired "
            "with `surface_savr`; together they produce `treesss.dat` in the "
            "output zip."
        ),
    )
    surface_fuel_load: FieldSource = Field(
        ...,
        description=(
            "2D surface fuel load (kg/m**2). Typically a lookup-grid band, e.g. "
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
        description=("2D surface SAVR (1/m), optional. Pairs with `canopy_savr`."),
    )
    topography: FieldSource | None = Field(
        default=None,
        description=("2D elevation (m), optional. When provided, produces `topo.dat`."),
    )

    rhof_merge: Literal["sum"] = Field(
        default="sum",
        description=(
            "How to combine canopy and surface bulk density at the bottom slab "
            "(k=0). Currently only 'sum' is supported: "
            "`merged[0] = canopy[0] + surface_load / dz`. Mass-additive."
        ),
    )
    moist_merge: Literal["max", "weighted_avg"] = Field(
        default="max",
        description=(
            "How to combine canopy and surface fuel moisture at the bottom "
            "slab (k=0). "
            "`'max'` (default, v1-parity): "
            "`merged[0] = max(canopy_moist[0], surface_moist)`. "
            "`'weighted_avg'`: "
            "`merged[0] = (canopy_rhof[0] * canopy_moist[0] + surface_rhof_layer * surface_moist) / "
            "(canopy_rhof[0] + surface_rhof_layer)` (both moistures already "
            "converted to fraction)."
        ),
    )
    savr_merge: Literal["weighted_avg"] = Field(
        default="weighted_avg",
        description=(
            "How to combine canopy and surface SAVR at the bottom slab. "
            "Currently only 'weighted_avg' is supported (mass-weighted SAVR, "
            "then converted to particle size scale `2/SAVR` before write). "
            "Only applies when both `canopy_savr` and `surface_savr` are set."
        ),
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

    alignment: QUICFireExportAlignmentSpec

    canopy_bulk_density: FieldSource
    canopy_moisture: FieldSource
    canopy_savr: FieldSource | None = None
    surface_fuel_load: FieldSource
    surface_fuel_depth: FieldSource
    surface_moisture: FieldSource
    surface_savr: FieldSource | None = None
    topography: FieldSource | None = None

    rhof_merge: Literal["sum"] = "sum"
    moist_merge: Literal["max", "weighted_avg"] = "max"
    savr_merge: Literal["weighted_avg"] = "weighted_avg"

    resolved: dict = Field(
        ...,
        description=(
            "Snapshot of CRS, transform, shape, and band units per role at "
            "request time. Used by the exporter to consume pre-validated data."
        ),
    )
