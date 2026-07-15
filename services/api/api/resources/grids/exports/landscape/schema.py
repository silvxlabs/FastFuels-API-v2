"""
api/v2/resources/grids/exports/landscape/schema.py

Schema models for the landscape combined export endpoint.

A landscape export assembles terrain + surface fuel model + canopy grids into
an 8-band LANDFIRE-style landscape GeoTIFF for operational fire behavior tools
(FlamMap, IFTDSS, WFDSS). Every band is a `{grid_id, band}` role.

Used at: POST /v2/domains/{domain_id}/grids/exports/landscape
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class LandscapeFieldSource(BaseModel):
    """A single landscape band drawn from one band on one grid."""

    grid_id: str = Field(..., description="Grid containing the source band.")
    band: str = Field(..., description="Band key on that grid (e.g. 'elevation').")


class LandscapeExportAlignmentDomainTarget(BaseModel):
    """Anchor the landscape to the Domain bounding box.

    Output cells tile the Domain bbox at `resolution`, padded outward if the
    bbox isn't already a whole multiple. The default 30 m matches LANDFIRE's
    native resolution.
    """

    target: Literal["domain"] = "domain"
    resolution: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Landscape cell size in meters. Defaults to 30 m, LANDFIRE's "
            "native resolution."
        ),
    )


class LandscapeExportAlignmentGridTarget(BaseModel):
    """Anchor the landscape to an existing grid's lattice.

    Useful when role grids share a non-Domain-anchored lattice (e.g. all
    chained off a `target="native"` master grid). The referenced grid's
    CRS, transform, and shape become the landscape lattice.
    """

    target: Literal["grid"]
    grid_id: str = Field(
        ...,
        description=(
            "Existing grid whose lattice (CRS, transform, shape) the "
            "landscape should match exactly."
        ),
    )


LandscapeExportAlignmentSpec = Annotated[
    LandscapeExportAlignmentDomainTarget | LandscapeExportAlignmentGridTarget,
    Field(discriminator="target"),
]


class LandscapeExportRequest(BaseModel):
    """Request body for creating a landscape export.

    Eight required roles produce an 8-band landscape GeoTIFF in LANDFIRE band
    order: elevation, slope, aspect, fuel model, canopy cover, canopy height,
    canopy base height, canopy bulk density. This is the shape modern fire
    behavior tools consume — IFTDSS requires all eight bands for upload.

    The landscape lattice is defined by the `alignment` field — either the
    Domain bounding box tiled at `resolution` (default 30 m, LANDFIRE-native),
    or the lattice of an existing grid. Every role grid must be lattice-aligned
    to the landscape and cover its full extent; otherwise the request is
    rejected with 422. The exporter only crops oversized roles by integer
    slicing — it never resamples or reprojects. To change a grid's resolution
    or anchor, use `POST /v2/domains/{domain_id}/grids/{grid_id}/resample`.
    """

    alignment: LandscapeExportAlignmentSpec = Field(
        default_factory=LandscapeExportAlignmentDomainTarget,
        description=(
            "How the landscape lattice is defined. Discriminated by `target`: "
            "`'domain'` (default) tiles the Domain bbox at `resolution`; "
            "`'grid'` matches an existing grid's lattice exactly. Omit for "
            "the default Domain-anchored 30 m landscape."
        ),
    )
    fire_behavior_fuel_model: Literal["fbfm13", "fbfm40"] = Field(
        ...,
        description=(
            "How the `fuel_model` band's codes should be interpreted: "
            "`'fbfm40'` (Scott-Burgan 40) or `'fbfm13'` (Anderson 13). "
            "Recorded in the landscape file so fire behavior tools apply "
            "the right classification."
        ),
    )

    elevation: LandscapeFieldSource = Field(
        ...,
        description="2D elevation (m), e.g. a topography-grid 'elevation' band.",
    )
    slope: LandscapeFieldSource = Field(
        ...,
        description="2D slope (deg), e.g. a topography-grid 'slope' band.",
    )
    aspect: LandscapeFieldSource = Field(
        ...,
        description=(
            "2D aspect (deg, azimuth from north), e.g. a topography-grid 'aspect' band."
        ),
    )
    fuel_model: LandscapeFieldSource = Field(
        ...,
        description=(
            "2D categorical fire behavior fuel model codes, e.g. an "
            "fbfm40-grid 'fbfm' band. Interpreted per "
            "`fire_behavior_fuel_model`."
        ),
    )
    canopy_cover: LandscapeFieldSource = Field(
        ...,
        description="2D canopy cover (%), e.g. a canopy-grid 'cc' band.",
    )

    canopy_height: LandscapeFieldSource = Field(
        ...,
        description="2D canopy height (m), e.g. a canopy-grid 'chm' band.",
    )
    canopy_base_height: LandscapeFieldSource = Field(
        ...,
        description="2D canopy base height (m), e.g. a canopy-grid 'cbh' band.",
    )
    canopy_bulk_density: LandscapeFieldSource = Field(
        ...,
        description=(
            "2D canopy bulk density (kg/m**3), e.g. a canopy-grid 'cbd' band."
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

    @model_validator(mode="before")
    @classmethod
    def _default_alignment_target(cls, data: object) -> object:
        """Default ``alignment.target`` to ``"domain"`` when omitted.

        ``alignment`` is a discriminated union on ``target``, so Pydantic
        requires the tag even though the domain target is the only default.
        Filling it here lets callers pass just ``{"resolution": 10}`` instead
        of repeating ``"target": "domain"``.
        """
        if isinstance(data, dict) and isinstance(data.get("alignment"), dict):
            data["alignment"].setdefault("target", "domain")
        return data


class LandscapeExportSource(BaseModel):
    """Stored source metadata for a landscape export, recorded in `Export.source`.

    `resolved` snapshots the landscape lattice at request time so the exporter
    is a pure consumer and the export is reproducible even if a source grid is
    later modified or deleted.
    """

    name: Literal["landscape"] = "landscape"
    domain_id: str

    alignment: LandscapeExportAlignmentSpec
    fire_behavior_fuel_model: Literal["fbfm13", "fbfm40"]

    elevation: LandscapeFieldSource
    slope: LandscapeFieldSource
    aspect: LandscapeFieldSource
    fuel_model: LandscapeFieldSource
    canopy_cover: LandscapeFieldSource
    canopy_height: LandscapeFieldSource
    canopy_base_height: LandscapeFieldSource
    canopy_bulk_density: LandscapeFieldSource

    resolved: dict = Field(
        ...,
        description=(
            "Snapshot of the landscape lattice (CRS, transform, shape) at "
            "request time. Used by the exporter to consume pre-validated data."
        ),
    )
