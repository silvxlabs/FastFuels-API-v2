"""
api/v2/resources/grids/canopy/inventory/schema.py

Schema models for deriving 2D canopy fuel grids directly from a tree
inventory: FuelCalc-style profile construction and reduction, computed
per output cell.

For each tree, available canopy fuel is estimated from biomass allometry,
distributed vertically over the crown, and attributed horizontally to
output cells. Per-cell vertical profiles are then reduced to the canonical
canopy bands (``cbd``, ``cbh``, ``chm``, ``cc``, and optionally ``cfl``),
matching the band keys and units of the LANDFIRE canopy source so the
result is a drop-in substitute anywhere a LANDFIRE canopy grid works —
including the landscape export.

The modeling choices are exposed as independent fields, not presets: the
biomass source, the available-fuel definition, species inclusion,
crown-class adjustment, the vertical and horizontal distributions, and a
per-band reduction method. Defaults follow the FuelCalc method with NSVB
allometry; FuelCalc/LANDFIRE-compatibility combinations are documented as
OpenAPI examples.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from api.resources.grids.alignment import (
    GridAlignmentDomainTarget,
    GridAlignmentSpecification,
)
from api.resources.grids.canopy.schema import (
    LANDFIRE_CANOPY_BAND_DEFS,
    LandfireCanopyFuelBand,
)
from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import Band, BandType, validate_no_duplicates
from api.resources.grids.voxelize.inventory.tree.schema import (
    BiomassUnit,
    InventoryColumnMaxCrownRadiusSource,
)

# Cell size applied when the request's domain-target alignment omits
# `resolution`. An inventory has no native raster cell size to inherit, so
# the router resolves the default here before persisting the source.
DEFAULT_INVENTORY_CANOPY_RESOLUTION_M = 30.0


class InventoryCanopyBand(StrEnum):
    """Selectable bands on the inventory canopy source.

    The first four share keys and units with the LANDFIRE canopy source, so
    downstream consumers (including the landscape export) accept either
    interchangeably. ``cfl`` (canopy fuel load) is additionally available
    here because the per-cell fuel profile is already assembled; it is not
    part of the LANDFIRE canopy band set and is not a landscape file role.
    """

    cbd = "cbd"
    cbh = "cbh"
    chm = "chm"
    cc = "cc"
    cfl = "cfl"


CFL_BAND_DEF = {
    "key": "cfl",
    "name": "Canopy Fuel Load",
    "description": "Total available canopy fuel per unit ground area.",
    "type": BandType.continuous,
    "unit": "kg/m**2",
}


def build_inventory_canopy_bands(
    requested: list[InventoryCanopyBand],
) -> list[Band]:
    """Build Band objects for requested bands with indices in request order.

    The four LANDFIRE-parity bands reuse LANDFIRE_CANOPY_BAND_DEFS; ``cfl``
    uses its own definition.
    """
    bands = []
    for i, band in enumerate(requested):
        if band is InventoryCanopyBand.cfl:
            bands.append(Band(index=i, **CFL_BAND_DEF))
        else:
            band_def = LANDFIRE_CANOPY_BAND_DEFS[LandfireCanopyFuelBand(band.value)]
            bands.append(Band(index=i, **band_def))
    return bands


class CanopyBiomassEquations(StrEnum):
    """Allometric equation families for estimating crown biomass.

    A superset of the voxelize biomass equations: ``brown_1978`` (Brown,
    "Weight and Density of Crowns of Rocky Mountain Conifers", INT-197) is
    offered here because it is the equation set behind FuelCalc and the
    LANDFIRE canopy layers. It is scoped to Interior West conifers and is
    intended for compatibility studies, not as a national default.
    """

    nsvb = "nsvb"
    jenkins = "jenkins"
    brown_1978 = "brown_1978"


class CanopyCrownWidthEquations(StrEnum):
    """Allometric equation families for a tree's maximum crown radius.

    ``purves`` (Purves et al. 2007) is the national default and the one
    the tree voxelization endpoint uses; it varies with height and
    crown ratio as well as diameter. ``crookston_stage`` (Crookston &
    Stage 1999, RMRS-GTR-24, reached through FVS) is the crown width
    behind FuelCalc's canopy cover — diameter alone above breast
    height, with regionally fitted coefficients — and is offered for
    compatibility studies.
    """

    purves = "purves"
    crookston_stage = "crookston_stage"


class CanopyAllometryMaxCrownRadiusSource(BaseModel):
    """Compute each tree's maximum crown radius from allometric equations.

    Supersets the voxelize allometry source with an `equations` choice,
    because canopy cover and crown biomass are separate axes here: which
    allometry supplies the radius is independent of how a cover method
    treats crowns that overlap, so a run can vary either without
    confounding the other.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["allometry"] = "allometry"
    equations: CanopyCrownWidthEquations = Field(
        default=CanopyCrownWidthEquations.purves,
        description=(
            "Allometric equation family for maximum crown radius. "
            "`purves` is the national default, consistent with the tree "
            "voxelization endpoint. `crookston_stage` reproduces the "
            "crown widths FuelCalc uses for canopy cover."
        ),
    )


CanopyMaxCrownRadiusSource = Annotated[
    CanopyAllometryMaxCrownRadiusSource | InventoryColumnMaxCrownRadiusSource,
    Field(discriminator="type"),
]


class AllometryCanopyBiomassSource(BaseModel):
    """Estimate each tree's crown biomass from allometric equations.

    The equations produce crown component biomass (foliage, branchwood);
    the `available_fuel` settings then reduce those components to the
    available canopy fuel used in the profile.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["allometry"] = "allometry"
    equations: CanopyBiomassEquations = Field(
        default=CanopyBiomassEquations.nsvb,
        description=(
            "Allometric equation family. `nsvb` is the national FIA-backed "
            "default, consistent with the tree voxelization endpoint. "
            "`brown_1978` reproduces the FuelCalc / LANDFIRE biomass basis "
            "(Interior West conifers)."
        ),
    )


class InventoryColumnCanopyBiomassSource(BaseModel):
    """Read each tree's available canopy fuel directly from an inventory column.

    The column value is used as-is: it must already be the per-tree mass of
    canopy fuel available to a crown fire (foliage plus the burnable fine
    branchwood). This bypasses allometry, `available_fuel`, species
    inclusion, and crown-class adjustment for fuel magnitude.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["inventory_column"] = "inventory_column"
    column: str = Field(
        description="Inventory column holding per-tree available canopy fuel."
    )
    unit: BiomassUnit = BiomassUnit.kg


CanopyBiomassSource = Annotated[
    AllometryCanopyBiomassSource | InventoryColumnCanopyBiomassSource,
    Field(discriminator="type"),
]


class CanopyBranchwoodSizePartition(StrEnum):
    """How branchwood is reduced to the size basis the fraction applies to.

    `equations` uses the equation family's own size classes: Brown (1978)
    reports the 0-1/4 inch branch class directly, so no external partition
    is applied. Only valid with `brown_1978`.

    `brown_proportions` partitions the family's total branchwood estimate
    using the Brown (1978) / Snell & Little (1983) crown-weight proportions
    (via the FuelCalc species crosswalk). This is how NSVB and Jenkins
    totals — which carry no size information — are reduced to the fine
    class. The tabulated proportions are accumulative: P1 is the foliage
    share of crown weight and P2 is foliage plus fine (0-1/4 inch)
    branchwood, so the fine share applied to a branchwood total is
    `(P2 - P1) / (1 - P1)`.

    `none` applies no size partition: the fraction applies to the family's
    total branchwood, so it expresses the fine-branch share and the consumed
    share as one combined number.
    """

    equations = "equations"
    brown_proportions = "brown_proportions"
    none = "none"


class CanopyBranchwood(BaseModel):
    """Branchwood availability: the size basis and how much of it counts.

    `fraction` multiplies the branchwood mass `size_partition` produces —
    the fine (0-1/4 inch) class under `equations` and `brown_proportions`,
    or total branchwood under `none` — so the fraction's referent is always
    an explicit choice, never an artifact of the biomass source.
    """

    model_config = ConfigDict(extra="forbid")

    size_partition: CanopyBranchwoodSizePartition | None = Field(
        default=None,
        description=(
            "Size basis for the branchwood fraction. Omitted (`null`), it "
            "resolves by equation family: `equations` for `brown_1978` "
            "(which reports the fine class directly), `brown_proportions` "
            "for `nsvb` and `jenkins` (which report only total branchwood). "
            "`none` skips the partition so `fraction` applies to total "
            "branchwood."
        ),
    )
    fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the size basis counted as available fuel. Against "
            "the fine class, 0.5 is the Brown & Reinhardt (1991) / FuelCalc "
            "assumption; Conrad et al. (2024) measured species-specific "
            "consumable fractions ranging from 0.0 to 0.99. Against total "
            'branchwood (`size_partition: "none"`), it combines the '
            "fine-branch share and the consumed share in one number."
        ),
    )


class CanopyAvailableFuel(BaseModel):
    """Which crown biomass counts as available canopy fuel.

    Available fuel is the mass consumed in the flaming front of a crown
    fire: foliage plus a fraction of the fine (0-1/4 inch) branchwood.
    Fine branchwood is the finest class any published crown allometry
    resolves, so the size line is fixed; the caller-adjustable choices are
    the fractions and how the fine class is obtained from the biomass
    equations.
    """

    model_config = ConfigDict(extra="forbid")

    foliage_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of foliage biomass counted as available fuel.",
    )
    branchwood: CanopyBranchwood = Field(
        default_factory=CanopyBranchwood,
        description=(
            "Fine-branchwood availability: the partition producing the fine "
            "class and the fraction of it counted as available fuel."
        ),
    )


class CanopySpeciesInclusion(StrEnum):
    """Which species contribute canopy fuel to the profile.

    FuelCalc excludes most hardwoods from canopy fuel calculations by
    default; FastFuels includes every species unless told otherwise. With
    `fuelcalc_default`, the persisted source records the exclusion table
    the worker applied.
    """

    all_species = "all_species"
    fuelcalc_default = "fuelcalc_default"


class CanopyNoCrownClassAdjustment(BaseModel):
    """Apply no crown-class adjustment: every tree keeps its full crown weight."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["none"] = "none"


class CanopyFuelcalcCrownClassAdjustment(BaseModel):
    """Multiply crown weight by the FuelCalc species x crown-class factors.

    Tree inventories do not carry a crown-class column, so every tree
    receives the factor selected by `missing_crown_class`. With the
    `other_none` fallback that is a global 0.5 multiplier for most species —
    a large, deliberate reduction that reproduces how FuelCalc treats trees
    of unknown crown class.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["fuelcalc_table"] = "fuelcalc_table"
    missing_crown_class: Literal["other_none"] = Field(
        default="other_none",
        description=(
            "Factor column applied to trees without a crown class — "
            "currently every tree. `other_none` is FuelCalc's Other/none "
            "column (0.5 for most species)."
        ),
    )


CanopyCrownClassAdjustment = Annotated[
    CanopyNoCrownClassAdjustment | CanopyFuelcalcCrownClassAdjustment,
    Field(discriminator="method"),
]


class CanopyVerticalDistribution(StrEnum):
    """How each tree's available fuel is distributed from crown base to top.

    `reinhardt_2006` uses the species cumulative-fraction cubics fit from
    destructive sampling (Reinhardt et al. 2006, CJFR 36), applied through
    the published FuelCalc species crosswalk; species the crosswalk maps to
    no cubic (including all hardwoods) distribute uniformly. `uniform`
    spreads mass evenly over the crown for every tree — the FFE-FVS
    assumption.
    """

    reinhardt_2006 = "reinhardt_2006"
    uniform = "uniform"


class CanopyHorizontalDistribution(StrEnum):
    """How a tree's available fuel is attributed to output cells.

    `crown_projected` splits each tree's mass across the cells its
    projected crown overlaps, in proportion to the overlap area.
    `stem` assigns the whole mass to the cell containing the stem — the
    closest analogue to FuelCalc's plot computation.
    """

    crown_projected = "crown_projected"
    stem = "stem"


class CanopyRunningMeanEdge(StrEnum):
    """What a running mean takes to lie past the ends of the profile.

    The three answers in use agree everywhere except the lowest and
    highest few layers, which is exactly where the threshold heights are
    read, so the choice can move a reported `cbh` or `chm` by a layer
    and shift `cbd` in a canopy that reaches the ground.

    `fixed_depth` divides by the window depth at every height, padding
    with zeros at both ends, so a slab of fuel reports the same bulk
    density wherever it sits — the reading Reinhardt et al. (2006)
    define. `ground_clamped` shortens the window where it would run
    below the ground and divides by what it actually covered, while
    still dividing by the full depth above the canopy; this is
    FuelCalc's, and it concentrates density against the ground while
    letting it fall away above the crowns. `truncated` divides by
    whatever the window covered at either end, which is FFE-FVS's and
    inflates the topmost layers.
    """

    fixed_depth = "fixed_depth"
    ground_clamped = "ground_clamped"
    truncated = "truncated"


class CanopyCbdDepth(StrEnum):
    """Depth definitions for the load-over-depth CBD method.

    `canopy_depth` is chm - cbh (the van Wagner convention), computed from
    the requested `cbh`/`chm` threshold settings — those bands must be
    requested with the `bulk_density_threshold` method so the depth's
    definition is recorded on the grid. `mean_crown_length` is the mean of
    per-tree crown lengths (the depth Cruz et al. 2003 used).
    `biomass_percentile` is the height span holding the central 80% of
    canopy biomass (10th to 90th percentile; Albini 1996).
    `height_percentile` is the 90th-percentile tree height minus the
    median crown base height.
    """

    canopy_depth = "canopy_depth"
    mean_crown_length = "mean_crown_length"
    biomass_percentile = "biomass_percentile"
    height_percentile = "height_percentile"


class CanopyCbdRunningMean(BaseModel):
    """CBD as the maximum of a running mean of the vertical fuel profile.

    The effective-CBD convention used by FuelCalc, FFE-FVS, and NEXUS.
    Published window depths vary by implementation: 1.524 m (5 ft, the
    FuelCalc 1.7 User Guide; its Appendix D reads unsmoothed layers, i.e.
    `null`), 3.9624 m (13 ft, FFE-FVS), 4.572 m (15 ft, the original
    FuelCalc method in RMRS-P-41; early NEXUS used 4.5 m), and 3.0 m
    (Reinhardt et al. 2006; current NEXUS). The default 3.0 m follows
    Reinhardt et al. (2006).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["maximum_running_mean"] = "maximum_running_mean"
    window: Annotated[float, Field(gt=0.0, le=15.0)] | None = Field(
        default=3.0,
        description=(
            "Running-mean window depth in meters, rounded internally to a "
            "whole number of profile layers. `null` disables smoothing, "
            "making CBD the maximum single-layer value."
        ),
    )
    edge: CanopyRunningMeanEdge = Field(
        default=CanopyRunningMeanEdge.ground_clamped,
        description=(
            "What the mean takes to lie past the ends of the profile. "
            "Ignored when `window` is `null`. `fixed_depth` is the reading "
            "that pairs with the 3.0 m Reinhardt et al. (2006) window; "
            "`ground_clamped` is FuelCalc's and is what its 1.524 m window "
            "reproduces."
        ),
    )


class CanopyCbdLoadOverDepth(BaseModel):
    """CBD as canopy fuel load divided by a canopy depth.

    The van Wagner-consistent average-density convention (load over
    depth). Produces systematically lower values than the running-mean
    maximum. Cruz et al. (2003) used this convention with
    `mean_crown_length` as the depth and foliage-only available fuel.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["load_over_depth"] = "load_over_depth"
    depth: CanopyCbdDepth = Field(
        default=CanopyCbdDepth.canopy_depth,
        description="Which canopy depth divides the canopy fuel load.",
    )


CanopyCbdMethod = Annotated[
    CanopyCbdRunningMean | CanopyCbdLoadOverDepth,
    Field(discriminator="method"),
]


class CanopyProfileThreshold(BaseModel):
    """A height where the vertical fuel profile crosses a bulk-density threshold.

    Shared by `cbh` (lowest crossing) and `chm` (highest crossing), so the
    two bands are consistent by construction (cbh <= chm). The effective
    threshold is ``min(relative_threshold_fraction * CBD_max, threshold)``
    — FuelCalc's rule, in which the relative branch engages whenever the
    cell's maximum profile density is below threshold / fraction (0.12
    kg/m**3 at the defaults). `CBD_max` is the maximum of the profile
    this method scans, so `smoothing_window` changes it; it is
    independent of the `cbd` band's own window, which is a separate
    setting.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["bulk_density_threshold"] = "bulk_density_threshold"
    threshold: float = Field(
        default=0.012,
        gt=0.0,
        le=1.0,
        description=(
            "Bulk-density threshold ceiling in kg/m**3. Default 0.012, the "
            "FuelCalc value, also used by LANDFIRE to define canopy base "
            "height. Published alternatives: 0.011 (Scott & Reinhardt 2001 "
            "/ FFE-FVS, 30 lb/acre/ft), 0.037 (Sando & Wick 1972, 100 "
            "lb/acre/ft), 0.074 (Williams 1977)."
        ),
    )
    relative_threshold_fraction: Annotated[float, Field(gt=0.0, le=1.0)] | None = Field(
        default=0.1,
        description=(
            "Fraction of the cell's maximum profile density used as the "
            "threshold when that is lower than `threshold`. `null` applies "
            "`threshold` flat, with no relative branch."
        ),
    )
    smoothing_window: Annotated[float, Field(gt=0.0, le=15.0)] | None = Field(
        default=None,
        description=(
            "Running-mean window in meters applied to the profile before "
            "locating threshold crossings. `null` reads raw layers. "
            "FuelCalc 1.7 crosses a 1.524 m (5 ft) running mean, the same "
            "window it reduces CBD over; FFE-FVS crosses 0.9144 m (3 ft); "
            "the original FuelCalc method (RMRS-P-41) crosses its full "
            "4.572 m (15 ft) window."
        ),
    )
    smoothing_edge: CanopyRunningMeanEdge = Field(
        default=CanopyRunningMeanEdge.ground_clamped,
        description=(
            "What the smoothing mean takes to lie past the ends of the "
            "profile. Ignored when `smoothing_window` is `null`. "
            "`truncated` is FFE-FVS's and reports `chm` a layer higher on "
            "the same profile, because dividing the topmost window by a "
            "short denominator carries it over the threshold."
        ),
    )


class CanopyCbhMeanCrownBase(BaseModel):
    """CBH as the available-fuel-weighted mean of per-tree crown base heights.

    Kept for comparison work; per-tree averaging is known to misrepresent
    crown-fire initiation in multi-storied stands.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["mean_crown_base"] = "mean_crown_base"


CanopyCbhMethod = Annotated[
    CanopyProfileThreshold | CanopyCbhMeanCrownBase,
    Field(discriminator="method"),
]


class CanopyChmHeightPercentile(BaseModel):
    """Canopy height as a percentile of tree heights in the cell."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["height_percentile"] = "height_percentile"
    percentile: float = Field(
        default=99.0,
        ge=1.0,
        le=100.0,
        description="Tree-height percentile reported as canopy height.",
    )


CanopyChmMethod = Annotated[
    CanopyProfileThreshold | CanopyChmHeightPercentile,
    Field(discriminator="method"),
]


class CanopyCcCrownUnion(BaseModel):
    """Canopy cover as the geometric union of projected crown areas.

    Measures the vertically projected crown cover directly from stem
    positions and crown radii, respecting the inventory's actual spatial
    pattern (clumping, gaps).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["crown_union"] = "crown_union"


class CanopyCcCrownOverlap(BaseModel):
    """Canopy cover from the Crookston-Stage random-overlap correction.

    ``100 * (1 - exp(-total_crown_area / cell_area))`` — the expected cover
    if stems were randomly placed. This is the FuelCalc estimator; it
    ignores actual stem positions.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["crown_overlap"] = "crown_overlap"


class CanopyCcCoverFraction(BaseModel):
    """Canopy cover as the fraction of the cell with canopy above a height.

    A CHM-style cover measure, distinct from projected crown cover: it
    reports where canopy *surface* exceeds `height_threshold` rather than
    the projection of all suspended canopy.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["cover_fraction"] = "cover_fraction"
    height_threshold: float = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description="Height in meters above which canopy counts as cover.",
    )


CanopyCcMethod = Annotated[
    CanopyCcCrownUnion | CanopyCcCrownOverlap | CanopyCcCoverFraction,
    Field(discriminator="method"),
]


class InventoryCanopySourceBase(BaseModel):
    """Modeling choices shared by the request and the persisted source.

    Per-band method fields are `None` when the band is not requested; the
    request validator resolves defaults for requested bands so the stored
    source always records the concrete choices that produced the grid.
    """

    biomass_source: CanopyBiomassSource = Field(
        default_factory=AllometryCanopyBiomassSource,
        description=(
            "Where each tree's crown fuel mass comes from: allometric "
            "equations (default: NSVB) or an inventory column carrying "
            "precomputed available canopy fuel."
        ),
    )
    available_fuel: CanopyAvailableFuel | None = Field(
        default_factory=CanopyAvailableFuel,
        description=(
            "How crown biomass reduces to available canopy fuel. Applies "
            "only with an allometry biomass source; resolved to `null` when "
            "the biomass source is an inventory column."
        ),
    )
    species_inclusion: CanopySpeciesInclusion = Field(
        default=CanopySpeciesInclusion.all_species,
        description=(
            "Which species contribute canopy fuel. `all_species` (default) "
            "includes everything; `fuelcalc_default` applies FuelCalc's "
            "hardwood exclusion table."
        ),
    )
    crown_class_adjustment: CanopyCrownClassAdjustment = Field(
        default_factory=CanopyNoCrownClassAdjustment,
        description=(
            "Crown-weight adjustment for canopy position. `none` (default) "
            "applies no adjustment; `fuelcalc_table` applies the FuelCalc "
            "species x crown-class factors."
        ),
    )
    min_tree_height: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description=(
            "Trees shorter than this height in meters contribute no canopy "
            "fuel. Default 0.0 includes every tree. FFE-FVS and the "
            "original FuelCalc method (RMRS-P-41) exclude trees under "
            "1.83 m (6 ft) as surface fuel; FuelCalc 1.7 includes small "
            "trees down to 0.3 m via Brown's small-tree tables."
        ),
    )
    vertical_distribution: CanopyVerticalDistribution = Field(
        default=CanopyVerticalDistribution.reinhardt_2006,
        description=(
            "How each tree's available fuel is distributed over its crown length."
        ),
    )
    layer_depth: float = Field(
        default=0.3048,
        ge=0.01,
        le=5.0,
        description=(
            "Vertical profile layer depth in meters. Default 0.3048 m "
            "(1 ft, the FuelCalc layer). Affects CBD smoothing and where "
            "threshold crossings land."
        ),
    )
    horizontal_distribution: CanopyHorizontalDistribution = Field(
        default=CanopyHorizontalDistribution.crown_projected,
        description=(
            "How a tree's fuel reaches output cells: split over the cells "
            "its crown projects onto (default), or wholly to the stem cell."
        ),
    )
    max_crown_radius_source: CanopyMaxCrownRadiusSource = Field(
        default_factory=CanopyAllometryMaxCrownRadiusSource,
        description=(
            "Source of each tree's maximum crown radius, used by "
            "`crown_projected` attribution and the geometric canopy cover "
            "methods. Defaults to the `purves` allometry; set "
            '`{"type": "allometry", "equations": "crookston_stage"}` for '
            "the crown widths FuelCalc uses, or "
            '`{"type": "inventory_column", "column": ...}` to read a '
            "per-tree radius in meters (e.g. derived from LiDAR)."
        ),
    )

    cbd: CanopyCbdMethod | None = Field(
        default=None,
        description=(
            "Canopy bulk density method. Defaults to the maximum 3.0 m "
            "running mean of the profile when the `cbd` band is requested."
        ),
    )
    cbh: CanopyCbhMethod | None = Field(
        default=None,
        description=(
            "Canopy base height method. Defaults to the lowest "
            "profile-threshold crossing when the `cbh` band is requested."
        ),
    )
    chm: CanopyChmMethod | None = Field(
        default=None,
        description=(
            "Canopy height method. Defaults to the highest "
            "profile-threshold crossing when the `chm` band is requested."
        ),
    )
    cc: CanopyCcMethod | None = Field(
        default=None,
        description=(
            "Canopy cover method. Defaults to the geometric crown union "
            "when the `cc` band is requested."
        ),
    )


def _default_bands() -> list[InventoryCanopyBand]:
    return [
        InventoryCanopyBand.cbd,
        InventoryCanopyBand.cbh,
        InventoryCanopyBand.chm,
        InventoryCanopyBand.cc,
    ]


# Maps each band with a configurable reduction to its method field and the
# default method applied when the band is requested without one. `cfl` is
# absent: it is always the summed available fuel over cell area.
_BAND_METHOD_DEFAULTS: dict[InventoryCanopyBand, type[BaseModel]] = {
    InventoryCanopyBand.cbd: CanopyCbdRunningMean,
    InventoryCanopyBand.cbh: CanopyProfileThreshold,
    InventoryCanopyBand.chm: CanopyProfileThreshold,
    InventoryCanopyBand.cc: CanopyCcCrownUnion,
}


class CreateInventoryCanopyRequest(InventoryCanopySourceBase):
    """Request body for creating a canopy fuel grid from a tree inventory.

    Only live trees contribute canopy fuel: the worker reads live
    inventory records only, matching FuelCalc, which excludes dead trees
    from all calculations.

    Does not extend CreateGridRequestBase: like DUET and the 3D voxel
    grids, inventory-derived grids do not support modifications — apply
    treatments and modifications to the inventory before deriving canopy
    metrics.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_inventory_id: str = Field(
        description=(
            "ID of a completed tree inventory in this domain to derive "
            "canopy metrics from."
        ),
    )
    bands: list[InventoryCanopyBand] = Field(
        default_factory=_default_bands,
        min_length=1,
        description=(
            "Which output bands to produce. Defaults to the four "
            "LANDFIRE-parity canopy bands; add `cfl` for canopy fuel load."
        ),
    )
    alignment: GridAlignmentSpecification = Field(
        default_factory=GridAlignmentDomainTarget,
        description=(
            "Output lattice. Against the domain (`target: 'domain'`, the "
            "default) `resolution` defaults to 30 m — an inventory has no "
            "native cell size to inherit. Against another grid "
            "(`target: 'grid'`) omitting `resolution` matches that grid's "
            "lattice exactly. `target: 'native'` is not supported."
        ),
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(
        cls, v: list[InventoryCanopyBand]
    ) -> list[InventoryCanopyBand]:
        return validate_no_duplicates(v)

    @model_validator(mode="after")
    def resolve_band_methods(self) -> Self:
        """Fill method defaults for requested bands; reject config for others.

        A method supplied for an unrequested band would be silently ignored
        by the worker, so it is rejected instead. Requested bands without an
        explicit method receive the documented default, making the persisted
        source a complete record of the choices applied.
        """
        for band, default_method in _BAND_METHOD_DEFAULTS.items():
            field = band.value
            requested = band in self.bands
            # An explicit null reads as absent, so `{"bands": ["cc"],
            # "cbd": null}` is a valid way to say "no cbd".
            provided = (
                field in self.model_fields_set and getattr(self, field) is not None
            )
            if provided and not requested:
                raise ValueError(
                    f"A {field!r} method was provided, but the {field!r} band "
                    "was not requested. Add it to `bands` or remove the method."
                )
            if requested and getattr(self, field) is None:
                setattr(self, field, default_method())
            elif not requested:
                setattr(self, field, None)
        return self

    @model_validator(mode="after")
    def resolve_available_fuel(self) -> Self:
        """Tie `available_fuel` to the allometry biomass source.

        An inventory-column biomass source already carries available canopy
        fuel, so availability settings would be double-counted; they are
        rejected if supplied and nulled otherwise.

        With an allometry source, the branchwood size partition resolves by
        equation family: `brown_1978` reports the fine class directly
        (`equations`); `nsvb` and `jenkins` report only total branchwood,
        which Brown's species proportions reduce to the fine class
        (`brown_proportions`). Requesting `equations` with a family that
        carries no size classes is rejected rather than silently
        approximated. `none` — fraction of total branchwood — is valid with
        every family.
        """
        if isinstance(self.biomass_source, InventoryColumnCanopyBiomassSource):
            if (
                "available_fuel" in self.model_fields_set
                and self.available_fuel is not None
            ):
                raise ValueError(
                    "available_fuel cannot be combined with an "
                    "inventory_column biomass source: the column already "
                    "contains available canopy fuel."
                )
            self.available_fuel = None
            return self

        if self.available_fuel is None:
            self.available_fuel = CanopyAvailableFuel()

        native = self.biomass_source.equations is CanopyBiomassEquations.brown_1978
        partition = self.available_fuel.branchwood.size_partition
        if partition is None:
            self.available_fuel.branchwood.size_partition = (
                CanopyBranchwoodSizePartition.equations
                if native
                else CanopyBranchwoodSizePartition.brown_proportions
            )
        elif partition is CanopyBranchwoodSizePartition.equations and not native:
            raise ValueError(
                "branchwood size_partition 'equations' requires biomass "
                "equations that report fine branchwood by size class "
                f"(brown_1978). '{self.biomass_source.equations.value}' "
                "reports only total branchwood; use 'brown_proportions' or "
                "'none'."
            )
        return self

    @model_validator(mode="after")
    def validate_canopy_depth_dependencies(self) -> Self:
        """Require recorded thresholds for `load_over_depth` + `canopy_depth`.

        That depth is chm - cbh, so the thresholds defining it must be part
        of the request — otherwise the stored source could not reproduce
        the grid's CBD.
        """
        if (
            isinstance(self.cbd, CanopyCbdLoadOverDepth)
            and self.cbd.depth is CanopyCbdDepth.canopy_depth
        ):
            missing = [
                field
                for field in ("cbh", "chm")
                if not isinstance(getattr(self, field), CanopyProfileThreshold)
            ]
            if missing:
                raise ValueError(
                    "cbd method 'load_over_depth' with depth 'canopy_depth' "
                    "divides canopy fuel load by chm - cbh, so 'cbh' and "
                    "'chm' must be requested bands using the "
                    "'bulk_density_threshold' method — otherwise the "
                    "thresholds defining the depth would not be recorded on "
                    "the grid. Missing or using another method: "
                    f"{', '.join(missing)}."
                )
        return self


class InventoryCanopySource(CanopySource, InventoryCanopySourceBase):
    """Source metadata stored on the Grid document for reproducibility.

    Records the inventory the grid was derived from and every resolved
    modeling choice, including defaults the caller omitted, so the grid can
    be exactly reproduced from the resource alone. Method fields are `null`
    for bands that were not requested.
    """

    model_config = ConfigDict(extra="forbid")

    product: Literal["inventory"] = "inventory"
    description: Literal["Canopy fuel metrics computed from a tree inventory"] = (
        "Canopy fuel metrics computed from a tree inventory"
    )

    source_inventory_id: str
    source_inventory_checksum: str | None = Field(
        default=None,
        description=(
            "The source inventory's `checksum` at the time this grid was "
            "created from it. Compare it against the inventory's current "
            "`checksum` to tell whether the source has changed since."
        ),
    )
    bands: list[InventoryCanopyBand]
