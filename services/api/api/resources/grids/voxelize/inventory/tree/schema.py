"""
api/v2/resources/grids/voxelize/inventory/tree/schema.py

Schema models for voxelizing a tree inventory into a 3D tree fuel grid.

Includes the band vocabulary, crown profile / biomass / moisture model
configuration types, and the request and persisted-source schemas.
"""

from enum import StrEnum
from random import randint
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from api.resources.grids.schema import Band, BandType, validate_no_duplicates


class TreeBand(StrEnum):
    """Available output bands for tree fuel grids.

    Each value is the band's dot-notation key as it appears in the stored
    Grid document. Band definitions (type, unit) live in TREE_BAND_DEFS.
    """

    bulk_density_foliage_live = "bulk_density.foliage.live"
    bulk_density_foliage_dead = "bulk_density.foliage.dead"
    bulk_density_branchwood_live = "bulk_density.branchwood.live"
    bulk_density_branchwood_dead = "bulk_density.branchwood.dead"
    bulk_density_fine_live = "bulk_density.fine.live"
    bulk_density_fine_dead = "bulk_density.fine.dead"
    fuel_moisture_live = "fuel_moisture.live"
    fuel_moisture_dead = "fuel_moisture.dead"
    savr_foliage = "savr.foliage"
    spcd = "spcd"
    tree_id = "tree_id"
    volume_fraction = "volume_fraction"


TREE_BAND_DEFS: dict[TreeBand, dict] = {
    TreeBand.bulk_density_foliage_live: {
        "key": "bulk_density.foliage.live",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.bulk_density_foliage_dead: {
        "key": "bulk_density.foliage.dead",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.bulk_density_branchwood_live: {
        "key": "bulk_density.branchwood.live",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.bulk_density_branchwood_dead: {
        "key": "bulk_density.branchwood.dead",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.bulk_density_fine_live: {
        "key": "bulk_density.fine.live",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.bulk_density_fine_dead: {
        "key": "bulk_density.fine.dead",
        "type": BandType.continuous,
        "unit": "kg/m**3",
    },
    TreeBand.fuel_moisture_live: {
        "key": "fuel_moisture.live",
        "type": BandType.continuous,
        "unit": "%",
    },
    TreeBand.fuel_moisture_dead: {
        "key": "fuel_moisture.dead",
        "type": BandType.continuous,
        "unit": "%",
    },
    TreeBand.savr_foliage: {
        "key": "savr.foliage",
        "type": BandType.continuous,
        "unit": "1/m",
    },
    TreeBand.spcd: {
        "key": "spcd",
        "type": BandType.categorical,
        "unit": None,
    },
    TreeBand.tree_id: {
        "key": "tree_id",
        "type": BandType.categorical,
        "unit": None,
    },
    TreeBand.volume_fraction: {
        "key": "volume_fraction",
        "type": BandType.continuous,
        "unit": None,
    },
}


class CrownProfileModel(StrEnum):
    """Crown geometry models — which voxels a tree's crown occupies."""

    purves = "purves"
    beta = "beta"


class BiomassEquations(StrEnum):
    """Allometric equation families for estimating biomass components."""

    nsvb = "nsvb"
    jenkins = "jenkins"


class BiomassComponent(StrEnum):
    """Biomass components that can be requested or supplied."""

    foliage = "foliage"
    branchwood = "branchwood"
    fine = "fine"


class BiomassUnit(StrEnum):
    """Accepted inventory biomass units."""

    kg = "kg"


class InventoryBiomassColumn(BaseModel):
    """Inventory column containing per-tree biomass for one component."""

    model_config = ConfigDict(extra="forbid")

    column: str
    unit: BiomassUnit = BiomassUnit.kg


class FineBiomassConfig(BaseModel):
    """Configuration for derived fine biomass."""

    model_config = ConfigDict(extra="forbid")

    recipe: Literal["foliage_plus_branchwood_fraction"]
    branchwood_fraction: float = Field(gt=0, le=1)


class BiomassComponentState(BaseModel):
    """Live/dead partition for one biomass component."""

    model_config = ConfigDict(extra="forbid")

    live: float = Field(default=1.0, ge=0, le=1)
    dead: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_partition_sums_to_one(self):
        if abs((self.live + self.dead) - 1.0) > 1e-9:
            raise ValueError("Component live/dead fractions must sum to 1.0.")
        return self


class BiomassSourceBase(BaseModel):
    """Common biomass component request behavior."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_components(self):
        if not self.components:
            raise ValueError("At least one biomass component is required.")

        seen: set[BiomassComponent] = set()
        duplicates: list[str] = []
        for component in self.components:
            if component in seen:
                duplicates.append(component.value)
            seen.add(component)
        if duplicates:
            raise ValueError(
                "Duplicate biomass components are not allowed: "
                f"{', '.join(duplicates)}."
            )

        if self.fine is not None and BiomassComponent.fine not in self.components:
            raise ValueError("fine configuration requires 'fine' in components.")

        unknown_states = [
            component.value
            for component in self.component_states
            if component not in self.components
        ]
        if unknown_states:
            raise ValueError(
                "component_states keys must also be listed in components: "
                f"{', '.join(unknown_states)}."
            )

        for component in self.components:
            self.component_states.setdefault(component, BiomassComponentState())

        return self


class AllometryBiomassSource(BiomassSourceBase):
    """Estimate biomass from allometric equations."""

    type: Literal["allometry"] = "allometry"
    equations: BiomassEquations = BiomassEquations.nsvb
    components: list[BiomassComponent] = Field(
        default_factory=lambda: [BiomassComponent.foliage],
        min_length=1,
    )
    component_states: dict[BiomassComponent, BiomassComponentState] = Field(
        default_factory=dict,
        description="Per-component live/dead biomass partition fractions.",
    )
    fine: FineBiomassConfig | None = None

    @model_validator(mode="after")
    def validate_fine_definition(self):
        if BiomassComponent.fine in self.components and self.fine is None:
            raise ValueError("Allometry fine biomass requires a fine configuration.")
        return self


class InventoryColumnsBiomassSource(BiomassSourceBase):
    """Read per-tree component biomass from inventory columns."""

    type: Literal["inventory_columns"] = "inventory_columns"
    columns: dict[BiomassComponent, InventoryBiomassColumn] = Field(
        min_length=1,
        description="Per-component inventory columns. Values must be per-tree kg.",
    )
    components: list[BiomassComponent] = Field(
        default_factory=lambda: [BiomassComponent.foliage],
        min_length=1,
    )
    component_states: dict[BiomassComponent, BiomassComponentState] = Field(
        default_factory=dict,
        description="Per-component live/dead biomass partition fractions.",
    )
    fine: FineBiomassConfig | None = None

    @model_validator(mode="after")
    def validate_source_supports_components(self):
        for component in self.components:
            if component == BiomassComponent.fine and self.fine is not None:
                missing = [
                    required
                    for required in (
                        BiomassComponent.foliage,
                        BiomassComponent.branchwood,
                    )
                    if required not in self.columns
                ]
                if missing:
                    missing_names = ", ".join(m.value for m in missing)
                    raise ValueError(
                        "Fine biomass recipe requires inventory columns for: "
                        f"{missing_names}."
                    )
            elif component not in self.columns:
                raise ValueError(
                    f"Inventory biomass source is missing a {component.value!r} column."
                )

        return self


BiomassSource = Annotated[
    AllometryBiomassSource | InventoryColumnsBiomassSource,
    Field(discriminator="type"),
]


class MaxCrownRadiusUnit(StrEnum):
    """Accepted inventory max crown radius units."""

    m = "m"


class AllometryMaxCrownRadiusSource(BaseModel):
    """Use the crown profile model's allometric max crown radius (default)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["allometry"] = "allometry"


class InventoryColumnMaxCrownRadiusSource(BaseModel):
    """Read per-tree max crown radius from an inventory column.

    The crown profile model still drives the crown shape — the supplied
    radius rescales it so the maximum radius matches the per-tree value.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["inventory_column"] = "inventory_column"
    column: str
    unit: MaxCrownRadiusUnit = MaxCrownRadiusUnit.m


MaxCrownRadiusSource = Annotated[
    AllometryMaxCrownRadiusSource | InventoryColumnMaxCrownRadiusSource,
    Field(discriminator="type"),
]


class UniformMoistureValue(BaseModel):
    """Uniform fuel moisture for one fuel state."""

    method: Literal["uniform"] = "uniform"
    value: float = Field(
        default=100.0,
        description="Fuel moisture content (%), applied uniformly.",
    )


class MoistureModel(BaseModel):
    """Live/dead fuel moisture settings."""

    model_config = ConfigDict(extra="forbid")

    live: UniformMoistureValue | None = None
    dead: UniformMoistureValue | None = None


def build_tree_bands(requested: list[TreeBand]) -> list[Band]:
    """Build Band objects for requested tree bands with indices in request order."""
    return [Band(index=i, **TREE_BAND_DEFS[band]) for i, band in enumerate(requested)]


def _bulk_density_component(band: TreeBand) -> BiomassComponent | None:
    """Return the biomass component a `bulk_density.<component>.<state>` band
    references, or None for bands that aren't bulk-density bands.
    """
    parts = band.value.split(".")
    if len(parts) == 3 and parts[0] == "bulk_density":
        try:
            return BiomassComponent(parts[1])
        except ValueError:
            return None
    return None


def validate_bulk_density_bands_have_components(
    bands: list[TreeBand],
    biomass_source: "AllometryBiomassSource | InventoryColumnsBiomassSource",
) -> None:
    """Reject requests where a bulk_density band's component isn't configured.

    Without this gate the worker either fails opaquely (priority dispatch picks
    the unimplemented component → NotImplementedError) or — once branchwood/fine
    compute lands — silently emits a zero band for the unconfigured component.
    """
    configured = set(biomass_source.components)
    missing = [
        (band.value, component.value)
        for band in bands
        if (component := _bulk_density_component(band)) is not None
        and component not in configured
    ]
    if missing:
        details = ", ".join(f"{b!r} requires {c!r}" for b, c in missing)
        raise ValueError(
            "Requested bulk_density bands reference components not in "
            f"biomass_source.components: {details}"
        )


class Resolution3D(BaseModel):
    """Voxel resolution for a 3D grid.

    `horizontal` applies to both x and y (fastfuels-core requires isotropic
    horizontal resolution). `vertical` is independent.
    """

    model_config = ConfigDict(extra="forbid")

    horizontal: PositiveFloat = Field(description="Cell size in x and y, meters.")
    vertical: PositiveFloat = Field(description="Cell size in z, meters.")

    @model_validator(mode="before")
    @classmethod
    def _reject_list_shape(cls, data):
        if isinstance(data, (list, tuple)):
            raise ValueError(
                'resolution must be an object with "horizontal" and '
                '"vertical" fields (in meters).'
            )
        return data


def _generate_random_seed() -> int:
    return randint(1, 1_000_000_000)


def _default_resolution() -> Resolution3D:
    return Resolution3D(horizontal=2.0, vertical=1.0)


def _default_bands() -> list[TreeBand]:
    return [TreeBand.bulk_density_foliage_live]


class TreeInventoryVoxelizationSource(BaseModel):
    """Source metadata stored on the Grid document for reproducibility.

    Records the inventory that was voxelized and every resolved model choice
    so the grid can be exactly reproduced.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal["voxelize"] = "voxelize"
    input: Literal["inventory"] = "inventory"
    entity: Literal["tree"] = "tree"

    source_inventory_id: str
    resolution: Resolution3D = Field(
        description="Voxel resolution (horizontal x/y, vertical z) in meters.",
    )
    bands: list[TreeBand]
    crown_profile_model: CrownProfileModel
    biomass_source: BiomassSource
    max_crown_radius_source: MaxCrownRadiusSource = AllometryMaxCrownRadiusSource()
    moisture_model: MoistureModel | None = None
    seed: int = Field(
        description=(
            "Random seed that drove stochastic sampling during voxelization. "
            "Persisted so the grid can be exactly reproduced."
        ),
    )

    @model_validator(mode="after")
    def validate_band_components_configured(self) -> Self:
        validate_bulk_density_bands_have_components(self.bands, self.biomass_source)
        return self


class CreateTreeInventoryRequest(BaseModel):
    """Request body for creating a tree fuel grid from a tree inventory.

    Does not extend CreateGridRequestBase because 3D grids do not support
    modifications — modifications must be applied to the inventory before
    voxelization, not to the resulting voxel grid.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)

    source_inventory_id: str = Field(
        description="ID of a completed tree inventory to voxelize.",
    )
    resolution: Resolution3D = Field(
        default_factory=_default_resolution,
        description="Voxel resolution (horizontal x/y, vertical z) in meters.",
    )
    bands: list[TreeBand] = Field(
        default_factory=_default_bands,
        min_length=1,
        description=(
            "Which output bands to produce. Defaults to `bulk_density.foliage.live`."
        ),
    )
    crown_profile_model: CrownProfileModel = Field(
        default=CrownProfileModel.purves,
        description="Crown geometry model. Default: purves.",
    )
    biomass_source: BiomassSource = Field(
        default_factory=AllometryBiomassSource,
        description="Biomass source and requested biomass components.",
    )
    max_crown_radius_source: MaxCrownRadiusSource = Field(
        default_factory=AllometryMaxCrownRadiusSource,
        description=(
            "Source of each tree's maximum crown radius. Defaults to the "
            "crown profile model's allometric value. Use "
            '`{"type": "inventory_column", "column": ...}` to read a '
            "per-tree maximum crown radius (m) from an inventory column "
            "(e.g. derived from LiDAR); the crown profile model still "
            "controls the crown shape — only the peak radius is rescaled."
        ),
    )
    moisture_model: MoistureModel | None = Field(
        default=None,
        description=(
            "Live/dead fuel moisture model. Applied only when matching "
            "fuel_moisture bands are requested. Live defaults to uniform "
            "100.0; dead defaults to uniform 10.0."
        ),
    )
    seed: int = Field(
        default_factory=_generate_random_seed,
        description=(
            "Random seed for reproducibility. Controls stochastic tree "
            "voxel sampling and biomass distribution. Generated randomly "
            "if omitted; persisted on the grid document either way so "
            "re-running a grid always yields bit-identical output."
        ),
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TreeBand]) -> list[TreeBand]:
        return validate_no_duplicates(v)

    @model_validator(mode="after")
    def resolve_conditional_defaults(self) -> Self:
        # moisture_model is only meaningful for requested moisture bands.
        # Populate state-specific uniform defaults when requested without a
        # model, and drop unrequested states so stored source reflects what
        # was actually applied.
        live_requested = TreeBand.fuel_moisture_live in self.bands
        dead_requested = TreeBand.fuel_moisture_dead in self.bands
        if live_requested or dead_requested:
            model = self.moisture_model or MoistureModel()
            self.moisture_model = MoistureModel(
                live=(
                    model.live or UniformMoistureValue(value=100.0)
                    if live_requested
                    else None
                ),
                dead=(
                    model.dead or UniformMoistureValue(value=10.0)
                    if dead_requested
                    else None
                ),
            )
        else:
            self.moisture_model = None

        return self

    @model_validator(mode="after")
    def validate_band_components_configured(self) -> Self:
        validate_bulk_density_bands_have_components(self.bands, self.biomass_source)
        return self
