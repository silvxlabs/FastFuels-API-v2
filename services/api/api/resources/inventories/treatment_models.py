"""
api/v2/resources/inventories/treatment_models.py

Schema models for silvicultural treatments applied to tree inventories.

A treatment thins the stand toward a target metric — a diameter-at-breast-height
limit, a residual basal area, or a minimum crown spacing. Treatments are a
create-time field on inventory create requests (parallel to modifications),
stored on the inventory for reproducibility. Standgen applies them, mapping the
variants to fastfuels-core's DirectionalThinTo{DiameterLimit, StandBasalArea},
ProportionalThinToBasalArea, and ThinToCrownSpacing.

Modeled as a **discriminated union on ``metric``** so each variant carries
exactly the fields and methods it supports — e.g. ``proportional`` exists only
for a basal-area target, so it cannot be expressed against a diameter limit at
all (no runtime cross-field validation needed).
"""

from enum import StrEnum
from typing import Annotated, ClassVar, Literal

import pint
from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.crown_radius import (
    CanopyAllometryMaxCrownRadiusSource,
    CanopyMaxCrownRadiusSource,
)
from api.resources.inventories.modification_models import InventorySpatialCondition
from lib.inventory_io import treatment_required_columns
from lib.units import validate_unit

ureg = pint.UnitRegistry()


class InventoryTreatmentMethod(StrEnum):
    """Tree-selection strategy for a silvicultural treatment.

    - from_below: low thinning — remove smaller/suppressed trees first
    - from_above: crown thinning — remove larger/dominant trees first
    - proportional: remove across all diameter classes proportionally
    """

    from_below = "from_below"
    from_above = "from_above"
    proportional = "proportional"


class InventoryCrownSpacingRank(StrEnum):
    """Column that decides which tree survives a crown-spacing conflict.

    - dbh: the larger diameter is kept
    - height: the taller tree is kept
    """

    dbh = "dbh"
    height = "height"


class _InventoryTreatmentBase(BaseModel):
    """Fields shared by every treatment variant. Not used directly."""

    unit: str | None = Field(
        default=None,
        description=(
            "Optional unit for `value`. Must be canonical and dimensionally "
            "compatible with the metric's native unit; converted before the "
            "treatment is applied."
        ),
    )
    conditions: list[InventorySpatialCondition] = Field(
        default_factory=list,
        description=(
            "Spatial conditions restricting the treatment to a region "
            "(within/outside/intersects a geometry or Feature). An empty list "
            "applies the treatment to the entire inventory."
        ),
    )

    # Native unit for `value`; set by each variant.
    native_unit: ClassVar[str]

    @field_validator("conditions", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Accept a single condition object in place of a list, for convenience."""
        if value is not None and not isinstance(value, list):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_unit_compatibility(self):
        """If a unit is supplied, require canonical form and compatibility with
        the variant's native unit."""
        if self.unit is None:
            return self

        # Canonical UDUNITS form (raises ValueError if not canonical).
        validate_unit(self.unit)

        try:
            user_unit = ureg.parse_expression(self.unit)
        except pint.UndefinedUnitError:
            raise ValueError(f"Unknown unit: '{self.unit}'")

        if not user_unit.is_compatible_with(ureg.parse_expression(self.native_unit)):
            raise ValueError(
                f"Unit '{self.unit}' is not compatible with the native unit "
                f"'{self.native_unit}' for this treatment."
            )
        return self

    def required_columns(self) -> set[str]:
        """Inventory columns this treatment reads."""
        return treatment_required_columns(self.model_dump(mode="json"))


class InventoryDiameterTreatment(_InventoryTreatmentBase):
    """Thin to a diameter-at-breast-height limit.

    A hard cutoff: ``from_below`` removes trees smaller than ``value``,
    ``from_above`` removes trees larger than ``value``. ``proportional`` does not
    apply to a diameter limit and is not an option here.
    """

    metric: Literal["diameter"] = "diameter"
    method: Literal[
        InventoryTreatmentMethod.from_below, InventoryTreatmentMethod.from_above
    ] = Field(
        ...,
        description=(
            "`from_below` removes trees below the limit; `from_above` removes "
            "trees above the limit."
        ),
    )
    value: float = Field(
        ...,
        gt=0,
        description="Diameter-at-breast-height limit, in cm unless `unit` is set.",
    )

    native_unit: ClassVar[str] = "cm"


class InventoryBasalAreaTreatment(_InventoryTreatmentBase):
    """Thin to a residual basal area.

    ``from_below``/``from_above`` remove the smallest/largest trees first until
    the target is reached; ``proportional`` removes across all diameter classes.
    """

    metric: Literal["basal_area"] = "basal_area"
    method: InventoryTreatmentMethod = Field(
        ...,
        description=(
            "`from_below`: remove smaller/suppressed trees first. `from_above`: "
            "remove larger/dominant trees first. `proportional`: remove across "
            "all diameter classes."
        ),
    )
    value: float = Field(
        ...,
        gt=0,
        description="Target residual basal area, in m**2/ha unless `unit` is set.",
    )

    native_unit: ClassVar[str] = "m**2/ha"


class InventoryCrownSpacingTreatment(_InventoryTreatmentBase):
    """Thin to a minimum crown spacing.

    Leaves every retained crown at least ``value`` apart, edge to edge, in plan
    view. Trees are visited from the largest ``rank_by`` value down, and each is
    kept only if its crown clears every crown already kept.

    With ``conditions``, only trees whose stem is in the region can be removed.
    Trees outside the region are never removed, but trees inside it must still
    clear their crowns.
    """

    metric: Literal["crown_spacing"] = "crown_spacing"
    value: float = Field(
        ...,
        gt=0,
        description=(
            "Minimum horizontal gap between retained crowns, crown edge to crown "
            "edge, in m unless `unit` is set."
        ),
    )
    rank_by: InventoryCrownSpacingRank = Field(
        default=InventoryCrownSpacingRank.dbh,
        description=(
            "Which tree is kept when two crowns are too close: `dbh` keeps the "
            "larger diameter, `height` keeps the taller tree. Use `height` for "
            "inventories without a `dbh` column."
        ),
    )
    max_crown_radius_source: CanopyMaxCrownRadiusSource = Field(
        default_factory=CanopyAllometryMaxCrownRadiusSource,
        description=(
            "Source of each tree's crown radius. Defaults to the `purves` "
            "allometry, the same default as tree voxelization. Set "
            '`{"type": "allometry", "equations": "crookston_stage"}` for '
            'FuelCalc crown widths, or `{"type": "inventory_column", "column": '
            '"crown_radius"}` to read a measured radius in meters. Trees with no '
            "value in the column use their `purves` radius. To keep thinned "
            "crowns from overlapping in a voxelized or canopy grid, build the "
            "grid with the same radius source."
        ),
    )

    native_unit: ClassVar[str] = "m"


InventoryTreatment = Annotated[
    InventoryDiameterTreatment
    | InventoryBasalAreaTreatment
    | InventoryCrownSpacingTreatment,
    Field(discriminator="metric"),
]
