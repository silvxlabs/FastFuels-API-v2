"""
api/v2/resources/inventories/treatment_models.py

Schema models for silvicultural treatments applied to tree inventories.

A treatment thins the stand toward a target residual state — a diameter limit
or a basal area — using a tree-selection method. Treatments are applied as a
create-time field on inventory create requests (parallel to modifications) and
are stored on the inventory for reproducibility. Thinning execution lives in
standgen (#103); it maps from_below/from_above to fastfuels-core's
DirectionalThinTo{DiameterLimit,StandBasalArea} and proportional to
ProportionalThinToBasalArea.
"""

from enum import StrEnum

import pint
from pydantic import BaseModel, Field, field_validator, model_validator

from api.resources.inventories.modification_models import InventorySpatialCondition
from lib.units import validate_unit

ureg = pint.UnitRegistry()

# Native unit for each target metric; user-supplied units convert to these.
TARGET_NATIVE_UNITS = {
    "diameter": "cm",
    "basal_area": "m**2/ha",
}


class InventoryTreatmentMethod(StrEnum):
    """Tree-selection strategy for a silvicultural treatment.

    - from_below: low thinning — remove smaller/suppressed trees first
    - from_above: crown thinning — remove larger/dominant trees first
    - proportional: remove across all diameter classes proportionally
    """

    from_below = "from_below"
    from_above = "from_above"
    proportional = "proportional"


class InventoryTreatmentTarget(BaseModel):
    """The residual stand state a treatment thins toward.

    Specify **exactly one** metric:

    - ``diameter`` — a diameter-at-breast-height limit (default cm). Paired with
      ``from_below``/``from_above`` it keeps trees above/below the limit.
    - ``basal_area`` — a residual basal area (default m**2/ha).

    An optional ``unit`` overrides the metric's default unit (e.g. ``in`` for a
    diameter limit, ``ft**2/acre`` for basal area); it is converted to the
    metric's native unit before the treatment is applied.
    """

    diameter: float | None = Field(
        default=None,
        gt=0,
        description="Diameter-at-breast-height limit, in cm unless `unit` is set.",
    )
    basal_area: float | None = Field(
        default=None,
        gt=0,
        description="Target residual basal area, in m**2/ha unless `unit` is set.",
    )
    unit: str | None = Field(
        default=None,
        description=(
            "Optional unit for the chosen metric, e.g. `in` for a diameter "
            "limit or `ft**2/acre` for basal area. Must be canonical and "
            "dimensionally compatible with the metric's native unit."
        ),
    )

    @model_validator(mode="after")
    def validate_exactly_one_metric(self):
        """Exactly one of diameter / basal_area must be set."""
        set_metrics = [
            m for m in ("diameter", "basal_area") if getattr(self, m) is not None
        ]
        if len(set_metrics) != 1:
            raise ValueError(
                "Specify exactly one treatment target metric: "
                "'diameter' or 'basal_area'."
            )
        return self

    @model_validator(mode="after")
    def validate_unit_compatibility(self):
        """If a unit is supplied, require canonical form and compatibility with
        the chosen metric's native unit."""
        if self.unit is None:
            return self

        metric = "diameter" if self.diameter is not None else "basal_area"
        native = TARGET_NATIVE_UNITS[metric]

        # Canonical UDUNITS form (raises ValueError if not canonical).
        validate_unit(self.unit)

        try:
            user_unit = ureg.parse_expression(self.unit)
        except pint.UndefinedUnitError:
            raise ValueError(f"Unknown unit: '{self.unit}'")

        native_unit = ureg.parse_expression(native)
        if not user_unit.is_compatible_with(native_unit):
            raise ValueError(
                f"Unit '{self.unit}' is not compatible with the native unit "
                f"'{native}' for target metric '{metric}'."
            )
        return self


class InventoryTreatment(BaseModel):
    """A silvicultural treatment applied to a tree inventory.

    Thins the stand toward ``target`` using the chosen ``method``. ``conditions``
    optionally restrict the treatment to a spatial region (a treatment-unit
    boundary); an empty list treats the whole inventory.
    """

    method: InventoryTreatmentMethod = Field(
        ...,
        description=(
            "Tree-selection strategy. `from_below`: low thinning — remove "
            "smaller/suppressed trees first. `from_above`: crown thinning — "
            "remove larger/dominant trees first. `proportional`: remove across "
            "all diameter classes proportionally (basal-area target only)."
        ),
    )
    target: InventoryTreatmentTarget
    conditions: list[InventorySpatialCondition] = Field(
        default_factory=list,
        description=(
            "Spatial conditions restricting the treatment to a region "
            "(within/outside/intersects a geometry or Feature). An empty list "
            "applies the treatment to the entire inventory."
        ),
    )

    @field_validator("conditions", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Accept a single condition object in place of a list, for convenience."""
        if value is not None and not isinstance(value, list):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_method_target_compatibility(self):
        """Proportional thinning only reaches a basal-area target.

        There is no proportional-to-diameter operation — a diameter limit is a
        hard threshold, applied directionally (from_below/from_above).
        """
        if (
            self.method == InventoryTreatmentMethod.proportional
            and self.target.diameter is not None
        ):
            raise ValueError(
                "The 'proportional' method requires a 'basal_area' target, "
                "not 'diameter'."
            )
        return self
