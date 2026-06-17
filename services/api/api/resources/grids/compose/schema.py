"""Schema models for grid compose operations."""

from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from api.resources.grids.modification_models import (
    GridFeatureSpatialCondition,
    GridGeometrySpatialCondition,
    GridModification,
)
from api.resources.grids.schema import Band, BandType, validate_no_duplicates
from lib.units import validate_unit


class ComposeOperator(StrEnum):
    """Operators available for compose computations."""

    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"
    max = "max"
    min = "min"
    average = "average"


class ComposeComparisonOperator(StrEnum):
    """Comparison operators for compose attribute conditions."""

    eq = "eq"
    ne = "ne"
    gt = "gt"
    lt = "lt"
    ge = "ge"
    le = "le"
    in_ = "in"


# Operators that accept two or more operands; the rest are strictly binary.
VARIADIC_OPERATORS = frozenset(
    {
        ComposeOperator.add,
        ComposeOperator.multiply,
        ComposeOperator.average,
        ComposeOperator.min,
        ComposeOperator.max,
    }
)
# Comparison operators valid on categorical bands (ordering is meaningless).
CATEGORICAL_CONDITION_OPERATORS = frozenset(
    {
        ComposeComparisonOperator.eq,
        ComposeComparisonOperator.ne,
        ComposeComparisonOperator.in_,
    }
)
# Ordering comparisons require a single scalar operand, never a list.
_ORDERING_OPERATORS = frozenset(
    {
        ComposeComparisonOperator.gt,
        ComposeComparisonOperator.lt,
        ComposeComparisonOperator.ge,
        ComposeComparisonOperator.le,
    }
)


class ComposeInput(BaseModel):
    """A source grid participating in a compose request."""

    grid_id: str
    alias: str = Field(
        ...,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="Short alias used to reference bands, e.g. `a.fuel_load.1hr`.",
    )


class ComposeSourceInput(ComposeInput):
    """Persisted source-grid input with provenance checksum."""

    source_grid_checksum: str | None = Field(
        default=None,
        description=(
            "The input grid's checksum at compose request time. Compare against "
            "the source grid's current checksum to detect staleness."
        ),
    )


class ComposeOutputBand(BaseModel):
    """Explicit output band metadata for compose-created grids."""

    key: str = Field(..., description="Output dot-notation band key.")
    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=2000)
    type: BandType
    unit: str | None = Field(
        None,
        description="Canonical UDUNITS-2 unit string, or null for categorical/dimensionless bands.",
    )

    @field_validator("unit")
    @classmethod
    def _check_canonical_unit(cls, v: str | None) -> str | None:
        validate_unit(v)
        return v


class ComposeLiteral(BaseModel):
    """Typed literal value for compose operands and fallback values."""

    type: Literal["literal"] = "literal"
    value: StrictInt | StrictFloat | str
    unit: str | None = Field(
        None,
        description="Canonical unit for numeric values. Must be null for string literals.",
    )

    @field_validator("unit")
    @classmethod
    def _check_canonical_unit(cls, v: str | None) -> str | None:
        validate_unit(v)
        return v

    @model_validator(mode="after")
    def _string_literals_are_unitless(self):
        if isinstance(self.value, str) and self.unit is not None:
            raise ValueError("String literals must not include a unit.")
        return self


BareNumber = StrictInt | StrictFloat
ComposeOperand = str | BareNumber | ComposeLiteral


def _validate_computation_operands(
    operator: ComposeOperator, operands: list[ComposeOperand]
) -> None:
    """Structural operand checks shared by computes and inline computes.

    These depend only on the request body (operator arity, the presence of a
    band operand, operand types); unit and band-type compatibility are checked
    by the router against the loaded source grids.
    """
    if operator in VARIADIC_OPERATORS:
        if len(operands) < 2:
            raise ValueError(f"Operator '{operator}' requires at least two operands.")
    elif len(operands) != 2:
        raise ValueError(f"Operator '{operator}' requires exactly two operands.")

    if not any(isinstance(operand, str) for operand in operands):
        raise ValueError("Compute operations must include at least one band operand.")
    if any(
        isinstance(operand, ComposeLiteral) and isinstance(operand.value, str)
        for operand in operands
    ):
        raise ValueError("String literals are not valid compute operands.")


class InlineCompute(BaseModel):
    """A computation body: an operator over operands.

    Usable on its own as a conditional-fallback value; `ComposeCompute`
    extends it with an output target and optional conditions.
    """

    operator: ComposeOperator
    operands: list[ComposeOperand] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_operands(self):
        _validate_computation_operands(self.operator, self.operands)
        return self


ComposeElseValue = str | BareNumber | ComposeLiteral | InlineCompute


class ComposeAttributeCondition(BaseModel):
    """Attribute condition using an alias-qualified input band reference."""

    band: str = Field(..., description="Alias-qualified band ref, e.g. `a.fbfm`.")
    operator: ComposeComparisonOperator
    value: StrictInt | StrictFloat | str | list[StrictInt | StrictFloat | str]

    @model_validator(mode="after")
    def _check_operator_value_shape(self):
        is_list = isinstance(self.value, list)
        if self.operator == ComposeComparisonOperator.in_ and not is_list:
            raise ValueError("The 'in' condition operator requires a list value.")
        if self.operator in _ORDERING_OPERATORS and is_list:
            raise ValueError(
                f"Operator '{self.operator}' does not support list values."
            )
        return self


ComposeCondition = (
    ComposeAttributeCondition
    | GridGeometrySpatialCondition
    | GridFeatureSpatialCondition
)


class ComposeSelect(BaseModel):
    """Select one input band into an output band, optionally conditionally."""

    model_config = ConfigDict(populate_by_name=True)

    output: str
    from_: str = Field(
        ..., alias="from", description="Alias-qualified source band ref."
    )
    conditions: list[ComposeCondition] | None = None
    else_: ComposeElseValue | None = Field(default=None, alias="else")

    @model_validator(mode="after")
    def _else_required_with_conditions(self):
        if self.conditions and self.else_ is None:
            raise ValueError("`else` is required when `conditions` are present.")
        return self


class ComposeCompute(InlineCompute):
    """Compute an output band from one or more operands."""

    model_config = ConfigDict(populate_by_name=True)

    output: str
    conditions: list[ComposeCondition] | None = None
    else_: ComposeElseValue | None = Field(default=None, alias="else")

    @model_validator(mode="after")
    def _else_required_with_conditions(self):
        if self.conditions and self.else_ is None:
            raise ValueError("`else` is required when `conditions` are present.")
        return self


class ComposeSource(BaseModel):
    """Persisted source metadata for grids created via compose."""

    name: Literal["compose"] = "compose"
    inputs: list[ComposeSourceInput]
    bands: list[Band]
    select: list[ComposeSelect] = Field(default_factory=list)
    compute: list[ComposeCompute] = Field(default_factory=list)


class CreateComposeRequest(BaseModel):
    """Request to create a grid by composing one or more existing grids."""

    inputs: list[ComposeInput] = Field(..., min_length=1)
    bands: list[ComposeOutputBand] = Field(..., min_length=1)
    select: list[ComposeSelect] = Field(default_factory=list)
    compute: list[ComposeCompute] = Field(default_factory=list)
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[GridModification] = Field(default_factory=list)

    @field_validator("inputs")
    @classmethod
    def _inputs_are_unique(cls, v: list[ComposeInput]) -> list[ComposeInput]:
        validate_no_duplicates([inp.alias for inp in v])
        validate_no_duplicates([inp.grid_id for inp in v])
        return v

    @field_validator("bands")
    @classmethod
    def _bands_are_unique(cls, v: list[ComposeOutputBand]) -> list[ComposeOutputBand]:
        validate_no_duplicates([band.key for band in v])
        return v

    @model_validator(mode="after")
    def _has_operations_and_matching_outputs(self):
        operation_outputs = [op.output for op in self.select] + [
            op.output for op in self.compute
        ]
        if not operation_outputs:
            raise ValueError("At least one select or compute operation is required.")
        validate_no_duplicates(operation_outputs)

        band_keys = [band.key for band in self.bands]
        if set(band_keys) != set(operation_outputs):
            raise ValueError(
                "`bands` keys must exactly match select/compute output keys."
            )
        return self


def build_compose_bands(outputs: list[ComposeOutputBand]) -> list[Band]:
    """Build indexed Grid Band models from explicit compose output metadata."""

    return [Band(index=i, **output.model_dump()) for i, output in enumerate(outputs)]
