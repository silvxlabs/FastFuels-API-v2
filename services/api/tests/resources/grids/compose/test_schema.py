"""Unit tests for compose schema models."""

import pytest
from api.resources.grids.compose.examples import ALL_COMPOSE_EXAMPLE_VALUES
from api.resources.grids.compose.schema import (
    ComposeAttributeCondition,
    ComposeCompute,
    ComposeLiteral,
    ComposeSelect,
    CreateComposeRequest,
    InlineCompute,
)
from pydantic import ValidationError


class TestCreateComposeRequest:
    def test_minimal_valid_request(self):
        request = CreateComposeRequest(
            inputs=[{"grid_id": "grid_a", "alias": "a"}],
            operations=[
                {
                    "type": "compute",
                    "output": "fuel_load.1hr",
                    "operator": "multiply",
                    "operands": ["a.fuel_load.1hr", 0.5],
                }
            ],
        )

        assert request.inputs[0].alias == "a"
        assert isinstance(request.operations[0], ComposeCompute)
        assert request.operations[0].operands[1] == 0.5

    @pytest.mark.parametrize("operations", [None, []])
    def test_at_least_one_operation_required(self, operations):
        body = {"inputs": [{"grid_id": "grid_a", "alias": "a"}]}
        if operations is not None:
            body["operations"] = operations
        with pytest.raises(ValidationError, match="operations"):
            CreateComposeRequest(**body)

    def test_operation_type_is_required(self):
        with pytest.raises(ValidationError, match="type"):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a"}],
                operations=[{"output": "fbfm", "from": "a.fbfm"}],
            )

    def test_operations_keep_order_and_discriminate(self):
        request = CreateComposeRequest(
            inputs=[{"grid_id": "grid_a", "alias": "a"}],
            operations=[
                {
                    "type": "compute",
                    "output": "fuel_load.1hr",
                    "operator": "multiply",
                    "operands": ["a.fuel_load.1hr", 0.5],
                },
                {"type": "select", "output": "fbfm", "from": "a.fbfm"},
            ],
        )

        assert [type(op) for op in request.operations] == [
            ComposeCompute,
            ComposeSelect,
        ]

    def test_operations_required_in_json_schema(self):
        schema = CreateComposeRequest.model_json_schema()
        assert "operations" in schema["required"]
        assert schema["properties"]["operations"]["minItems"] == 1
        assert "select" not in schema["properties"]
        assert "compute" not in schema["properties"]

    def test_alias_must_be_identifier_like(self):
        with pytest.raises(ValidationError):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a.bad"}],
                operations=[
                    {
                        "type": "compute",
                        "output": "fuel_load.1hr",
                        "operator": "multiply",
                        "operands": ["a.fuel_load.1hr", 0.5],
                    }
                ],
            )

    def test_duplicate_outputs_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a"}],
                operations=[
                    {"type": "select", "output": "fbfm", "from": "a.fbfm"},
                    {
                        "type": "compute",
                        "output": "fbfm",
                        "operator": "multiply",
                        "operands": ["a.fuel_load.1hr", 0.5],
                    },
                ],
            )

    def test_else_required_with_conditions(self):
        with pytest.raises(ValidationError, match="else"):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a"}],
                operations=[
                    {
                        "type": "select",
                        "output": "fuel_load.1hr",
                        "from": "a.fuel_load.1hr",
                        "conditions": [
                            {"band": "a.fbfm", "operator": "eq", "value": 91}
                        ],
                    }
                ],
            )


class TestComposeLiteral:
    def test_typed_literal_accepts_canonical_unit(self):
        literal = ComposeLiteral(value=7, unit="kg/m**2")
        assert literal.type == "literal"
        assert literal.value == 7

    def test_typed_literal_rejects_noncanonical_unit(self):
        with pytest.raises(ValidationError):
            ComposeLiteral(value=7, unit="kg/m^2")

    def test_string_literal_must_be_unitless(self):
        with pytest.raises(ValidationError):
            ComposeLiteral(value="NB1", unit="%")


class TestComputationOperands:
    """Operand arity and structure are enforced by the schema, not the router."""

    def test_binary_operator_requires_exactly_two_operands(self):
        with pytest.raises(ValidationError, match="exactly two operands"):
            InlineCompute(operator="divide", operands=["a.x", "a.y", "a.z"])

    def test_variadic_operator_requires_at_least_two_operands(self):
        with pytest.raises(ValidationError, match="at least two operands"):
            InlineCompute(operator="add", operands=["a.x"])

    def test_requires_at_least_one_band_operand(self):
        with pytest.raises(ValidationError, match="at least one band operand"):
            InlineCompute(operator="add", operands=[1, 2])

    def test_string_literal_is_not_a_valid_operand(self):
        with pytest.raises(ValidationError, match="String literals are not valid"):
            InlineCompute(
                operator="add",
                operands=["a.x", {"type": "literal", "value": "NB1"}],
            )

    def test_compose_compute_inherits_operand_validation(self):
        with pytest.raises(ValidationError, match="exactly two operands"):
            ComposeCompute(
                type="compute",
                output="fuel_load",
                operator="subtract",
                operands=["a.x"],
            )


class TestAttributeConditionShape:
    """`in`/ordering operator value shape is enforced by the schema."""

    def test_in_operator_requires_list_value(self):
        with pytest.raises(ValidationError, match="requires a list value"):
            ComposeAttributeCondition(band="a.fbfm", operator="in", value=101)

    def test_ordering_operator_rejects_list_value(self):
        with pytest.raises(ValidationError, match="does not support list values"):
            ComposeAttributeCondition(band="a.x", operator="gt", value=[1, 2])

    def test_equality_accepts_scalar_or_list(self):
        assert ComposeAttributeCondition(band="a.x", operator="eq", value=1).value == 1
        assert ComposeAttributeCondition(
            band="a.x", operator="eq", value=[1, 2]
        ).value == [1, 2]


class TestExamplesValidateAgainstSchema:
    @pytest.mark.parametrize("example_name,example_value", ALL_COMPOSE_EXAMPLE_VALUES)
    def test_example_validates_against_schema(self, example_name, example_value):
        request = CreateComposeRequest(**example_value)
        assert request.inputs, example_name
        assert request.operations, example_name


class TestOutputKey:
    def test_unknown_quantity_rejected(self):
        with pytest.raises(ValidationError, match="not a band key"):
            ComposeSelect.model_validate(
                {"type": "select", "output": "moisture", "from": "a.x"}
            )
