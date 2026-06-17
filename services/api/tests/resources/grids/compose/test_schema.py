"""Unit tests for compose schema models."""

import pytest
from api.resources.grids.compose.examples import ALL_COMPOSE_EXAMPLE_VALUES
from api.resources.grids.compose.schema import (
    ComposeAttributeCondition,
    ComposeCompute,
    ComposeLiteral,
    CreateComposeRequest,
    InlineCompute,
)
from pydantic import ValidationError


class TestCreateComposeRequest:
    def test_minimal_valid_request(self):
        request = CreateComposeRequest(
            inputs=[{"grid_id": "grid_a", "alias": "a"}],
            compute=[
                {
                    "output": "fuel_load.1hr",
                    "operator": "multiply",
                    "operands": ["a.fuel_load.1hr", 0.5],
                }
            ],
        )

        assert request.inputs[0].alias == "a"
        assert request.compute[0].operands[1] == 0.5

    def test_at_least_one_operation_required(self):
        with pytest.raises(ValidationError, match="select or compute"):
            CreateComposeRequest(inputs=[{"grid_id": "grid_a", "alias": "a"}])

    def test_alias_must_be_identifier_like(self):
        with pytest.raises(ValidationError):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a.bad"}],
                compute=[
                    {
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
                select=[
                    {"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"},
                    {"output": "fuel_load.1hr", "from": "a.fuel_load.10hr"},
                ],
            )

    def test_else_required_with_conditions(self):
        with pytest.raises(ValidationError, match="else"):
            CreateComposeRequest(
                inputs=[{"grid_id": "grid_a", "alias": "a"}],
                select=[
                    {
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
            ComposeCompute(output="z", operator="subtract", operands=["a.x"])


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
        assert request.select or request.compute, example_name
