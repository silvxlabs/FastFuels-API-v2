"""Router for grid compose endpoints."""

import math
import uuid
from datetime import datetime
from typing import Annotated, Any

import pint
from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import firestore_client, get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.grids.compose.examples import CREATE_COMPOSE_OPENAPI_EXAMPLES
from api.resources.grids.compose.schema import (
    CATEGORICAL_CONDITION_OPERATORS,
    ComposeAttributeCondition,
    ComposeCompute,
    ComposeElseValue,
    ComposeInput,
    ComposeLiteral,
    ComposeOperator,
    ComposeSelect,
    ComposeSource,
    ComposeSourceInput,
    CreateComposeRequest,
    InlineCompute,
)
from api.resources.grids.modification_models import (
    GridFeatureSpatialCondition,
)
from api.resources.grids.schema import CHUNK_SHAPE, Band, BandType, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    resolve_modification_fuel_model_labels,
    validate_feature_modifications,
    validate_grid_has_band,
    validate_grid_has_georeference,
)
from api.resources.modifications import stringify_modification_coordinates
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    FEATURES_COLLECTION,
    GRIDDLE_QUEUE,
    GRIDDLE_SERVICE,
    GRIDS_COLLECTION,
)
from lib.fuel_models import UnknownFuelModelError, resolve_fuel_model_value
from lib.units import canonicalize_unit

router = APIRouter()

COLLECTION = GRIDS_COLLECTION
_ureg = pint.UnitRegistry()

# Operators whose output carries the same unit as its operands (as opposed to
# multiply/divide, which derive a new unit). Operands must be unit-compatible
# with the output band.
_UNIT_MATCHED_OPERATORS = frozenset(
    {
        ComposeOperator.add,
        ComposeOperator.subtract,
        ComposeOperator.average,
        ComposeOperator.min,
        ComposeOperator.max,
    }
)


def _http_422(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=message
    )


def _split_band_ref(ref: str, aliases: set[str]) -> tuple[str, str]:
    alias, sep, band = ref.partition(".")
    if not sep or not alias or not band or alias not in aliases:
        raise _http_422(
            f"Band reference {ref!r} must use an input alias, e.g. 'a.fuel_load.1hr'."
        )
    return alias, band


def _band_by_key(grid_data: dict, grid_id: str, key: str) -> dict:
    validate_grid_has_band(grid_data, grid_id, key)
    return next(b for b in grid_data.get("bands", []) if b["key"] == key)


def _band_ref_metadata(
    ref: str,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> dict:
    alias, band_key = _split_band_ref(ref, set(source_grids))
    grid_data = source_grids[alias]
    return _band_by_key(grid_data, input_by_alias[alias].grid_id, band_key)


def _shape_rank(grid_data: dict, grid_id: str) -> int:
    validate_grid_has_georeference(grid_data, grid_id)
    shape = grid_data["georeference"].get("shape", [])
    rank = len(shape)
    if rank != 2:
        raise _http_422(f"Grid '{grid_id}' is {rank}D. Compose supports 2D grids only.")
    return rank


def _validate_alignment(
    source_grids: dict[str, dict], input_by_alias: dict[str, ComposeInput]
) -> None:
    first_alias = next(iter(source_grids))
    first_grid = source_grids[first_alias]
    _shape_rank(first_grid, input_by_alias[first_alias].grid_id)
    first_georef = first_grid["georeference"]
    first_transform = tuple(first_georef["transform"])

    for alias, grid_data in source_grids.items():
        grid_id = input_by_alias[alias].grid_id
        _shape_rank(grid_data, grid_id)
        georef = grid_data["georeference"]
        if georef.get("crs") != first_georef.get("crs"):
            raise _http_422("All compose input grids must have the same CRS.")
        if tuple(georef.get("shape", ())) != tuple(first_georef.get("shape", ())):
            raise _http_422("All compose input grids must have the same shape.")
        transform = tuple(georef.get("transform", ()))
        if len(transform) != len(first_transform) or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
            for a, b in zip(transform, first_transform, strict=True)
        ):
            raise _http_422("All compose input grids must have the same transform.")


async def _load_source_grids(
    inputs: list[ComposeInput],
    owner_id: str,
    domain_id: str,
) -> dict[str, dict]:
    source_grids: dict[str, dict] = {}
    for inp in inputs:
        _, snapshot = await get_document_async(
            COLLECTION,
            inp.grid_id,
            owner_id=owner_id,
            domain_id=domain_id,
            document_status="completed",
        )
        source_grids[inp.alias] = snapshot.to_dict()
    return source_grids


async def _validate_compose_feature_conditions(
    select: list[ComposeSelect],
    compute: list[ComposeCompute],
    owner_id: str,
    domain_id: str,
) -> None:
    feature_ids: list[str] = []
    seen: set[str] = set()
    for operation in [*select, *compute]:
        for condition in operation.conditions or []:
            if (
                isinstance(condition, GridFeatureSpatialCondition)
                and condition.feature_id not in seen
            ):
                seen.add(condition.feature_id)
                feature_ids.append(condition.feature_id)

    if not feature_ids:
        return

    refs = [
        firestore_client.collection(FEATURES_COLLECTION).document(fid)
        for fid in feature_ids
    ]
    snapshots = {snap.id: snap async for snap in firestore_client.get_all(refs)}

    for fid in feature_ids:
        snap = snapshots.get(fid)
        data = snap.to_dict() if snap is not None and snap.exists else None
        if (
            data is None
            or data.get("owner_id") != owner_id
            or data.get("domain_id") != domain_id
        ):
            raise _http_422(
                f"Compose condition references feature_id {fid!r}, which does not exist in this domain."
            )
        feature_status = data.get("status")
        if feature_status != "completed":
            raise _http_422(
                f"Compose condition references feature_id {fid!r} whose status is "
                f"{feature_status!r}, expected 'completed'."
            )


def _unit_object(unit: str | None):
    if unit is None:
        return _ureg.dimensionless
    return _ureg.parse_units(unit)


def _canonical_unit_from_pint(unit) -> str | None:
    formatted = f"{unit:~C}"
    if formatted == "%":
        return "%"
    if unit.dimensionless:
        return None
    return canonicalize_unit(formatted)


def _operand_metadata(
    operand: Any,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> dict | None:
    if isinstance(operand, str):
        return _band_ref_metadata(operand, source_grids, input_by_alias)
    return None


def _literal_unit(operand: Any) -> str | None:
    if isinstance(operand, ComposeLiteral):
        return operand.unit
    return None


def _is_bare_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _resolve_labels(value: Any) -> Any:
    """Resolve FBFM40 string labels to integer codes, raising 422 on unknowns."""
    try:
        return resolve_fuel_model_value(value)
    except UnknownFuelModelError as exc:
        raise _http_422(str(exc))


def _value_items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _all_numeric_values(value: Any) -> bool:
    return all(_is_bare_number(item) for item in _value_items(value))


def _units_compatible(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return _unit_object(left).is_compatible_with(_unit_object(right))


def _derive_compute_unit(
    op: ComposeCompute | InlineCompute,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> str | None:
    """Validate operand band types and return the compute's output unit.

    Operand arity and structural rules are enforced by the schema; this needs
    the loaded source-grid metadata. The output unit defaults to the unit
    derived from the operands and may be overridden (on `ComposeCompute`) with
    any dimensionally compatible unit, which the worker converts to.
    """
    raster_bands = [
        metadata
        for operand in op.operands
        if (metadata := _operand_metadata(operand, source_grids, input_by_alias))
        is not None
    ]
    for band in raster_bands:
        if band["type"] != BandType.continuous.value:
            raise _http_422("Compute operands must reference continuous bands.")

    typed_literal_units = [
        operand.unit for operand in op.operands if isinstance(operand, ComposeLiteral)
    ]
    override = getattr(op, "unit", None)

    if op.operator in _UNIT_MATCHED_OPERATORS:
        units = [band.get("unit") for band in raster_bands] + typed_literal_units
        has_unitful = any(unit is not None for unit in units)
        has_unitless = any(unit is None for unit in units)
        if has_unitful and has_unitless:
            raise _http_422(
                f"Operator '{op.operator}' cannot mix unitless and unitful operands."
            )
        # The operands all share a unit; default the output to the first band's.
        output_unit = override if override is not None else raster_bands[0].get("unit")
        if any(not _units_compatible(unit, output_unit) for unit in units):
            raise _http_422(
                f"Operator '{op.operator}' requires operands compatible with output unit "
                f"{output_unit!r}."
            )
        return output_unit

    if op.operator == ComposeOperator.multiply:
        unit = _ureg.dimensionless
        for band in raster_bands:
            unit *= _unit_object(band.get("unit"))
        for literal_unit in typed_literal_units:
            unit *= _unit_object(literal_unit)
        derived_unit = _canonical_unit_from_pint(unit)
    else:
        numerator, denominator = op.operands

        def operand_unit(operand: Any):
            metadata = _operand_metadata(operand, source_grids, input_by_alias)
            if metadata is not None:
                return _unit_object(metadata.get("unit"))
            return _unit_object(_literal_unit(operand))

        derived_unit = _canonical_unit_from_pint(
            operand_unit(numerator) / operand_unit(denominator)
        )

    if override is not None and not _units_compatible(override, derived_unit):
        raise _http_422(
            f"Output unit {override!r} is not compatible with the unit "
            f"{derived_unit!r} derived for operator '{op.operator}'."
        )
    return override if override is not None else derived_unit


def _validate_else_value(
    value: ComposeElseValue,
    output_band: Band,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> None:
    if isinstance(value, InlineCompute):
        else_unit = _derive_compute_unit(value, source_grids, input_by_alias)
        if not _units_compatible(else_unit, output_band.unit):
            raise _http_422(
                f"Inline `else` compute produces unit {else_unit!r}, incompatible "
                f"with output {output_band.key!r} unit {output_band.unit!r}."
            )
        return

    metadata = (
        _band_ref_metadata(value, source_grids, input_by_alias)
        if isinstance(value, str) and value.partition(".")[0] in source_grids
        else None
    )
    if metadata is not None:
        if (
            metadata["type"] != output_band.type.value
            or metadata.get("unit") != output_band.unit
        ):
            raise _http_422(
                f"Else band reference {value!r} is not compatible with output {output_band.key!r}."
            )
        return

    if isinstance(value, ComposeLiteral):
        if isinstance(value.value, str):
            raise _http_422(
                "String fallbacks are only supported as FBFM fuel-model labels "
                "for categorical output bands."
            )
        elif value.unit != output_band.unit:
            raise _http_422(
                f"Literal fallback unit {value.unit!r} does not match output unit {output_band.unit!r}."
            )
        return

    if isinstance(value, str):
        raise _http_422(
            "String fallbacks are only supported as FBFM fuel-model labels "
            "for categorical output bands."
        )


def _resolve_else_labels(value: ComposeElseValue, output_band: Band, aliases: set[str]):
    """Resolve an FBFM label `else` fallback for a categorical output to a code.

    Band references (`alias.band`) and non-categorical outputs are left as-is.
    """
    if output_band.type != BandType.categorical:
        return value
    if isinstance(value, str) and value.partition(".")[0] not in aliases:
        return _resolve_labels(value)
    if isinstance(value, ComposeLiteral) and isinstance(value.value, str):
        value.value = _resolve_labels(value.value)
    return value


def _validate_conditions(
    conditions: list[Any] | None,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> None:
    for condition in conditions or []:
        if not isinstance(condition, ComposeAttributeCondition):
            continue
        band = _band_ref_metadata(condition.band, source_grids, input_by_alias)
        if band["type"] == BandType.categorical.value:
            if condition.operator not in CATEGORICAL_CONDITION_OPERATORS:
                raise _http_422(
                    "Categorical compose conditions support only eq, ne, and in."
                )
            # FBFM labels (e.g. "GR1") resolve to their stored integer code.
            condition.value = _resolve_labels(condition.value)
        elif not _all_numeric_values(condition.value):
            raise _http_422("Continuous compose conditions require numeric values.")


def _build_output_bands(
    body: CreateComposeRequest,
    source_grids: dict[str, dict],
    input_by_alias: dict[str, ComposeInput],
) -> list[Band]:
    """Derive the ordered output band list from the compose operations.

    Select outputs inherit the source band's type and unit; compute outputs are
    always continuous with a derived (or compatibly overridden) unit.
    `name`/`description` come from the operation. Conditions and `else`
    fallbacks are validated here against the loaded source grids, and FBFM
    label fallbacks are resolved in place.
    """
    bands: list[Band] = []
    for operation in [*body.select, *body.compute]:
        if isinstance(operation, ComposeSelect):
            source_band = _band_ref_metadata(
                operation.from_, source_grids, input_by_alias
            )
            band = Band(
                index=len(bands),
                key=operation.output,
                type=BandType(source_band["type"]),
                unit=source_band.get("unit"),
                name=operation.name,
                description=operation.description,
            )
        else:
            band = Band(
                index=len(bands),
                key=operation.output,
                type=BandType.continuous,
                unit=_derive_compute_unit(operation, source_grids, input_by_alias),
                name=operation.name,
                description=operation.description,
            )

        _validate_conditions(operation.conditions, source_grids, input_by_alias)
        if operation.else_ is not None:
            operation.else_ = _resolve_else_labels(
                operation.else_, band, set(source_grids)
            )
            _validate_else_value(operation.else_, band, source_grids, input_by_alias)
        bands.append(band)
    return bands


def _dump_operations_for_firestore(
    operations: list[ComposeSelect] | list[ComposeCompute],
) -> list[dict]:
    data = [op.model_dump(by_alias=True, exclude_none=True) for op in operations]
    return stringify_modification_coordinates(data)


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid by composing existing grids",
    responses=QUOTA_429_RESPONSE,
)
async def create_compose_grid(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateComposeRequest,
        Body(openapi_examples=CREATE_COMPOSE_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Compose Grid

    Creates a new grid by selecting bands, computing bands, and applying
    optional conditional fallback rules across one or more completed grids.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_modifications(body.modifications, owner_id, domain_id)
    await _validate_compose_feature_conditions(
        body.select, body.compute, owner_id, domain_id
    )

    source_grids = await _load_source_grids(body.inputs, owner_id, domain_id)
    input_by_alias = {inp.alias: inp for inp in body.inputs}
    _validate_alignment(source_grids, input_by_alias)

    bands = _build_output_bands(body, source_grids, input_by_alias)
    output_band_types = {band.key: band.type.value for band in bands}
    resolve_modification_fuel_model_labels(body.modifications, output_band_types)

    source = ComposeSource(
        inputs=[
            ComposeSourceInput(
                grid_id=inp.grid_id,
                alias=inp.alias,
                source_grid_checksum=source_grids[inp.alias].get("checksum"),
            )
            for inp in body.inputs
        ],
        bands=bands,
        select=body.select,
        compute=body.compute,
    )

    source_data = source.model_dump(by_alias=True)
    source_data["select"] = _dump_operations_for_firestore(body.select)
    source_data["compute"] = _dump_operations_for_firestore(body.compute)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source_data,
        "modifications": dump_modifications_for_firestore(body.modifications),
        "bands": [band.model_dump() for band in bands],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
