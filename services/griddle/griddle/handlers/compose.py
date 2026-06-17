"""Compose handler for Griddle."""

import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pint
import rioxarray  # noqa: F401
import xarray as xr

from griddle.modifications import _evaluate_spatial_condition, _resolve_band
from griddle.storage import load_zarr
from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError, get_document

COMPARISON_OPERATORS: dict[str, Callable] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
}
_ureg = pint.UnitRegistry()


def compose_grid(
    grid: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Compose a new Dataset from one or more existing grid Zarr stores."""

    progress("Loading compose inputs...", 10)
    inputs = _load_inputs(source["inputs"], grid)
    _validate_alignment(inputs)
    _normalize_input_coords(inputs)

    base_ds = next(iter(inputs.values()))["dataset"]
    feature_cache: dict[tuple[str, float], object] = {}
    output_vars: dict[str, xr.DataArray] = {}
    output_bands = {band["key"]: band for band in grid["bands"]}

    progress("Evaluating compose operations...", 50)
    for operation in source.get("select", []):
        output_band = output_bands[operation["output"]]
        output_vars[operation["output"]] = _evaluate_select(
            operation,
            output_band,
            inputs,
            base_ds,
            grid["domain_id"],
            feature_cache,
        )

    for operation in source.get("compute", []):
        output_band = output_bands[operation["output"]]
        output_vars[operation["output"]] = _evaluate_compute(
            operation,
            output_band,
            inputs,
            base_ds,
            grid["domain_id"],
            feature_cache,
        )

    missing = [band["key"] for band in grid["bands"] if band["key"] not in output_vars]
    if missing:
        raise ProcessingError(
            code="COMPOSE_OUTPUT_MISSING",
            message=f"Compose did not produce expected output bands: {missing}.",
            suggestion="Check the compose source metadata.",
        )

    ordered_vars = {band["key"]: output_vars[band["key"]] for band in grid["bands"]}
    result = xr.Dataset(ordered_vars)
    result = result.rio.write_crs(base_ds.rio.crs)
    result = result.rio.write_transform(base_ds.rio.transform())
    progress("Compose complete.", 80)
    return result


def _load_inputs(inputs: list[dict], output_grid: dict) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for inp in inputs:
        grid_id = inp["grid_id"]
        alias = inp["alias"]
        try:
            _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
        except DocumentNotFoundError:
            raise ProcessingError(
                code="COMPOSE_INPUT_NOT_FOUND",
                message=f"Compose input grid '{grid_id}' no longer exists.",
                suggestion="Recreate the compose grid with existing input grids.",
            )
        grid_doc = snapshot.to_dict()
        _validate_input_grid_doc(inp, grid_doc, output_grid)
        loaded[alias] = {
            "grid": grid_doc,
            "dataset": load_zarr(grid_id),
        }
    return loaded


def _validate_input_grid_doc(inp: dict, grid_doc: dict, output_grid: dict) -> None:
    grid_id = inp["grid_id"]
    status = grid_doc.get("status")
    if status != "completed":
        raise ProcessingError(
            code="COMPOSE_INPUT_NOT_COMPLETED",
            message=(
                f"Compose input grid '{grid_id}' has status {status!r}, "
                "expected 'completed'."
            ),
            suggestion="Recreate the compose grid after all input grids complete.",
        )

    domain_id = output_grid.get("domain_id")
    if domain_id is not None and grid_doc.get("domain_id") != domain_id:
        raise ProcessingError(
            code="COMPOSE_INPUT_DOMAIN_MISMATCH",
            message=f"Compose input grid '{grid_id}' is not in the output domain.",
            suggestion="Recreate the compose grid with same-domain input grids.",
        )

    owner_id = output_grid.get("owner_id")
    if owner_id is not None and grid_doc.get("owner_id") != owner_id:
        raise ProcessingError(
            code="COMPOSE_INPUT_OWNER_MISMATCH",
            message=f"Compose input grid '{grid_id}' is not owned by the output owner.",
            suggestion="Recreate the compose grid with owned input grids.",
        )

    expected_checksum = inp.get("source_grid_checksum")
    current_checksum = grid_doc.get("checksum")
    if expected_checksum is not None and current_checksum != expected_checksum:
        raise ProcessingError(
            code="COMPOSE_INPUT_CHANGED",
            message=(
                f"Compose input grid '{grid_id}' changed after this compose "
                "request was created."
            ),
            suggestion="Recreate the compose grid so provenance matches input data.",
        )


def _validate_alignment(inputs: dict[str, dict]) -> None:
    first_alias = next(iter(inputs))
    first_ds = inputs[first_alias]["dataset"]
    _validate_dataset_is_2d(first_ds, first_alias)
    if first_ds.rio.crs is None:
        raise ProcessingError(
            code="COMPOSE_ALIGNMENT_MISMATCH",
            message="All compose input grids must have a CRS.",
        )
    first_transform = tuple(first_ds.rio.transform())[:6]
    first_shape = (first_ds.rio.height, first_ds.rio.width)

    for alias, item in inputs.items():
        ds = item["dataset"]
        _validate_dataset_is_2d(ds, alias)
        if ds.rio.crs != first_ds.rio.crs:
            raise ProcessingError(
                code="COMPOSE_ALIGNMENT_MISMATCH",
                message="All compose input grids must have the same CRS.",
            )
        if (ds.rio.height, ds.rio.width) != first_shape:
            raise ProcessingError(
                code="COMPOSE_ALIGNMENT_MISMATCH",
                message="All compose input grids must have the same shape.",
            )
        transform = tuple(ds.rio.transform())[:6]
        if any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
            for a, b in zip(transform, first_transform, strict=True)
        ):
            raise ProcessingError(
                code="COMPOSE_ALIGNMENT_MISMATCH",
                message="All compose input grids must have the same transform.",
            )


def _validate_dataset_is_2d(ds: xr.Dataset, alias: str) -> None:
    if not ds.data_vars:
        raise ProcessingError(
            code="COMPOSE_INPUT_EMPTY",
            message=f"Compose input '{alias}' has no data variables.",
        )
    for name, da in ds.data_vars.items():
        if da.ndim != 2:
            raise ProcessingError(
                code="COMPOSE_UNSUPPORTED_DIMENSIONALITY",
                message=(
                    f"Compose input '{alias}' variable '{name}' is {da.ndim}D. "
                    "Compose currently supports 2D grids only."
                ),
            )


def _normalize_input_coords(inputs: dict[str, dict]) -> None:
    """Snap every input onto the first input's spatial coordinates.

    Alignment validation only proves the inputs share a shape and a transform
    to within ``abs_tol=1e-9``. xarray arithmetic and ``xr.where`` align by
    coordinate *label*, so sub-nanometer drift between grids built by different
    code paths (e.g. a LANDFIRE lattice vs. a reprojected raster) could
    otherwise inner-join to NaN or empty results. Assigning one shared set of
    ``y``/``x`` coordinates makes every later combination broadcast
    cell-for-cell.
    """
    first_ds = inputs[next(iter(inputs))]["dataset"]
    y_dim, x_dim = first_ds.rio.y_dim, first_ds.rio.x_dim
    ref_coords = {
        dim: first_ds.coords[dim] for dim in (y_dim, x_dim) if dim in first_ds.coords
    }
    if not ref_coords:
        return
    for item in inputs.values():
        ds = item["dataset"]
        shared = {dim: coord for dim, coord in ref_coords.items() if dim in ds.coords}
        if shared:
            item["dataset"] = ds.assign_coords(shared)


def _split_ref(ref: str, inputs: dict[str, dict]) -> tuple[str, str]:
    alias, sep, band_key = ref.partition(".")
    if not sep or alias not in inputs or not band_key:
        raise ProcessingError(
            code="COMPOSE_INVALID_BAND_REF",
            message=f"Invalid compose band reference: {ref!r}.",
            suggestion="Use an alias-qualified reference such as 'a.fuel_load.1hr'.",
        )
    return alias, band_key


def _band_da(inputs: dict[str, dict], ref: str) -> xr.DataArray:
    alias, band_key = _split_ref(ref, inputs)
    ds = inputs[alias]["dataset"]
    var_name, band_coord_val = _resolve_band(ds, band_key)
    da = ds[var_name]
    if band_coord_val is not None:
        da = da.sel(band=band_coord_val)
    return da


def _band_metadata(inputs: dict[str, dict], ref: str) -> dict:
    alias, band_key = _split_ref(ref, inputs)
    for band in inputs[alias]["grid"].get("bands", []):
        if band["key"] == band_key:
            return band
    raise ProcessingError(
        code="COMPOSE_INVALID_BAND_REF",
        message=f"Compose input band '{ref}' is missing from input grid metadata.",
    )


def _unit_object(unit: str | None):
    if unit is None:
        return _ureg.dimensionless
    return _ureg.parse_units(unit)


def _conversion_factor(source_unit: str | None, target_unit: str | None) -> float:
    return float(
        (1 * _unit_object(source_unit)).to(_unit_object(target_unit)).magnitude
    )


def _is_known_band_ref(value: Any, inputs: dict[str, dict]) -> bool:
    return isinstance(value, str) and value.partition(".")[0] in inputs


def _literal_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "literal":
        return value["value"]
    return value


def _literal_numeric_value(value: dict, target_unit: str | None = None) -> int | float:
    literal_value = value["value"]
    literal_unit = value.get("unit")
    if literal_unit is not None:
        return literal_value * _conversion_factor(literal_unit, target_unit)
    return literal_value


def _nodata_mask(da: xr.DataArray) -> np.ndarray:
    arr = da.values
    nodata = da.rio.nodata
    if nodata is None:
        if np.issubdtype(arr.dtype, np.floating):
            return np.isnan(arr)
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(arr)
    return arr == nodata


def _template_da(base_ds: xr.Dataset) -> xr.DataArray:
    first_name = next(iter(base_ds.data_vars))
    return xr.zeros_like(base_ds[first_name], dtype=float)


def _data_array_from_values(template: xr.DataArray, values: Any) -> xr.DataArray:
    if isinstance(values, xr.DataArray):
        return values
    return xr.DataArray(values, dims=template.dims, coords=template.coords)


def _condition_mask(
    conditions: list[dict] | None,
    inputs: dict[str, dict],
    base_ds: xr.Dataset,
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> np.ndarray:
    mask = np.ones((base_ds.rio.height, base_ds.rio.width), dtype=bool)
    for condition in conditions or []:
        if "source" in condition:
            cond_mask = _evaluate_spatial_condition(
                base_ds, condition, domain_id, feature_cache
            )
        else:
            cond_mask = _attribute_condition_mask(condition, inputs)
        mask &= cond_mask
    return mask


def _attribute_condition_mask(condition: dict, inputs: dict[str, dict]) -> np.ndarray:
    da = _band_da(inputs, condition["band"])
    arr = da.values
    op_name = condition["operator"]
    value = condition["value"]
    _validate_condition_value(condition, inputs)

    if op_name == "in":
        result = np.isin(arr, value)
    elif isinstance(value, list):
        if op_name == "eq":
            result = np.isin(arr, value)
        elif op_name == "ne":
            result = ~np.isin(arr, value)
        else:
            raise ProcessingError(
                code="COMPOSE_INVALID_CONDITION",
                message=f"Operator '{op_name}' does not support list values.",
            )
    else:
        result = COMPARISON_OPERATORS[op_name](arr, value)
    return result & ~_nodata_mask(da)


def _condition_value_items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _validate_condition_value(condition: dict, inputs: dict[str, dict]) -> None:
    op_name = condition["operator"]
    value = condition["value"]
    metadata = _band_metadata(inputs, condition["band"])
    if op_name == "in" and not isinstance(value, list):
        raise ProcessingError(
            code="COMPOSE_INVALID_CONDITION",
            message="The 'in' compose condition operator requires a list value.",
        )
    if op_name not in {"eq", "ne", "in"} and isinstance(value, list):
        raise ProcessingError(
            code="COMPOSE_INVALID_CONDITION",
            message=f"Operator '{op_name}' does not support list values.",
        )
    if metadata.get("type") == "categorical" and any(
        isinstance(item, str) for item in _condition_value_items(value)
    ):
        raise ProcessingError(
            code="COMPOSE_INVALID_CONDITION",
            message=(
                "Categorical compose conditions require numeric stored codes, "
                "not string labels."
            ),
        )


def _evaluate_value(
    value: Any,
    output_band: dict,
    inputs: dict[str, dict],
    base_ds: xr.Dataset,
    used_mask: np.ndarray | None = None,
) -> xr.DataArray | int | float | str:
    if _is_known_band_ref(value, inputs):
        return _band_da(inputs, value)
    if isinstance(value, dict) and value.get("type") == "literal":
        return value["value"]
    if isinstance(value, dict) and "operator" in value:
        return _evaluate_inline_compute(
            value,
            output_band,
            inputs,
            base_ds,
            used_mask=used_mask,
        )
    return value


def _evaluate_select(
    operation: dict,
    output_band: dict,
    inputs: dict[str, dict],
    base_ds: xr.Dataset,
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> xr.DataArray:
    selected = _band_da(inputs, operation["from"])
    if not operation.get("conditions"):
        return selected.copy()

    mask = _condition_mask(
        operation["conditions"], inputs, base_ds, domain_id, feature_cache
    )
    fallback = _evaluate_value(
        operation["else"], output_band, inputs, base_ds, used_mask=~mask
    )
    mask_da = xr.DataArray(mask, dims=selected.dims, coords=selected.coords)
    return xr.where(mask_da, selected, fallback)


def _evaluate_compute(
    operation: dict,
    output_band: dict,
    inputs: dict[str, dict],
    base_ds: xr.Dataset,
    domain_id: str,
    feature_cache: dict[tuple[str, float], object],
) -> xr.DataArray:
    if not operation.get("conditions"):
        return _evaluate_inline_compute(operation, output_band, inputs, base_ds)

    # Compute the mask first so the non-finite check only fires on cells that
    # actually keep the computed branch — a divide-by-zero in a cell the
    # conditions exclude (and that takes the fallback) must not fail the job.
    mask = _condition_mask(
        operation["conditions"], inputs, base_ds, domain_id, feature_cache
    )
    computed = _evaluate_inline_compute(
        operation, output_band, inputs, base_ds, used_mask=mask
    )
    fallback = _evaluate_value(
        operation["else"], output_band, inputs, base_ds, used_mask=~mask
    )
    mask_da = xr.DataArray(mask, dims=computed.dims, coords=computed.coords)
    return xr.where(mask_da, computed, fallback)


def _evaluate_inline_compute(
    operation: dict,
    output_band: dict,
    inputs: dict[str, dict],
    base_ds: xr.Dataset,
    used_mask: np.ndarray | None = None,
) -> xr.DataArray:
    template = _template_da(base_ds)
    values: list[xr.DataArray | int | float] = []
    valid_mask = np.ones((base_ds.rio.height, base_ds.rio.width), dtype=bool)
    op_name = operation["operator"]
    output_unit = output_band.get("unit")

    for operand in operation["operands"]:
        if isinstance(operand, str):
            raw = _band_da(inputs, operand)
            # Read nodata from the band as loaded — before astype (which need
            # not preserve the nodata metadata) and before any unit scaling
            # (which would shift a sentinel past `== nodata` recognition).
            valid_mask &= ~_nodata_mask(raw)
            da = raw.astype(float)
            if op_name in {"add", "subtract", "average", "max", "min"}:
                source_unit = _band_metadata(inputs, operand).get("unit")
                factor = _conversion_factor(source_unit, output_unit)
                da = da * factor
            values.append(da)
        elif isinstance(operand, dict) and operand.get("type") == "literal":
            if op_name in {"add", "subtract", "average", "max", "min"}:
                values.append(_literal_numeric_value(operand, output_unit))
            else:
                values.append(_literal_value(operand))
        else:
            values.append(operand)

    if op_name == "add":
        result = values[0]
        for value in values[1:]:
            result = result + value
    elif op_name == "subtract":
        result = values[0] - values[1]
    elif op_name == "multiply":
        result = values[0]
        for value in values[1:]:
            result = result * value
    elif op_name == "divide":
        result = values[0] / values[1]
    elif op_name == "average":
        result = sum(values) / len(values)
    elif op_name == "max":
        result = values[0]
        for value in values[1:]:
            result = np.maximum(result, value)
    elif op_name == "min":
        result = values[0]
        for value in values[1:]:
            result = np.minimum(result, value)
    else:
        raise ProcessingError(
            code="COMPOSE_UNKNOWN_OPERATOR",
            message=f"Unknown compose operator: {op_name}.",
        )

    if op_name in {"multiply", "divide"}:
        result = result * _operation_conversion_factor(operation, inputs, output_unit)

    result_da = _data_array_from_values(template, result)
    check_mask = valid_mask if used_mask is None else (valid_mask & used_mask)
    _raise_on_non_finite_result(result_da, check_mask)
    valid_da = xr.DataArray(valid_mask, dims=result_da.dims, coords=result_da.coords)
    result_da = xr.where(valid_da, result_da, np.nan)
    return result_da.rio.write_nodata(np.nan)


def _operation_conversion_factor(
    operation: dict,
    inputs: dict[str, dict],
    output_unit: str | None,
) -> float:
    op_name = operation["operator"]
    if op_name == "multiply":
        unit = _ureg.dimensionless
        for operand in operation["operands"]:
            unit *= _operand_unit(operand, inputs)
    elif op_name == "divide":
        numerator, denominator = operation["operands"]
        unit = _operand_unit(numerator, inputs) / _operand_unit(denominator, inputs)
    else:
        return 1.0
    return float((1 * unit).to(_unit_object(output_unit)).magnitude)


def _operand_unit(operand: Any, inputs: dict[str, dict]):
    if isinstance(operand, str):
        return _unit_object(_band_metadata(inputs, operand).get("unit"))
    if isinstance(operand, dict) and operand.get("type") == "literal":
        return _unit_object(operand.get("unit"))
    return _ureg.dimensionless


def _raise_on_non_finite_result(
    result_da: xr.DataArray, valid_mask: np.ndarray
) -> None:
    arr = np.asarray(result_da.values, dtype=float)
    invalid = valid_mask & ~np.isfinite(arr)
    if np.any(invalid):
        raise ProcessingError(
            code="COMPOSE_NON_FINITE_RESULT",
            message=(
                "Compose computation produced non-finite values, such as inf "
                "or NaN, in valid cells."
            ),
            suggestion=(
                "Check divide operations for zero denominators, or add a "
                "condition (with an else fallback) that routes those cells to "
                "the fallback instead of the computed result."
            ),
        )
