"""
Shared utilities for Griddle handlers.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from affine import Affine
from xarray import DataArray

_IDENTITY_TRANSFORM = Affine(1, 0, 0, 0, 1, 0)


def infer_nodata(dtype: np.dtype, da: DataArray | None = None) -> int | float:
    """Return a nodata sentinel for a given numpy dtype.

    If a DataArray is provided and already has nodata declared, that value
    is returned as-is. Otherwise the sentinel is inferred from the dtype.

    Convention:
        float dtypes   → dtype-preserving NaN (float32 nan, not float64 nan)
        integer dtypes → dtype max (iinfo.max), covers signed and unsigned

    Args:
        dtype: numpy dtype to infer a sentinel for.
        da: Optional DataArray. If provided and da.rio.nodata is not None,
            that value is returned directly without inspecting dtype.

    Returns:
        Sentinel value matching the dtype.

    Raises:
        ValueError: If the dtype is neither floating nor integer.

    Examples:
        # Preserve existing nodata from source raster:
        da = da.rio.write_nodata(infer_nodata(da.dtype, da))

        # Infer a fill value for a constructed array:
        nodata_val = infer_nodata(da.dtype)
        values[invalid_mask] = nodata_val
    """
    if da is not None and da.rio.nodata is not None:
        # Cast to the array's dtype so the sentinel matches the data type
        # e.g. float64 nan on a float32 array becomes float32 nan
        da_dtype = np.dtype(da.dtype)
        return da_dtype.type(da.rio.nodata)

    da_dtype = np.dtype(dtype)
    if np.issubdtype(da_dtype, np.floating):
        # dtype.type("nan") produces a dtype-preserving NaN.
        # np.nan is float64; writing it onto a float32 array causes a silent
        # dtype mismatch in rioxarray's nodata encoding.
        return da_dtype.type("nan")
    if np.issubdtype(da_dtype, np.integer):
        # Covers signed (int16→32767, int32→2147483647) and
        # unsigned (uint32→4294967295, int64→9223372036854775807).
        return np.iinfo(da_dtype).max
    raise ValueError(
        f"Cannot infer nodata sentinel for dtype {da_dtype!r}. "
        "Only floating-point and integer dtypes are supported."
    )


def to_dataset(
    variables: dict[str, DataArray],
    crs=None,
    transform=None,
) -> xr.Dataset:
    """Build an xr.Dataset from named DataArrays, writing CRS, transform, and nodata.

    Shared construction point for all handlers — ensures CRS, transform, and
    nodata are always declared consistently regardless of how the source data
    was fetched or constructed.

    Every DataArray must have da.rio.nodata set before calling. Nodata is
    stored in da.attrs["_FillValue"] and read back correctly when the Zarr
    store is opened with mask_and_scale=False.

    Args:
        variables: Mapping of variable name to DataArray. All arrays must
            share the same CRS and transform. Every DataArray must have
            da.rio.nodata set before calling.
        crs: Optional CRS to write onto the Dataset. If omitted, the CRS is
            read from the first DataArray.
        transform: Optional transform to write onto the Dataset. If omitted,
            the transform is read from the first DataArray.

    Returns:
        Dataset with CRS, transform, and nodata written on every variable.

    Raises:
        ValueError: If variables is empty, any DataArray is missing nodata,
            or CRS or transform cannot be resolved.
    """
    if not variables:
        raise ValueError("variables must not be empty")

    missing_nodata = [k for k, da in variables.items() if da.rio.nodata is None]
    if missing_nodata:
        raise ValueError(
            f"Variable(s) {missing_nodata} have no nodata set. "
            "Call da.rio.write_nodata(infer_nodata(da.dtype)) before building "
            "the variables dict."
        )

    first = next(iter(variables.values()))
    resolved_crs = crs if crs is not None else first.rio.crs
    resolved_transform = transform if transform is not None else first.rio.transform()

    if resolved_crs is None:
        raise ValueError(
            "No CRS available. Either set it on the DataArrays or pass crs explicitly."
        )
    if resolved_transform == _IDENTITY_TRANSFORM:
        raise ValueError(
            "No transform available. Either set it on the DataArrays or pass transform explicitly."
        )

    ds = xr.Dataset(variables)
    ds = ds.rio.write_crs(resolved_crs)
    ds = ds.rio.write_transform(resolved_transform)

    return ds
