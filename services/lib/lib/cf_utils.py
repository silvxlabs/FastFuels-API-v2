"""CF (Climate and Forecast) metadata helpers for netCDF export.

CF-1.13 is the dominant netCDF convention in earth science; it specifies
how to attach CRS metadata, units, axes, and vertical-direction info so
generic CF-aware tooling (CDO, NCO, ArcGIS Pro, GeoZarr readers) can
interpret the file without out-of-band knowledge.

These helpers stamp the in-memory xarray Dataset right before
`ds.to_netcdf(...)`. The stored zarr is left alone — CF metadata is only
required at the netCDF I/O boundary; rioxarray auto-discovers CRS from
the `spatial_ref` coord on zarr reopen via `decode_coords="all"`.
"""

from __future__ import annotations

import xarray as xr

from lib.units import validate_unit

CF_CONVENTIONS = "CF-1.13"


def stamp_cf(
    ds: xr.Dataset,
    *,
    bands: list[dict],
    vertical: bool,
) -> xr.Dataset:
    """Add CF-1.13 metadata to an in-memory Dataset before netCDF write.

    Args:
        ds: Dataset with `spatial_ref` coord already set via
            `ds.rio.write_crs(...)`. Mutated in place and returned for
            chaining.
        bands: The Grid document's bands list — each entry has at least
            `{"key": str, "unit": str | None}`. Only entries whose `key`
            matches a data variable in `ds` are applied; extras are
            ignored.
        vertical: True if the Dataset has a `z` dimension. Controls
            whether the z-axis CF attrs are stamped.

    Returns the same Dataset for chaining.
    """
    ds.attrs["Conventions"] = CF_CONVENTIONS

    if "x" in ds.coords:
        ds["x"].attrs.update(
            standard_name="projection_x_coordinate", units="m", axis="X"
        )
    if "y" in ds.coords:
        ds["y"].attrs.update(
            standard_name="projection_y_coordinate", units="m", axis="Y"
        )
    if vertical and "z" in ds.coords:
        ds["z"].attrs.update(axis="Z", positive="up", units="m")

    unit_by_key = {b["key"]: b.get("unit") for b in bands}
    for var in ds.data_vars:
        ds[var].attrs["grid_mapping"] = "spatial_ref"
        ds[var].attrs.setdefault("long_name", var)
        unit = unit_by_key.get(var)
        if unit:
            validate_unit(unit)
            ds[var].attrs["units"] = unit

    return ds
