"""CF (Climate and Forecast) metadata helpers for netCDF export.

CF-1.13 is the dominant netCDF convention in earth science; it specifies
how to attach CRS metadata, units, axes, and vertical-direction info so
generic CF-aware tooling (CDO, NCO, ArcGIS Pro, GeoZarr readers) can
interpret the file without out-of-band knowledge.

These helpers stamp the in-memory xarray Dataset right before
`ds.to_netcdf(...)`. The stored zarr is left alone — CF metadata is only
required at the netCDF I/O boundary; rioxarray auto-discovers CRS from
the `spatial_ref` coord on zarr reopen via `decode_coords="all"`.

## Convention: `grid_mapping` lives in `var.encoding`, never `var.attrs`

Per the CF spec and xarray docs (user-guide/weather-climate.rst):

> CF variable attributes — `coordinates`, `bounds`, `grid_mapping` — are
> parsed by xarray. The attribute values are decoded into ENCODING
> information, and the variables in those values are interpreted as
> non-dimension coordinates.

`rio.write_crs(...)` and `open_*(..., decode_coords="all")` both place
`grid_mapping` in `var.encoding`. xarray's CF encoder migrates it to the
on-disk attrs at `to_netcdf` / `to_zarr` time via
`pop_to(var.encoding, var.attrs, "grid_mapping")` — which **raises
ValueError** if both `encoding` and `attrs` carry the key. So `stamp_cf`
does **not** write `attrs["grid_mapping"]`: doing so would clash with
the encoder on any Dataset that came from a zarr round-trip.

## The `encoding=` kwarg trap

`Dataset.to_netcdf(..., encoding={k: {...}})` (and the equivalent for
`to_zarr`) **replaces** `variables[k].encoding` wholesale — see
`xarray/backends/writers.py` and the encoding setter at
`xarray/core/variable.py`. If you pass `encoding={'fbfm': {'zlib': True}}`
to write a CRS-bearing dataset, the kwarg-supplied dict has no
`grid_mapping`, the original `var.encoding["grid_mapping"]` is wiped, and
the resulting file has no CRS reference. **Do not pass the encoding
kwarg on CRS-bearing datasets** — mutate `var.encoding` in place
instead. See `services/exporter/exporter/handlers/netcdf.py` for the
compression pattern.
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
        # Do not stamp grid_mapping into attrs. Per CF + xarray conventions,
        # grid_mapping belongs in var.encoding (rio.write_crs and
        # decode_coords="all" both put it there). xarray's CF encoder at
        # to_netcdf/to_zarr time runs pop_to(encoding, attrs, "grid_mapping")
        # to materialize the attribute on disk. Stamping attrs ourselves
        # causes the encoder to raise "grid_mapping already exists in attrs".
        ds[var].attrs.setdefault("long_name", var)
        unit = unit_by_key.get(var)
        if unit:
            validate_unit(unit)
            ds[var].attrs["units"] = unit

    return ds
