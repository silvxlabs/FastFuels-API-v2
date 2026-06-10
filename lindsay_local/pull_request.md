# Standardize nodata conventions across all handlers

## Problem

- Nodata conventions were inconsistent across handlers — some used NaN, some integer sentinels, some declared nothing
- `_to_dataset` existed in `landfire.py` and `threedep.py` but was duplicated and didn't handle nodata

## Changes

### `griddle/utils.py` (new)
- `infer_nodata(dtype, da=None)` — returns a dtype-appropriate sentinel; preserves existing nodata if already set on the DataArray
- `to_dataset(variables, crs=None, transform=None)` — single construction point for all handler output Datasets; reads CRS and transform from the first DataArray unless passed explicitly; raises if nodata, CRS, or transform cannot be resolved
### Handlers
- All handlers now call `da.rio.write_nodata(infer_nodata(da.dtype, da))`
- All handlers return `to_dataset(variables)` instead of building Datasets inline
- Private `_to_dataset` helpers in `landfire.py` and `threedep.py` deleted
- `_scale_canopy_band` — drops `encoded=True` which caused a conflict between attrs and encoding during `to_zarr`
- `pim.plt_cn` — uses `0` as fill value explicitly since FIA CNs are large positive integers and this was used before

### Zarr round-trip (`lib/zarr_utils.py`)
- `load_zarr` now uses `mask_and_scale=False` — xarray's CF decoding was consuming `_FillValue` on load and conflicting with GDAL-written identity `scale_factor`/`add_offset` attrs, causing `rio.nodata` to return `None` after round-trip
- Nodata is stored in `da.attrs["_FillValue"]` and read back correctly without being consumed
- API tests pass — worth double checking that nothing else relies on xarray doing the masking/scaling on load
- `griddle_runner` fixture updated to use `load_zarr` instead of `xr.open_zarr` directly

### `GRIDDLE.md`
- Section 2 updated to show the new `to_dataset` usage pattern
- Section 3 (new) documents the nodata conventions


### Tests
- `test_utils.py` (new) — unit tests for `infer_nodata` and `to_dataset`
- `test_nodata_declared` added to every handler test class
- `rio.nodata is not None` assertion added to every integration test
- `test_plt_cn_nodata_declared` asserts the specific value `0`
- `_assert_valid_data` in `test_threedep.py` simplified — now asserts nodata is not None rather than falling back silently
- `test_pim.py` integration — dtype assertion corrected to unsigned integer types only since `mask_and_scale=False` preserves raw dtype on load
- Mock helpers updated to write nodata matching their source dtype