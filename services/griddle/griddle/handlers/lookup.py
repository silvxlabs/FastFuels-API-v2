"""
Lookup table handlers for Griddle.

Converts categorical fuel model codes to continuous fuel parameters
using standard lookup tables with pint for unit conversion.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pint
import xarray as xr

from griddle.storage import load_zarr
from lib.config import TABLES_BUCKET
from lib.errors import ProcessingError

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

DATA_DIR = Path(__file__).parent.parent / "data"

# Maximum valid FBFM keys
MAX_FBFM13_KEY = 99
MAX_FBFM40_KEY = 204

# All valid FBFM13 codes: NB (91-99) plus the Anderson 13 fuel models (1-13)
VALID_FBFM13_KEYS = frozenset([91, 92, 93, 98, 99] + list(range(1, 14)))

# All valid FBFM40 codes (46 fuel models)
VALID_FBFM40_KEYS = frozenset(
    [
        91,
        92,
        93,
        98,
        99,  # NB
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,  # GR
        121,
        122,
        123,
        124,  # GS
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,  # SH
        161,
        162,
        163,
        164,
        165,  # TU
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,  # TL
        201,
        202,
        203,
        204,  # SB
    ]
)

# Imperial-to-metric unit mapping per quantity prefix
UNIT_CONVERSIONS = {
    "fuel_load": ("short_ton / acre", "kg / m**2"),
    "savr": ("1 / ft", "1 / m"),
    "fuel_depth": ("ft", "m"),
    "duff_depth": ("in", "m"),
}

# Quantity columns in the CSV
_DEAD_FUEL_COLUMNS = [
    "fuel_load_1hr",
    "fuel_load_10hr",
    "fuel_load_100hr",
    "savr_1hr",
    "savr_10hr",
    "savr_100hr",
    "fuel_depth",
]

FBFM40_QUANTITY_COLUMNS = _DEAD_FUEL_COLUMNS + [
    "fuel_load_live_herb",
    "fuel_load_live_woody",
    "savr_live_herb",
    "savr_live_woody",
]

FBFM13_QUANTITY_COLUMNS = _DEAD_FUEL_COLUMNS + [
    "fuel_load_live_foliage",
    "savr_live_foliage",
]

FCCS_QUANTITY_COLUMNS = [
    "fuel_load_litter",
    "fuel_load_duff",
    "duff_depth",
    "fuel_load_live_shrub",
    "fuel_load_live_herb",
    "fuel_load_1hr",
    "fuel_load_10hr",
    "fuel_load_100hr",
    "fuel_load_1000hr_sound",
    "fuel_load_1000hr_rotten",
    "fuel_load_live_foliage",
    "fuel_load_live_branch",
]


def _get_conversion_key(column_name: str) -> str:
    """Get the unit conversion key for a given CSV column name."""
    if column_name.startswith("fuel_load"):
        return "fuel_load"
    if column_name.startswith("savr"):
        return "savr"
    return column_name


def _band_key_to_column(columns: list[str]) -> dict[str, str]:
    """Derive dot-notation band keys from CSV column names.

    'fuel_load_1hr' -> 'fuel_load.1hr', 'savr_live_foliage' -> 'savr.live_foliage',
    'fuel_depth' -> 'fuel_depth' (no prefix, key equals column name).
    """
    keys = {}
    for col in columns:
        conv_key = _get_conversion_key(col)
        if conv_key == col:
            band_key = col
        else:
            band_key = f"{conv_key}.{col[len(conv_key) + 1 :]}"
        keys[band_key] = col
    return keys


# Map from band key (dot-notation) to CSV column name
FBFM40_BAND_KEY_TO_COLUMN = _band_key_to_column(FBFM40_QUANTITY_COLUMNS)
FBFM13_BAND_KEY_TO_COLUMN = _band_key_to_column(FBFM13_QUANTITY_COLUMNS)
FCCS_BAND_KEY_TO_COLUMN = _band_key_to_column(FCCS_QUANTITY_COLUMNS)


def _load_fbfm13_table() -> dict[str, np.ndarray]:
    """Load Anderson 13 lookup table from CSV into numpy arrays.

    Returns a dict mapping column name to a numpy array indexed by FBFM13
    key, indices 0 through MAX_FBFM13_KEY, with zeros for missing keys.
    """
    csv_path = DATA_DIR / "fbfm13_lookup.csv"

    arrays = {
        col: np.zeros(MAX_FBFM13_KEY + 1, dtype=np.float32)
        for col in FBFM13_QUANTITY_COLUMNS
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row["key"])
            if key > MAX_FBFM13_KEY:
                continue
            for col in FBFM13_QUANTITY_COLUMNS:
                arrays[col][key] = float(row[col])

    return arrays


def _load_sb40_table() -> dict[str, np.ndarray]:
    """Load SB40 lookup table from CSV into numpy arrays.

    Returns a dict mapping column name to a numpy array indexed by FBFM key.
    Index 0 through MAX_FBFM40_KEY, with zeros for missing keys.
    """
    csv_path = DATA_DIR / "sb40_fbfm40.csv"

    # Initialize arrays with zeros (size MAX_FBFM40_KEY + 1 for direct indexing)
    arrays = {
        col: np.zeros(MAX_FBFM40_KEY + 1, dtype=np.float32)
        for col in FBFM40_QUANTITY_COLUMNS
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row["key"])
            if key > MAX_FBFM40_KEY:
                continue
            for col in FBFM40_QUANTITY_COLUMNS:
                arrays[col][key] = float(row[col])

    return arrays


def _load_fccs_table() -> dict:
    """Load the FCCS fuelbed lookup table from gs://{TABLES_BUCKET}.

    Unlike the FBFM13/FBFM40 tables, FCCS codes are sparse across a huge
    range (observed up to ~54.8M), so this does not build a dense array
    indexed directly by code — it returns arrays sorted by code, meant to
    be paired with np.searchsorted for vectorized lookup.

    Called fresh on every fccs_lookup() call, matching fetch_treemap's
    pattern for the TreeMap tree table in griddle/handlers/pim.py — no
    caching.

    Returns a dict with:
      - "codes": sorted int32 array of every FCCS code present in the table
      - "base_codes": frozenset of valid base FCCSID values (code // 10000,
        plus 0 for bare ground) — used only to distinguish a genuinely
        invalid code from a real code that's simply missing a table row
      - one array per FCCS_QUANTITY_COLUMNS, aligned to "codes"
    """
    table_url = f"gs://{TABLES_BUCKET}/fccs_parameter_lookup.parquet"
    df = pd.read_parquet(table_url)
    df = df.sort_values("fccs_id").reset_index(drop=True)

    codes = df["fccs_id"].to_numpy(dtype=np.int32)
    arrays = {col: df[col].to_numpy(dtype=np.float32) for col in FCCS_QUANTITY_COLUMNS}
    base_codes = frozenset(int(c) // 10_000 for c in codes)

    return {"codes": codes, "base_codes": base_codes, **arrays}


# Load tables once at module level
_FBFM13_TABLE = _load_fbfm13_table()
_SB40_TABLE = _load_sb40_table()


def _convert_to_metric(values: np.ndarray, column_name: str) -> np.ndarray:
    """Convert imperial values to metric using pint.

    Args:
        values: Array of imperial-unit values
        column_name: CSV column name (determines which conversion to apply)

    Returns:
        Array of metric-unit values
    """
    conv_key = _get_conversion_key(column_name)
    src_unit, dst_unit = UNIT_CONVERSIONS[conv_key]

    if src_unit is None:
        return values

    return Q_(values, src_unit).to(dst_unit).magnitude


def fbfm13_lookup(
    source_grid_id: str,
    bands: list[dict],
    progress,
) -> xr.Dataset:
    """Convert FBFM13 codes to fuel parameters using the Anderson 13 lookup table.

    Args:
        source_grid_id: ID of the grid containing FBFM13 codes
        bands: List of band dicts with "key" fields (dot-notation band keys)
        progress: Callback for progress reporting

    Returns:
        Dataset with one variable per band, each with dims (y, x)

    """
    progress("Loading source grid...", 20)

    try:
        source_ds = load_zarr(source_grid_id)
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Could not load source grid {source_grid_id}: {e}",
            suggestion="Ensure the source grid exists and has been processed.",
        )

    try:
        var_names = list(source_ds.data_vars)
        if not var_names:
            raise ValueError("Dataset has no data variables")
        fbfm_codes = source_ds[var_names[0]].values
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_READ_ERROR",
            message=f"Could not read FBFM codes from source grid: {e}",
            suggestion="Ensure the source grid contains valid FBFM13 data.",
        )

    if fbfm_codes.ndim == 3 and fbfm_codes.shape[0] == 1:
        fbfm_codes = fbfm_codes[0]

    nodata = source_ds[var_names[0]].rio.nodata
    nodata_mask = (
        np.zeros(fbfm_codes.shape, dtype=bool)
        if nodata is None
        else (fbfm_codes == nodata)
    )

    fbfm_codes = np.where(nodata_mask, 0, fbfm_codes).astype(np.int32)

    unique_codes = set(np.unique(fbfm_codes[~nodata_mask]))
    invalid_codes = unique_codes - VALID_FBFM13_KEYS
    if invalid_codes:
        raise ProcessingError(
            code="INVALID_FBFM_CODES",
            message=(
                f"Source grid contains {len(invalid_codes)} invalid FBFM13 code(s): "
                f"{sorted(int(c) for c in invalid_codes)}"
            ),
            suggestion=(
                "Valid FBFM13 codes are 91-99 (NB) and 1-13 (Anderson 13 "
                "fuel models). Ensure the source grid contains only valid "
                "FBFM13 fuel model codes."
            ),
        )

    progress("Looking up fuel parameters...", 40)

    band_keys = [b["key"] for b in bands]
    result_bands = []

    for band_key in band_keys:
        column = FBFM13_BAND_KEY_TO_COLUMN.get(band_key)
        if column is None:
            raise ProcessingError(
                code="UNKNOWN_BAND",
                message=f"Unknown lookup band: {band_key}",
                suggestion=f"Available bands: {list(FBFM13_BAND_KEY_TO_COLUMN.keys())}",
            )

        imperial_vals = _FBFM13_TABLE[column][fbfm_codes]
        metric_vals = _convert_to_metric(imperial_vals, column).astype(np.float32)
        metric_vals[nodata_mask] = np.nan

        result_bands.append(metric_vals)

    progress("Building output dataset...", 70)

    source_var = source_ds[var_names[0]]
    y_coords = source_var.coords["y"].values
    x_coords = source_var.coords["x"].values

    variables = {}
    for band_key, band_data in zip(band_keys, result_bands):
        da = xr.DataArray(
            data=band_data,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        variables[band_key] = da.rio.write_nodata(np.nan)

    result = xr.Dataset(variables)

    if hasattr(source_var, "rio") and source_var.rio.crs is not None:
        result = result.rio.write_crs(source_var.rio.crs)
        transform = source_var.rio.transform()
        if transform is not None:
            result = result.rio.write_transform(transform)

    progress("Lookup complete.", 80)

    return result


def fbfm40_lookup(
    source_grid_id: str,
    bands: list[dict],
    progress,
) -> xr.Dataset:
    """Convert FBFM40 codes to fuel parameters using SB40 lookup tables.

    Args:
        source_grid_id: ID of the grid containing FBFM40 codes
        bands: List of band dicts with "key" fields (dot-notation band keys)
        progress: Callback for progress reporting

    Returns:
        Dataset with one variable per band, each with dims (y, x)
    """
    progress("Loading source grid...", 20)

    try:
        source_ds = load_zarr(source_grid_id)
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Could not load source grid {source_grid_id}: {e}",
            suggestion="Ensure the source grid exists and has been processed.",
        )

    # Extract the FBFM code array from the dataset
    try:
        # load_zarr returns a Dataset; get the single data variable
        var_names = list(source_ds.data_vars)
        if not var_names:
            raise ValueError("Dataset has no data variables")
        fbfm_codes = source_ds[var_names[0]].values
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_READ_ERROR",
            message=f"Could not read FBFM codes from source grid: {e}",
            suggestion="Ensure the source grid contains valid FBFM40 data.",
        )

    # Handle multi-dimensional source: squeeze out band dim if present
    if fbfm_codes.ndim == 3 and fbfm_codes.shape[0] == 1:
        fbfm_codes = fbfm_codes[0]

    # Cells with no fuel model (the source nodata sentinel) are not looked up;
    # they pass through as nodata (NaN) in every output band. Grids load raw
    # (mask_and_scale=False), so nodata appears as the integer sentinel here.
    nodata = source_ds[var_names[0]].rio.nodata
    nodata_mask = (
        np.zeros(fbfm_codes.shape, dtype=bool)
        if nodata is None
        else (fbfm_codes == nodata)
    )

    # Replace nodata cells with 0 (an in-range index) so they neither trip
    # validation nor overflow the lookup table; their output is masked to NaN.
    fbfm_codes = np.where(nodata_mask, 0, fbfm_codes).astype(np.int32)

    # Validate the actual fuel-model codes (nodata cells excluded).
    unique_codes = set(np.unique(fbfm_codes[~nodata_mask]))
    invalid_codes = unique_codes - VALID_FBFM40_KEYS
    if invalid_codes:
        raise ProcessingError(
            code="INVALID_FBFM_CODES",
            message=(
                f"Source grid contains {len(invalid_codes)} invalid FBFM40 code(s): "
                f"{sorted(int(c) for c in invalid_codes)}"
            ),
            suggestion=(
                "Valid FBFM40 codes are 91-99 (NB), 101-109 (GR), 121-124 (GS), "
                "141-149 (SH), 161-165 (TU), 181-189 (TL), 201-204 (SB). "
                "Ensure the source grid contains only valid FBFM40 fuel model codes."
            ),
        )

    progress("Looking up fuel parameters...", 40)

    band_keys = [b["key"] for b in bands]
    result_bands = []

    for band_key in band_keys:
        column = FBFM40_BAND_KEY_TO_COLUMN.get(band_key)
        if column is None:
            raise ProcessingError(
                code="UNKNOWN_BAND",
                message=f"Unknown lookup band: {band_key}",
                suggestion=f"Available bands: {list(FBFM40_BAND_KEY_TO_COLUMN.keys())}",
            )

        # Vectorized lookup: imperial values
        imperial_vals = _SB40_TABLE[column][fbfm_codes]

        # Convert to metric, then mask no-fuel-model cells back out to NaN.
        metric_vals = _convert_to_metric(imperial_vals, column).astype(np.float32)
        metric_vals[nodata_mask] = np.nan

        result_bands.append(metric_vals)

    progress("Building output dataset...", 70)

    # Get spatial coordinates from source
    source_var = source_ds[var_names[0]]
    y_coords = source_var.coords["y"].values
    x_coords = source_var.coords["x"].values

    # Build Dataset with each band as a named variable
    variables = {}
    for band_key, band_data in zip(band_keys, result_bands):
        da = xr.DataArray(
            data=band_data,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        variables[band_key] = da.rio.write_nodata(np.nan)

    result = xr.Dataset(variables)

    # Copy spatial metadata from source
    if hasattr(source_var, "rio") and source_var.rio.crs is not None:
        result = result.rio.write_crs(source_var.rio.crs)
        transform = source_var.rio.transform()
        if transform is not None:
            result = result.rio.write_transform(transform)

    progress("Lookup complete.", 80)

    return result


def fccs_lookup(
    source_grid_id: str,
    bands: list[dict],
    progress,
) -> xr.Dataset:
    """Convert FCCS codes to fuel parameters using the FOFEM lookup table.

    Args:
        source_grid_id: ID of the grid containing FCCS codes
        bands: List of band dicts with "key" fields (dot-notation band keys)
        progress: Callback for progress reporting

    Returns:
        Dataset with one variable per band, each with dims (y, x)
    """
    progress("Loading source grid...", 20)

    try:
        source_ds = load_zarr(source_grid_id)
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_NOT_FOUND",
            message=f"Could not load source grid {source_grid_id}: {e}",
            suggestion="Ensure the source grid exists and has been processed.",
        )

    try:
        var_names = list(source_ds.data_vars)
        if not var_names:
            raise ValueError("Dataset has no data variables")
        fccs_codes = source_ds[var_names[0]].values
    except Exception as e:
        raise ProcessingError(
            code="SOURCE_GRID_READ_ERROR",
            message=f"Could not read FCCS codes from source grid: {e}",
            suggestion="Ensure the source grid contains valid FCCS data.",
        )

    if fccs_codes.ndim == 3 and fccs_codes.shape[0] == 1:
        fccs_codes = fccs_codes[0]

    nodata = source_ds[var_names[0]].rio.nodata
    fccs_codes = fccs_codes.astype(np.int64)  # headroom during arithmetic below
    nodata_mask = (
        np.zeros(fccs_codes.shape, dtype=bool)
        if nodata is None
        else (fccs_codes == nodata)
    )

    # Replace nodata cells with 0 (bare ground, an in-range code) so they
    # don't trip validation; their output is masked to NaN below regardless.
    fccs_codes = np.where(nodata_mask, 0, fccs_codes)

    progress("Loading FCCS parameter table...", 30)
    table = _load_fccs_table()
    sorted_codes = table["codes"]

    idx = np.searchsorted(sorted_codes, fccs_codes)
    idx_clipped = np.clip(idx, 0, len(sorted_codes) - 1)
    in_table_mask = (sorted_codes[idx_clipped] == fccs_codes) & ~nodata_mask

    # Split codes not found in the table into "not a real FCCS code" vs
    # "real code, just missing a row in this parameter table."
    unmatched = ~in_table_mask & ~nodata_mask
    if unmatched.any():
        unmatched_codes = set(np.unique(fccs_codes[unmatched]))
        invalid_codes = {
            code
            for code in unmatched_codes
            if (code // 10_000) not in table["base_codes"]
        }
        missing_codes = unmatched_codes - invalid_codes

        if invalid_codes:
            raise ProcessingError(
                code="INVALID_FCCS_CODES",
                message=(
                    f"Source grid contains {len(invalid_codes)} invalid FCCS "
                    f"code(s): {sorted(int(c) for c in invalid_codes)}"
                ),
                suggestion=(
                    "These codes don't correspond to any known FCCS fuelbed. "
                    "Ensure the source grid contains only valid FCCS codes."
                ),
            )

        if missing_codes:
            progress(
                f"{len(missing_codes)} valid FCCS code(s) have no matching "
                f"row in the FOFEM lookup table and will be output as NaN: "
                f"{sorted(int(c) for c in missing_codes)}",
                35,
            )

    progress("Looking up fuel parameters...", 40)

    band_keys = [b["key"] for b in bands]
    result_bands = []

    for band_key in band_keys:
        column = FCCS_BAND_KEY_TO_COLUMN.get(band_key)
        if column is None:
            raise ProcessingError(
                code="UNKNOWN_BAND",
                message=f"Unknown lookup band: {band_key}",
                suggestion=f"Available bands: {list(FCCS_BAND_KEY_TO_COLUMN.keys())}",
            )

        imperial_vals = np.where(
            in_table_mask,
            table[column][idx_clipped],
            np.nan,
        ).astype(np.float32)

        metric_vals = _convert_to_metric(imperial_vals, column).astype(np.float32)

        result_bands.append(metric_vals)

    progress("Building output dataset...", 70)

    source_var = source_ds[var_names[0]]
    y_coords = source_var.coords["y"].values
    x_coords = source_var.coords["x"].values

    variables = {}
    for band_key, band_data in zip(band_keys, result_bands):
        da = xr.DataArray(
            data=band_data,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        variables[band_key] = da.rio.write_nodata(np.nan)

    result = xr.Dataset(variables)

    if hasattr(source_var, "rio") and source_var.rio.crs is not None:
        result = result.rio.write_crs(source_var.rio.crs)
        transform = source_var.rio.transform()
        if transform is not None:
            result = result.rio.write_transform(transform)

    progress("Lookup complete.", 80)

    return result
