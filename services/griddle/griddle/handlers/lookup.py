"""
Lookup table handlers for Griddle.

Converts categorical fuel model codes to continuous fuel parameters
using standard lookup tables with pint for unit conversion.
"""

import csv
from pathlib import Path

import numpy as np
import pint
import xarray as xr

from griddle.storage import load_zarr
from lib.errors import ProcessingError

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

DATA_DIR = Path(__file__).parent.parent / "data"

# Maximum valid FBFM40 key (SB4 = 204)
MAX_FBFM40_KEY = 204

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
    "moisture_of_extinction": (None, None),
    "heat_content": ("BTU / lb", "kJ / kg"),
    "is_dynamic": (None, None),
}

# All quantity columns in the CSV
QUANTITY_COLUMNS = [
    "fuel_load_1hr",
    "fuel_load_10hr",
    "fuel_load_100hr",
    "fuel_load_live_herb",
    "fuel_load_live_woody",
    "savr_1hr",
    "savr_10hr",
    "savr_100hr",
    "savr_live_herb",
    "savr_live_woody",
    "fuel_depth",
    "moisture_of_extinction",
    "heat_content",
    "is_dynamic",
]

# Map from band key (dot-notation) to CSV column name
BAND_KEY_TO_COLUMN = {
    "fuel_load.1hr": "fuel_load_1hr",
    "fuel_load.10hr": "fuel_load_10hr",
    "fuel_load.100hr": "fuel_load_100hr",
    "fuel_load.live_herb": "fuel_load_live_herb",
    "fuel_load.live_woody": "fuel_load_live_woody",
    "savr.1hr": "savr_1hr",
    "savr.10hr": "savr_10hr",
    "savr.100hr": "savr_100hr",
    "savr.live_herb": "savr_live_herb",
    "savr.live_woody": "savr_live_woody",
    "fuel_depth": "fuel_depth",
    "moisture_of_extinction": "moisture_of_extinction",
    "heat_content": "heat_content",
    "is_dynamic": "is_dynamic",
}


def _get_conversion_key(column_name: str) -> str:
    """Get the unit conversion key for a given CSV column name."""
    if column_name.startswith("fuel_load"):
        return "fuel_load"
    if column_name.startswith("savr"):
        return "savr"
    return column_name


def _load_sb40_table() -> dict[str, np.ndarray]:
    """Load SB40 lookup table from CSV into numpy arrays.

    Returns a dict mapping column name to a numpy array indexed by FBFM key.
    Index 0 through MAX_FBFM40_KEY, with zeros for missing keys.
    """
    csv_path = DATA_DIR / "sb40_fbfm40.csv"

    # Initialize arrays with zeros (size MAX_FBFM40_KEY + 1 for direct indexing)
    arrays = {
        col: np.zeros(MAX_FBFM40_KEY + 1, dtype=np.float32) for col in QUANTITY_COLUMNS
    }

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row["key"])
            if key > MAX_FBFM40_KEY:
                continue
            for col in QUANTITY_COLUMNS:
                if col == "is_dynamic":
                    arrays[col][key] = (
                        1.0 if row[col].strip().lower() == "true" else 0.0
                    )
                else:
                    arrays[col][key] = float(row[col])

    return arrays


# Load table once at module level
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
                f"{sorted(invalid_codes)}"
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
        column = BAND_KEY_TO_COLUMN.get(band_key)
        if column is None:
            raise ProcessingError(
                code="UNKNOWN_BAND",
                message=f"Unknown lookup band: {band_key}",
                suggestion=f"Available bands: {list(BAND_KEY_TO_COLUMN.keys())}",
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
