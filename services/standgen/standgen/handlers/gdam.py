"""
GDAM allometry imputation handler.

Fills the missing morphology columns (dbh, crown ratio, species) of a tree
inventory by calling the external GDAM batch-prediction API.

This module is split into two layers:
- pure marshalling functions (v2 <-> GDAM units/shape), unit-tested directly;
- the `handle_gdam` orchestration (added separately) that chunks, calls GDAM,
  merges, and saves.

Unit and coordinate conventions (verified against a live GDAM probe):
- v2 parquet: x/y in the domain CRS (m), height m, dbh cm, crown_ratio fraction
  0-1, fia_species_code FIA integer.
- GDAM: Lat/Lon (EPSG:4326), HT feet, DIA inches, CR **percent 0-100** (its
  docstring says 0-1, but the live API returns 0-100), SPCD FIA integer (returned
  as a float). Response columns are name-keyed, reordered vs the request, and omit
  Lat/Lon; rows align to the request via the returned `index`.
"""

import logging

import dask.dataframe as dd
import httpx
import pandas as pd
import pint
import pyproj

from lib.errors import ProcessingError
from standgen import config
from standgen.storage import load_inventory_parquet, save_parquet

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# v2 morphology columns GDAM imputes (only these are filled / consumed back).
IMPUTED_COLUMNS = ("dbh", "crown_ratio", "fia_species_code")

# Columns GDAM needs on every source row.
_REQUIRED_COLUMNS = ("x", "y", "height")


def _reproject_to_lonlat(x, y, source_crs):
    """Reproject domain-CRS x/y arrays to (lon, lat) in EPSG:4326.

    The transformer is built here (not cached at module scope) with the CRS passed
    as a string — the pickle-safe pattern used across standgen reprojection.
    """
    transformer = pyproj.Transformer.from_crs(
        str(source_crs), "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(x, y)
    return lon, lat


def _to_split_payload(df: pd.DataFrame) -> dict:
    """Serialize a DataFrame to pandas split orientation with JSON-safe values.

    NaN becomes null and SPCD is emitted as an int (GDAM's SPCD field is an
    integer). Every other value is a float.
    """
    split = {
        "columns": list(df.columns),
        "index": [int(i) for i in df.index],
        "data": [],
    }
    for _, row in df.iterrows():
        out_row = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                out_row.append(None)
            elif col == "SPCD":
                out_row.append(int(value))
            else:
                out_row.append(float(value))
        split["data"].append(out_row)
    return split


def build_batch_payload(df: pd.DataFrame, source_crs) -> dict:
    """Build a GDAM /predict/batch request body from a v2 inventory chunk.

    Reprojects x/y to Lat/Lon and converts height to feet (both required). Any
    present morphology columns are sent as conditioning inputs (dbh cm->in,
    crown_ratio fraction->percent, fia_species_code->SPCD); absent ones are
    omitted. Environmental fields (elevation/slope/aspect) are never sent — GDAM
    auto-extracts them.
    """
    lon, lat = _reproject_to_lonlat(df["x"].to_numpy(), df["y"].to_numpy(), source_crs)
    columns = {
        "Lat": lat,
        "Lon": lon,
        "HT": Q_(df["height"].to_numpy(), "m").to("ft").magnitude,
    }
    if "dbh" in df.columns:
        columns["DIA"] = Q_(df["dbh"].to_numpy(), "cm").to("inch").magnitude
    if "crown_ratio" in df.columns:
        columns["CR"] = df["crown_ratio"].to_numpy() * 100.0  # fraction -> percent
    if "fia_species_code" in df.columns:
        columns["SPCD"] = df["fia_species_code"].to_numpy()

    payload_df = pd.DataFrame(columns, index=df.index)
    return {"trees": _to_split_payload(payload_df)}


def parse_gdam_response(response: dict) -> pd.DataFrame:
    """Convert a GDAM /predict/batch response into v2-unit imputed columns.

    Selects DIA/CR/SPCD by **name** (response column order differs from the
    request) and indexes the result by the returned `index` so callers join by
    index, not position. Converts DIA in->cm, CR percent->fraction, SPCD float->int.
    """
    predictions = response["predictions"]
    frame = pd.DataFrame(
        predictions["data"],
        columns=predictions["columns"],
        index=predictions["index"],
    )
    out = pd.DataFrame(index=frame.index)
    out["dbh"] = Q_(frame["DIA"].to_numpy(dtype=float), "inch").to("cm").magnitude
    out["crown_ratio"] = (
        frame["CR"].to_numpy(dtype=float) / 100.0
    )  # percent -> fraction
    spcd = frame["SPCD"].to_numpy(dtype=float).round()  # float -> int (nullable)
    out["fia_species_code"] = pd.array(spcd, dtype="Int64")
    return out


def fill_missing(source_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
    """Fill only the originally-missing morphology cells from GDAM predictions.

    Existing non-null values in `source_df` are preserved verbatim; cells that are
    null (or whose column is absent) are taken from `predicted_df`. Rows are aligned
    by index. x/y/height and any other source columns are untouched.
    """
    result = source_df.copy()
    for col in IMPUTED_COLUMNS:
        predicted = predicted_df[col].reindex(result.index)
        if col in result.columns:
            result[col] = result[col].where(result[col].notna(), predicted)
        else:
            result[col] = predicted
    return result


def _post_batch(payload: dict, chunk_number: int) -> dict:
    """POST one chunk to GDAM and return the parsed JSON.

    Any transport/HTTP failure is terminal (`GDAM_REQUEST_FAILED`): re-running the
    whole task won't fix a bad request, and the 120s timeout already absorbs GDAM
    cold starts, so a timeout here is a real outage rather than a transient blip.
    """
    try:
        response = httpx.post(
            f"{config.GDAM_API_URL}/predict/batch",
            json=payload,
            timeout=config.GDAM_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise ProcessingError(
            code="GDAM_REQUEST_FAILED",
            message=f"GDAM prediction request failed on chunk {chunk_number}: {e}",
            suggestion="Retry shortly; if it persists, the GDAM service may be down.",
        ) from e


def _predict_chunk(payload: dict, sent_index, chunk_number: int) -> dict:
    """Predict one chunk, retrying once if GDAM returns a partial result.

    GDAM is expected to return one prediction per sent tree. The validity check
    (every sent index present in the response) guards against a silent partial
    result; a single retry absorbs a one-off hiccup before failing the inventory.
    """
    expected = {int(i) for i in sent_index}
    returned: set = set()
    for _ in range(2):  # initial attempt + one retry
        data = _post_batch(payload, chunk_number)
        returned = {int(i) for i in data.get("predictions", {}).get("index", [])}
        if expected <= returned:
            return data
    raise ProcessingError(
        code="PARTIAL_PREDICTION",
        message=(
            f"GDAM returned an incomplete prediction for chunk {chunk_number}: "
            f"expected {len(expected)} trees, got {len(expected & returned)}."
        ),
        suggestion="Retry the inventory; if it persists, contact the GDAM team.",
    )


def handle_gdam(inventory: dict, source: dict, domain_gdf, progress) -> dict:
    """Fill a tree inventory's missing morphology via GDAM.

    Loads the source inventory parquet, sends it to GDAM in batches (converting to
    GDAM units and lat/lon), fills only the originally-missing dbh/crown_ratio/
    species, and writes a new parquet under this inventory's id.

    Returns ``{"georeference": ...}`` for main.py to persist.
    """
    inventory_id = inventory["id"]
    source_id = source["source_tree_inventory_id"]

    progress("Loading source inventory...", 10)
    try:
        ddf = load_inventory_parquet(source_id)
        df = ddf.compute().reset_index(drop=True)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="SOURCE_INVENTORY_NOT_FOUND",
            message=f"Source tree inventory '{source_id}' has no data.",
            suggestion="Ensure the source inventory completed before running GDAM.",
        ) from e

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ProcessingError(
            code="MISSING_REQUIRED_COLUMNS",
            message=f"Source inventory is missing required column(s): {missing_cols}.",
            suggestion="GDAM needs x, y, and height columns on the source inventory.",
        )

    null_height = int(df["height"].isna().sum())
    if null_height:
        raise ProcessingError(
            code="MISSING_REQUIRED_HEIGHT",
            message=(
                f"{null_height} tree(s) have no height; GDAM requires a height for "
                f"every tree."
            ),
            suggestion="Remove or fix rows with a missing height before running GDAM.",
        )

    source_crs = domain_gdf.crs
    batch_size = config.GDAM_BATCH_SIZE
    total = len(df)
    filled_chunks = []
    for start in range(0, total, batch_size):
        chunk = df.iloc[start : start + batch_size]
        chunk_number = start // batch_size + 1
        progress(
            f"Predicting trees ({min(start + batch_size, total)}/{total})...",
            30 + int(60 * start / total),
        )
        payload = build_batch_payload(chunk, source_crs)
        data = _predict_chunk(payload, chunk.index, chunk_number)
        predicted = parse_gdam_response(data)
        filled_chunks.append(fill_missing(chunk, predicted))

    result = pd.concat(filled_chunks) if filled_chunks else df
    logger.info(
        f"GDAM imputed {len(result)} trees",
        extra={"inventory_id": inventory_id},
    )

    progress("Writing inventory...", 90)
    result_ddf = dd.from_pandas(result, npartitions=max(ddf.npartitions, 1))
    save_parquet(inventory_id, result_ddf)

    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in domain_gdf.total_bounds],
    }
    progress("Complete", 100)
    return {"georeference": georeference}
