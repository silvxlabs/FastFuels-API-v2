"""
GDAM allometry imputation handler.

Fills the missing morphology columns (dbh, crown ratio, species) of a tree
inventory by calling the external GDAM batch-prediction API with its
flow-matching model, pinned to one version (``GDAM_MODEL``). The crown base
height is not a GDAM prediction: it comes from the core uncompacted crown ratio
model, run on the filled species, crown ratio, dbh and height.

This module is split into two layers:
- pure marshalling functions (v2 <-> GDAM units/shape), unit-tested directly;
- the `handle_gdam` orchestration that maps each dask partition through GDAM
  (convert -> predict -> fill-missing) and saves, so the whole inventory is never
  held in memory at once.

Unit and coordinate conventions (verified against a live GDAM probe):
- v2 parquet: x/y in the domain CRS (m), height m, dbh cm, crown_ratio fraction
  0-1, fia_species_code FIA integer.
- GDAM: Lat/Lon (EPSG:4326), HT feet, DIA inches, CR **percent 0-100** (its
  docstring says 0-1, but the live API returns 0-100), SPCD FIA integer (returned
  as a float). Response columns are name-keyed, reordered vs the request, and omit
  Lat/Lon; rows align to the request via the returned `index`.

Reproducibility: each predicted value is one random draw. The draws depend on the
model version, the seed and the trees of the request, so each partition sends a
seed derived from the inventory id and the partition number (``batch_seed``).
Reprocessing an inventory re-forms the same partitions from the same source and
so reproduces its values.
"""

import hashlib
import json
import logging
import time

import httpx
import pandas as pd
import pint
import pyproj

from lib.errors import ProcessingError
from standgen import config
from standgen.crown_base import crown_base_height
from standgen.region import in_pacific_northwest
from standgen.storage import load_inventory_parquet, save_parquet_with_summary

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# v2 morphology columns GDAM predicts.
GDAM_COLUMNS = ("dbh", "crown_ratio", "fia_species_code")
# Every imputable column, with the dtype each carries in the output. Order
# matters: columns absent from a partition are appended in this order, and the
# result `meta` must match.
IMPUTED_COLUMNS = (*GDAM_COLUMNS, "crown_base_height")
_IMPUTED_DTYPES = {
    "dbh": "float64",
    "crown_ratio": "float64",
    "fia_species_code": "Int64",
    "crown_base_height": "float64",
}
# Inputs of the crown base height model.
_CROWN_BASE_INPUTS = ("fia_species_code", "crown_ratio", "dbh", "height")

# Columns GDAM needs on every source row.
_REQUIRED_COLUMNS = ("x", "y", "height")

# The GDAM model every request names. The version is pinned so a change of GDAM's
# default can't change our output. "overhead": every tree is visible from above,
# as trees detected on a canopy height model are.
GDAM_MODEL = {
    "name": "flow_matching",
    "version": "v2_west.2",
    "tree_observation": "overhead",
}


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


def batch_seed(inventory_id: str, partition: int) -> int:
    """GDAM seed for one partition of an inventory, in [0, 2**32 - 1].

    Stable for the same inventory and partition (so a rerun reproduces the draws)
    and different across partitions (so batches don't reuse the same noise).
    """
    digest = hashlib.sha256(f"{inventory_id}/{partition}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def build_batch_payload(df: pd.DataFrame, source_crs, seed: int) -> dict:
    """Build a GDAM /predict/batch request body from a v2 inventory chunk.

    Names ``GDAM_MODEL`` with ``seed``. Reprojects x/y to Lat/Lon and converts
    height to feet (both required). Any present morphology columns are sent as
    conditioning inputs (dbh cm->in, crown_ratio fraction->percent,
    fia_species_code->SPCD); absent ones are omitted. Environmental fields (elevation/slope/aspect) are never sent — GDAM
    auto-extracts them.

    Column dtypes are trusted from the inventory: the upload path — the only way
    users supply an inventory — runs a pandera schema that coerces species to a
    nullable ``Int64``, and that dtype survives the parquet round-trip. So
    ``to_json`` serializes each column correctly on its own (ints/null for SPCD,
    floats/null elsewhere) with no per-value casting. The species column is kept
    as-is rather than ``.to_numpy()``-ed, so its ``Int64`` dtype is preserved
    through serialization instead of collapsing to a float.
    """
    lon, lat = _reproject_to_lonlat(df["x"].to_numpy(), df["y"].to_numpy(), source_crs)
    payload = pd.DataFrame({"Lat": lat, "Lon": lon}, index=df.index)
    payload["HT"] = Q_(df["height"].to_numpy(), "m").to("ft").magnitude
    if "dbh" in df.columns:
        payload["DIA"] = Q_(df["dbh"].to_numpy(), "cm").to("inch").magnitude
    if "crown_ratio" in df.columns:
        payload["CR"] = df["crown_ratio"].to_numpy() * 100.0  # fraction -> percent
    if "fia_species_code" in df.columns:
        payload["SPCD"] = df["fia_species_code"]  # keep Int64 -> serializes as int

    return {
        "trees": json.loads(payload.to_json(orient="split", double_precision=15)),
        "model": {**GDAM_MODEL, "seed": seed},
    }


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


def fill_missing(
    source_df: pd.DataFrame, predicted_df: pd.DataFrame, columns=GDAM_COLUMNS
) -> pd.DataFrame:
    """Fill only the originally-missing morphology cells from GDAM predictions.

    Only the requested `columns` are filled (default: all GDAM columns);
    columns left out are untouched, so the caller can impute a subset. Within a
    filled column, existing non-null values in `source_df` are preserved verbatim;
    cells that are null (or whose column is absent) are taken from `predicted_df`.
    Rows are aligned by index. x/y/height and any other source columns are
    untouched.
    """
    result = source_df.copy()
    for col in columns:
        predicted = predicted_df[col].reindex(result.index)
        if col in result.columns:
            result[col] = result[col].where(result[col].notna(), predicted)
        else:
            result[col] = predicted
    return result


# httpx transport errors worth retrying: the connection dropped or the server
# disconnected mid-response (e.g. the SSL UNEXPECTED_EOF seen when GDAM recycles
# instances under load). These fail fast, so a retry is cheap. Read/write and
# connect *timeouts* are deliberately excluded — they already burned the
# per-request budget, so retrying only risks the caller's Cloud Run deadline.
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)
# Statuses worth retrying: 429 (GDAM's rate limit at the load balancer) and the
# gateway errors a scaling or redeploying Cloud Run service returns. A 500 is
# terminal: GDAM returns it for failures a retry won't fix.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_RETRY_BACKOFF_BASE_S = 0.5
# Longest Retry-After honored; standgen's whole request has a deadline.
_MAX_RETRY_AFTER_S = 30.0


def _retry_delay(attempt: int, response=None) -> float:
    """Seconds to wait before the next attempt: the response's Retry-After (in
    seconds, capped) when it has one, else exponential backoff."""
    if response is not None:
        try:
            return min(float(response.headers["Retry-After"]), _MAX_RETRY_AFTER_S)
        except (KeyError, TypeError, ValueError):
            pass
    return _RETRY_BACKOFF_BASE_S * (2**attempt)


def _post_batch(payload: dict) -> dict:
    """POST one partition's trees to GDAM and return the parsed JSON.

    A transient error — a dropped/reset connection, a 429 or a 502/503/504 — is
    retried up to ``GDAM_MAX_ATTEMPTS`` times, waiting for the response's
    Retry-After or with exponential backoff, so one blip doesn't fail the whole
    inventory. Any other non-2xx status or a timeout is terminal
    (`GDAM_REQUEST_FAILED`): a bad request or a sustained-unhealthy GDAM won't be
    fixed by re-running the task, and the 120s timeout already absorbs cold starts.
    """
    last_exc = None
    for attempt in range(config.GDAM_MAX_ATTEMPTS):
        retry_response = None
        try:
            response = httpx.post(
                f"{config.GDAM_API_URL}/predict/batch",
                json=payload,
                timeout=config.GDAM_REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            return response.json()
        except _RETRYABLE_TRANSPORT_ERRORS as e:
            last_exc = e
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUSES:
                raise ProcessingError(
                    code="GDAM_REQUEST_FAILED",
                    message=f"GDAM prediction request failed: {e} {e.response.text}",
                    suggestion="Retry shortly; if it persists, the GDAM service may be down.",
                ) from e
            last_exc = e
            retry_response = e.response
        except httpx.HTTPError as e:
            raise ProcessingError(
                code="GDAM_REQUEST_FAILED",
                message=f"GDAM prediction request failed: {e}",
                suggestion="Retry shortly; if it persists, the GDAM service may be down.",
            ) from e
        if attempt < config.GDAM_MAX_ATTEMPTS - 1:
            logger.warning(
                f"Transient GDAM error on attempt "
                f"{attempt + 1}/{config.GDAM_MAX_ATTEMPTS}, retrying: {last_exc}"
            )
            time.sleep(_retry_delay(attempt, retry_response))

    raise ProcessingError(
        code="GDAM_REQUEST_FAILED",
        message=(
            f"GDAM prediction request failed after {config.GDAM_MAX_ATTEMPTS} "
            f"attempts: {last_exc}"
        ),
        suggestion="Retry shortly; if it persists, the GDAM service may be down.",
    ) from last_exc


def _predict(payload: dict, expected_count: int) -> dict:
    """Predict one partition, retrying once if GDAM returns a partial result.

    GDAM returns one prediction per sent tree, 0-based and aligned to the sent
    batch by position. The validity check (returned count == expected count)
    guards against a silent partial result; a single retry absorbs a one-off
    hiccup before failing the inventory. An exact match (not ``>=``) is required
    so a degenerate over-long response can't slip through and misalign the
    position->index mapping in ``_process_partition``.
    """
    for _ in range(2):
        data = _post_batch(payload)
        returned_count = len(data.get("predictions", {}).get("index", []))
        if returned_count == expected_count:
            return data
    raise ProcessingError(
        code="PARTIAL_PREDICTION",
        message=(
            f"GDAM returned an incomplete prediction: expected {expected_count} "
            f"trees, got {returned_count}."
        ),
        suggestion="Retry the inventory; if it persists, contact the GDAM team.",
    )


def fill_crown_base_height(df: pd.DataFrame, pacific_northwest: bool) -> pd.DataFrame:
    """Fill null (or absent) ``crown_base_height`` cells from the crown base
    height model on the frame's species, crown ratio, dbh and height. Existing
    values are kept, as in ``fill_missing``."""
    modeled = pd.Series(
        crown_base_height(
            *(df[c] for c in _CROWN_BASE_INPUTS), pacific_northwest=pacific_northwest
        ),
        index=df.index,
    )
    result = df.copy()
    if "crown_base_height" in result.columns:
        current = result["crown_base_height"]
        result["crown_base_height"] = current.where(current.notna(), modeled)
    else:
        result["crown_base_height"] = modeled
    return result


def _process_partition(
    pdf: pd.DataFrame,
    source_crs_str: str,
    pacific_northwest: bool,
    columns,
    inventory_id: str,
    partition_info=None,
) -> pd.DataFrame:
    """Impute one dask partition: call GDAM and fill the selected morphology.

    Runs on a dask worker (one partition at a time), so the whole inventory is
    never held in memory. When any GDAM column is selected, raises
    ``MISSING_REQUIRED_HEIGHT`` if any tree in the partition lacks a height and
    ``NON_POSITIVE_HEIGHT`` if any height is not above zero (GDAM requires a
    positive height). A selected ``crown_base_height`` is then filled from the
    crown base height model on the filled columns; with no GDAM column selected
    no GDAM request is made, and a tree without a positive height gets no crown
    base. Always surfaces the selected `columns` so the partition's schema
    matches the declared ``meta``.

    Dask passes ``partition_info``; its partition number seeds the draws
    (``batch_seed``).

    GDAM returns predictions 0-based and aligned to the sent batch by position
    (it ignores the index it was sent). Those positions are mapped back onto
    this partition's original index before ``fill_missing`` so the join is by
    index — correct even when the partition's index isn't 0-based (every
    partition after the first) and even if GDAM returns rows in a different
    order than they were sent.
    """
    gdam_columns = [c for c in columns if c in GDAM_COLUMNS]
    if len(pdf) == 0:
        # No trees to predict; still surface the selected columns (empty) so the
        # partition's schema matches `meta`.
        out = pdf.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = pd.Series([], dtype=_IMPUTED_DTYPES[col], index=out.index)
        return out

    out = pdf
    if gdam_columns:
        out = _impute_gdam(
            pdf, source_crs_str, gdam_columns, inventory_id, partition_info
        )
    if "crown_base_height" in columns:
        out = fill_crown_base_height(out, pacific_northwest)
    return out


def _impute_gdam(
    pdf: pd.DataFrame, source_crs_str: str, columns, inventory_id: str, partition_info
) -> pd.DataFrame:
    """Check heights, call GDAM for one non-empty partition and fill `columns`."""
    null_height = int(pdf["height"].isna().sum())
    if null_height:
        raise ProcessingError(
            code="MISSING_REQUIRED_HEIGHT",
            message=(
                f"{null_height} tree(s) have no height; GDAM requires a height for "
                f"every tree."
            ),
            suggestion="Remove or fix rows with a missing height before running GDAM.",
        )
    non_positive = int((pdf["height"] <= 0).sum())
    if non_positive:
        raise ProcessingError(
            code="NON_POSITIVE_HEIGHT",
            message=(
                f"{non_positive} tree(s) have a height of zero or less; GDAM requires "
                f"a positive height for every tree."
            ),
            suggestion="Remove or fix rows with a zero height before running GDAM.",
        )

    seed = batch_seed(inventory_id, partition_info["number"])
    payload = build_batch_payload(pdf, source_crs_str, seed)
    data = _predict(payload, len(pdf))
    predicted = parse_gdam_response(data)
    predicted.index = pdf.index[predicted.index.to_numpy()]
    return fill_missing(pdf, predicted, columns)


def _result_meta(ddf, columns) -> pd.DataFrame:
    """Empty DataFrame describing _process_partition's output schema for dask.

    Source columns, plus the selected imputed columns appended wherever the
    source lacks them, in ``IMPUTED_COLUMNS`` order — matching
    ``_process_partition``.
    """
    meta = ddf._meta.copy()
    for col in columns:
        if col not in meta.columns:
            meta[col] = pd.Series([], dtype=_IMPUTED_DTYPES[col])
    return meta


def handle_gdam(inventory: dict, source: dict, domain_gdf, progress) -> dict:
    """Fill a tree inventory's missing morphology via GDAM.

    Loads the source inventory parquet lazily, repartitions it so each partition
    is ~``GDAM_BATCH_SIZE`` trees, and maps every partition through GDAM
    (convert -> predict -> fill-missing) with ``map_partitions``, then fills a
    selected crown base height from the crown base height model. The lazy graph
    executes once, at ``save_parquet_with_summary`` — so the whole inventory is
    never held in memory and each GDAM request covers one partition.

    Dict with 'georeference', 'columns' with per-column summary statistics,
    and 'forestry_metrics' with stand-level forestry scalars or None.
    """
    inventory_id = inventory["id"]
    source_id = source["source_tree_inventory_id"]

    progress("Loading source inventory...", 10)
    try:
        ddf = load_inventory_parquet(source_id)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="SOURCE_INVENTORY_NOT_FOUND",
            message=f"Source tree inventory '{source_id}' has no data.",
            suggestion="Ensure the source inventory completed before running GDAM.",
        ) from e

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in ddf.columns]
    if missing_cols:
        raise ProcessingError(
            code="MISSING_REQUIRED_COLUMNS",
            message=f"Source inventory is missing required column(s): {missing_cols}.",
            suggestion="GDAM needs x, y, and height columns on the source inventory.",
        )

    # Which morphology columns to impute (default: all). Normalize to canonical
    # order so the partition output and `meta` agree on column ordering.
    requested = source.get("impute_columns") or list(IMPUTED_COLUMNS)
    impute_columns = [c for c in IMPUTED_COLUMNS if c in requested]

    if "crown_base_height" in impute_columns:
        missing_inputs = [
            c
            for c in _CROWN_BASE_INPUTS
            if c not in ddf.columns and c not in impute_columns
        ]
        if missing_inputs:
            raise ProcessingError(
                code="MISSING_CROWN_BASE_INPUTS",
                message=(
                    f"Crown base height needs column(s) {missing_inputs}, which the "
                    f"source inventory lacks and are not imputed."
                ),
                suggestion="Also impute the missing column(s).",
            )

    # Size partitions to the GDAM batch target. Row counts come from the parquet
    # footer (cheap) — not by loading the data.
    total = int(len(ddf))
    batch_size = config.GDAM_BATCH_SIZE
    npartitions = max(1, (total + batch_size - 1) // batch_size)
    ddf = ddf.repartition(npartitions=npartitions)

    progress("Imputing missing morphology via GDAM...", 40)
    # Pass the CRS as a string so the pyproj transformer is rebuilt inside each
    # partition (pickle-safe — see standgen reprojection convention).
    source_crs_str = str(domain_gdf.crs)
    pacific_northwest = in_pacific_northwest(domain_gdf)
    result_ddf = ddf.map_partitions(
        _process_partition,
        source_crs_str,
        pacific_northwest,
        impute_columns,
        inventory_id,
        meta=_result_meta(ddf, impute_columns),
    )

    # save_parquet triggers the single execution of the lazy graph above (this is
    # where the GDAM calls actually run).
    progress("Writing inventory...", 90)
    _, stats, forestry_metrics = save_parquet_with_summary(
        inventory_id, result_ddf, inventory["columns"], inventory["type"], domain_gdf
    )
    logger.info(
        f"Imputed {impute_columns} for {total} trees in {npartitions} batch(es) with "
        f"{GDAM_MODEL['name']}/{GDAM_MODEL['version']}, seeds from "
        f"batch_seed(inventory_id, partition)",
        extra={"inventory_id": inventory_id},
    )

    georeference = {
        "crs": str(domain_gdf.crs),
        "bounds": [float(b) for b in domain_gdf.total_bounds],
    }
    progress("Complete", 100)
    return {
        "georeference": georeference,
        "columns": [
            {**col, "summary": stats.get(col["key"])} for col in inventory["columns"]
        ],
        "forestry_metrics": forestry_metrics,
    }
