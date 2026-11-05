"""
Inventory upload handler for the Uploader service.

Processes uploaded inventory files (CSV, GeoJSON, GeoPackage), validates
against the inventory schema, and writes Parquet to INVENTORIES_BUCKET.

The upload is streamed from GCS in batches, and the trees kept in the domain
are sorted once, by 1 km tile along a Hilbert curve, so each Parquet row group
covers a compact area.
"""

import math
from contextlib import suppress
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pandera.errors import SchemaErrors
from pandera.pandas import DataFrameModel, Field, dataframe_check
from pandera.typing import Series

from lib.config import DOMAINS_COLLECTION, INVENTORIES_BUCKET, INVENTORIES_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.errors import CancelledException, ProcessingError
from lib.firestore import get_document
from lib.gcs import delete_file, storage_size
from uploader.handlers.inventory_failures import FailureCases
from uploader.handlers.inventory_reader import check_exists, csv_batches, geo_batches
from uploader.main import update_resource

_V2_COLUMNS = {
    "tree_id",
    "x",
    "y",
    "height",
    "fia_species_code",
    "fia_status_code",
    "fia_crown_class_code",
    "dbh",
    "crown_ratio",
    "crown_base_height",
    "crown_radius",
}

# Doc metadata (type, unit) per v2 column, in canonical order. Only x, y, and
# height are required in an upload, so the inventory document's `columns` field
# is written from the columns the file actually provided (the Parquet pads the
# optional ones with nulls) — the API's treatments endpoint relies on it to
# tell whether an inventory has a `dbh` column to thin against.
_COLUMN_METADATA = {
    "tree_id": ("categorical", None),
    "x": ("continuous", "m"),
    "y": ("continuous", "m"),
    "fia_species_code": ("categorical", None),
    "fia_status_code": ("categorical", None),
    "fia_crown_class_code": ("categorical", None),
    "dbh": ("continuous", "cm"),
    "height": ("continuous", "m"),
    "crown_ratio": ("continuous", None),
    "crown_base_height": ("continuous", "m"),
    "crown_radius": ("continuous", "m"),
}

_OPTIONAL_COLUMNS = [
    c for c in _COLUMN_METADATA if c not in ("tree_id", "x", "y", "height")
]


# tree_id range: 1 … the int32 maximum (the voxel `tree_id` band uses 0 as nodata).
MAX_TREE_ID = 2_147_483_647
MAX_CROWN_RADIUS_M = 20.0

# The API serves one Parquet row group per inventory data partition (#763).
ROW_GROUP_SIZE = 50_000

# Rows are written in Hilbert order of square tiles this wide.
TILE_SIZE_M = 1000.0


class _InventorySchema(DataFrameModel):
    # int64 so out-of-range IDs reach the `le` check before the int32 cast.
    tree_id: Series[np.int64] | None = Field(ge=1, le=MAX_TREE_ID, unique=True)
    x: Series[float]
    y: Series[float]
    height: Series[float] = Field(ge=0, le=116)
    fia_species_code: Series[int] | None = Field(nullable=True)
    fia_status_code: Series[int] | None = Field(isin=[0, 1, 2, 3], nullable=True)
    fia_crown_class_code: Series[int] | None = Field(
        isin=[1, 2, 3, 4, 5], nullable=True
    )
    dbh: Series[float] | None = Field(ge=0, nullable=True)
    crown_ratio: Series[float] | None = Field(ge=0, le=1, nullable=True)
    crown_base_height: Series[float] | None = Field(ge=0, nullable=True)
    crown_radius: Series[float] | None = Field(
        gt=0, le=MAX_CROWN_RADIUS_M, nullable=True
    )

    @dataframe_check(error="crown_base_height must not exceed height")
    def crown_base_height_at_most_height(cls, df: pd.DataFrame) -> pd.Series:
        if "crown_base_height" not in df.columns:
            return pd.Series(True, index=df.index)
        cbh = df["crown_base_height"]
        return cbh.isna() | (cbh <= df["height"])

    class Config:
        coerce = True


# Batches are validated one at a time; tree_id uniqueness is checked across
# the whole file instead.
_BATCH_SCHEMA = _InventorySchema.to_schema().update_column("tree_id", unique=False)

# Persisted columns in canonical order. The pandas metadata makes readers
# restore the Int64 (not float64) codes.
_PARQUET_SCHEMA = pa.Schema.from_pandas(
    pd.DataFrame(
        {
            "tree_id": pd.Series(dtype="int32"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
            "fia_species_code": pd.Series(dtype="Int64"),
            "fia_status_code": pd.Series(dtype="Int64"),
            "fia_crown_class_code": pd.Series(dtype="Int64"),
            "dbh": pd.Series(dtype="float64"),
            "height": pd.Series(dtype="float64"),
            "crown_ratio": pd.Series(dtype="float64"),
            "crown_base_height": pd.Series(dtype="float64"),
            "crown_radius": pd.Series(dtype="float64"),
        }
    ),
    preserve_index=False,
)


def handle_inventory(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    source = doc["source"]
    fmt = source["format"]
    col_map = source.get("columns", {})
    domain_id = doc["domain_id"]

    try:
        _, domain_snap = get_document(DOMAINS_COLLECTION, domain_id)
        domain_data = domain_snap.to_dict()
        domain_gdf = parse_domain_gdf(domain_data)
        domain_crs_str = _extract_crs_string(domain_data)
        bounds = tuple(float(v) for v in domain_gdf.total_bounds)

        table = _read_table(
            fmt, f"gs://{bucket}/{object_name}", col_map, domain_crs_str, bounds
        )
        if table.num_rows == 0:
            xmin, ymin, xmax, ymax = bounds
            raise ProcessingError(
                code="EMPTY_AFTER_FILTER",
                message="No trees remain after filtering to domain bounds.",
                suggestion=(
                    f"Domain extent in {domain_crs_str}: "
                    f"x=[{xmin:.1f}, {xmax:.1f}], y=[{ymin:.1f}, {ymax:.1f}]. "
                    "For CSV files, verify coordinates are in the domain CRS. "
                    "For GeoJSON/GeoPackage, verify features overlap the domain's geographic extent."
                ),
            )

        path = f"gs://{INVENTORIES_BUCKET}/{resource_id}"
        _write_parquet(table, bounds, path)

        x, y = table.column("x"), table.column("y")
        georeference = {
            "crs": domain_crs_str,
            "bounds": [
                pc.min(x).as_py(),
                pc.min(y).as_py(),
                pc.max(x).as_py(),
                pc.max(y).as_py(),
            ],
        }
        # Record the columns the file actually provided (the table holds only
        # those, plus tree_id), each with a summary over the persisted
        # (domain-filtered) data. The create endpoint wrote a provisional full
        # column list, and the Parquet pads the missing optional columns with
        # nulls for schema compatibility — but an all-null dbh is not a column
        # treatments can thin against.
        columns = [
            {
                "key": key,
                "type": col_type,
                "unit": unit,
                "summary": _column_summary(table.column(key), key, col_type),
            }
            for key, (col_type, unit) in _COLUMN_METADATA.items()
            if key in table.column_names
        ]
        update_resource(
            INVENTORIES_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": georeference,
                "columns": columns,
                "size_bytes": storage_size(path),
                "progress": {"message": "Complete", "percent": 100},
            },
        )

    except (ProcessingError, CancelledException, FileNotFoundError):
        # Terminal in main.py. Any other error is retried by Eventarc and the
        # retry needs the upload, so it is kept.
        with suppress(Exception):
            delete_file(f"gs://{bucket}/{object_name}")
        raise
    else:
        with suppress(Exception):
            delete_file(f"gs://{bucket}/{object_name}")


def _read_table(
    fmt: str, src: str, col_map: dict, domain_crs_str: str, bounds: tuple
) -> pa.Table:
    """Stream the upload into a validated, domain-filtered Arrow table.

    col_map maps v2 column names → user column names in the file.

    Returns:
        The trees inside ``bounds``, one chunk per batch: ``tree_id`` plus
        the v2 columns the file provided, in canonical order.

    Raises:
        FileNotFoundError: If the upload no longer exists.
        ProcessingError: ``SCHEMA_VALIDATION_ERROR``, or a reader error.
    """
    check_exists(src)
    rename = {user_col: v2_name for v2_name, user_col in col_map.items()}
    if fmt != "csv":
        batches = geo_batches(fmt, src, rename, _V2_COLUMNS, domain_crs_str)
        return _collect(batches, bounds)
    try:
        return _collect(csv_batches(src, rename, _V2_COLUMNS), bounds)
    except pa.ArrowInvalid:
        pass
    # A cell is not a number: re-read as strings so pandera reports it. Outside
    # the except block, so the first pass's batches are freed.
    return _collect(csv_batches(src, rename, _V2_COLUMNS, as_strings=True), bounds)


def _collect(batches, bounds: tuple) -> pa.Table:
    """Validate, domain-filter and number each batch, keeping only its
    compact typed columns.

    Generated tree IDs are 1 … N over the trees kept in the domain, in file
    order. Once any batch fails validation the rest are still validated, for
    the report, but no data is kept.
    """
    xmin, ymin, xmax, ymax = bounds
    failures = FailureCases(list(_BATCH_SCHEMA.columns))
    tables: list[pa.Table] = []
    tree_ids: list[tuple[int, np.ndarray]] = []
    provided: list[str] | None = None
    next_id = 1
    for df in batches:
        if provided is None:
            provided = [c for c in _COLUMN_METADATA if c in df.columns]
            data_schema = pa.schema(
                [
                    f
                    for f in _PARQUET_SCHEMA
                    if f.name in provided and f.name != "tree_id"
                ]
            )
        for col in _OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        try:
            df = _BATCH_SCHEMA.validate(df, lazy=True)
        except SchemaErrors as e:
            failures.add(e.failure_cases, list(e.data.columns))
            tables.clear()
            df = e.data
        has_ids = "tree_id" in df.columns
        if has_ids and pd.api.types.is_integer_dtype(df["tree_id"]):
            first = int(df.index[0]) if len(df) else 0
            tree_ids.append((first, df["tree_id"].to_numpy("int64")))
        if failures:
            continue

        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        # Filter in Arrow, so each kept column is its own allocation rather
        # than a view of a pandas block.
        table = pa.Table.from_pandas(
            df[data_schema.names], schema=data_schema, preserve_index=False
        ).filter(pa.array(inside))
        if has_ids:
            ids = df["tree_id"].to_numpy("int64")[inside].astype("int32")
        else:
            ids = np.arange(next_id, next_id + table.num_rows, dtype="int32")
            next_id += table.num_rows
        tables.append(table.add_column(0, "tree_id", pa.array(ids)))

    if "tree_id" in (provided or []):
        failures.add_duplicate_tree_ids(tree_ids)
    if failures:
        total, cases = failures.report()
        raise ProcessingError(
            code="SCHEMA_VALIDATION_ERROR",
            message=f"Schema validation failed with {total} error(s).",
            suggestion=str(cases),
        )
    return pa.concat_tables(tables)


def _finite_or_none(value) -> float | None:
    """Coerce a reduction result to float, mapping null/NaN/inf to None.

    A single-value continuous column yields a null sample std (ddof=1) and an
    all-null column yields null min/max/mean/std. NaN/inf are not
    JSON-serializable — the API serves inventories with a JSONResponse that
    uses ``allow_nan=False`` — so they must not reach Firestore.
    """
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _column_summary(column: pa.ChunkedArray, key: str, col_type: str) -> dict:
    """Per-column summary matching the API's ColumnSummary schema.

    Statistics are over non-null values; non-finite continuous stats are
    sanitized to None. Mirrors the summary standgen writes for generated
    inventories so both sources present an identical shape. Computed with
    Arrow kernels, so float statistics match pandas to rounding, not bit for
    bit.
    """
    count = pc.count(column).as_py()
    null_count = column.null_count
    if col_type == "continuous":
        min_max = pc.min_max(column)
        return {
            "type": "continuous",
            "count": count,
            "null_count": null_count,
            "min": _finite_or_none(min_max["min"].as_py()),
            "max": _finite_or_none(min_max["max"].as_py()),
            "mean": _finite_or_none(pc.mean(column).as_py()),
            "std": _finite_or_none(pc.stddev(column, ddof=1).as_py()),
        }
    return {
        "type": "categorical",
        "count": count,
        "null_count": null_count,
        # tree_id is unique by construction; skip the distinct count.
        "unique_count": count
        if key == "tree_id"
        else pc.count_distinct(column, mode="only_valid").as_py(),
    }


def _write_parquet(table: pa.Table, bounds: tuple, path: str) -> None:
    """Write ``table`` as one Parquet file with an aggregated ``_metadata`` file.

    Rows are sorted by their 1 km tile's position on a Hilbert curve over the
    domain ``bounds``, then by y, then x, so each row group covers a compact
    area. Row groups are capped at ``ROW_GROUP_SIZE`` rows; each is an API data
    partition (#763). Columns the upload did not provide are written as nulls.
    """
    tiles = np.concatenate(
        [
            _tile_index(b.column(0).to_numpy(), b.column(1).to_numpy(), bounds)
            for b in table.select(["x", "y"]).to_batches()
        ]
    )
    order = pc.sort_indices(
        pa.table({"tile": tiles, "y": table.column("y"), "x": table.column("x")}),
        sort_keys=[("tile", "ascending"), ("y", "ascending"), ("x", "ascending")],
    )

    fs, root = pafs.FileSystem.from_uri(path)
    collector: list[pq.FileMetaData] = []
    with pq.ParquetWriter(
        f"{root}/part.0.parquet",
        _PARQUET_SCHEMA,
        filesystem=fs,
        metadata_collector=collector,
    ) as writer:
        for start in range(0, len(order), ROW_GROUP_SIZE):
            rows = table.take(order[start : start + ROW_GROUP_SIZE])
            arrays = [
                rows.column(f.name)
                if f.name in rows.column_names
                else pa.nulls(rows.num_rows, f.type)
                for f in _PARQUET_SCHEMA
            ]
            writer.write_table(
                pa.Table.from_arrays(arrays, schema=_PARQUET_SCHEMA),
                row_group_size=ROW_GROUP_SIZE,
            )
    collector[0].set_file_path("part.0.parquet")
    pq.write_metadata(_PARQUET_SCHEMA, f"{root}/_common_metadata", filesystem=fs)
    pq.write_metadata(
        _PARQUET_SCHEMA,
        f"{root}/_metadata",
        metadata_collector=collector,
        filesystem=fs,
    )


def _tile_index(x: np.ndarray, y: np.ndarray, bounds: tuple) -> np.ndarray:
    """Position of each point's 1 km tile on a Hilbert curve covering ``bounds``."""
    tx0 = math.floor(bounds[0] / TILE_SIZE_M)
    ty0 = math.floor(bounds[1] / TILE_SIZE_M)
    extent = 1 + max(
        math.floor(bounds[2] / TILE_SIZE_M) - tx0,
        math.floor(bounds[3] / TILE_SIZE_M) - ty0,
    )
    tx = np.floor(x / TILE_SIZE_M).astype(np.int64) - tx0
    ty = np.floor(y / TILE_SIZE_M).astype(np.int64) - ty0
    return _hilbert(tx, ty, extent)


def _hilbert(x: np.ndarray, y: np.ndarray, extent: int) -> np.ndarray:
    """Hilbert-curve index of non-negative integer cells in an ``extent``-wide grid."""
    n = 1
    while n < extent:
        n *= 2
    d = np.zeros(len(x), np.int64)
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        flip = rx & ~ry
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        x, y = np.where(ry, x, y), np.where(ry, y, x)
        s //= 2
    return d


def _extract_crs_string(domain_data: dict) -> str:
    """Extract the CRS EPSG string from a domain document."""
    crs_field = domain_data.get("crs")
    if isinstance(crs_field, dict):
        return crs_field["properties"]["name"]
    return crs_field or "EPSG:4326"
