"""
Inventory upload handler for the Uploader service.

Processes uploaded inventory files (CSV, GeoJSON, GeoPackage), validates
against the inventory schema, and writes Parquet to INVENTORIES_BUCKET.
"""

import os
from datetime import UTC, datetime

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from lib.config import DOMAINS_COLLECTION, INVENTORIES_BUCKET, INVENTORIES_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.errors import ProcessingError
from lib.firestore import get_document, update_document
from lib.gcs import delete_file, download_file

_V2_COLUMNS = {
    "x",
    "y",
    "height",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "crown_ratio",
}

# Doc metadata (type, unit) per v2 column, in canonical order. Only x, y, and
# height are required in an upload, so the inventory document's `columns` field
# is written from the columns the file actually provided — the API's treatments
# and voxelize endpoints rely on it to tell whether an inventory carries the
# columns an operation needs (e.g. a `dbh` column to thin against).
_COLUMN_METADATA = {
    "x": ("continuous", "m"),
    "y": ("continuous", "m"),
    "fia_species_code": ("categorical", None),
    "fia_status_code": ("categorical", None),
    "dbh": ("continuous", "cm"),
    "height": ("continuous", "m"),
    "crown_ratio": ("continuous", None),
}


class _InventorySchema(pa.DataFrameModel):
    x: Series[float]
    y: Series[float]
    height: Series[float] = pa.Field(ge=0, le=116)
    fia_species_code: Series[int] | None = pa.Field(nullable=True)
    fia_status_code: Series[int] | None = pa.Field(isin=[0, 1, 2, 3], nullable=True)
    dbh: Series[float] | None = pa.Field(ge=0, nullable=True)
    crown_ratio: Series[float] | None = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        coerce = True


def handle_inventory(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    source = doc["source"]
    fmt = source["format"]
    col_map = source.get("columns", {})
    domain_id = doc["domain_id"]

    local_filename = object_name.rsplit("/", 1)[-1]
    local_path = f"/tmp/{local_filename}"
    download_file(f"gs://{bucket}/{object_name}", local_path)

    try:
        _, domain_snap = get_document(DOMAINS_COLLECTION, domain_id)
        domain_data = domain_snap.to_dict()
        domain_gdf = parse_domain_gdf(domain_data)
        domain_crs_str = _extract_crs_string(domain_data)

        df = _parse(fmt, local_path, col_map, domain_crs_str)
        # Record which columns the file actually provided. _validate no longer
        # pads missing optionals, so the set is stable across validation — but
        # capture it here so the intent (file-provided columns only) is explicit.
        provided_columns = [c for c in _COLUMN_METADATA if c in df.columns]
        df = _validate(df)

        xmin, ymin, xmax, ymax = domain_gdf.total_bounds
        df = df[
            (df["x"] >= xmin)
            & (df["x"] <= xmax)
            & (df["y"] >= ymin)
            & (df["y"] <= ymax)
        ]
        if df.empty:
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
        dd.from_pandas(df.reset_index(drop=True), npartitions=1).to_parquet(
            path, write_metadata_file=True
        )

        georeference = {
            "crs": domain_crs_str,
            "bounds": [
                float(df["x"].min()),
                float(df["y"].min()),
                float(df["x"].max()),
                float(df["y"].max()),
            ],
        }
        # Record the columns the file actually provided. The create endpoint
        # wrote a provisional full column list; overwrite it with the real set
        # so the metadata matches the Parquet, which now carries exactly these
        # columns (no all-null padding).
        columns = [
            {"key": key, "type": col_type, "unit": unit}
            for key, (col_type, unit) in _COLUMN_METADATA.items()
            if key in provided_columns
        ]
        update_document(
            INVENTORIES_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": georeference,
                "columns": columns,
                "progress": {"message": "Complete", "percent": 100},
            },
        )

    finally:
        try:
            delete_file(f"gs://{bucket}/{object_name}")
        except Exception:
            pass
        if os.path.exists(local_path):
            os.remove(local_path)


def _parse(
    fmt: str, local_path: str, col_map: dict, domain_crs_str: str
) -> pd.DataFrame:
    """Parse an uploaded file into a normalized pandas DataFrame.

    col_map maps v2 column names → user column names in the file.
    Builds an inverse rename dict {user_col: v2_name} and applies it.
    """
    rename = {user_col: v2_name for v2_name, user_col in col_map.items()}

    if fmt == "csv":
        df = pd.read_csv(local_path)
        df = df.rename(columns=rename)
        return df[[col for col in df.columns if col in _V2_COLUMNS]]

    # GeoJSON or GeoPackage
    gdf = gpd.read_file(local_path)
    if gdf.empty:
        raise ProcessingError(
            code="EMPTY_FILE",
            message="The uploaded file contains no features.",
        )

    geom_types = set(gdf.geometry.geom_type.unique())
    unsupported = geom_types - {"Point", "MultiPoint"}
    if unsupported:
        raise ProcessingError(
            code="INVALID_GEOMETRY_TYPE",
            message=f"Expected Point or MultiPoint geometries, found: {unsupported}.",
        )

    if "MultiPoint" in geom_types:
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # GeoJSON is always EPSG:4326 per spec; set CRS if missing
    if gdf.crs is None and fmt == "geojson":
        gdf = gdf.set_crs("EPSG:4326")

    if gdf.crs is not None:
        gdf = gdf.to_crs(domain_crs_str)

    df = pd.DataFrame({"x": gdf.geometry.x.values, "y": gdf.geometry.y.values})
    for col in gdf.columns:
        if col == "geometry":
            continue
        v2_col = rename.get(col, col)
        if v2_col in _V2_COLUMNS and v2_col not in ("x", "y"):
            df[v2_col] = gdf[col].values

    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the parsed DataFrame against the inventory schema.

    Only the columns the file actually provided are written to the Parquet.
    Missing optional columns stay absent rather than being padded with all-null
    placeholders: absence is loud and checkable downstream (the document's
    `columns` metadata records it, and consumers reject operations that need a
    column the inventory doesn't have), whereas a present-but-null column is
    silently wrong everywhere. Pandera passes a bare x/y/height frame because the
    optional fields are `Series[...] | None`.
    """
    try:
        return _InventorySchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        cases = e.failure_cases.head(100).to_dict("records")
        raise ProcessingError(
            code="SCHEMA_VALIDATION_ERROR",
            message=f"Schema validation failed with {len(e.failure_cases)} error(s).",
            suggestion=str(cases),
        )


def _extract_crs_string(domain_data: dict) -> str:
    """Extract the CRS EPSG string from a domain document."""
    crs_field = domain_data.get("crs")
    if isinstance(crs_field, dict):
        return crs_field["properties"]["name"]
    return crs_field or "EPSG:4326"
