"""
Inventory export handlers.

Loads inventory data from partitioned Parquet, optionally selects a column
subset, and writes to the requested format (Parquet ZIP, CSV, GeoJSON,
GeoPackage).
"""

import json
import logging
import os
import tempfile
import time
import traceback
import zipfile
from collections.abc import Callable

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyogrio
import shapely

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_inventory_parquet, open_upload, upload_file
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)

# Rows per GeoJSON/GeoPackage write; bounds what one export holds at once (#608).
WRITE_BATCH_ROWS = 50_000


def _load_and_select(
    source: dict,
    progress: Callable[[str, int | None], None],
) -> pd.DataFrame:
    """Load inventory data and optionally select a column subset.

    Args:
        source: Source configuration with inventory_id and optional columns
        progress: Progress callback

    Returns:
        DataFrame with inventory data (all or selected columns)

    Raises:
        ProcessingError: If inventory cannot be loaded or columns are missing
    """
    inventory_id = source["inventory_id"]
    column_subset = source.get("columns")

    progress("Loading inventory data...", 20)
    try:
        df = load_inventory_parquet(inventory_id)
    except Exception as e:
        raise ProcessingError(
            code="INVENTORY_LOAD_ERROR",
            message=f"Failed to load inventory {inventory_id}: {e}",
            suggestion="Ensure the inventory exists and has completed processing.",
            traceback=traceback.format_exc(),
        )

    if column_subset:
        progress("Selecting columns...", 40)
        missing = [c for c in column_subset if c not in df.columns]
        if missing:
            raise ProcessingError(
                code="COLUMN_NOT_FOUND",
                message=f"Columns not found in inventory: {missing}",
                suggestion=f"Available columns: {list(df.columns)}",
            )
        df = df[column_subset]

    return df


def _require_xy(df: pd.DataFrame) -> None:
    """Raise if the DataFrame lacks the x and y columns.

    Raises:
        ProcessingError: If x or y columns are missing
    """
    if "x" not in df.columns or "y" not in df.columns:
        raise ProcessingError(
            code="MISSING_COORDINATES",
            message="GeoJSON and GeoPackage exports require 'x' and 'y' columns.",
            suggestion=(
                "Include 'x' and 'y' in the columns list, or omit the columns "
                "parameter to export all columns."
            ),
        )


def _to_geodataframe(df: pd.DataFrame, crs: str | None) -> gpd.GeoDataFrame:
    """Convert a DataFrame to a GeoDataFrame using x and y columns.

    Args:
        df: DataFrame that must contain 'x' and 'y' columns
        crs: Coordinate reference system string (e.g. "EPSG:32611")

    Returns:
        GeoDataFrame with Point geometry, without the x and y columns
    """
    geometry = shapely.points(df["x"].to_numpy(), df["y"].to_numpy())
    return gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=geometry, crs=crs)


def export_parquet(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export inventory to a zipped Parquet file.

    Args:
        export: Export document from Firestore
        source: Source configuration
        progress: Progress callback

    Returns:
        GCS path to the exported ZIP file
    """
    df = _load_and_select(source, progress)
    export_id = export["id"]

    filename = sanitize_filename(export.get("name", ""), ".zip")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    # Same member metadata ZipFile.writestr() sets; open() defaults to 1980.
    member_info = zipfile.ZipInfo(
        sanitize_filename(export.get("name", ""), ".parquet"),
        date_time=time.localtime()[:6],
    )
    member_info.compress_type = zipfile.ZIP_DEFLATED
    member_info.external_attr = 0o600 << 16

    progress("Writing Parquet...", 60)
    with (
        open_upload(gcs_path) as f,
        zipfile.ZipFile(f, "w") as zf,
        zf.open(member_info, "w", force_zip64=True) as member,
    ):
        df.to_parquet(member, index=False)

    return gcs_path


def export_csv(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export inventory to CSV.

    Args:
        export: Export document from Firestore
        source: Source configuration
        progress: Progress callback

    Returns:
        GCS path to the exported CSV file
    """
    df = _load_and_select(source, progress)
    export_id = export["id"]

    filename = sanitize_filename(export.get("name", ""), ".csv")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    progress("Writing CSV...", 70)
    with open_upload(gcs_path) as f:
        df.to_csv(f, index=False)

    return gcs_path


def export_geojson(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export inventory to GeoJSON.

    Args:
        export: Export document from Firestore
        source: Source configuration
        progress: Progress callback

    Returns:
        GCS path to the exported GeoJSON file
    """
    df = _load_and_select(source, progress)
    export_id = export["id"]
    crs = source.get("crs")

    _require_xy(df)

    filename = sanitize_filename(export.get("name", ""), ".geojson")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    progress("Writing GeoJSON...", 80)
    # Same bytes as GeoDataFrame.to_json(), written in batches instead of as
    # one string.
    envelope = gpd.GeoDataFrame(geometry=[], crs=crs).to_json()
    head, tail = envelope.split('"features": []')
    with open_upload(gcs_path) as f:
        f.write(f'{head}"features": ['.encode())
        for start in range(0, len(df), WRITE_BATCH_ROWS):
            batch = _to_geodataframe(df.iloc[start : start + WRITE_BATCH_ROWS], crs)
            features = ", ".join(json.dumps(ft) for ft in batch.iterfeatures(na="null"))
            f.write(((", " if start else "") + features).encode())
        f.write(f"]{tail}".encode())

    return gcs_path


def export_geopackage(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export inventory to GeoPackage.

    Args:
        export: Export document from Firestore
        source: Source configuration
        progress: Progress callback

    Returns:
        GCS path to the exported GeoPackage file
    """
    df = _load_and_select(source, progress)
    export_id = export["id"]
    crs = source.get("crs")
    _require_xy(df)

    filename = sanitize_filename(export.get("name", ""), ".gpkg")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    props = df.drop(columns=["x", "y"])
    props_schema = pa.Schema.from_pandas(props, preserve_index=False)
    schema = props_schema.append(pa.field("geometry", pa.binary()))

    def batches():
        for start in range(0, len(df), WRITE_BATCH_ROWS):
            rows = slice(start, start + WRITE_BATCH_ROWS)
            table = pa.RecordBatch.from_pandas(
                props.iloc[rows], schema=props_schema, preserve_index=False
            )
            points = shapely.points(
                df["x"].iloc[rows].to_numpy(), df["y"].iloc[rows].to_numpy()
            )
            geometry = pa.array(shapely.to_wkb(points), pa.binary())
            yield pa.RecordBatch.from_arrays([*table.columns, geometry], schema=schema)

    progress("Writing GeoPackage...", 80)
    # GPKG is SQLite and needs a local path; on Cloud Run that is tmpfs.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "export.gpkg")
        pyogrio.write_arrow(
            pa.RecordBatchReader.from_batches(schema, batches()),
            tmp_path,
            layer="export",
            driver="GPKG",
            geometry_name="geometry",
            geometry_type="Point",
            crs=crs,
        )
        upload_file(tmp_path, gcs_path)

    return gcs_path
