"""
Inventory export handlers.

Loads inventory data from partitioned Parquet, optionally selects a column
subset, and writes to the requested format (Parquet ZIP, CSV, GeoJSON,
GeoPackage).
"""

import io
import logging
import os
import tempfile
import traceback
import zipfile
from collections.abc import Callable

import gcsfs
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_inventory_parquet
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)


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


def _to_geodataframe(df: pd.DataFrame, crs: str | None) -> gpd.GeoDataFrame:
    """Convert a DataFrame to a GeoDataFrame using x and y columns.

    Args:
        df: DataFrame that must contain 'x' and 'y' columns
        crs: Coordinate reference system string (e.g. "EPSG:32611")

    Returns:
        GeoDataFrame with Point geometry

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

    geometry = [Point(xy) for xy in zip(df["x"], df["y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    # Drop x/y columns since they're now in the geometry
    gdf = gdf.drop(columns=["x", "y"])
    return gdf


def _upload_bytes(data: bytes, gcs_path: str) -> None:
    """Upload bytes to GCS.

    Args:
        data: Raw bytes to upload
        gcs_path: Full GCS path (gs://bucket/path/to/file)
    """
    fs = gcsfs.GCSFileSystem()
    with fs.open(gcs_path, "wb") as f:
        f.write(data)


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

    progress("Writing Parquet...", 60)
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_bytes = parquet_buffer.getvalue()

    progress("Creating ZIP archive...", 80)
    zip_buffer = io.BytesIO()
    inner_name = sanitize_filename(export.get("name", ""), ".parquet")
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, parquet_bytes)
    zip_bytes = zip_buffer.getvalue()

    filename = sanitize_filename(export.get("name", ""), ".zip")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    _upload_bytes(zip_bytes, gcs_path)

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

    progress("Writing CSV...", 70)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    filename = sanitize_filename(export.get("name", ""), ".csv")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    _upload_bytes(csv_bytes, gcs_path)

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

    progress("Converting to GeoDataFrame...", 60)
    gdf = _to_geodataframe(df, crs)

    progress("Writing GeoJSON...", 80)
    geojson_bytes = gdf.to_json().encode("utf-8")

    filename = sanitize_filename(export.get("name", ""), ".geojson")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    _upload_bytes(geojson_bytes, gcs_path)

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

    progress("Converting to GeoDataFrame...", 60)
    gdf = _to_geodataframe(df, crs)

    progress("Writing GeoPackage...", 80)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "export.gpkg")
        gdf.to_file(tmp_path, driver="GPKG")
        with open(tmp_path, "rb") as f:
            gpkg_bytes = f.read()

    filename = sanitize_filename(export.get("name", ""), ".gpkg")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"
    _upload_bytes(gpkg_bytes, gcs_path)

    return gcs_path
