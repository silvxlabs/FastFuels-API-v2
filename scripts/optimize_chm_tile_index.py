"""
Optimize a GeoParquet tile index for fast remote spatial lookups.

Reads a source GeoParquet file from GCS and writes an optimized plain
parquet with the suffix ``_optimized``. The source file is never modified.

Optimizations applied:
  1. Strip to only the columns needed at runtime
  2. Replace geometry with flat bbox columns (xmin, ymin, xmax, ymax)
  3. Hilbert-sort rows for spatial locality
  4. Write with small row groups for efficient partial reads

The output is a plain parquet file (not GeoParquet) with no geometry column.

Usage:
    cd services/griddle
    uv run --active python ../../scripts/optimize_tile_index.py <gcs_path> --columns col1,col2
    uv run --active python ../../scripts/optimize_tile_index.py --all

Examples:
    # Optimize a single file
    uv run --active python ../../scripts/optimize_tile_index.py \\
        gs://bucket/naip_chm_index.parquet --columns chm_url,scale_factor

    # Optimize all known CHM tile indexes
    uv run --active python ../../scripts/optimize_tile_index.py --all
"""

import argparse
import io
import sys

import gcsfs
import geopandas as gpd

from lib.config import TABLES_BUCKET

ROW_GROUP_SIZE = 500

# Registry of all tile indexes and the columns the runtime needs.
TILE_INDEXES = {
    f"gs://{TABLES_BUCKET}/naip_chm_index.parquet": ["chm_url", "scale_factor"],
    f"gs://{TABLES_BUCKET}/Meta2024_chm_index.parquet": ["tile"],
    f"gs://{TABLES_BUCKET}/Meta_chmv2_index.parquet": ["tile"],
}


def _optimized_path(gcs_path: str) -> str:
    """gs://bucket/foo.parquet -> gs://bucket/foo_optimized.parquet"""
    return gcs_path.replace(".parquet", "_optimized.parquet")


def optimize(gcs_path: str, columns: list[str] | None = None) -> None:
    """Read a source GeoParquet tile index and write an optimized copy."""
    fs = gcsfs.GCSFileSystem()
    src = gcs_path.removeprefix("gs://")
    dst = _optimized_path(gcs_path).removeprefix("gs://")

    # Download source
    print(f"\n  source: gs://{src}")
    raw = fs.cat(src)
    gdf = gpd.read_parquet(io.BytesIO(raw))
    print(
        f"    read:  {len(gdf)} rows, {len(gdf.columns)} cols, {len(raw) / 1024 / 1024:.1f} MB"
    )

    # Strip to needed data columns
    if columns:
        keep = [c for c in columns if c in gdf.columns]
        data_df = gdf[keep].copy()
        print(f"    cols:  kept {keep}")
    else:
        data_df = gdf.drop(columns=["geometry"]).copy()

    # Replace geometry with flat bbox columns
    bounds = gdf.geometry.bounds
    data_df["bbox_xmin"] = bounds["minx"].values
    data_df["bbox_ymin"] = bounds["miny"].values
    data_df["bbox_xmax"] = bounds["maxx"].values
    data_df["bbox_ymax"] = bounds["maxy"].values

    # Hilbert sort (uses the original GeoDataFrame for spatial indexing)
    order = gdf.hilbert_distance(total_bounds=gdf.total_bounds)
    data_df = data_df.iloc[order.argsort()].reset_index(drop=True)

    # Write optimized plain parquet
    with fs.open(dst, "wb") as f:
        data_df.to_parquet(f, row_group_size=ROW_GROUP_SIZE, index=False)

    info = fs.info(dst)
    n_groups = (len(data_df) + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE
    print(f"    dest:  gs://{dst}")
    print(
        f"    wrote: {len(data_df)} rows, {n_groups} row groups, "
        f"{info['size'] / 1024 / 1024:.1f} MB"
    )


def main():
    parser = argparse.ArgumentParser(description="Optimize a GeoParquet tile index.")
    parser.add_argument(
        "gcs_path", nargs="?", help="gs:// path to the source parquet file"
    )
    parser.add_argument(
        "--columns",
        help="Comma-separated list of data columns to keep (bbox is always added)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize all known CHM tile indexes",
    )
    args = parser.parse_args()

    if args.all:
        for path, cols in TILE_INDEXES.items():
            optimize(path, columns=cols)
    elif args.gcs_path:
        columns = args.columns.split(",") if args.columns else None
        optimize(args.gcs_path, columns=columns)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
