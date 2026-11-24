# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=2.0",
#     "gcsfs>=2024.6",
#     "pyarrow>=17",
#     "python-dotenv>=1.0",
# ]
# ///
"""
Build the NAIP-CHM v1 tile index used by griddle and the API.

Reads NTSG's STAC GeoParquet catalog of NAIP-CHM v1 rasters
(``https://data.ntsg.umt.edu/naip-chm-v1/stac/items.parquet``) and writes a
compact, location-sorted index that can be queried remotely by reading only
the parquet footer and the one to three row groups that intersect a domain.

When to run: by hand, whenever NTSG publishes new NAIP-CHM data. There is no
scheduled job. Compare the catalog's current ``ETag`` with the
``naip_chm:source_etag`` recorded in the uploaded index to tell whether a
rebuild is needed.

Usage (from the repo root):

    # Build locally (downloads items.parquet, ~330 MB)
    uv run scripts/build_naip_chm_index.py --output naip_chm_v1_index.parquet

    # Build from an already-downloaded items.parquet
    uv run scripts/build_naip_chm_index.py --source items.parquet \\
        --output naip_chm_v1_index.parquet

    # Build and upload to gs://{TABLES_BUCKET}/naip_chm_v1_index.parquet
    uv run scripts/build_naip_chm_index.py --upload

``TABLES_BUCKET`` is read from the environment, falling back to the repo-root
``.env``. Uploading uses Application Default Credentials.

Index contents (one row per raster; rows with ``chm:valid_frac == 0`` are
dropped because they contain no valid canopy heights):

    chm_url      string     assets.chm.href
    year         int16      naip:year
    bbox_xmin    float64    bbox.xmin (EPSG:4326)
    bbox_ymin    float64    bbox.ymin
    bbox_xmax    float64    bbox.xmax
    bbox_ymax    float64    bbox.ymax
    footprint    binary     geometry as WKB (EPSG:4326)
    datetime     timestamp[us, UTC]  acquisition datetime
    source_gsd   float64    chm:source_gsd (NAIP imagery GSD, m)
    pipeline     string     chm:pipeline
    model        string     chm:model

Layout: rows are sorted by the Hilbert index of footprint centroids (extent
taken from the data, not hard-coded) so that every vintage of one place sits
in the same row group. 5,000 rows per row group, zstd, column statistics on,
so readers prune row groups with the bbox column min/max in the footer.

Provenance is recorded in the parquet key-value metadata:

    naip_chm:source_url            catalog URL
    naip_chm:source_etag           catalog ETag from a HEAD request
    naip_chm:source_last_modified  catalog Last-Modified from a HEAD request
    naip_chm:built_at              build time, UTC ISO-8601
    naip_chm:row_count             rows in the index
    naip_chm:years                 JSON list of distinct years, ascending

With ``--source``, the remote catalog is still HEADed and its ETag and
Last-Modified are recorded only if the local file's size equals the remote
``Content-Length``; otherwise the build fails, because the local copy cannot
be tied to the current catalog version.
"""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE_URL = "https://data.ntsg.umt.edu/naip-chm-v1/stac/items.parquet"
INDEX_NAME = "naip_chm_v1_index.parquet"
ROW_GROUP_SIZE = 5000
REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = pa.schema(
    [
        ("chm_url", pa.string()),
        ("year", pa.int16()),
        ("bbox_xmin", pa.float64()),
        ("bbox_ymin", pa.float64()),
        ("bbox_xmax", pa.float64()),
        ("bbox_ymax", pa.float64()),
        ("footprint", pa.binary()),
        ("datetime", pa.timestamp("us", tz="UTC")),
        ("source_gsd", pa.float64()),
        ("pipeline", pa.string()),
        ("model", pa.string()),
    ]
)


def head_source(url: str) -> dict[str, str]:
    """Return the catalog's ETag, Last-Modified and Content-Length."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        headers = resp.headers
    return {
        "etag": headers["ETag"],
        "last_modified": headers["Last-Modified"],
        "content_length": headers["Content-Length"],
    }


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` without holding it in memory."""
    with urllib.request.urlopen(url, timeout=600) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)


def build_table(source: Path) -> tuple[pa.Table, int]:
    """Read the catalog, filter and Hilbert-sort it; return (table, rows_in)."""
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    src = str(source).replace("'", "''")
    con.execute(
        f"""
        CREATE TEMP VIEW items AS
        SELECT
            assets.chm.href AS chm_url,
            CAST("naip:year" AS SMALLINT) AS year,
            bbox.xmin AS bbox_xmin,
            bbox.ymin AS bbox_ymin,
            bbox.xmax AS bbox_xmax,
            bbox.ymax AS bbox_ymax,
            ST_GeomFromWKB(geometry::BLOB) AS geom,
            datetime,
            "chm:source_gsd" AS source_gsd,
            "chm:pipeline" AS pipeline,
            "chm:model" AS model,
            "chm:valid_frac" AS valid_frac
        FROM read_parquet('{src}')
        """
    )
    rows_in = con.sql("SELECT count(*) FROM items").fetchall()[0][0]
    table = con.sql(
        """
        WITH kept AS (SELECT * FROM items WHERE valid_frac > 0),
        extent AS (
            SELECT ST_Extent(ST_MakeEnvelope(
                min(bbox_xmin), min(bbox_ymin), max(bbox_xmax), max(bbox_ymax)
            )) AS box FROM kept
        )
        SELECT
            chm_url, year, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
            ST_AsWKB(geom)::BLOB AS footprint,
            datetime, source_gsd, pipeline, model
        FROM kept, extent
        ORDER BY ST_Hilbert(ST_Centroid(geom), extent.box), chm_url
        """
    ).to_arrow_table()
    return table.cast(SCHEMA), rows_in


def write_index(table: pa.Table, dest: Path, provenance: dict[str, str]) -> None:
    metadata = {
        "naip_chm:source_url": SOURCE_URL,
        "naip_chm:source_etag": provenance["etag"],
        "naip_chm:source_last_modified": provenance["last_modified"],
        "naip_chm:built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "naip_chm:row_count": str(table.num_rows),
        "naip_chm:years": json.dumps(
            sorted(pc.unique(table.column("year")).to_pylist())
        ),
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(
        table,
        dest,
        row_group_size=ROW_GROUP_SIZE,
        compression="zstd",
        write_statistics=True,
    )


def load_tables_bucket() -> str:
    if "TABLES_BUCKET" not in os.environ:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    bucket = os.environ.get("TABLES_BUCKET")
    if not bucket:
        raise SystemExit("TABLES_BUCKET is not set (environment or repo .env)")
    return bucket.removeprefix("gs://").rstrip("/")


def upload(local: Path, bucket: str) -> str:
    import gcsfs

    remote = f"{bucket}/{INDEX_NAME}"
    gcsfs.GCSFileSystem().put_file(str(local), remote)
    return f"gs://{remote}"


def print_summary(rows_in: int, path: Path) -> None:
    pf = pq.ParquetFile(path)
    years = pq.read_table(path, columns=["year"]).column("year")
    distinct = sorted(set(years.to_pylist()))
    print(f"rows in:     {rows_in:,}")
    print(f"rows out:    {pf.metadata.num_rows:,}")
    print(f"dropped:     {rows_in - pf.metadata.num_rows:,} (chm:valid_frac == 0)")
    print(f"years:       {distinct[0]}-{distinct[-1]} ({len(distinct)}): {distinct}")
    print(f"file size:   {path.stat().st_size / 1e6:.1f} MB")
    print(f"row groups:  {pf.metadata.num_row_groups}")
    for k, v in pf.schema_arrow.metadata.items():
        print(f"{k.decode()}: {v.decode()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the NAIP-CHM v1 tile index from NTSG's items.parquet."
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Local copy of items.parquet (skips the download; size must "
        "match the remote Content-Length).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Local output path (default: a temp file when --upload, "
        f"else ./{INDEX_NAME}).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Upload to gs://{{TABLES_BUCKET}}/{INDEX_NAME}.",
    )
    args = parser.parse_args()

    bucket = load_tables_bucket() if args.upload else None
    provenance = head_source(SOURCE_URL)
    print(
        f"source: {SOURCE_URL} etag={provenance['etag']} "
        f"last-modified={provenance['last_modified']}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if args.source:
            source = args.source
            size = source.stat().st_size
            if str(size) != provenance["content_length"]:
                raise SystemExit(
                    f"{source} is {size} bytes but the remote catalog is "
                    f"{provenance['content_length']} bytes; it is not the "
                    "current catalog version. Re-download or omit --source."
                )
        else:
            source = tmpdir / "items.parquet"
            print("downloading items.parquet ...")
            download(SOURCE_URL, source)

        output = args.output or (
            tmpdir / INDEX_NAME if args.upload else Path(INDEX_NAME)
        )
        table, rows_in = build_table(source)
        write_index(table, output, provenance)
        print_summary(rows_in, output)

        if args.upload:
            print(f"uploaded:    {upload(output, bucket)}")


if __name__ == "__main__":
    main()
