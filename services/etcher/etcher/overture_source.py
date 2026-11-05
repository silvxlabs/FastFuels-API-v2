"""Read Overture Maps buildings directly from Overture's public S3 bucket.

Overture publishes each release as GeoParquet on anonymous S3:

    s3://overturemaps-us-west-2/release/<YYYY-MM-DD.N>/theme=buildings/type=building/*.parquet

and indexes it with a STAC catalog at https://stac.overturemaps.org. The catalog
names the latest release, and each release's ``collections.parquet`` lists every
data file with its bbox. Only about the last two months of releases stay online,
so features always read the latest release and record it; reproducibility comes
from the stored feature, not from re-reading a pinned release.

A read first picks the files whose bbox overlaps the ROI from the index (one or
two of the ~500 building files), then scans only those, filtering on each row's
``bbox`` struct (``xmin``/``xmax``/``ymin``/``ymax``, EPSG:4326) so pyarrow skips
row groups whose statistics fall outside the ROI. Scanning the whole release
instead holds every file's parsed footer for the length of the scan, several
GiB outside Arrow's memory pool.
"""

import json
import logging
import re
import urllib.request
from datetime import date

import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from shapely.geometry import box

from etcher.errors import ProcessingError

logger = logging.getLogger(__name__)

OVERTURE_REGION = "us-west-2"
STAC_URL = "https://stac.overturemaps.org"
STAC_TIMEOUT_S = 60
BUILDING_COLLECTION = "building"
BUILDING_COLUMNS = ["id", "geometry", "height", "sources"]

RELEASE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d+)$")

OVERTURE_ATTRIBUTION = {
    "dataset": "Overture Maps Foundation",
    "license_name": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "citation": "© OpenStreetMap contributors, Overture Maps Foundation",
    "access_url": "https://docs.overturemaps.org/attribution/",
}

READ_FAILED_SUGGESTION = (
    "Overture Maps data could not be read; this is usually a transient network "
    "error. Retry the feature request."
)


def overture_attribution() -> dict:
    """The Overture Maps Foundation attribution for a feature built now."""
    return {**OVERTURE_ATTRIBUTION, "accessed_on": date.today().isoformat()}


def s3_filesystem() -> pafs.S3FileSystem:
    """Anonymous filesystem for Overture's public bucket."""
    return pafs.S3FileSystem(anonymous=True, region=OVERTURE_REGION)


def fetch(url: str) -> bytes:
    """GET a file from Overture's STAC catalog."""
    with urllib.request.urlopen(url, timeout=STAC_TIMEOUT_S) as response:
        return response.read()


def parse_release(name: str) -> tuple[date, int] | None:
    """Sort key for a release name like ``2026-10-21.0``; None if it isn't one."""
    match = RELEASE_PATTERN.match(name)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def latest_release() -> str:
    """The newest Overture release, as named by the STAC catalog root.

    Raises:
        ProcessingError: if the catalog cannot be read or names no valid release.
    """
    try:
        catalog = json.loads(fetch(f"{STAC_URL}/catalog.json"))
    except Exception as e:
        raise ProcessingError(
            code="OVERTURE_SOURCE_READ_FAILED",
            message="Failed to read the Overture Maps release catalog.",
            suggestion=READ_FAILED_SUGGESTION,
        ) from e

    release = catalog.get("latest") if isinstance(catalog, dict) else None
    if not isinstance(release, str) or parse_release(release) is None:
        raise ProcessingError(
            code="OVERTURE_RELEASE_NOT_FOUND",
            message="No Overture Maps release is currently published.",
            suggestion=READ_FAILED_SUGGESTION,
        )
    return release


def bbox_filter(bbox_4326: tuple) -> pc.Expression:
    """Row filter keeping rows whose ``bbox`` struct overlaps the ROI."""
    west, south, east, north = bbox_4326
    return (
        (pc.field("bbox", "xmin") <= east)
        & (pc.field("bbox", "xmax") >= west)
        & (pc.field("bbox", "ymin") <= north)
        & (pc.field("bbox", "ymax") >= south)
    )


def building_files(bbox_4326: tuple, release: str) -> list[str]:
    """S3 paths (``bucket/key``) of the release's building files overlapping a bbox.

    Read from the release's ``collections.parquet`` STAC index, which has one row
    per data file with the file's bbox.
    """
    index = pq.read_table(
        pa.BufferReader(fetch(f"{STAC_URL}/{release}/collections.parquet")),
        columns=["assets"],
        filters=(pc.field("collection") == BUILDING_COLLECTION)
        & bbox_filter(bbox_4326),
    )
    hrefs = [
        assets["aws"]["alternate"]["s3"]["href"]
        for assets in index["assets"].to_pylist()
    ]
    return sorted(href.removeprefix("s3://") for href in hrefs)


def read_overture_buildings(bbox_4326: tuple, release: str) -> gpd.GeoDataFrame:
    """Read Overture buildings intersecting a bbox.

    Args:
        bbox_4326: (minx, miny, maxx, maxy) in EPSG:4326.
        release: Overture release name, e.g. from :func:`latest_release`.

    Returns:
        EPSG:4326 GeoDataFrame with columns ``id``, ``height`` and ``sources``
        (list of ``{dataset, property, license, ...}`` dicts) and the footprint
        geometry. Empty if no building intersects the bbox.

    Raises:
        ProcessingError: if the release cannot be read.
    """
    bbox = tuple(bbox_4326)
    try:
        paths = building_files(bbox, release)
        if paths:
            dataset = ds.dataset(paths, filesystem=s3_filesystem(), format="parquet")
            table = dataset.to_table(columns=BUILDING_COLUMNS, filter=bbox_filter(bbox))
    except Exception as e:
        raise ProcessingError(
            code="OVERTURE_SOURCE_READ_FAILED",
            message=f"Failed to read Overture Maps buildings (release {release}).",
            suggestion=READ_FAILED_SUGGESTION,
        ) from e

    if not paths:
        logger.info(
            "No Overture building files overlap the bbox in release %s", release
        )
        return gpd.GeoDataFrame(
            {"id": [], "height": [], "sources": []}, geometry=[], crs="EPSG:4326"
        )

    df = table.drop_columns(["geometry"]).to_pandas()
    geometry = gpd.GeoSeries.from_wkb(
        table.column("geometry").to_numpy(zero_copy_only=False), crs="EPSG:4326"
    )
    gdf = gpd.GeoDataFrame(df, geometry=geometry.values, crs="EPSG:4326")
    gdf = gdf[gdf.intersects(box(*bbox))].reset_index(drop=True)
    logger.info(
        "Read %d Overture buildings from %d file(s) of release %s",
        len(gdf),
        len(paths),
        release,
    )
    return gdf
