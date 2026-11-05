"""Read Overture Maps buildings directly from Overture's public S3 bucket.

Overture publishes each release as GeoParquet on anonymous S3:

    s3://overturemaps-us-west-2/release/<YYYY-MM-DD.N>/theme=buildings/type=building/*.parquet

Only about the last two months of releases stay online, so features always read
the latest release (found by listing ``release/``) and record it; reproducibility
comes from the stored feature, not from re-reading a pinned release.

Each row carries a ``bbox`` struct (``xmin``/``xmax``/``ymin``/``ymax``, EPSG:4326).
Filtering the dataset on it lets pyarrow skip row groups whose statistics fall
outside the ROI, so a domain-sized read fetches only a few row groups out of a
release that is hundreds of GB.
"""

import logging
import re
from datetime import date

import geopandas as gpd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from shapely.geometry import box

from etcher.errors import ProcessingError

logger = logging.getLogger(__name__)

OVERTURE_BUCKET = "overturemaps-us-west-2"
OVERTURE_REGION = "us-west-2"
RELEASE_PREFIX = f"{OVERTURE_BUCKET}/release"
BUILDINGS_PATH = "theme=buildings/type=building"
BUILDING_COLUMNS = ["id", "geometry", "height", "sources"]

# Files whose row-group statistics are checked concurrently. Every scan opens all
# ~500 files of a release to test their bbox statistics, so latency, not bytes,
# dominates. Measured locally on the Bigfork domain: ~25-50 s at pyarrow's
# default of 4, ~13-15 s at 16-64.
FRAGMENT_READAHEAD = 32

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


def parse_release(name: str) -> tuple[date, int] | None:
    """Sort key for a release name like ``2026-10-21.0``; None if it isn't one."""
    match = RELEASE_PATTERN.match(name)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def newest_release(names: list[str]) -> str | None:
    """The newest valid release name, ordered by (date, N); None if none are valid."""
    keyed = [(key, name) for name in names if (key := parse_release(name))]
    if not keyed:
        return None
    return max(keyed)[1]


def latest_release() -> str:
    """The newest Overture release currently published on S3.

    Raises:
        ProcessingError: if the bucket listing fails or holds no releases.
    """
    try:
        infos = s3_filesystem().get_file_info(
            pafs.FileSelector(RELEASE_PREFIX, recursive=False)
        )
    except Exception as e:
        raise ProcessingError(
            code="OVERTURE_SOURCE_READ_FAILED",
            message="Failed to list Overture Maps releases.",
            suggestion=READ_FAILED_SUGGESTION,
        ) from e

    names = [info.base_name for info in infos if info.type == pafs.FileType.Directory]
    release = newest_release(names)
    if release is None:
        raise ProcessingError(
            code="OVERTURE_RELEASE_NOT_FOUND",
            message="No Overture Maps release is currently published.",
            suggestion=READ_FAILED_SUGGESTION,
        )
    return release


def bbox_filter(bbox_4326: tuple) -> pc.Expression:
    """Row filter keeping features whose ``bbox`` struct overlaps the ROI."""
    west, south, east, north = bbox_4326
    return (
        (pc.field("bbox", "xmin") <= east)
        & (pc.field("bbox", "xmax") >= west)
        & (pc.field("bbox", "ymin") <= north)
        & (pc.field("bbox", "ymax") >= south)
    )


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
    path = f"{RELEASE_PREFIX}/{release}/{BUILDINGS_PATH}/"
    try:
        dataset = ds.dataset(path, filesystem=s3_filesystem(), format="parquet")
        table = dataset.scanner(
            columns=BUILDING_COLUMNS,
            filter=bbox_filter(bbox),
            fragment_readahead=FRAGMENT_READAHEAD,
        ).to_table()
    except Exception as e:
        raise ProcessingError(
            code="OVERTURE_SOURCE_READ_FAILED",
            message=f"Failed to read Overture Maps buildings (release {release}).",
            suggestion=READ_FAILED_SUGGESTION,
        ) from e

    df = table.drop_columns(["geometry"]).to_pandas()
    geometry = gpd.GeoSeries.from_wkb(
        table.column("geometry").to_numpy(zero_copy_only=False), crs="EPSG:4326"
    )
    gdf = gpd.GeoDataFrame(df, geometry=geometry.values, crs="EPSG:4326")
    gdf = gdf[gdf.intersects(box(*bbox))].reset_index(drop=True)
    logger.info("Read %d Overture buildings from release %s", len(gdf), release)
    return gdf
