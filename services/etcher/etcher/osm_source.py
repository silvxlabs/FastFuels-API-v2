"""Read OSM road/water features from per-state FlatGeobuf layers on GCS.

Replaces the runtime Overpass API dependency (which now refuses our Cloud Run
egress IP). Layers live on the OSM bucket as per-state FlatGeobuf:

    {OSM_BUCKET}/road/<state-slug>.fgb    cols: osm_id, highway, name
    {OSM_BUCKET}/water/<state-slug>.fgb   cols: osm_id, waterway, name
    {OSM_BUCKET}/index/states.fgb         cols: slug, name, geometry

Reads are lazy: a ``gs://`` path + a bbox makes GDAL (via pyogrio) fetch only the
bytes for the ROI through HTTP range requests — the same way ``lib.raster`` reads
hosted COGs. The tiny states index maps an ROI to the state file(s) it touches; a
domain straddling a border reads multiple state files and dedupes on ``osm_id``
(a way crossing a state line appears in both states' Geofabrik extracts).

The bucket name is injected via the ``OSM_BUCKET`` env var (see ``lib.config``)
and never hardcoded — this repo is public.
"""

import logging
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

from etcher.errors import ProcessingError
from lib.config import OSM_BUCKET
from lib.gcs.blobs import exists

logger = logging.getLogger(__name__)

# Tune GDAL for remote FlatGeobuf range reads, mirroring lib.raster.GDAL_COG_CONFIG
# for COGs: skip the directory listing on open, restrict the VSI curl handler to
# .fgb, merge adjacent ranges, and cache fetched blocks. etcher's only GDAL reads
# are these FlatGeobuf reads, so the config is applied process-wide via pyogrio.
# Measured on a Montana road layer (139 MB, 334k features): ~22% fewer range
# requests for a small ROI. https://gdal.org/en/stable/user/configoptions.html
FGB_GDAL_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".fgb",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}
pyogrio.set_gdal_config_options(FGB_GDAL_CONFIG)

STATES_INDEX = f"gs://{OSM_BUCKET}/index/states.fgb"


@lru_cache(maxsize=1)
def _states() -> gpd.GeoDataFrame:
    """The US states index, read once per process and cached."""
    return gpd.read_file(STATES_INDEX)


def _intersecting_state_slugs(bbox_4326: tuple) -> list[str]:
    """Geofabrik slugs of the states whose polygon intersects the ROI bbox."""
    roi = box(*bbox_4326)
    states = _states()
    return states.loc[states.intersects(roi), "slug"].tolist()


def _layer_exists(path: str) -> bool:
    """Whether a per-state layer object is present on GCS.

    Consulted only on the read-error path, to tell a genuinely-absent layer
    (safe to skip — a neighbouring state still contributes) from a real read
    failure (corrupt file, transient GCS error, auth) that must not be swallowed.
    """
    return exists(path)


def read_osm_features(bbox_4326: tuple, feature_type: str) -> gpd.GeoDataFrame:
    """Drop-in replacement for ``ox.features_from_polygon``.

    Args:
        bbox_4326: (minx, miny, maxx, maxy) in EPSG:4326
            (e.g. ``tuple(domain_gdf_4326.total_bounds)``).
        feature_type: ``"road"`` or ``"water"``.

    Returns:
        EPSG:4326 GeoDataFrame with osmnx-compatible columns
        (road: ``highway``, ``name``; water: ``waterway``, ``name``).
        Empty GeoDataFrame if no features fall in the ROI.

    Raises:
        ValueError: if ``feature_type`` is not ``"road"`` or ``"water"``.
        ProcessingError: if a per-state layer that exists on GCS fails to read.
            A genuinely-absent layer is skipped instead.
    """
    if feature_type not in ("road", "water"):
        raise ValueError(
            f"Please provide a valid feature type: {feature_type} is not implemented"
        )

    bbox = tuple(bbox_4326)
    parts = []
    for slug in _intersecting_state_slugs(bbox):
        path = f"gs://{OSM_BUCKET}/{feature_type}/{slug}.fgb"
        try:
            parts.append(gpd.read_file(path, bbox=bbox))
        except Exception as e:
            # A genuinely-absent layer is safe to skip; any other failure
            # (corrupt file, transient GCS error, auth) must surface as a failed
            # job rather than a silently-empty "successful" result.
            if _layer_exists(path):
                raise ProcessingError(
                    code="OSM_SOURCE_READ_FAILED",
                    message=f"Failed to read OSM {feature_type} data for state '{slug}'.",
                    suggestion=(
                        "The layer exists but could not be read; this is usually a "
                        "transient storage error. Retry the feature request."
                    ),
                ) from e
            # Log the state/feature only; never surface the bucket path.
            logger.info("OSM layer absent for %s/%s; skipping", feature_type, slug)
            continue

    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    if "osm_id" in gdf.columns:
        gdf = gdf.drop_duplicates("osm_id").reset_index(drop=True)
    return gdf
