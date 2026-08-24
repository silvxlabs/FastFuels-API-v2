"""
LANDFIRE Product Service (LFPS) on-demand API client.

LFPS is an ArcGIS GeoProcessing Server: submit a job, poll until it finishes,
then download the output from a URL the job carries once done.

Only the request shape our own pipeline needs is exposed. `Resample_Resolution`,
`Edit_Rule`, `Edit_Mask`, `Include_Layer_List_XML_File`, `Output_Projection`,
and the Map Zone AOI form are deliberately not parameters here: resampling,
reprojection, and non-burnable reclassification already happen in our own
post-fetch pipeline, so an LFPS fetch goes through the same path as a
stored-COG raster rather than being edited twice.

All functions are synchronous. The API wraps blocking I/O calls with
asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import geopandas as gpd
import requests
from shapely.geometry.base import BaseGeometry

from lib.config import RASTERS_BUCKET
from lib.landfire.config import LANDFIRE_USER_EMAIL, SEASON_CODES

logger = logging.getLogger(__name__)

BASE_URL = "https://lfps.usgs.gov/api"

# LFPS's own timeout guidance puts typical jobs at 12s-7min and a hard fail at
# 2 hours; this is just the HTTP request timeout for each individual call.
_REQUEST_TIMEOUT = 60

# LANDFIRE updates its product catalog only a handful of times a year,
# so caching for an hour costs nothing in staleness
LFPS_PRODUCTS_TTL_SECONDS = float(os.getenv("LFPS_PRODUCTS_TTL_SECONDS", 3600))

# Matches the trailing season+year suffix on a Seasonal Fuels layer name,
# e.g. "_SP26" in "LF2025_FBFM40_SP26". The two digits are the season's own
# two-digit year -- the calendar year the seasonal fuels represent, which
# LANDFIRE currently sets one ahead of the base layer's `version` year but
# which we read from the layer name rather than assume.
_SEASON_PATTERN = re.compile(rf"_({'|'.join(SEASON_CODES)})(\d{{2}})$")

_products: list[LfpsProduct] | None = None
_products_fetched_on: datetime | None = None


def _parse_season_suffix(layer_name: str) -> tuple[str | None, int | None]:
    """Extract the season code and four-digit year from a seasonal layer name,
    e.g. ("SP", 2026) from "LF2025_FBFM40_SP26". (None, None) for a
    non-seasonal (annual) layer."""
    match = _SEASON_PATTERN.search(layer_name)
    if match is None:
        return None, None
    return match.group(1), 2000 + int(match.group(2))


class LfpsJobFailedError(Exception):
    """An LFPS job finished with status ``Failed``.

    Carries LFPS's own error text unchanged -- LFPS reports the failure
    reason as one of the free-text ``messages`` entries on the job status
    response, not a structured error code, so there is nothing to map it to.
    """


class LfpsJobTimeoutError(Exception):
    """The caller gave up polling an LFPS job before it reached a final status.

    Distinct from :class:`LfpsJobFailedError`: this is our own decision to
    stop waiting, not an error LFPS reported.
    """


@dataclass(frozen=True)
class LfpsProduct:
    """One entry from the LFPS product catalog (``GET /api/products``).

    `ak`, `hi`, and `prvi` coverage flags are not carried here -- FastFuels is
    CONUS-only, so a product's coverage outside CONUS is not relevant.
    """

    layer_name: str
    product_name: str
    theme: str
    acronym: str
    version: str
    conus: bool
    geo_areas: str
    season: str | None
    season_year: int | None


@dataclass(frozen=True)
class LfpsJob:
    """An LFPS job, as returned by ``job/submit`` or ``job/status``.

    `output_file` and `geo_area` are only populated once LFPS has them: a
    freshly submitted or still-running job has no `output_file`.
    """

    job_id: str
    status: str
    output_file: str | None = None
    geo_area: str | None = None


def list_products(refresh: bool = False) -> list[LfpsProduct]:
    """List LFPS's available products, caching the result in the process.

    Args:
        refresh: Re-fetch even if the cached copy is still within
            ``LFPS_PRODUCTS_TTL_SECONDS``.

    Returns:
        Every product LFPS currently serves.
    """
    global _products, _products_fetched_on

    fresh = (
        _products_fetched_on is not None
        and (datetime.now(UTC) - _products_fetched_on).total_seconds()
        < LFPS_PRODUCTS_TTL_SECONDS
    )
    if _products is not None and fresh and not refresh:
        return _products

    response = requests.get(f"{BASE_URL}/products", timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    raw_products = response.json()["products"]

    _products = []
    for p in raw_products:
        season, season_year = _parse_season_suffix(p["layerName"])
        _products.append(
            LfpsProduct(
                layer_name=p["layerName"],
                product_name=p["productName"],
                theme=p["theme"],
                acronym=p["acronym"],
                version=p["version"],
                conus=p["conus"],
                geo_areas=p["geoAreas"],
                season=season,
                season_year=season_year,
            )
        )
    _products_fetched_on = datetime.now(UTC)
    logger.info(f"Loaded LFPS product catalog: {len(_products)} products")
    return _products


def submit_job(layers: list[str], aoi: str) -> LfpsJob:
    """Submit an LFPS clip job.

    Args:
        layers: LFPS layer names (e.g. ``LF2024_FBFM40``) to include in the
            job's output.
        aoi: Area of interest as a space-separated ``W S E N`` bbox in WGS84.

    Returns:
        The newly submitted job.
    """
    params = {
        "Email": LANDFIRE_USER_EMAIL,
        "Layer_List": ";".join(layers),
        "Area_of_Interest": aoi,
    }
    response = requests.get(
        f"{BASE_URL}/job/submit", params=params, timeout=_REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return _parse_job(response.json())


def poll_status(job_id: str) -> LfpsJob:
    """Check an LFPS job's current status once.

    Does not loop or wait -- the caller owns retry/re-check timing.

    Args:
        job_id: The job to check.

    Returns:
        The job's current state.

    Raises:
        LfpsJobFailedError: The job's status is ``Failed``.
    """
    response = requests.get(
        f"{BASE_URL}/job/status",
        params={"JobId": job_id},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "Failed":
        descriptions = [m.get("description", "") for m in data.get("messages", [])]
        error_lines = [d for d in descriptions if d.strip().startswith("ERROR")]
        message = "\n".join(error_lines) if error_lines else "\n".join(descriptions)
        raise LfpsJobFailedError(message or "LFPS job failed with no message.")

    return _parse_job(data)


def download(job: LfpsJob) -> bytes:
    """Download a succeeded job's output.

    Args:
        job: A job with ``output_file`` set (i.e. one whose status is
            ``Succeeded``).

    Returns:
        The raw response content -- a zip of GeoTIFFs. Unzipping is left to
        the caller.
    """
    response = requests.get(job.output_file, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def _parse_job(data: dict) -> LfpsJob:
    """Build an `LfpsJob` from a `job/submit` or `job/status` response body."""
    return LfpsJob(
        job_id=data["jobId"],
        status=data["status"],
        output_file=data.get("outputFile"),
        geo_area=data.get("geoArea"),
    )


# --- Coverage check --------------------------------------------
#
# Checks whether a domain falls inside LANDFIRE's current coverage for the
# requested product/version. CONUS-wide products skip the GeoArea check.
#
# Boundaries are loaded at import time in EPSG:5070. Callers must pass
# `geometry` already reprojected to that CRS.

_BOUNDARY_CRS_EPSG = 5070


def _load_boundaries(filename: str, key: str) -> dict[str, BaseGeometry]:
    """Load a GeoParquet boundary file into a {key property value: geometry} dict.

    Reprojects to EPSG:5070 if the source CRS differs.
    """
    gdf = gpd.read_parquet(f"gs://{RASTERS_BUCKET}/{filename}")
    if gdf.crs is not None and gdf.crs.to_epsg() != _BOUNDARY_CRS_EPSG:
        gdf = gdf.to_crs(epsg=_BOUNDARY_CRS_EPSG)
    return dict(zip(gdf[key], gdf.geometry))


GEOAREA_BOUNDARIES: dict[str, BaseGeometry] = _load_boundaries(
    "LF_GeoAreas.parquet", key="geoArea"
)

SEASONAL_BOUNDARIES: dict[str, BaseGeometry] = _load_boundaries(
    "LF_SeasonalRegions.parquet", key="season"
)


class CoverageStatus(StrEnum):
    """How much of a geometry falls inside a covered area.

    NO_SUCH_PRODUCT is distinct from NONE: NONE means LFPS serves this
    product/version (or product/version/season), just not where the
    geometry is; NO_SUCH_PRODUCT means LFPS isn't serving it anywhere at
    all right now (e.g. a season that hasn't been published yet).
    """

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    NO_SUCH_PRODUCT = "no_such_product"


def covers_annual(product: str, version: str, geometry: BaseGeometry) -> CoverageStatus:
    """Check LFPS coverage for an annual product/version against a geometry.

    Args:
        product: Product acronym, matched case-insensitively against
            `LfpsProduct.acronym` (e.g. "fbfm40" matches "FBFM40").
        version: LANDFIRE version year (e.g. "2025") -- matched against
            LFPS's "LF{version}" format.
        geometry: Domain geometry, already reprojected to EPSG:5070.

    Returns:
        CoverageStatus. NO_SUCH_PRODUCT if no non-Seasonal-Fuels LFPS
        entry matches product/version at all. FULL if `geo_areas` is
        "All", or if every GeoArea the geometry touches is served. NONE
        if it touches no served GeoArea. PARTIAL otherwise.
    """
    matching_product = next(
        (
            p
            for p in list_products()
            if p.theme != "Seasonal Fuels"
            and p.acronym.lower() == product.lower()
            and p.version == f"LF{version}"
        ),
        None,
    )
    if matching_product is None:
        return CoverageStatus.NO_SUCH_PRODUCT

    if matching_product.geo_areas.strip() == "All":
        return CoverageStatus.FULL

    covered_geoareas = {area.strip() for area in matching_product.geo_areas.split(",")}

    remaining = geometry
    touched_covered = False
    for code, boundary in GEOAREA_BOUNDARIES.items():
        if not boundary.intersects(geometry):
            continue
        if code in covered_geoareas:
            touched_covered = True
            if not remaining.is_empty:
                remaining = remaining.difference(boundary)

    if not touched_covered:
        return CoverageStatus.NONE
    return CoverageStatus.FULL if remaining.is_empty else CoverageStatus.PARTIAL


def resolve_seasonal_product(
    product: str, version: str, season: str
) -> LfpsProduct | None:
    """Find the live Seasonal Fuels catalog entry for a product/version/season.

    Returns the matching `LfpsProduct` (carrying its real `layer_name` and
    `season_year`), or None if LFPS isn't currently serving that
    product/version/season. This is the single matcher the API and griddle
    use to resolve the actual layer and year from the catalog instead of
    assuming the season year is `version + 1`.

    Args:
        product: Product acronym, matched case-insensitively against
            `LfpsProduct.acronym` (e.g. "fbfm40" matches "FBFM40").
        version: LANDFIRE version year (e.g. "2025") -- matched against
            LFPS's "LF{version}" format.
        season: A season code (e.g. "SP").
    """
    return next(
        (
            p
            for p in list_products()
            if p.season == season
            and p.acronym.lower() == product.lower()
            and p.version == f"LF{version}"
        ),
        None,
    )


def covers_seasonal(
    product: str, version: str, season: str, geometry: BaseGeometry
) -> CoverageStatus:
    """Check Seasonal Fuels coverage for a product/version/season against a geometry.

    Not every season is necessarily live at a given time -- e.g. fall may
    not yet be published -- so live availability is checked first, via
    `LfpsProduct.season`, before geometry is even looked at.

    Args:
        product: Product acronym, matched case-insensitively against
            `LfpsProduct.acronym` (e.g. "fbfm40" matches "FBFM40").
        version: LANDFIRE version year (e.g. "2025") -- matched against
            LFPS's "LF{version}" format.
        season: A key in SEASONAL_BOUNDARIES (e.g. "SP").
        geometry: Domain geometry, already reprojected to EPSG:5070.

    Returns:
        CoverageStatus. NO_SUCH_PRODUCT if `season` isn't currently live
        for `product`/`version`. Otherwise FULL, PARTIAL, or NONE.
    """
    if resolve_seasonal_product(product, version, season) is None:
        return CoverageStatus.NO_SUCH_PRODUCT

    boundary = SEASONAL_BOUNDARIES[season]
    if not boundary.intersects(geometry):
        return CoverageStatus.NONE
    remaining = geometry.difference(boundary)
    return CoverageStatus.FULL if remaining.is_empty else CoverageStatus.PARTIAL
