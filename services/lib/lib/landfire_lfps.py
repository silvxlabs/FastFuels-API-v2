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

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from lib.config import LANDFIRE_USER_EMAIL

logger = logging.getLogger(__name__)

BASE_URL = "https://lfps.usgs.gov/api"

# LFPS's own timeout guidance puts typical jobs at 12s-7min and a hard fail at
# 2 hours; this is just the HTTP request timeout for each individual call.
_REQUEST_TIMEOUT = 60

# How long a loaded product list is reused, in seconds. LFPS coverage doesn't
# change intra-day, so a short TTL is only there to eventually pick up a new
# product/version without a process restart.
LFPS_PRODUCTS_TTL_SECONDS = float(os.getenv("LFPS_PRODUCTS_TTL_SECONDS", 3600))

_products: list["LfpsProduct"] | None = None
_products_fetched_on: datetime | None = None


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

    _products = [
        LfpsProduct(
            layer_name=p["layerName"],
            product_name=p["productName"],
            theme=p["theme"],
            acronym=p["acronym"],
            version=p["version"],
            conus=p["conus"],
            geo_areas=p["geoAreas"],
        )
        for p in raw_products
    ]
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
