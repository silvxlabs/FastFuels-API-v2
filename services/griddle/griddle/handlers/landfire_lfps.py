"""
LANDFIRE Product Service (LFPS) job orchestration.

Submits an on-demand LFPS job and blocks until it finishes, then hands
back a path to the downloaded, unzipped result. Griddle's Cloud Run
timeout is set long enough to accommodate the wait (see
.github/workflows/griddle.yml's --timeout=1200).

landfire.py uses this module to get the file and turn it into an
aligned Dataset.
"""

import io
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from lib.alignment import resolve_alignment_destination
from lib.errors import ProcessingError
from lib.landfire import (
    SEASON_CODES,
    LfpsJobFailedError,
    download,
    poll_status,
    resolve_lf_product,
    submit_job,
)
from lib.raster import REPROJECTION_GUARD_CELLS

# Policy knobs for the LFPS submit/poll loop -- specific to this module's
# orchestration, not the LFPS client itself.
_LFPS_POLL_DELAY_SECONDS = 10
_LFPS_JOB_TIMEOUT_SECONDS = 1200  # 20 min
_LFPS_NATIVE_RESOLUTION = 30.0  # meters; all LANDFIRE products share this grid
_LFPS_AOI_PADDING_METERS = (
    500  # extra safety margin beyond the known reprojection guard
)


def _lfps_aoi_bbox(
    roi: gpd.GeoDataFrame,
    alignment: dict,
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
) -> str:
    """Compute the WGS84 "W S E N" bbox to submit as the LFPS job's AOI.

    Sized from the same destination lattice the eventual fetch will
    reproject into (via `resolve_alignment_destination`, using LANDFIRE's
    known 30m native resolution). `native` alignment defers its buffer to
    `RasterConnection.extract_window` instead of baking it into a
    destination lattice, so `extent_buffer_cells` is applied here
    directly for that case. Padded by `extract_window`'s exact
    reprojection guard margin plus extra safety margin, so LFPS always
    delivers enough for that later clip.
    """
    dest = resolve_alignment_destination(
        alignment,
        roi,
        target_grid_doc,
        _LFPS_NATIVE_RESOLUTION,
        extent_buffer_cells=extent_buffer_cells,
    )
    if "destination_transform" in dest:
        minx, miny, maxx, maxy = array_bounds(
            *dest["destination_shape"], dest["destination_transform"]
        )
        crs = dest["destination_crs"]
    else:
        buf = extent_buffer_cells * (
            alignment.get("resolution") or _LFPS_NATIVE_RESOLUTION
        )
        minx, miny, maxx, maxy = roi.total_bounds
        minx, miny, maxx, maxy = minx - buf, miny - buf, maxx + buf, maxy + buf
        crs = roi.crs

    pad = REPROJECTION_GUARD_CELLS * _LFPS_NATIVE_RESOLUTION + _LFPS_AOI_PADDING_METERS
    w, s, e, n = transform_bounds(
        crs, "EPSG:4326", minx - pad, miny - pad, maxx + pad, maxy + pad
    )
    return f"{w:.6f} {s:.6f} {e:.6f} {n:.6f}"


def _lfps_layer_name(product: str, version: str, season: str | None = None) -> str:
    """Build the LFPS layer name for a fetch request.

    Most products use "LF{version}_{PRODUCT}", e.g. "LF2025_FBFM40".

    Because LFPS names them differently, FBFM40 seasonal layers and
    annual_disturbance come from the live LFPS catalog via resolve_lf_product().
    """
    if product.lower() == "annual_disturbance":
        match = resolve_lf_product("LDist", version)
        if match is None:
            raise ProcessingError(
                code="ANNUAL_DISTURBANCE_NOT_AVAILABLE",
                message=(
                    "LANDFIRE Product Service isn't currently serving a "
                    f"Limited Annual Disturbance layer for version {version}."
                ),
            )
        return match.layer_name

    product = product.upper()
    if season is None:
        return f"LF{version}_{product}"
    if product != "FBFM40":
        raise ProcessingError(
            code="SEASONAL_NOT_SUPPORTED",
            message=f"LANDFIRE Seasonal Fuels only publishes FBFM40, not {product!r}.",
        )
    if season not in SEASON_CODES:
        raise ProcessingError(
            code="INVALID_SEASON",
            message=f"Unknown LANDFIRE season: {season!r}",
            suggestion=f"Supported seasons: {', '.join(SEASON_CODES)}",
        )
    match = resolve_lf_product(product, version, season)
    if match is None:
        raise ProcessingError(
            code="SEASONAL_NOT_AVAILABLE",
            message=(
                f"LANDFIRE Product Service isn't currently serving a {season} "
                f"seasonal FBFM40 layer for version {version}."
            ),
        )
    return match.layer_name


@contextmanager
def fetch_lfps(
    roi: gpd.GeoDataFrame,
    product: str,
    version: str,
    alignment: dict,
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
    progress: Callable[[str, int | None], None],
    season: str | None = None,
) -> Iterator[str]:
    """Submit an LFPS job for LANDFIRE `product`, wait for it,
    and yield a local path to the downloaded, unzipped GeoTIFF.

    Blocks for the whole job -- typically <30s, capped at 20 minutes.
    The temp directory holding the unzipped file is cleaned up once the
    caller's `with` block exits.

    Args:
        roi: GeoDataFrame defining the region of interest.
        product: LANDFIRE product acronym as LFPS names it (e.g. "FBFM40").
        version: LANDFIRE version year.
        alignment: Alignment specification dict, used to size the AOI
            submitted to LFPS so it covers the eventual fetch's destination.
        target_grid_doc: Loaded grid document used when
            `alignment["target"] == "grid"`.
        extent_buffer_cells: Result-grid cells of buffer around the ROI,
            also factored into the AOI size.
        progress: Progress callback, called at each major step (submit,
            poll, download).
        season: LANDFIRE season code (e.g. "SP"). `None` fetches the plain
            annual layer instead of a seasonal one.

    Yields:
        Local path to the downloaded, unzipped GeoTIFF.
    """
    layer = _lfps_layer_name(product, version, season)
    aoi = _lfps_aoi_bbox(roi, alignment, target_grid_doc, extent_buffer_cells)
    progress(f"Submitting LANDFIRE Product Service job for {layer}...", 10)
    job = submit_job([layer], aoi)

    deadline = time.monotonic() + _LFPS_JOB_TIMEOUT_SECONDS
    try:
        while job.status != "Succeeded":
            if time.monotonic() > deadline:
                raise ProcessingError(
                    code="LFPS_TIMEOUT",
                    message=(
                        f"LFPS job {job.job_id} did not finish within "
                        f"{_LFPS_JOB_TIMEOUT_SECONDS}s."
                    ),
                    suggestion="Try again later, or contact support if this persists.",
                )
            progress(f"Waiting on LANDFIRE Product Service ({job.status})...", 30)
            time.sleep(_LFPS_POLL_DELAY_SECONDS)
            job = poll_status(job.job_id)
    except LfpsJobFailedError as e:
        raise ProcessingError(code="LFPS_JOB_FAILED", message=str(e))

    progress("LANDFIRE Product Service job complete, downloading...", 70)
    zip_bytes = download(job)

    work = Path(tempfile.mkdtemp(prefix="lfps_"))
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(work)

        # LFPS zips also carry .tfw/.aux.xml/VAT sidecar files alongside
        # the one GeoTIFF -- filtering by suffix cleanly excludes those.
        tifs = list(work.glob("*.tif"))
        if not tifs:
            raise ProcessingError(
                code="LFPS_OUTPUT_INVALID",
                message="LFPS job output zip contained no GeoTIFF.",
            )

        yield str(tifs[0])
    finally:
        shutil.rmtree(work, ignore_errors=True)
