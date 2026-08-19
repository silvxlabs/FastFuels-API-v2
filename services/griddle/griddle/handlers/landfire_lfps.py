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
from shapely.geometry import box

from lib.alignment import resolve_alignment_destination
from lib.errors import ProcessingError
from lib.landfire import SEASON_CODES, LfpsJobFailedError, download, poll_status, submit_job

# Policy knobs for the LFPS submit/poll loop -- specific to this module's
# orchestration, not the LFPS client itself.
_LFPS_POLL_DELAY_SECONDS = 10
_LFPS_JOB_TIMEOUT_SECONDS = 1200  # 20 min
_LFPS_NATIVE_RESOLUTION = 30.0  # meters; all LANDFIRE products share this grid
_LFPS_AOI_PADDING_METERS = 500  # guard margin beyond the alignment destination


def _lfps_aoi_bbox(
    roi: gpd.GeoDataFrame,
    alignment: dict,
    target_grid_doc: dict | None,
    extent_buffer_cells: int,
) -> str:
    """Compute the WGS84 "W S E N" bbox to submit as the LFPS job's AOI.

    Sized from the same destination lattice the eventual fetch will
    reproject into (via `resolve_alignment_destination`, using LANDFIRE's
    known 30m native resolution), padded generously so the reprojection
    guard cells `RasterConnection.extract_window` applies always land
    inside what LFPS actually delivers.
    """
    dest = resolve_alignment_destination(
        alignment,
        roi,
        target_grid_doc,
        _LFPS_NATIVE_RESOLUTION,
        extent_buffer_cells=extent_buffer_cells,
    )
    if "destination_transform" in dest:
        transform = dest["destination_transform"]
        h, w = dest["destination_shape"]
        minx, maxy = transform.c, transform.f
        maxx, miny = minx + w * transform.a, maxy + h * transform.e
        bounds_crs = dest["destination_crs"]
    else:
        minx, miny, maxx, maxy = roi.total_bounds
        bounds_crs = roi.crs

    pad = _LFPS_AOI_PADDING_METERS
    geom = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    bbox_gdf = gpd.GeoDataFrame(geometry=[geom], crs=bounds_crs)
    w, s, e, n = bbox_gdf.to_crs("EPSG:4326").total_bounds
    return f"{w:.6f} {s:.6f} {e:.6f} {n:.6f}"


def _lfps_layer_name(product: str, version: str, season: str | None = None) -> str:
    """Build the LFPS layer name for a fuel-model request.

    E.g. product="FBFM40", version="2025", season="SP" -> "LF2025_FBFM40_SP26".
    With no season, returns the plain annual layer -> "LF2025_FBFM40".
    """
    product = product.upper()
    if season is None:
        return f"LF{version}_{product}"
    if product != "FBFM40":
        raise ProcessingError(
            code="SEASONAL_NOT_SUPPORTED",
            message = f"LANDFIRE Seasonal Fuels only publishes FBFM40, not {product!r}.",
        )
    if season not in SEASON_CODES:
        raise ProcessingError(
            code="INVALID_SEASON",
            message=f"Unknown LANDFIRE season: {season!r}",
            suggestion=f"Supported seasons: {', '.join(SEASON_CODES)}",
        )
    season_year = str(int(version) + 1)[-2:]
    return f"LF{version}_FBFM40_{season}{season_year}"


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