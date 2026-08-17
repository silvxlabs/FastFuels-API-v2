"""
LANDFIRE Product Service (LFPS) job orchestration.

Griddle processes one Cloud Task per invocation, so a multi-step LFPS job
(submit -> poll -> poll -> ... -> download) can't block one HTTP request.
`process_lfps_fbfm40` does one step per call and, if the job isn't done,
persists its state to Firestore and re-enqueues a delayed follow-up task
for the same grid (see `griddle.tasks`), raising `ProcessingDeferred` so
`main.py` stops without marking the grid complete or failed.

Coverage (whether LFPS actually serves a given product/version/season for
a domain) is checked by the API at request time, not here -- griddle
processing a grid means the API already decided the request was valid.
"""

import io
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from griddle import tasks
from griddle.handlers import landfire
from lib.alignment import resolve_alignment_destination
from lib.config import GRIDS_COLLECTION
from lib.errors import ProcessingDeferred, ProcessingError
from lib.firestore import update_document
from lib.landfire import (
    LANDFIRE_VERSIONS,
    SEASON_CODES,
    LfpsJobFailedError,
    download,
    poll_status,
    submit_job,
)

# Policy knobs for the LFPS submit/poll orchestration -- specific to
# griddle's re-enqueue loop, not the LFPS client itself.
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


def _lfps_fbfm40_layer_name(version: str, season: str) -> str:
    """Build the LFPS layer name for a seasonal FBFM40 request.

    E.g. version="2025", season="SP" -> "LF2025_FBFM40_SP26"
    """
    if season not in SEASON_CODES:
        raise ProcessingError(
            code="INVALID_SEASON",
            message=f"Unknown LANDFIRE season: {season!r}",
            suggestion=f"Supported seasons: {', '.join(SEASON_CODES)}",
        )
    season_year = str(int(version) + 1)[-2:]
    return f"LF{version}_FBFM40_{season}{season_year}"


def _fetch_downloaded_fbfm40(
    zip_bytes: bytes,
    roi: gpd.GeoDataFrame,
    grid_id: str,
    remove_non_burnable: list[str] | None,
    extent_buffer_cells: int,
    alignment: dict,
    target_grid_doc: dict | None,
) -> xr.Dataset:
    """Unzip an LFPS job's output and fetch it through the normal pipeline.

    LFPS zips also carry `.tfw`/`.aux.xml`/VAT sidecar files alongside the
    one GeoTIFF -- filtering by suffix cleanly excludes those. Uses
    `mkdtemp()` + `finally: rmtree()` rather than `TemporaryDirectory()`,
    since `landfire.fetch_fbfm40` needs the file to still exist while it
    reads it -- past the point where a `with`-block would already have
    cleaned it up.
    """
    work = Path(tempfile.mkdtemp(prefix=f"lfps_{grid_id}_"))
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(work)

        tifs = list(work.glob("*.tif"))
        if not tifs:
            raise ProcessingError(
                code="LFPS_OUTPUT_INVALID",
                message="LFPS job output zip contained no GeoTIFF.",
            )

        return landfire.fetch_fbfm40(
            roi,
            remove_non_burnable=remove_non_burnable,
            extent_buffer_cells=extent_buffer_cells,
            alignment=alignment,
            target_grid_doc=target_grid_doc,
            url=str(tifs[0]),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process_lfps_fbfm40(
    grid: dict,
    roi: gpd.GeoDataFrame,
    source: dict,
    target_grid_doc: dict | None,
    progress: Callable[[str, int | None], None],
) -> xr.Dataset:
    """Submit or poll an LFPS job for a seasonal FBFM40 grid.

    Griddle processes one Cloud Task per invocation, so a job that isn't
    done yet is handled by persisting state to Firestore, enqueuing a
    delayed follow-up task for the same grid, and raising
    `ProcessingDeferred` so `main.py` stops without marking the grid
    complete or failed. The next invocation re-reads `grid["lfps_job"]`
    fresh from Firestore and continues from there.
    """
    version = source.get("version", LANDFIRE_VERSIONS["fbfm40"]["default"])
    season = source["season"]
    extent_buffer_cells = source.get("extent_buffer_cells", 0)
    alignment = source.get("alignment") or {"target": "domain"}
    remove_non_burnable = source.get("remove_non_burnable")

    lfps_job = grid.get("lfps_job")

    if lfps_job is None:
        layer = _lfps_fbfm40_layer_name(version, season)
        aoi = _lfps_aoi_bbox(roi, alignment, target_grid_doc, extent_buffer_cells)
        progress(f"Submitting LANDFIRE Product Service job for {layer}...", 10)
        job = submit_job([layer], aoi)
        update_document(
            GRIDS_COLLECTION,
            grid["id"],
            {"lfps_job": {"job_id": job.job_id, "submitted_at": datetime.now(UTC)}},
        )
        tasks.enqueue_delayed_task(grid["id"], _LFPS_POLL_DELAY_SECONDS)
        raise ProcessingDeferred(
            f"LFPS job {job.job_id} submitted for grid {grid['id']}"
        )

    job_id = lfps_job["job_id"]
    submitted_at = lfps_job["submitted_at"]

    try:
        job = poll_status(job_id)
    except LfpsJobFailedError as e:
        raise ProcessingError(code="LFPS_JOB_FAILED", message=str(e))

    if job.status == "Succeeded":
        progress("LANDFIRE Product Service job complete, downloading...", 70)
        zip_bytes = download(job)
        return _fetch_downloaded_fbfm40(
            zip_bytes,
            roi,
            grid["id"],
            remove_non_burnable,
            extent_buffer_cells,
            alignment,
            target_grid_doc,
        )

    elapsed = (datetime.now(UTC) - submitted_at).total_seconds()
    if elapsed > _LFPS_JOB_TIMEOUT_SECONDS:
        raise ProcessingError(
            code="LFPS_TIMEOUT",
            message=f"LFPS job {job_id} did not finish within {_LFPS_JOB_TIMEOUT_SECONDS}s.",
            suggestion="Try again later, or contact support if this persists.",
        )

    progress(f"Waiting on LANDFIRE Product Service ({job.status})...", 30)
    tasks.enqueue_delayed_task(grid["id"], _LFPS_POLL_DELAY_SECONDS)
    raise ProcessingDeferred(f"LFPS job {job_id} still {job.status}, re-enqueued")
