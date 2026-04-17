"""Cloud Function entry for treevox — voxelize a tree inventory into a 3D zarr grid.

Absorbs what griddle splits across `main.py + dispatch.py + errors.py +
handlers/*.py`. With one source today a single file is clearer than a
scaffold; a `handlers/` subpackage can be introduced when a second source
type (lidar, treelist, ...) arrives.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import functions_framework
import pandas as pd
from flask import Request

from lib.config import DOMAINS_COLLECTION, GRIDS_COLLECTION
from lib.domain_utils import EmptyDomainError, InvalidGeometryError, parse_domain_gdf
from lib.firestore import DocumentNotFoundError, get_document, update_document
from lib.gcs import download_file
from treevox import storage, voxelize
from treevox._worker import run as worker_run

# Errors


class CancelledException(Exception):
    """Raised when a grid document is deleted during processing (user cancelled)."""


@dataclass
class ProcessingError(Exception):
    """Structured error with a user-friendly message.

    Codes emitted by treevox:
      INVENTORY_NOT_FOUND, EMPTY_INVENTORY, INVALID_RESOLUTION,
      UNKNOWN_SOURCE, VOXELIZATION_FAILED, DOMAIN_NOT_FOUND,
      EMPTY_DOMAIN, INVALID_GEOMETRY.
    """

    code: str
    message: str
    suggestion: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict:
        result: dict = {"code": self.code, "message": self.message}
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.traceback:
            result["traceback"] = self.traceback
        return result


# Logging


class StructuredLogHandler(logging.Handler):
    """JSON log handler for Cloud Logging; carries grid_id / domain_id via extra."""

    def emit(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("grid_id", "domain_id"):
            value = getattr(record, field, None)
            if value:
                log_entry[field] = value
        if record.exc_info:
            log_entry["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        print(json.dumps(log_entry), file=sys.stderr)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(StructuredLogHandler())


UNEXPECTED_FAILURE_MESSAGE = (
    "Job failed unexpectedly. Please try again or contact the development team."
)


# Firestore helpers


def load_grid(grid_id: str) -> dict:
    """Load grid document from Firestore."""
    _, snapshot = get_document(GRIDS_COLLECTION, grid_id)
    return snapshot.to_dict()


def update_progress(grid_id: str, message: str, percent: int | None = None) -> None:
    progress: dict = {"message": message}
    if percent is not None:
        progress["percent"] = percent
    try:
        update_document(
            GRIDS_COLLECTION,
            grid_id,
            {"progress": progress, "modified_on": datetime.now(UTC)},
        )
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def update_status(
    grid_id: str,
    status: str,
    georeference: dict | None = None,
    error: dict | None = None,
) -> None:
    data: dict = {"status": status, "modified_on": datetime.now(UTC)}
    if status == "completed":
        data["progress"] = {"message": "Complete", "percent": 100}
    elif status == "failed":
        data["progress"] = {"message": "Failed", "percent": 100}
    if georeference is not None:
        data["georeference"] = georeference
    if error is not None:
        data["error"] = error
    try:
        update_document(GRIDS_COLLECTION, grid_id, data)
    except DocumentNotFoundError:
        raise CancelledException(f"Grid {grid_id} was cancelled")


def make_progress_callback(grid_id: str) -> Callable[[str, int | None], None]:
    def callback(message: str, percent: int | None = None):
        update_progress(grid_id, message, percent)

    return callback


def _load_domain(domain_id: str):
    try:
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
        return parse_domain_gdf(snapshot.to_dict())
    except DocumentNotFoundError:
        raise ProcessingError(
            code="DOMAIN_NOT_FOUND",
            message=f"Domain {domain_id} not found.",
            suggestion="Ensure the domain exists before creating a grid.",
        )
    except EmptyDomainError:
        raise ProcessingError(
            code="EMPTY_DOMAIN",
            message="Domain has no geometry.",
            suggestion="Create a domain with at least one polygon feature.",
        )
    except InvalidGeometryError as e:
        raise ProcessingError(
            code="INVALID_GEOMETRY",
            message=str(e),
            suggestion="Ensure the domain has valid GeoJSON geometry.",
        )


# Inventory parquet IO


REQUIRED_COLUMNS = [
    "x",
    "y",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "height",
    "crown_ratio",
]


def download_inventory(inventory_id: str, tmpdir: str) -> pd.DataFrame:
    """Download and parse the inventory parquet from INVENTORIES_BUCKET."""
    from lib.config import INVENTORIES_BUCKET

    gcs_path = f"gs://{INVENTORIES_BUCKET}/{inventory_id}"
    local_path = os.path.join(tmpdir, "inventory.parquet")
    try:
        download_file(gcs_path, local_path)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Inventory {inventory_id} not found at {gcs_path}.",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e
    except Exception as e:
        # gcsfs raises FileNotFoundError or variants. Treat any I/O failure as
        # a missing inventory for user-facing purposes.
        raise ProcessingError(
            code="INVENTORY_NOT_FOUND",
            message=f"Could not read inventory {inventory_id}: {e}",
            suggestion="Verify the inventory ID exists and has completed processing.",
        ) from e
    return pd.read_parquet(local_path)


def filter_live(df: pd.DataFrame, biomass_column: str | None = None) -> pd.DataFrame:
    """Drop null required columns and retain only live trees (fia_status_code == 1)."""
    required = list(REQUIRED_COLUMNS)
    if biomass_column:
        required.append(biomass_column)
    df = df.dropna(subset=required)
    df = df[df["fia_status_code"] == 1].reset_index(drop=True)
    return df


def assign_tree_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a unique int32 `tree_id` per row."""
    out = df.copy()
    import numpy as np

    out["tree_id"] = np.arange(len(out), dtype="int32")
    return out


# Handler


@dataclass
class VoxelizationResult:
    gcs_path: str
    georeference: dict
    chunk_shape: list[int]


DEFAULT_MAX_WORKERS = 4
WORKER_MEMORY_ESTIMATE_BYTES = 500 * 1024 * 1024  # ~500 MB budget per worker


def _pick_worker_count() -> int:
    """Cap workers by available CPU and memory.

    Cloud Run container memory fluctuates; we read MemAvailable at startup
    rather than MemTotal so we don't oversubscribe.
    """
    try:
        cpu_cap = max(1, min(DEFAULT_MAX_WORKERS, len(os.sched_getaffinity(0))))
    except AttributeError:
        cpu_cap = max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(
                int(ln.split()[1]) for ln in f if ln.startswith("MemAvailable")
            )
        mem_cap = max(1, (avail_kb * 1024) // WORKER_MEMORY_ESTIMATE_BYTES)
    except (FileNotFoundError, StopIteration):
        # Non-Linux (local dev): fall back to CPU cap only.
        mem_cap = DEFAULT_MAX_WORKERS
    return int(min(cpu_cap, mem_cap))


def _build_payloads(
    batch: list[tuple[int, int]],
    union_ds,
    union_y: slice,
    union_x: slice,
    df: pd.DataFrame,
    requested_keys: list[str],
    dims: dict,
    source_config: dict,
    grid_id: str,
    chunk_xy: int,
) -> list[dict]:
    """Split a halo-extended union Dataset into per-chunk worker payloads.

    Each payload carries numpy buffers and scalar grid params only (no xarray
    or custom objects).
    """
    import numpy as np

    ny, nx, nz = dims["ny"], dims["nx"], dims["nz"]
    hr, vr = dims["hr"], dims["vr"]
    x_origin, y_origin = dims["x_origin"], dims["y_origin"]

    payloads: list[dict] = []
    for row, col in batch:
        chunk_y, chunk_x = voxelize.chunk_slice(
            (row, col), ny, nx, chunk_xy, overlap_cells=voxelize.OVERLAP_CELLS
        )
        # Relative slice into the union buffer.
        rel_y = slice(chunk_y.start - union_y.start, chunk_y.stop - union_y.start)
        rel_x = slice(chunk_x.start - union_x.start, chunk_x.stop - union_x.start)

        buffers: dict = {}
        for key in requested_keys:
            dtype, fill = storage.BAND_SPECS[key]
            chunk_shape = (
                nz,
                chunk_y.stop - chunk_y.start,
                chunk_x.stop - chunk_x.start,
            )
            existing = union_ds[key].values[:, rel_y, rel_x]
            buf = np.array(existing, dtype=dtype, copy=True)
            if buf.shape != chunk_shape:
                resized = np.full(chunk_shape, fill, dtype=dtype)
                z_n = min(chunk_shape[0], buf.shape[0])
                y_n = min(chunk_shape[1], buf.shape[1])
                x_n = min(chunk_shape[2], buf.shape[2])
                resized[:z_n, :y_n, :x_n] = buf[:z_n, :y_n, :x_n]
                buf = resized
            buffers[key] = buf

        trees_in_chunk = df[(df["row_chunk"] == row) & (df["col_chunk"] == col)]
        rng_seed = abs(hash((grid_id, row, col))) & 0xFFFFFFFF

        payloads.append(
            {
                "chunk_location": (row, col),
                "buffers": buffers,
                "trees": trees_in_chunk,
                "hr": hr,
                "vr": vr,
                "x_origin": x_origin,
                "y_origin": y_origin,
                "source_config": source_config,
                "chunk_y_start": chunk_y.start,
                "chunk_x_start": chunk_x.start,
                "y_slice": chunk_y,
                "x_slice": chunk_x,
                "rng_seed": int(rng_seed),
            }
        )
    return payloads


def handle_inventory(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> VoxelizationResult:
    """Orchestrate the full voxelization job: load → spec → batched mp → write."""
    source = grid["source"]
    grid_id = grid["id"]
    path = storage.gcs_path(grid_id)

    progress("Loading inventory...", 5)
    with tempfile.TemporaryDirectory() as tmp:
        df = download_inventory(source["source_inventory_id"], tmp)
    df = filter_live(df, source.get("biomass_column"))
    df = assign_tree_ids(df)

    if df.empty:
        raise ProcessingError(
            code="EMPTY_INVENTORY",
            message="Inventory has no live trees with complete measurements.",
            suggestion="Verify the inventory contains rows with fia_status_code == 1 "
            "and non-null dbh / height / crown_ratio.",
        )

    progress("Computing grid extent...", 10)
    try:
        dims = voxelize.compute_grid_dimensions(domain_gdf, df, source["resolution"])
    except voxelize.InvalidResolutionError as e:
        raise ProcessingError(
            code="INVALID_RESOLUTION",
            message=str(e),
            suggestion="Check the grid resolution against the domain bounds and "
            "inventory tree heights.",
        )

    requested_keys = [b["key"] for b in grid["bands"]]
    hr, vr = dims["hr"], dims["vr"]
    nx, ny, nz = dims["nx"], dims["ny"], dims["nz"]
    chunk_xy = max(1, int(voxelize.CHUNK_LENGTH_METERS / hr))
    chunk_xy = min(chunk_xy, nx, ny)
    chunk_shape: tuple[int, int, int] = (nz, chunk_xy, chunk_xy)

    progress("Initializing zarr store...", 15)
    storage.init_store(
        path,
        x_coords=dims["x_coords"],
        y_coords=dims["y_coords"],
        z_coords=dims["z_coords"],
        hr=hr,
        vr=vr,
        crs=dims["crs"],
        z_origin=dims["z_origin"],
        requested_keys=requested_keys,
        chunk_shape=chunk_shape,
    )

    # Pre-process trees: cache keys, chunk assignment, height sort.
    df["_cache_key"] = voxelize.compute_cache_keys(df)
    df = voxelize.assign_trees_to_chunks(
        df, dims["x_origin"], dims["y_origin"], hr, nx, ny, chunk_xy
    )

    num_row_chunks = int(math.ceil(ny / chunk_xy))
    num_col_chunks = int(math.ceil(nx / chunk_xy))
    chunk_locations = sorted(
        [(r, c) for r in range(num_row_chunks) for c in range(num_col_chunks)],
        key=lambda loc: (loc[0] // 2, loc[1] // 2, loc[0], loc[1]),
    )

    num_workers = _pick_worker_count()
    batch_size = max(1, num_workers)
    num_batches = max(1, int(math.ceil(len(chunk_locations) / batch_size)))
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        for i in range(num_batches):
            batch = chunk_locations[i * batch_size : (i + 1) * batch_size]
            if not batch:
                continue

            union_y, union_x = voxelize.batch_union_slices(
                batch, ny, nx, chunk_xy, overlap_cells=voxelize.OVERLAP_CELLS
            )
            union_ds = storage.read_union(path, union_y, union_x)
            assert not union_ds.chunks, (
                "read_union must materialize — workers cannot receive lazy dask arrays"
            )

            payloads = _build_payloads(
                batch,
                union_ds,
                union_y,
                union_x,
                df,
                requested_keys,
                dims,
                source,
                grid_id,
                chunk_xy,
            )

            results = pool.map(worker_run, payloads)

            for r in results:
                if "error" in r:
                    raise ProcessingError(
                        code="VOXELIZATION_FAILED",
                        message=f"Chunk {r['chunk_location']} failed during voxelization.",
                        suggestion="Check service logs for the worker traceback.",
                        traceback=r["error"],
                    )

            merged = storage.masked_merge(union_ds, results, union_y, union_x)
            storage.write_union(path, merged, union_y, union_x)

            pct = 15 + int(75 * (i + 1) / num_batches)
            progress(f"Voxelizing batch {i + 1}/{num_batches}...", pct)

    progress("Finalizing...", 95)
    storage.consolidate_metadata(path)

    georeference = {
        "crs": dims["crs"],
        "transform": list(dims["transform"]),
        "shape": [nz, ny, nx],
        "z_origin": dims["z_origin"],
        "z_resolution": vr,
    }

    return VoxelizationResult(
        gcs_path=path,
        georeference=georeference,
        chunk_shape=list(chunk_shape),
    )


# Dispatch


def dispatch_handler(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> VoxelizationResult:
    """Route on grid['source']['name']. Single-source today; extensible later."""
    source_name = grid["source"]["name"]
    match source_name:
        case "inventory":
            return handle_inventory(grid, domain_gdf, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown tree grid source: {source_name!r}",
                suggestion="Supported sources today: 'inventory'.",
            )


# HTTP entry


@functions_framework.http
def process_grid_request(request: Request):
    """Cloud Tasks HTTP trigger — expects {"id": grid_id}.

    Retry semantics (via `X-CloudTasks-TaskRetryCount`):
    - First attempt processes normally.
    - Any retry attempt marks the grid as failed and returns 200 to stop retries.
    """
    data = request.get_json(silent=True)
    grid_id = data.get("id") if data else None

    if not grid_id:
        logger.error("No id in request body")
        return "Missing id", 400

    retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
    if retry_count > 0:
        logger.error(
            "Failed on previous attempt, marking as failed",
            extra={"grid_id": grid_id},
        )
        try:
            update_status(
                grid_id,
                "failed",
                error={
                    "code": "UNEXPECTED_FAILURE",
                    "message": UNEXPECTED_FAILURE_MESSAGE,
                },
            )
            storage.delete_zarr(storage.gcs_path(grid_id))
        except (CancelledException, DocumentNotFoundError):
            pass
        return "OK", 200

    logger.info("Processing started", extra={"grid_id": grid_id})

    try:
        grid = load_grid(grid_id)
    except DocumentNotFoundError:
        logger.info("Grid not found (already deleted?)", extra={"grid_id": grid_id})
        return "OK", 200

    domain_id = grid.get("domain_id")
    ids = {"grid_id": grid_id, "domain_id": domain_id}

    try:
        update_status(grid_id, "running")
    except CancelledException:
        logger.info("Cancelled before processing started", extra=ids)
        return "OK", 200

    try:
        domain_gdf = _load_domain(grid["domain_id"])
        progress_callback = make_progress_callback(grid_id)
        result = dispatch_handler(grid, domain_gdf, progress_callback)

        update_document(GRIDS_COLLECTION, grid_id, {"chunk_shape": result.chunk_shape})
        update_status(grid_id, "completed", georeference=result.georeference)

        logger.info("Processing complete", extra=ids)
        return "OK", 200

    except CancelledException:
        logger.info("Cancelled during processing", extra=ids)
        storage.delete_zarr(storage.gcs_path(grid_id))
        return "OK", 200

    except ProcessingError as e:
        logger.error(f"Processing failed: {e.code} - {e.message}", extra=ids)
        # Clean up partial zarr store before marking failed.
        storage.delete_zarr(storage.gcs_path(grid_id))
        try:
            update_status(grid_id, "failed", error=e.to_dict())
        except CancelledException:
            pass
        return "OK", 200

    except Exception as e:
        logger.exception(f"Unexpected error: {e}", extra=ids)
        storage.delete_zarr(storage.gcs_path(grid_id))
        return "Internal error", 500


# Local dev harness


class MockRequest:
    """Simple mock request for local testing via `GRID_ID=... uv run treevox/main.py`."""

    def __init__(self, data: dict, headers: dict | None = None):
        self._json = data
        self.headers = headers or {}

    def get_json(self, silent: bool = False):
        return self._json


if __name__ == "__main__":
    grid_id = os.environ.get("GRID_ID")
    if grid_id:
        request = MockRequest({"id": grid_id})
        response, status_code = process_grid_request(request)
        print(f"Response: {response}, Status: {status_code}")
    else:
        print("Set GRID_ID environment variable for local testing")
