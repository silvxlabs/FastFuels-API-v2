"""
QUIC-Fire combined export handler.

Bundles canopy fuel + surface fuel + (optional) topography + (optional) SAVR
grids into a zip of trees*.dat files plus metadata.json and domain.geojson.

The handler is a pure consumer: every shape/CRS/transform decision was made
at request time by the API validator and snapshotted into
``source["resolved"]["fire_grid"]``. Per-role band selection is preserved
in ``source["<role>"]`` (each a ``{grid_id, band}`` dict).

Surface and canopy values are merged at the bottom slab (k=0) per the
additive policy locked in by the API merge fields:

- ``rhof`` (kg/m³): ``merged[0] = canopy[0] + surface_load / dz``
- ``moist`` (fraction, after dividing input % by 100):
  mass-weighted by canopy_rhof[0] and surface_rhof_layer
- ``fueldepth`` (m): ``merged[0] = surface_depth`` (canopy contributes 0)
- ``savr`` (m⁻¹, mass-weighted): converted to particle size scale (m)
  via ``2/SAVR`` before write

Output zip layout (flat):
    treesrhof.dat
    treesmoist.dat
    treesfueldepth.dat
    metadata.json
    domain.geojson
    topo.dat       (only when topography role provided)
    treesss.dat    (only when both SAVR roles provided)
"""

import json
import logging
import shutil
import tempfile
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from google.cloud import storage as gcs_storage
from scipy.io import FortranFile

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from lib.config import DOMAINS_COLLECTION, EXPORTS_BUCKET
from lib.firestore.documents import get_document

logger = logging.getLogger(__name__)

# Tiny constant guarding the divide in the mass-weighted average. Smaller than
# any plausible bulk density value (kg/m³); preserves the canopy-only and
# surface-only limits exactly when the corresponding mass term is zero.
_EPS = 1e-12


def export_quicfire(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Build a QUIC-Fire input zip and upload it to GCS."""
    fire_grid = source["resolved"]["fire_grid"]
    nx = fire_grid["nx"]
    ny = fire_grid["ny"]
    nz = fire_grid["nz"]
    dz = float(fire_grid["z_resolution"])

    grid_cache: dict[str, xr.Dataset] = {}

    def load_band(role: dict, *, rank: int) -> np.ndarray:
        grid_id = role["grid_id"]
        band = role["band"]
        if grid_id not in grid_cache:
            try:
                grid_cache[grid_id] = load_grid_zarr(grid_id)
            except Exception as e:
                raise ProcessingError(
                    code="GRID_LOAD_ERROR",
                    message=f"Failed to load grid {grid_id}: {e}",
                    suggestion="Ensure the grid exists and has completed processing.",
                    traceback=traceback.format_exc(),
                )
        ds = grid_cache[grid_id]
        if band not in ds.data_vars:
            raise ProcessingError(
                code="BAND_NOT_FOUND",
                message=f"Band '{band}' not found in grid {grid_id}",
                suggestion=f"Available bands: {list(ds.data_vars)}",
            )
        dims = ("z", "y", "x") if rank == 3 else ("y", "x")
        return ds[band].transpose(*dims).values.astype(np.float32, copy=False)

    progress("Loading canopy bands...", 10)
    canopy_rhof = load_band(source["canopy_bulk_density"], rank=3)
    canopy_moist = load_band(source["canopy_moisture"], rank=3) / 100.0

    progress("Loading surface bands...", 25)
    surf_load = load_band(source["surface_fuel_load"], rank=2)
    surf_depth = load_band(source["surface_fuel_depth"], rank=2)
    surf_moist = load_band(source["surface_moisture"], rank=2) / 100.0
    surf_rhof_layer = surf_load / dz

    if canopy_rhof.shape != (nz, ny, nx):
        raise ProcessingError(
            code="SHAPE_MISMATCH",
            message=(
                f"Canopy grid shape {canopy_rhof.shape} does not match "
                f"resolved fire grid (nz, ny, nx)=({nz}, {ny}, {nx})."
            ),
            suggestion="The API validator should have rejected this request.",
        )
    if surf_load.shape != (ny, nx):
        raise ProcessingError(
            code="SHAPE_MISMATCH",
            message=(
                f"Surface grid shape {surf_load.shape} does not match "
                f"resolved fire grid (ny, nx)=({ny}, {nx})."
            ),
            suggestion="The API validator should have rejected this request.",
        )

    progress("Stitching surface + canopy...", 40)
    rhof = canopy_rhof.copy()
    rhof[0] = canopy_rhof[0] + surf_rhof_layer

    total_rhof_k0 = canopy_rhof[0] + surf_rhof_layer + _EPS
    moist = canopy_moist.copy()
    moist[0] = (
        canopy_rhof[0] * canopy_moist[0] + surf_rhof_layer * surf_moist
    ) / total_rhof_k0

    fueldepth = np.zeros_like(canopy_rhof)
    fueldepth[0] = surf_depth

    progress("Building optional layers...", 55)
    treesss = None
    if source.get("canopy_savr") and source.get("surface_savr"):
        canopy_savr = load_band(source["canopy_savr"], rank=3)
        surf_savr = load_band(source["surface_savr"], rank=2)
        savr_arr = canopy_savr.copy()
        savr_arr[0] = (
            canopy_rhof[0] * canopy_savr[0] + surf_rhof_layer * surf_savr
        ) / total_rhof_k0
        # Convert SAVR (m⁻¹) → particle size scale (m): radius = 2/SAVR.
        # Zeros where SAVR is non-positive or NaN.
        with np.errstate(divide="ignore", invalid="ignore"):
            treesss = np.where(
                savr_arr > 0, 2.0 / np.maximum(savr_arr, _EPS), 0.0
            ).astype(np.float32)

    topo_arr = None
    if source.get("topography"):
        topo_arr = load_band(source["topography"], rank=2)

    progress("Writing output files...", 70)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "quicfire"
        out_dir.mkdir()

        try:
            _write_fortran_3d(out_dir / "treesrhof.dat", rhof)
            _write_fortran_3d(out_dir / "treesmoist.dat", moist)
            _write_fortran_3d(out_dir / "treesfueldepth.dat", fueldepth)
            if topo_arr is not None:
                _write_fortran_2d(out_dir / "topo.dat", topo_arr)
            if treesss is not None:
                _write_fortran_3d(out_dir / "treesss.dat", treesss)
            _write_metadata(out_dir / "metadata.json", export, source, fire_grid)
            _write_domain_geojson(out_dir / "domain.geojson", source["domain_id"])
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                code="QUICFIRE_WRITE_ERROR",
                message=f"Failed to write QUIC-Fire output files: {e}",
                suggestion="Check exporter logs for details.",
                traceback=traceback.format_exc(),
            )

        progress("Zipping...", 85)
        zip_base = str(Path(tmp) / "quicfire")
        zip_path = shutil.make_archive(zip_base, "zip", str(out_dir))

        progress("Uploading...", 95)
        gcs_path = _upload_zip(zip_path, export)

    return gcs_path


def _prep_3d(arr: np.ndarray) -> np.ndarray:
    """NaN→0, Y-flip, ascontiguous float32, ready for FortranFile.write_record."""
    cleaned = np.nan_to_num(arr.astype(np.float32, copy=False))
    flipped = np.flip(cleaned, axis=1)
    return np.ascontiguousarray(flipped)


def _prep_2d(arr: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(arr.astype(np.float32, copy=False))
    flipped = np.flipud(cleaned)
    return np.ascontiguousarray(flipped)


def _write_fortran_3d(path: Path, arr: np.ndarray) -> None:
    with FortranFile(str(path), "w") as f:
        f.write_record(_prep_3d(arr))


def _write_fortran_2d(path: Path, arr: np.ndarray) -> None:
    with FortranFile(str(path), "w") as f:
        f.write_record(_prep_2d(arr))


def _write_metadata(
    path: Path,
    export: dict,
    source: dict,
    fire_grid: dict,
) -> None:
    transform = fire_grid.get("transform") or []
    dx = abs(float(transform[0])) if len(transform) >= 1 else None
    dy = abs(float(transform[4])) if len(transform) >= 5 else None
    metadata = {
        "format": "quicfire",
        "exporter_version": "1",
        "completed_on": datetime.now(UTC).isoformat(),
        "fire_grid": {
            "nx": fire_grid["nx"],
            "ny": fire_grid["ny"],
            "nz": fire_grid["nz"],
            "dx": dx,
            "dy": dy,
            "dz": float(fire_grid["z_resolution"]),
            "z_origin": fire_grid.get("z_origin"),
            "transform": fire_grid.get("transform"),
            "crs": fire_grid.get("crs"),
        },
        "export_id": export.get("id"),
        "export_name": export.get("name"),
        "source": source,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _write_domain_geojson(path: Path, domain_id: str) -> None:
    """Read the domain doc from Firestore and dump its features as GeoJSON.

    Domain features are stored with JSON-stringified coordinates because
    Firestore can't represent nested arrays (see MEMORY.md). We parse them
    back to nested lists and serialize as standard GeoJSON.
    """
    try:
        _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
    except Exception as e:
        raise ProcessingError(
            code="DOMAIN_NOT_FOUND",
            message=f"Failed to load domain {domain_id}: {e}",
            suggestion="The export references a domain that no longer exists.",
            traceback=traceback.format_exc(),
        )
    doc = snapshot.to_dict() or {}
    features = doc.get("features", [])
    parsed_features = []
    for feature in features:
        f = dict(feature)
        geometry = dict(f.get("geometry") or {})
        coords = geometry.get("coordinates")
        if isinstance(coords, str):
            geometry["coordinates"] = json.loads(coords)
        f["geometry"] = geometry
        parsed_features.append(f)

    crs_name = (doc.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(parsed_features, crs=crs_name)
    gdf.to_file(str(path), driver="GeoJSON")


def _upload_zip(zip_path: str, export: dict) -> str:
    export_id = export["id"]
    filename = sanitize_filename(export.get("name", ""), ".zip")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    without_scheme = gcs_path.removeprefix("gs://")
    bucket_name, blob_path = without_scheme.split("/", 1)
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(zip_path)
    return gcs_path
