"""
QUIC-Fire combined export handler.

Bundles canopy fuel + surface fuel + (optional) topography + (optional) SAVR
grids into a zip of trees*.dat files plus metadata.json and domain.geojson.

The handler is a pure consumer: every shape/CRS/transform decision was made
at request time by the API validator and snapshotted into
``source["georeference"]``. Per-role band selection is preserved in
``source["<role>"]`` (each a ``{grid_id, band}`` dict).

Surface and canopy values are merged at the bottom slab (k=0):

- ``rhof`` (kg/m**3): ``merged[0] = canopy[0] + surface_load / dz``
- ``moist`` (fraction, after dividing input % by 100): ``max`` of canopy[0]
  and surface (v1-parity default); ``weighted_avg`` is opt-in via
  ``source["moist_merge"]``
- ``fueldepth`` (m): ``merged[0] = surface_depth`` (canopy contributes 0)
- ``savr`` (1/m, mass-weighted): converted to particle size scale (m)
  via ``2/SAVR`` before write

Oversized role grids are cropped to the fire-grid extent by integer
slicing — never resampled.

Canopy bands are read in z-slabs and the zip is streamed straight to GCS, so
peak memory scales with one slab, not with the fire grid.

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
import tempfile
import traceback
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from google.cloud import storage as gcs_storage

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from lib.config import DOMAINS_COLLECTION, EXPORTS_BUCKET
from lib.firestore.documents import get_document

logger = logging.getLogger(__name__)

# Cells per canopy z-slab (~64 MB as float32). Peak memory holds roughly ten
# slab-sized arrays; smaller slabs mean more passes over the zarr chunks.
_SLAB_CELLS = 16_000_000


def export_quicfire(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Build a QUIC-Fire input zip and stream it to GCS."""
    georeference = source["georeference"]
    nz, ny, nx = (int(n) for n in georeference["shape"])
    transform = [float(c) for c in georeference["transform"]]
    dx = abs(transform[0])
    dz = float(georeference["z_resolution"])
    fire_minx = transform[2]
    fire_maxy = transform[5]

    grid_cache: dict[str, xr.Dataset] = {}

    def open_band(role: dict, *, rank: int) -> xr.DataArray:
        """Lazily open a band cropped to the fire-grid extent.

        The validator already enforced lattice alignment and coverage, so the
        offsets here are integers within tolerance — `round` cleans the
        floating-point residual.
        """
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
        da = ds[band].transpose(*dims)

        # x coords ascend (west→east), y coords descend (north→south).
        # Coordinates are cell centers; offset back to cell origin by dx/2.
        role_minx = float(ds.x.values[0]) - dx / 2
        role_maxy = float(ds.y.values[0]) + dx / 2
        i0 = round((fire_minx - role_minx) / dx)
        j0 = round((role_maxy - fire_maxy) / dx)
        return da.isel(y=slice(j0, j0 + ny), x=slice(i0, i0 + nx))

    def read(da: xr.DataArray, **isel) -> np.ndarray:
        """Read a window of a band as float32.

        Nodata (NaN or the band's ``rio.nodata``) becomes 0 so a missing
        value contributes nothing to the k=0 merge.
        """
        raw = da.isel(**isel).values
        arr = np.nan_to_num(raw.astype(np.float32))
        if da.rio.nodata is not None:
            arr[raw == da.rio.nodata] = 0.0
        return arr

    # Open every role before the upload starts, so a missing grid or band
    # fails without touching GCS.
    canopy_rhof = open_band(source["canopy_bulk_density"], rank=3)
    canopy_moist = open_band(source["canopy_moisture"], rank=3)
    canopy_savr = None
    if source.get("canopy_savr") and source.get("surface_savr"):
        canopy_savr = open_band(source["canopy_savr"], rank=3)

    progress("Loading surface bands...", 10)
    surf_load = read(open_band(source["surface_fuel_load"], rank=2))
    surf_depth = read(open_band(source["surface_fuel_depth"], rank=2))
    surf_moist = read(open_band(source["surface_moisture"], rank=2)) / 100.0
    surf_rhof_layer = surf_load / dz
    surf_savr = None
    if canopy_savr is not None:
        surf_savr = read(open_band(source["surface_savr"], rank=2))
    topo_arr = None
    if source.get("topography"):
        topo_arr = read(open_band(source["topography"], rank=2))

    # The k=0 moisture and SAVR merges weight by the canopy bulk density there.
    canopy_rhof_k0 = read(canopy_rhof, z=0)
    total_rhof_k0 = canopy_rhof_k0 + surf_rhof_layer

    slab_nz = max(1, _SLAB_CELLS // (ny * nx))
    z_slabs = [slice(k, min(k + slab_nz, nz)) for k in range(0, nz, slab_nz)]

    def rhof_slabs() -> Iterator[np.ndarray]:
        for zs in z_slabs:
            slab = read(canopy_rhof, z=zs)
            if zs.start == 0:
                slab[0] = slab[0] + surf_rhof_layer
            yield slab

    def moist_slabs() -> Iterator[np.ndarray]:
        for zs in z_slabs:
            slab = read(canopy_moist, z=zs) / 100.0
            if zs.start == 0:
                if source.get("moist_merge", "max") == "max":
                    slab[0] = np.maximum(slab[0], surf_moist)
                else:
                    numerator = canopy_rhof_k0 * slab[0] + surf_rhof_layer * surf_moist
                    with np.errstate(divide="ignore", invalid="ignore"):
                        slab[0] = np.where(
                            total_rhof_k0 > 0, numerator / total_rhof_k0, 0.0
                        )
            yield slab

    def fueldepth_slabs() -> Iterator[np.ndarray]:
        for zs in z_slabs:
            slab = np.zeros((zs.stop - zs.start, ny, nx), dtype=np.float32)
            if zs.start == 0:
                slab[0] = surf_depth
            yield slab

    def treesss_slabs() -> Iterator[np.ndarray]:
        for zs in z_slabs:
            savr = read(canopy_savr, z=zs)
            with np.errstate(divide="ignore", invalid="ignore"):
                if zs.start == 0:
                    numerator = canopy_rhof_k0 * savr[0] + surf_rhof_layer * surf_savr
                    savr[0] = np.where(
                        total_rhof_k0 > 0, numerator / total_rhof_k0, 0.0
                    )
                # SAVR (1/m) → particle size scale (m): 2 / SAVR. Zero where
                # SAVR is non-positive or NaN — those cells carry no
                # resolvable fuel geometry.
                yield np.where(savr > 0, 2.0 / savr, 0.0).astype(np.float32)

    nbytes_3d = nz * ny * nx * np.dtype(np.float32).itemsize
    with (
        _open_upload(export) as (gcs_path, fp),
        zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zf,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            geojson_path = Path(tmp) / "domain.geojson"
            _write_domain_geojson(geojson_path, source["domain_id"])
            zf.write(geojson_path, "domain.geojson")

        try:
            progress("Writing treesrhof.dat...", 20)
            _write_record(zf, "treesrhof.dat", nbytes_3d, map(_prep_3d, rhof_slabs()))
            progress("Writing treesmoist.dat...", 40)
            _write_record(zf, "treesmoist.dat", nbytes_3d, map(_prep_3d, moist_slabs()))
            progress("Writing treesfueldepth.dat...", 55)
            _write_record(
                zf, "treesfueldepth.dat", nbytes_3d, map(_prep_3d, fueldepth_slabs())
            )
            if canopy_savr is not None:
                progress("Writing treesss.dat...", 65)
                _write_record(
                    zf, "treesss.dat", nbytes_3d, map(_prep_3d, treesss_slabs())
                )
            if topo_arr is not None:
                _write_record(zf, "topo.dat", topo_arr.size * 4, [_prep_2d(topo_arr)])
        except (ProcessingError, FileNotFoundError):
            raise
        except Exception as e:
            raise ProcessingError(
                code="QUICFIRE_WRITE_ERROR",
                message=f"Failed to write QUIC-Fire output files: {e}",
                suggestion="Check exporter logs for details.",
                traceback=traceback.format_exc(),
            )

        zf.writestr("metadata.json", _metadata_json(export, source, georeference))
        progress("Finishing upload...", 85)

    return gcs_path


def _prep_3d(arr: np.ndarray) -> np.ndarray:
    """NaN→0, Y-flip, ascontiguous float32, ready for a Fortran record."""
    cleaned = np.nan_to_num(arr.astype(np.float32, copy=False))
    flipped = np.flip(cleaned, axis=1)
    return np.ascontiguousarray(flipped)


def _prep_2d(arr: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(arr.astype(np.float32, copy=False))
    flipped = np.flipud(cleaned)
    return np.ascontiguousarray(flipped)


def _write_record(
    zf: zipfile.ZipFile, name: str, nbytes: int, parts: Iterable[np.ndarray]
) -> None:
    """Write one Fortran unformatted sequential record as a zip member.

    Byte-identical to ``scipy.io.FortranFile.write_record`` on the
    concatenated parts: a uint32 byte-count marker, the data, the marker
    again.
    """
    marker = np.array([nbytes], dtype=np.uint32).tobytes()
    written = 0
    # A member can exceed 2 GiB, and a streamed zip can't be patched to
    # ZIP64 after the fact.
    with zf.open(name, "w", force_zip64=True) as f:
        f.write(marker)
        for part in parts:
            f.write(memoryview(part).cast("B"))
            written += part.nbytes
        f.write(marker)
    if written != nbytes:
        raise RuntimeError(f"{name}: wrote {written} bytes, expected {nbytes}")


def _fire_grid_spec(georeference: dict) -> dict:
    """Expand the stored `Georeference3D` into the metadata.json `fire_grid` block.

    The sidecar is a file format read by whoever unzips the export, not the API
    contract, so it keeps its explicit cell-count / cell-size shape even though
    the API stores the equivalent georeference.
    """
    nz, ny, nx = (int(n) for n in georeference["shape"])
    transform = [float(c) for c in georeference["transform"]]
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": abs(transform[0]),
        "dy": abs(transform[4]),
        "dz": float(georeference["z_resolution"]),
        "transform": transform,
        "z_origin": float(georeference["z_origin"]),
        "crs": georeference["crs"],
    }


def _metadata_json(export: dict, source: dict, georeference: dict) -> str:
    metadata = {
        "format": "quicfire",
        "exporter_version": "1",
        "completed_on": datetime.now(UTC).isoformat(),
        "fire_grid": _fire_grid_spec(georeference),
        "export_id": export.get("id"),
        "export_name": export.get("name"),
        "source": source,
    }
    return json.dumps(metadata, indent=2, default=str)


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


@contextmanager
def _open_upload(export: dict) -> Iterator[tuple[str, BinaryIO]]:
    """Open a streaming writer for the export zip in GCS.

    An exception inside the block cancels the upload, so no partial object
    is left behind.
    """
    export_id = export["id"]
    filename = sanitize_filename(export.get("name", ""), ".zip")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    without_scheme = gcs_path.removeprefix("gs://")
    bucket_name, blob_path = without_scheme.split("/", 1)
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    # zipfile flushes when it finishes; the GCS writer raises on flush()
    # unless told to ignore it.
    with blob.open("wb", content_type="application/zip", ignore_flush=True) as fp:
        yield gcs_path, fp
