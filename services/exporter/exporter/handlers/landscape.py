"""
Landscape combined export handler.

Assembles terrain + surface fuel model + canopy grids into an 8-band
LANDFIRE-style landscape GeoTIFF for operational fire behavior tools
(FlamMap, IFTDSS, WFDSS).

The handler is a pure consumer: every shape/CRS/transform decision was made
at request time by the API validator and snapshotted into
``source["resolved"]["landscape_grid"]``. Per-role band selection is
preserved in ``source["<role>"]`` (each a ``{grid_id, band}`` dict).

Output conventions (per the LANDFIRE 2024 landscape product and the
LCP-to-GeoTIFF transition memo):

- Bands in LANDFIRE order: elevation, slope, aspect, fuel model, canopy
  cover, canopy height, canopy base height, canopy bulk density.
- All bands int16 with the LANDFIRE scaled encodings: canopy height and
  canopy base height in meters x 10, canopy bulk density in kg/m**3 x 100;
  everything else unscaled (m / deg / % / categorical codes).
- Nodata -9999; NaN cells become nodata.
- Band identity mirrors the mechanism LFPS-produced landscapes use (band
  description + a ``BandName`` GDAL metadata tag, readable by GDAL/QGIS,
  ignored by ESRI), plus a ``Units`` tag since LANDFIRE's units are
  otherwise implicit. Fire behavior tools read bands positionally.

Oversized role grids are cropped to the landscape extent by integer
slicing — never resampled.
"""

import logging
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from google.cloud import storage as gcs_storage
from rasterio.transform import Affine

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.storage import load_grid_zarr
from lib.config import EXPORTS_BUCKET

logger = logging.getLogger(__name__)

_NODATA = -9999

# (role, layer name, units label, scale factor) in LANDFIRE band order.
# The fuel model units label is filled per-request from
# source["fire_behavior_fuel_model"].
_BAND_SPECS = [
    ("elevation", "Elevation", "meters", 1),
    ("slope", "Slope", "degrees", 1),
    ("aspect", "Aspect", "degrees", 1),
    ("fuel_model", "Fuel Model", None, 1),
    ("canopy_cover", "Canopy Cover", "percent", 1),
    ("canopy_height", "Canopy Height", "meters * 10", 10),
    ("canopy_base_height", "Canopy Base Height", "meters * 10", 10),
    ("canopy_bulk_density", "Canopy Bulk Density", "kg/m^3 * 100", 100),
]

_FUEL_MODEL_UNITS = {
    "fbfm40": "Scott and Burgan Fire Behavior Fuel Models",
    "fbfm13": "Anderson Fire Behavior Fuel Models",
}


def export_landscape(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Build a landscape GeoTIFF and upload it to GCS."""
    lattice = source["resolved"]["landscape_grid"]
    nx = int(lattice["nx"])
    ny = int(lattice["ny"])
    dx = float(lattice["dx"])
    minx = float(lattice["transform"][2])
    maxy = float(lattice["transform"][5])

    grid_cache: dict[str, xr.Dataset] = {}

    def load_band(role: dict) -> np.ndarray:
        """Load a band, crop to the landscape extent, return a float64 array.

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
        arr = ds[band].transpose("y", "x").values.astype(np.float64, copy=False)

        # x coords ascend (west→east), y coords descend (north→south).
        # Coordinates are cell centers; offset back to cell origin by dx/2.
        role_minx = float(ds.x.values[0]) - dx / 2
        role_maxy = float(ds.y.values[0]) + dx / 2
        i0 = round((minx - role_minx) / dx)
        j0 = round((role_maxy - maxy) / dx)
        return arr[j0 : j0 + ny, i0 : i0 + nx]

    fuel_model_units = _FUEL_MODEL_UNITS[source["fire_behavior_fuel_model"]]

    progress("Loading and encoding bands...", 20)
    bands: list[tuple[np.ndarray, str, str]] = []
    for role_name, layer_name, units, scale in _BAND_SPECS:
        raw = load_band(source[role_name])
        scaled = np.rint(raw * scale)
        encoded = np.where(
            np.isnan(scaled),
            _NODATA,
            np.clip(scaled, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        ).astype(np.int16)
        bands.append((encoded, layer_name, units or fuel_model_units))

    progress("Writing GeoTIFF...", 60)
    transform = Affine(*[float(c) for c in lattice["transform"][:6]])
    profile = {
        "driver": "GTiff",
        "width": nx,
        "height": ny,
        "count": len(bands),
        "dtype": "int16",
        "crs": lattice["crs"],
        "transform": transform,
        "nodata": _NODATA,
        "compress": "lzw",
        "interleave": "pixel",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tif_path = Path(tmp) / "landscape.tif"
        try:
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.update_tags(Creator="FastFuels API")
                for index, (encoded, layer_name, units) in enumerate(bands, start=1):
                    dst.write(encoded, index)
                    dst.set_band_description(index, layer_name)
                    dst.update_tags(index, BandName=layer_name, Units=units)
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                code="LANDSCAPE_WRITE_ERROR",
                message=f"Failed to write landscape GeoTIFF: {e}",
                suggestion="Check exporter logs for details.",
                traceback=traceback.format_exc(),
            )

        progress("Uploading...", 90)
        gcs_path = _upload_tif(str(tif_path), export)

    return gcs_path


def _upload_tif(tif_path: str, export: dict) -> str:
    export_id = export["id"]
    filename = sanitize_filename(export.get("name", ""), ".tif")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    without_scheme = gcs_path.removeprefix("gs://")
    bucket_name, blob_path = without_scheme.split("/", 1)
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(tif_path)
    return gcs_path
