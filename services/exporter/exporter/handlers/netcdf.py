"""NetCDF grid export handler.

Loads a grid from Zarr, stamps CF-1.13 metadata onto the in-memory
Dataset, writes a compressed CF-conformant netCDF to /tmp, and uploads
it to EXPORTS_BUCKET.

Memory profile:
- RSS during write is bounded by `chunk × workers × dtype_size` (dask
  streams chunks through RAM one at a time).
- The full compressed netCDF accumulates on /tmp (tmpfs) before upload —
  HDF5 needs seekable random-access writes and is not a streamable
  container. The exporter Cloud Run revision's `--memory` must cover
  this; #110 tracks the proper fix (ephemeral-disk scratch).
"""

import logging
import os
import tempfile
import traceback
from collections.abc import Callable

from google.cloud import storage as gcs_storage

from exporter.errors import ProcessingError
from exporter.filename import sanitize_filename
from exporter.handlers.grid import _load_and_select_bands
from lib.cf_utils import stamp_cf
from lib.config import EXPORTS_BUCKET, GRIDS_COLLECTION
from lib.firestore import get_document

logger = logging.getLogger(__name__)


_INTERNAL_ATTRS_TO_STRIP = ("transform", "z_origin", "z_resolution")


def export_netcdf(
    export: dict,
    source: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Export a grid as a CF-1.13 conformant netCDF file.

    Args:
        export: Export document from Firestore.
        source: Source configuration (grid_id, bands).
        progress: Progress callback (message, percent).

    Returns:
        GCS path to the exported netCDF file.
    """
    export_id = export["id"]
    grid_id = source["grid_id"]

    ds = _load_and_select_bands(source, progress)

    progress("Reading grid metadata...", 55)
    try:
        _, grid_snapshot = get_document(GRIDS_COLLECTION, grid_id)
    except Exception as e:
        raise ProcessingError(
            code="GRID_DOC_NOT_FOUND",
            message=f"Failed to load grid {grid_id} from Firestore: {e}",
            suggestion="Ensure the grid exists in Firestore.",
            traceback=traceback.format_exc(),
        )
    grid_doc = grid_snapshot.to_dict() or {}
    bands = grid_doc.get("bands", [])

    progress("Stamping CF metadata...", 60)
    for k in _INTERNAL_ATTRS_TO_STRIP:
        ds.attrs.pop(k, None)
    stamp_cf(ds, bands=bands, vertical=("z" in ds.dims))

    progress("Writing netCDF...", 70)
    filename = sanitize_filename(export.get("name", ""), ".nc")
    gcs_path = f"gs://{EXPORTS_BUCKET}/{export_id}/{filename}"

    # Set compression on each var.encoding rather than passing the encoding
    # kwarg to to_netcdf. Passing encoding via kwarg REPLACES the variable's
    # entire encoding dict at write time (xarray/backends/writers.py), which
    # wipes the grid_mapping field that decode_coords="all" left there. By
    # mutating var.encoding in place we preserve grid_mapping so xarray's CF
    # encoder can migrate it to attrs on the netCDF.
    for var in ds.data_vars:
        ds[var].encoding["zlib"] = True
        ds[var].encoding["complevel"] = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, "export.nc")
        try:
            ds.to_netcdf(local_path, engine="h5netcdf")
        except Exception as e:
            raise ProcessingError(
                code="NETCDF_WRITE_ERROR",
                message=f"Failed to write netCDF: {e}",
                suggestion="This may indicate an issue with the grid data or metadata.",
                traceback=traceback.format_exc(),
            )

        progress("Uploading...", 90)
        _upload_file(local_path, gcs_path)

    return gcs_path


def _upload_file(local_path: str, gcs_path: str) -> None:
    """Upload a local file to a GCS path via multipart upload."""
    without_scheme = gcs_path.removeprefix("gs://")
    bucket_name, blob_path = without_scheme.split("/", 1)
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
