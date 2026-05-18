"""
Uploader handlers for grid uploads.

Reads uploaded grid files directly from GCS (no local download), validates
structure and CRS against the domain, clips to domain bounds, and writes a Zarr
store to GRIDS_BUCKET. Fires on google.cloud.storage.object.v1.finalized for
grids/. One function per supported input format:

- ``handle_grid_geotiff``: 2D GeoTIFF (`.tif`). Band metadata comes from the
  request body.
- ``handle_grid_netcdf``: CF-conformant 2D or 3D netCDF (`.nc`). Band metadata
  is derived from the file's data variables.
"""

from datetime import UTC, datetime

import gcsfs
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from lib.config import DOMAINS_COLLECTION, GRIDS_BUCKET, GRIDS_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.errors import ProcessingError
from lib.firestore import get_document, update_document
from lib.gcs import delete_file
from lib.grids import compute_chunks_doc
from lib.units import validate_unit
from lib.zarr_utils import save_zarr

_CHUNK_SHAPE = (512, 512)
_ALLOWED_NETCDF_DIMS = {("y", "x"), ("z", "y", "x")}


def handle_grid_geotiff(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    """Process an uploaded GeoTIFF and write it as a Zarr store.

    Args:
        resource_id: Grid document ID in Firestore.
        bucket: GCS bucket containing the uploaded file (UPLOADS_BUCKET).
        object_name: Full GCS object path, e.g. "grids/{id}/fuel_load.tif".
        doc: Grid document loaded from Firestore.
    """
    source = doc["source"]
    domain_id = doc["domain_id"]
    bands_spec = source["bands"]
    num_buffer_cells = source.get("num_buffer_cells", 0)

    gcs_path = f"gs://{bucket}/{object_name}"

    try:
        _, domain_snap = get_document(DOMAINS_COLLECTION, domain_id)
        domain_data = domain_snap.to_dict()
        domain_gdf = parse_domain_gdf(domain_data)
        domain_crs_str = domain_data["crs"]["properties"]["name"]

        dataset = _build_dataset(
            gcs_path,
            bands_spec,
            domain_crs_str,
            domain_gdf,
            num_buffer_cells=num_buffer_cells,
        )

        output_path = f"gs://{GRIDS_BUCKET}/{resource_id}"
        save_zarr(output_path, dataset, chunk_shape=_CHUNK_SHAPE)

        transform = dataset.rio.transform()
        grid_shape = (dataset.rio.height, dataset.rio.width)

        update_document(
            GRIDS_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": {
                    "crs": str(dataset.rio.crs),
                    "transform": list(transform)[:6],
                    "shape": list(grid_shape),
                },
                "chunks": compute_chunks_doc(grid_shape, _CHUNK_SHAPE),
                "progress": {"message": "Complete", "percent": 100},
            },
        )
    finally:
        try:
            delete_file(gcs_path)
        except Exception:
            pass


def _build_dataset(
    gcs_path: str,
    bands_spec: list[dict],
    domain_crs_str: str,
    domain_gdf,
    num_buffer_cells: int = 0,
) -> xr.Dataset:
    """Open a GeoTIFF from GCS and return a clipped, reprojected xarray Dataset.

    Reads lazily — pixel data is not loaded until save_zarr triggers dask computation.
    CRS and band count come from file header metadata only.

    Args:
        gcs_path: Full GCS path, e.g. "gs://bucket/grids/{id}/file.tif".
        bands_spec: List of band definitions from the Firestore source doc.
            Each entry has at least {"key": str, "unit": str | None}.
        domain_crs_str: Target CRS string, e.g. "EPSG:32611".
        domain_gdf: Domain GeoDataFrame used for bounds clipping.
        num_buffer_cells: Number of native-resolution cells to keep around the
            domain extent. The clip bounds are expanded by
            ``num_buffer_cells * native_pixel_size`` meters on each side.

    Returns:
        xr.Dataset with one variable per band, clipped to domain bounds.

    Raises:
        ProcessingError: MISSING_CRS if the GeoTIFF has no CRS.
        ProcessingError: BAND_COUNT_MISMATCH if band count doesn't match spec.
        ProcessingError: CRS_MISMATCH if the GeoTIFF CRS does not match the domain CRS.
        ProcessingError: NO_OVERLAP if the GeoTIFF does not intersect the domain.
    """
    # chunks={"x":512,"y":512} → dask-backed lazy arrays; pixel data deferred until compute.
    # lock=False → parallel chunk reads (safe for GCS with GDAL's /vsigs/).
    da = rioxarray.open_rasterio(
        gcs_path, chunks={"x": 512, "y": 512}, lock=False, masked=True
    )

    if da.rio.crs is None:
        raise ProcessingError(
            code="MISSING_CRS",
            message="GeoTIFF has no CRS. Set a coordinate reference system before uploading.",
            suggestion="Use gdalwarp -t_srs EPSG:<code> to assign a CRS to your GeoTIFF.",
        )

    if da.shape[0] != len(bands_spec):
        raise ProcessingError(
            code="BAND_COUNT_MISMATCH",
            message=(
                f"GeoTIFF has {da.shape[0]} band(s) but request defines {len(bands_spec)}. "
                "bands[i] must correspond to GeoTIFF band i+1."
            ),
        )

    if str(da.rio.crs) != domain_crs_str:
        raise ProcessingError(
            code="CRS_MISMATCH",
            message=(
                f"GeoTIFF CRS ({da.rio.crs}) does not match the domain CRS ({domain_crs_str}). "
                "Reproject your file before uploading."
            ),
            suggestion=f"Use gdalwarp -t_srs {domain_crs_str} input.tif output.tif to reproject.",
        )

    xmin, ymin, xmax, ymax = domain_gdf.total_bounds
    if num_buffer_cells > 0:
        pixel_size = abs(float(da.rio.transform().a))
        padding = num_buffer_cells * pixel_size
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
    try:
        da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
    except NoDataInBounds:
        raise ProcessingError(
            code="NO_OVERLAP",
            message="The uploaded GeoTIFF does not overlap the domain extent.",
            suggestion="Verify the GeoTIFF covers the domain region.",
        )

    data_vars = {}
    for i, band_spec in enumerate(bands_spec):
        band_da = da.sel(band=i + 1).drop_vars("band")
        if band_spec.get("unit"):
            band_da.attrs["units"] = band_spec["unit"]
        data_vars[band_spec["key"]] = band_da

    return xr.Dataset(data_vars)


def handle_grid_netcdf(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    """Process an uploaded CF-conformant netCDF and write it as a Zarr store.

    Supports 2D ``(y, x)`` and 3D ``(z, y, x)`` data variables. Variable names
    in the file become the band keys; units come from each variable's ``units``
    attribute. Z-chunking is intentionally absent — chunks tile only ``(y, x)``
    so QF-style column reads load one vertical profile per chunk.

    Args:
        resource_id: Grid document ID in Firestore.
        bucket: GCS bucket containing the uploaded file (UPLOADS_BUCKET).
        object_name: Full GCS object path, e.g. "grids/{id}/upload.nc".
        doc: Grid document loaded from Firestore.
    """
    source = doc["source"]
    domain_id = doc["domain_id"]
    num_buffer_cells = source.get("num_buffer_cells", 0)

    gcs_path = f"gs://{bucket}/{object_name}"

    try:
        _, domain_snap = get_document(DOMAINS_COLLECTION, domain_id)
        domain_data = domain_snap.to_dict()
        domain_gdf = parse_domain_gdf(domain_data)
        domain_crs_str = domain_data["crs"]["properties"]["name"]

        # gcsfs file-like read avoids /tmp staging (Cloud Run has no local
        # disk). The dataset is dask-backed by this file handle, so everything
        # that touches it must run inside the with-block.
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            dataset = _build_netcdf_dataset(
                f, domain_crs_str, domain_gdf, num_buffer_cells
            )

            output_path = f"gs://{GRIDS_BUCKET}/{resource_id}"
            is_3d = "z" in dataset.sizes
            save_zarr(output_path, dataset, chunk_shape=_CHUNK_SHAPE)

            ny = dataset.sizes["y"]
            nx = dataset.sizes["x"]
            transform = list(dataset.rio.transform())[:6]

            if is_3d:
                nz = dataset.sizes["z"]
                z_vals = dataset["z"].values
                dz = float(z_vals[1] - z_vals[0])
                georeference = {
                    "crs": str(dataset.rio.crs),
                    "transform": transform,
                    "shape": [nz, ny, nx],
                    "z_resolution": dz,
                    "z_origin": float(z_vals[0]) - dz / 2,
                }
                grid_shape = (nz, ny, nx)
                chunk_shape = (nz, _CHUNK_SHAPE[0], _CHUNK_SHAPE[1])
            else:
                georeference = {
                    "crs": str(dataset.rio.crs),
                    "transform": transform,
                    "shape": [ny, nx],
                }
                grid_shape = (ny, nx)
                chunk_shape = _CHUNK_SHAPE

            bands = []
            for i, var_name in enumerate(dataset.data_vars):
                da = dataset[var_name]
                band_type = (
                    "categorical"
                    if np.issubdtype(da.dtype, np.integer)
                    else "continuous"
                )
                bands.append(
                    {
                        "key": str(var_name),
                        "type": band_type,
                        "unit": da.attrs.get("units"),
                        "index": i,
                    }
                )

            update_document(
                GRIDS_COLLECTION,
                resource_id,
                {
                    "status": "completed",
                    "modified_on": datetime.now(UTC),
                    "bands": bands,
                    "georeference": georeference,
                    "chunks": compute_chunks_doc(grid_shape, chunk_shape),
                    "progress": {"message": "Complete", "percent": 100},
                },
            )
    finally:
        try:
            delete_file(gcs_path)
        except Exception:
            pass


def _build_netcdf_dataset(
    source,
    domain_crs_str: str,
    domain_gdf,
    num_buffer_cells: int,
) -> xr.Dataset:
    """Open a netCDF from a file-like or path, validate, and clip to the domain.

    The returned Dataset is dask-backed by ``source``. Callers must keep the
    underlying file open until all reads (e.g. ``save_zarr``) complete.

    Args:
        source: File-like or path argument accepted by ``xr.open_dataset``.
        domain_crs_str: Target CRS string, e.g. "EPSG:32611".
        domain_gdf: Domain GeoDataFrame used for bounds clipping.
        num_buffer_cells: Native-resolution cells to keep around the domain
            extent. Bounds expand by ``num_buffer_cells * pixel_size`` per side.

    Returns:
        Validated, clipped xr.Dataset stamped with ``grid_mapping="spatial_ref"``.

    Raises:
        ProcessingError: WRONG_DIMS, MISSING_CRS, CRS_MISMATCH, INVALID_UNITS,
            MISSING_Z_POSITIVE, SINGLE_Z_LAYER, NONUNIFORM_Z, or NO_OVERLAP
            per spec.
    """
    ds = xr.open_dataset(
        source,
        engine="h5netcdf",
        decode_coords="all",
        chunks={"y": _CHUNK_SHAPE[0], "x": _CHUNK_SHAPE[1]},
    )

    dim_sets = {da.dims for da in ds.data_vars.values()}
    if not dim_sets:
        raise ProcessingError(
            code="WRONG_DIMS",
            message="netCDF has no data variables.",
        )
    invalid = dim_sets - _ALLOWED_NETCDF_DIMS
    if invalid:
        raise ProcessingError(
            code="WRONG_DIMS",
            message=(
                f"netCDF data variables must have dims ('y','x') or "
                f"('z','y','x') in that order; got {sorted(invalid)}."
            ),
        )
    if len(dim_sets) > 1:
        raise ProcessingError(
            code="WRONG_DIMS",
            message=(
                "netCDF data variables must all have the same rank; "
                f"got mixed ranks {sorted(dim_sets)}."
            ),
        )

    if ds.rio.crs is None:
        raise ProcessingError(
            code="MISSING_CRS",
            message=(
                "netCDF has no CRS. Set a CF grid_mapping (typically "
                "'spatial_ref') on each data variable."
            ),
        )
    if str(ds.rio.crs) != domain_crs_str:
        raise ProcessingError(
            code="CRS_MISMATCH",
            message=(
                f"netCDF CRS ({ds.rio.crs}) does not match the domain CRS "
                f"({domain_crs_str}). Reproject your file before uploading."
            ),
        )

    for var_name, da in ds.data_vars.items():
        unit = da.attrs.get("units")
        if unit is None:
            continue
        try:
            validate_unit(unit)
        except ValueError as e:
            raise ProcessingError(
                code="INVALID_UNITS",
                message=f"Variable {var_name!r} has invalid unit {unit!r}: {e}",
            )

    is_3d = "z" in ds.sizes
    if is_3d:
        positive = ds["z"].attrs.get("positive")
        if positive != "up":
            raise ProcessingError(
                code="MISSING_Z_POSITIVE",
                message=(
                    f"netCDF z-coord must have attr positive='up'; got {positive!r}."
                ),
            )
        z_vals = ds["z"].values
        if len(z_vals) < 2:
            raise ProcessingError(
                code="SINGLE_Z_LAYER",
                message=(
                    "netCDF 3D variable has only one z level; z_resolution "
                    "(cell thickness) cannot be derived from a single coordinate. "
                    "Upload as 2D (drop the z dim) or include at least two z levels."
                ),
            )
        diffs = np.diff(z_vals)
        if not np.allclose(diffs, diffs[0]):
            raise ProcessingError(
                code="NONUNIFORM_Z",
                message=(
                    "netCDF z-coord must be uniformly spaced (z_resolution is scalar)."
                ),
            )

    xmin, ymin, xmax, ymax = domain_gdf.total_bounds
    if num_buffer_cells > 0:
        pixel_size = abs(float(ds.rio.transform().a))
        padding = num_buffer_cells * pixel_size
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
    try:
        ds = ds.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
    except NoDataInBounds:
        raise ProcessingError(
            code="NO_OVERLAP",
            message="The uploaded netCDF does not overlap the domain extent.",
        )

    # Don't stamp grid_mapping into var.attrs. decode_coords="all" above put
    # it in var.encoding already; xarray's CF encoder migrates it to the
    # zarr's on-disk attrs at to_zarr time via pop_to(encoding, attrs, ...).
    # Stamping attrs ourselves makes that pop_to raise.

    return ds
