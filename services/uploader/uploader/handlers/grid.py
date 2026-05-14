"""
Uploader handler for GeoTIFF grid uploads.

Reads the uploaded GeoTIFF directly from GCS (no local download), reprojects to
the domain CRS if needed, clips to domain bounds, and writes a Zarr store to
GRIDS_BUCKET. Fires on google.cloud.storage.object.v1.finalized for grids/.
"""

from datetime import UTC, datetime

import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from lib.config import DOMAINS_COLLECTION, GRIDS_BUCKET, GRIDS_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.errors import ProcessingError
from lib.firestore import get_document, update_document
from lib.gcs import delete_file
from lib.grids import compute_chunks_doc
from lib.zarr_utils import save_zarr

_CHUNK_SHAPE = (512, 512)


def handle_grid(resource_id: str, bucket: str, object_name: str, doc: dict) -> None:
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

    gcs_path = f"gs://{bucket}/{object_name}"

    try:
        _, domain_snap = get_document(DOMAINS_COLLECTION, domain_id)
        domain_data = domain_snap.to_dict()
        domain_gdf = parse_domain_gdf(domain_data)
        domain_crs_str = domain_data["crs"]["properties"]["name"]

        dataset = _build_dataset(gcs_path, bands_spec, domain_crs_str, domain_gdf)

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

    Returns:
        xr.Dataset with one variable per band, clipped to domain bounds.

    Raises:
        ProcessingError: MISSING_CRS if the GeoTIFF has no CRS.
        ProcessingError: BAND_COUNT_MISMATCH if band count doesn't match spec.
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
        da = da.rio.reproject(domain_crs_str)

    xmin, ymin, xmax, ymax = domain_gdf.total_bounds
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
