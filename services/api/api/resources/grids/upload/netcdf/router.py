"""
api/v2/resources/grids/upload/netcdf/router.py

Router for direct netCDF upload grid creation. Accepts CF-conformant 2D
or 3D netCDFs; variable names in the file become the band keys.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.schema import Grid
from api.resources.grids.upload.netcdf.examples import (
    CREATE_NETCDF_UPLOAD_OPENAPI_EXAMPLES,
)
from api.resources.grids.upload.netcdf.schema import CreateNetcdfUploadRequest
from api.resources.grids.upload.schema import (
    GridUploadCreatedResponse,
    GridUploadSpec,
)
from api.schema import JobStatus
from lib.config import GRIDS_COLLECTION, UPLOADS_BUCKET
from lib.gcs import generate_upload_signed_url

router = APIRouter()

MAX_GRID_SIZE_BYTES = 1_073_741_824  # 1 GB
_CONTENT_TYPE = "application/x-netcdf"


@router.post(
    "",
    response_model=GridUploadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from a direct netCDF upload",
)
async def create_netcdf_upload(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateNetcdfUploadRequest,
        Body(openapi_examples=CREATE_NETCDF_UPLOAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Upload Grid (netCDF)

    Creates a grid resource and returns a signed URL for uploading a
    CF-conformant netCDF directly to GCS. The upload must use HTTP PUT
    with `Content-Type: application/x-netcdf`.

    When the upload completes, the uploader service processes the file
    automatically via Eventarc and updates the grid status to `completed`
    (or `failed` on error).

    ## Bands

    Unlike the GeoTIFF route, the request body has **no `bands` field**.
    netCDF data variable names are the canonical band keys — they are
    extracted directly from the file and become the variable names in the
    output Zarr store. Per-band `units` (if set on the variable) and dtype
    drive the stored band metadata.

    ## Dimensions

    Each data variable must have dims exactly `("y","x")` (2D) or
    `("z","y","x")` (3D) in that order. Mixed-rank datasets are rejected
    with `WRONG_DIMS`.

    ## CRS

    The dataset must carry a CF `grid_mapping` (typically `spatial_ref`)
    that matches the domain CRS. Missing CRS fails with `MISSING_CRS`;
    mismatched CRS fails with `CRS_MISMATCH`. No auto-reproject.

    ## Units

    If a data variable has a `units` attribute it must be in canonical
    UDUNITS-2 ASCII form with `**` exponents (e.g. `kg/m**3`, `1/m`, `%`).
    Non-canonical forms (`kg/m³`, `kg/m^3`, `kg/m3`) fail with
    `INVALID_UNITS`. See docs/units.md.

    ## Z axis (3D only)

    - `z.attrs["positive"]` must equal `"up"`. `"down"` is rejected
      (`MISSING_Z_POSITIVE`).
    - z-coordinates must be uniformly spaced. Non-uniform spacing is
      rejected with `NONUNIFORM_Z`.

    ## Buffer cells

    `num_buffer_cells` (default 0) keeps extra cells around the domain
    extent in the stored grid. The uploaded netCDF must cover the domain
    bbox expanded by `num_buffer_cells * native_pixel_size` on each side;
    pixels beyond that expanded extent are clipped away.

    ## File requirements

    CF-conformant netCDF (`.nc`). Maximum 1 GB.
    """
    owner_id = request.state.id
    domain_id = domain["id"]
    grid_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    object_name = f"grids/{grid_id}/upload.nc"

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": {
            "name": "upload",
            "format": "netcdf",
            "object_name": object_name,
            "num_buffer_cells": body.num_buffer_cells,
        },
        "modifications": [],
        "bands": [],
        "georeference": None,
        "error": None,
        "chunks": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }
    await set_document_async(GRIDS_COLLECTION, grid_id, grid_data)

    expires_at = request_time + timedelta(minutes=60)
    url = generate_upload_signed_url(
        UPLOADS_BUCKET, object_name, _CONTENT_TYPE, MAX_GRID_SIZE_BYTES
    )

    return GridUploadCreatedResponse(
        grid=Grid(**grid_data),
        upload=GridUploadSpec(
            url=url,
            content_type=_CONTENT_TYPE,
            expires_at=expires_at,
            max_size_bytes=MAX_GRID_SIZE_BYTES,
        ),
    )
