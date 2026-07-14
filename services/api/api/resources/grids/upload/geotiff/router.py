"""
api/v2/resources/grids/upload/geotiff/router.py

Router for direct GeoTIFF upload grid creation.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.schema import Grid
from api.resources.grids.upload.geotiff.examples import (
    CREATE_GRID_UPLOAD_OPENAPI_EXAMPLES,
)
from api.resources.grids.upload.geotiff.schema import CreateGeoTIFFUploadRequest
from api.resources.grids.upload.schema import (
    GridUploadCreatedResponse,
    GridUploadSpec,
)
from api.schema import JobStatus
from lib.config import GRIDS_COLLECTION, UPLOADS_BUCKET
from lib.gcs import generate_upload_signed_url, upload_required_headers

router = APIRouter()

MAX_GRID_SIZE_BYTES = 1_073_741_824  # 1 GB
_CONTENT_TYPE = "image/tiff"


@router.post(
    "",
    response_model=GridUploadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from a direct GeoTIFF upload",
    responses=QUOTA_429_RESPONSE,
)
async def create_geotiff_upload(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGeoTIFFUploadRequest,
        Body(openapi_examples=CREATE_GRID_UPLOAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Upload Grid (GeoTIFF)

    Creates a grid resource and returns a signed URL for uploading a GeoTIFF
    directly to GCS. Upload with HTTP PUT, sending **every header in the
    response's `upload.headers`** exactly as given — the signed URL commits to
    them, and the upload is rejected if any is missing or altered. For example:

    ```bash
    curl -X PUT --upload-file grid.tif \
      -H "Content-Type: image/tiff" \
      -H "x-goog-content-length-range: 0,1073741824" \
      "<upload.url>"
    ```

    When the upload completes, the uploader service processes the file
    automatically via Eventarc and updates the grid status to `completed`
    (or `failed` on error).

    ## Band Definitions

    The `bands` array maps 1:1 to GeoTIFF raster bands in order:
    `bands[0]` → GeoTIFF band 1, `bands[1]` → GeoTIFF band 2, etc.
    Each band key becomes a variable name in the output Zarr store.

    ## CRS Handling

    The GeoTIFF must have a CRS set and must match the domain CRS. A mismatch
    fails with `CRS_MISMATCH`; reproject the GeoTIFF (e.g., `gdalwarp -t_srs`)
    before uploading.

    ## Buffer Cells

    `num_buffer_cells` (default 0) keeps extra cells around the domain extent
    in the stored grid. The uploaded GeoTIFF must cover the domain bbox
    expanded by `num_buffer_cells * native_pixel_size` on each side; pixels
    beyond that expanded extent are clipped away.

    ## File requirements

    Single or multi-band GeoTIFF (`.tif`, `.tiff`). Maximum 1 GB.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(GRIDS_COLLECTION, request)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    object_name = f"grids/{grid_id}/upload.tif"

    bands = [
        {"key": b.key, "type": b.type.value, "unit": b.unit, "index": i}
        for i, b in enumerate(body.bands)
    ]

    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": {
            "name": "upload",
            "format": "geotiff",
            "object_name": object_name,
            "bands": [b.model_dump() for b in body.bands],
            "num_buffer_cells": body.num_buffer_cells,
        },
        "modifications": [],
        "bands": bands,
        "georeference": None,
        "error": None,
        "chunks": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }
    await set_document_async(GRIDS_COLLECTION, grid_id, grid_data)
    register_dispatch(request, response, background_tasks)

    expires_at = request_time + timedelta(minutes=60)
    url = generate_upload_signed_url(
        UPLOADS_BUCKET, object_name, _CONTENT_TYPE, MAX_GRID_SIZE_BYTES
    )

    return GridUploadCreatedResponse(
        grid=Grid(**grid_data),
        upload=GridUploadSpec(
            url=url,
            headers=upload_required_headers(_CONTENT_TYPE, MAX_GRID_SIZE_BYTES),
            content_type=_CONTENT_TYPE,
            expires_at=expires_at,
            max_size_bytes=MAX_GRID_SIZE_BYTES,
        ),
    )
