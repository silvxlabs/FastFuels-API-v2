"""
api/v2/resources/grids/upload/router.py

Router for direct GeoTIFF upload grid creation.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.schema import Grid
from api.resources.grids.upload.examples import CREATE_GRID_UPLOAD_OPENAPI_EXAMPLES
from api.resources.grids.upload.schema import (
    CreateGridUploadRequest,
    GridUploadCreatedResponse,
    GridUploadSpec,
)
from api.schema import JobStatus
from lib.config import GRIDS_COLLECTION, UPLOADS_BUCKET
from lib.gcs import generate_upload_signed_url

router = APIRouter()

MAX_GRID_SIZE_BYTES = 1_073_741_824  # 1 GB
_CONTENT_TYPE = "image/tiff"


@router.post(
    "",
    response_model=GridUploadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from a direct GeoTIFF upload",
)
async def create_grid_upload(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateGridUploadRequest,
        Body(openapi_examples=CREATE_GRID_UPLOAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Upload Grid

    Creates a grid resource and returns a signed URL for uploading a GeoTIFF
    directly to GCS. The upload must use HTTP PUT with Content-Type: image/tiff.

    When the upload completes, the uploader service processes the file
    automatically via Eventarc and updates the grid status to `completed`
    (or `failed` on error).

    ## Band Definitions

    The `bands` array maps 1:1 to GeoTIFF raster bands in order:
    `bands[0]` → GeoTIFF band 1, `bands[1]` → GeoTIFF band 2, etc.
    Each band key becomes a variable name in the output Zarr store.

    ## CRS Handling

    The GeoTIFF must have a CRS set. If the GeoTIFF CRS differs from the
    domain CRS, the uploader reprojects automatically.

    ## Supported Format

    - **geotiff**: Single or multi-band GeoTIFF (`.tif`, `.tiff`). Maximum 1 GB.
    """
    owner_id = request.state.id
    domain_id = domain["id"]
    grid_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    object_name = f"grids/{grid_id}/upload.tif"

    bands = [
        {"key": b.key, "type": b.type.value, "unit": b.unit, "index": i}
        for i, b in enumerate(body.bands)
    ]

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
            "format": body.format.value,
            "object_name": object_name,
            "bands": [b.model_dump() for b in body.bands],
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
