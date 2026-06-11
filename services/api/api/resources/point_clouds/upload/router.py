"""
api/v2/resources/point_clouds/upload/router.py

Router for creating a point cloud from a direct file upload.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.point_clouds.schema import PointCloud
from api.resources.point_clouds.upload.examples import (
    CREATE_UPLOAD_OPENAPI_EXAMPLES,
)
from api.resources.point_clouds.upload.schema import (
    CreatePointCloudUploadRequest,
    PointCloudUploadCreatedResponse,
    PointCloudUploadSpec,
)
from api.schema import JobStatus
from lib.config import POINT_CLOUDS_COLLECTION, UPLOADS_BUCKET
from lib.gcs import generate_upload_signed_url

router = APIRouter()

_CONTENT_TYPE = "application/octet-stream"
# Caps the signed upload. The worker streams the file with bounded chunk
# memory, but a rewritten (reprojected/recompressed) cloud is built in an
# in-memory buffer, so this cap also bounds the worker's peak RAM. Tunable.
MAX_POINT_CLOUD_SIZE_BYTES = 1_073_741_824  # 1 GiB


@router.post(
    "",
    response_model=PointCloudUploadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a point cloud from a direct file upload",
)
async def create_point_cloud_upload(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreatePointCloudUploadRequest,
        Body(openapi_examples=CREATE_UPLOAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Upload a Point Cloud

    Creates a point cloud resource and returns a signed URL for uploading the
    source file directly to storage. Upload is a two-step flow:

    1. **POST** this request to create the point cloud and receive an `upload`
       spec containing a signed URL.
    2. **PUT** your file to `upload.url` with a `Content-Type` header equal to
       `upload.content_type`. The file must not exceed `upload.max_size_bytes`,
       and the upload must complete before `upload.expires_at`.

    The point cloud is returned immediately with `status` = `pending`. Once the
    file finishes uploading it is ingested in the background: `status` becomes
    `running`, then `completed` after the cloud is validated and its
    `georeference` and `summary` are filled in — or `failed` if the file cannot
    be read as a point cloud with a coordinate reference system. Poll
    `GET /domains/{domain_id}/pointclouds/{id}` to follow progress.

    ## Supported formats

    Upload an uncompressed **LAS** or compressed **LAZ** file (including Cloud
    Optimized Point Clouds, which are valid LAZ). The format is detected from
    the file itself — there is nothing to declare.

    ## Coordinate reference system

    The file must carry a coordinate reference system; uploads without one are
    rejected during ingestion. A cloud in a different CRS than its domain is
    automatically reprojected to the domain CRS (horizontal coordinates only —
    elevations are preserved as-is), so the stored cloud is always in the
    domain CRS.
    """
    owner_id = request.state.id
    domain_id = domain["id"]
    point_cloud_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    object_name = f"pointclouds/{point_cloud_id}/upload"

    point_cloud_data = {
        "id": point_cloud_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": body.type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": {
            "name": "upload",
            "object_name": object_name,
        },
        "georeference": None,
        "summary": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }
    await set_document_async(POINT_CLOUDS_COLLECTION, point_cloud_id, point_cloud_data)

    expires_at = request_time + timedelta(minutes=60)
    url = generate_upload_signed_url(
        UPLOADS_BUCKET, object_name, _CONTENT_TYPE, MAX_POINT_CLOUD_SIZE_BYTES
    )

    return PointCloudUploadCreatedResponse(
        point_cloud=PointCloud(**point_cloud_data),
        upload=PointCloudUploadSpec(
            url=url,
            content_type=_CONTENT_TYPE,
            expires_at=expires_at,
            max_size_bytes=MAX_POINT_CLOUD_SIZE_BYTES,
        ),
    )
