"""
api/v2/resources/inventories/tree/upload/router.py

Router for direct file upload inventory creation.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.inventories.schema import BASE_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.upload.examples import (
    CREATE_UPLOAD_OPENAPI_EXAMPLES,
)
from api.resources.inventories.tree.upload.schema import (
    CreateInventoryUploadRequest,
    InventoryUploadCreatedResponse,
    InventoryUploadSpec,
)
from api.schema import JobStatus
from lib.config import INVENTORIES_COLLECTION, UPLOADS_BUCKET
from lib.gcs import generate_upload_signed_url, upload_required_headers

router = APIRouter()

_CONTENT_TYPES = {
    "csv": "text/csv",
    "geojson": "application/geo+json",
    "geopackage": "application/geopackage+sqlite3",
}
_FORMAT_EXTENSIONS = {
    "csv": "csv",
    "geojson": "geojson",
    "geopackage": "gpkg",
}
MAX_INVENTORY_SIZE_BYTES = 524_288_000  # 500 MB


@router.post(
    "",
    response_model=InventoryUploadCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an inventory from a direct file upload",
    responses=QUOTA_429_RESPONSE,
)
async def create_inventory_upload(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateInventoryUploadRequest,
        Body(openapi_examples=CREATE_UPLOAD_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Upload Inventory

    Creates an inventory resource and returns a signed URL for uploading the
    source file directly to GCS. Upload with HTTP PUT, sending **every header
    in the response's `upload.headers`** exactly as given — the signed URL
    commits to them, and the upload is rejected if any is missing or altered.
    For example:

    ```bash
    curl -X PUT --upload-file trees.csv \
      -H "Content-Type: text/csv" \
      -H "x-goog-content-length-range: 0,524288000" \
      "<upload.url>"
    ```

    When the upload completes, the uploader service processes the file
    automatically via Eventarc and updates the inventory status to
    `completed` (or `failed` on error).

    ## Supported Formats

    - **csv**: Comma-separated values. Coordinates must already be in the
      domain's CRS.
    - **geojson**: GeoJSON FeatureCollection with Point or MultiPoint
      geometries. Reprojected to domain CRS automatically.
    - **geopackage**: OGC GeoPackage. Reprojected to domain CRS automatically.

    ## Column Mapping

    Use the `columns` field to map v2 column names to the column names in
    your file. Omit entries where the file already uses v2 names. Required
    in the file: `x`, `y`, `height`.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(INVENTORIES_COLLECTION, request)

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    fmt = body.format.value
    object_name = f"inventories/{inventory_id}/upload.{_FORMAT_EXTENSIONS[fmt]}"
    content_type = _CONTENT_TYPES[fmt]

    inventory_data = {
        "id": inventory_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": "tree",
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": {
            "name": "upload",
            "format": fmt,
            "object_name": object_name,
            "columns": body.columns.model_dump(exclude_none=True),
        },
        "modifications": [],
        "columns": [c.model_dump() for c in BASE_INVENTORY_COLUMNS],
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }
    await set_document_async(INVENTORIES_COLLECTION, inventory_id, inventory_data)

    expires_at = request_time + timedelta(minutes=60)
    url = generate_upload_signed_url(
        UPLOADS_BUCKET, object_name, content_type, MAX_INVENTORY_SIZE_BYTES
    )

    return InventoryUploadCreatedResponse(
        inventory=Inventory(**inventory_data),
        upload=InventoryUploadSpec(
            url=url,
            headers=upload_required_headers(content_type, MAX_INVENTORY_SIZE_BYTES),
            content_type=content_type,
            expires_at=expires_at,
            max_size_bytes=MAX_INVENTORY_SIZE_BYTES,
        ),
    )
