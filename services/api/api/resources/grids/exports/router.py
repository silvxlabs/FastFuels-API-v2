"""
api/v2/resources/grids/exports/router.py

Router for grid export creation endpoints.

URL pattern: POST /domains/{domain_id}/grids/{grid_id}/exports/{format}

Lifecycle endpoints (list, get, patch, delete) are at /v2/exports/.
"""

import uuid
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.exports.schema import (
    Export,
    ExportGridRequest,
    GridExportFormat,
    GridExportSource,
)
from api.resources.grids.exports.examples import CREATE_GRID_EXPORT_OPENAPI_EXAMPLES
from api.resources.grids.utils import validate_grid_has_band
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    EXPORTER_QUEUE,
    EXPORTER_SERVICE,
    EXPORTS_COLLECTION,
    GRIDS_COLLECTION,
)

router = APIRouter()


@router.post(
    "/{format}",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export a grid",
)
async def create_grid_export(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    format: GridExportFormat,
    body: Annotated[
        ExportGridRequest,
        Body(openapi_examples=CREATE_GRID_EXPORT_OPENAPI_EXAMPLES),
    ],
):
    """Export a grid to the specified format.

    Supported formats: `geotiff`, `zarr` (zipped), `netcdf` (CF-1.13).

    The grid must belong to this domain and have status `completed`.
    If `bands` is specified, only those bands are included; otherwise
    all bands are exported.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to get the
    signed download URL.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Validate grid exists, is owned, belongs to domain, and is completed
    _, grid_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    grid_data = grid_snapshot.to_dict()

    # Validate band subset if provided
    if body.bands:
        validate_grid_has_band(grid_data, grid_id, body.bands)

    # Build source metadata
    source = GridExportSource(
        name=format.value,
        grid_id=grid_id,
        bands=body.bands,
    )

    # Create export document
    export_id = uuid.uuid4().hex
    request_time = datetime.now()

    export_data = {
        "id": export_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "signed_url": None,
        "expiration_days": body.expiration_days,
        "expires_on": request_time + timedelta(days=body.expiration_days),
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(EXPORTS_COLLECTION, export_id, export_data)
    await create_http_task_async(EXPORTER_QUEUE, EXPORTER_SERVICE, export_id)

    return Export(**export_data)
