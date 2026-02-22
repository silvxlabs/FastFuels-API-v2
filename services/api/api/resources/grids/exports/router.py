"""
api/v2/resources/grids/exports/router.py

Router for grid export creation endpoints.

Two routers for two URL patterns:
- router:             POST /domains/{domain_id}/grids/exports/geotiff
- single_grid_router: POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff

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
    ExportGeoTiffRequest,
    ExportSingleGridGeoTiffRequest,
    GeoTiffExportSource,
)
from api.resources.grids.exports.examples import (
    CREATE_GEOTIFF_OPENAPI_EXAMPLES,
    CREATE_SINGLE_GRID_GEOTIFF_OPENAPI_EXAMPLES,
)
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
single_grid_router = APIRouter()


async def _create_export(
    owner_id: str,
    domain_id: str,
    grid_ids: list[str],
    bands: list[str] | None,
    expiration_days: int,
    name: str,
    description: str,
    tags: list[str],
) -> Export:
    """Shared logic for creating a GeoTIFF export.

    Validates all grids, creates the export document, and enqueues processing.
    """
    for grid_id in grid_ids:
        _, grid_snapshot = await get_document_async(
            GRIDS_COLLECTION,
            grid_id,
            owner_id=owner_id,
            domain_id=domain_id,
            document_status="completed",
        )
        grid_data = grid_snapshot.to_dict()

        if bands:
            validate_grid_has_band(grid_data, grid_id, bands)

    export_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = GeoTiffExportSource(grid_ids=grid_ids, bands=bands)

    export_data = {
        "id": export_id,
        "domain_id": domain_id,
        "name": name,
        "description": description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "signed_url": None,
        "expiration_days": expiration_days,
        "expires_on": request_time + timedelta(days=expiration_days),
        "error": None,
        "tags": tags,
        "owner_id": owner_id,
    }

    await set_document_async(EXPORTS_COLLECTION, export_id, export_data)
    await create_http_task_async(EXPORTER_QUEUE, EXPORTER_SERVICE, export_id)

    return Export(**export_data)


@router.post(
    "/geotiff",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export grids to GeoTIFF",
)
async def create_geotiff_export(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        ExportGeoTiffRequest,
        Body(openapi_examples=CREATE_GEOTIFF_OPENAPI_EXAMPLES),
    ],
):
    """Export one or more grids to a GeoTIFF file.

    Provide a list of grid IDs in the request body. All grids must belong to
    this domain and have status `completed`. If `bands` is specified, only
    those bands are included; otherwise all bands from each grid are exported.

    To export a single grid without specifying its ID in the body, use the
    per-grid endpoint: `POST /domains/{domain_id}/grids/{grid_id}/exports/geotiff`.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to get the
    signed download URL.
    """
    return await _create_export(
        owner_id=request.state.id,
        domain_id=domain["id"],
        grid_ids=body.grid_ids,
        bands=body.bands,
        expiration_days=body.expiration_days,
        name=body.name,
        description=body.description,
        tags=body.tags,
    )


@single_grid_router.post(
    "/geotiff",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export a grid to GeoTIFF",
)
async def create_single_grid_geotiff_export(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    body: Annotated[
        ExportSingleGridGeoTiffRequest,
        Body(openapi_examples=CREATE_SINGLE_GRID_GEOTIFF_OPENAPI_EXAMPLES),
    ],
):
    """Export a single grid to a GeoTIFF file.

    The grid is identified by the `{grid_id}` path parameter. It must belong
    to this domain and have status `completed`. If `bands` is specified, only
    those bands are included; otherwise all bands are exported.

    To export multiple grids at once, use the domain-level endpoint:
    `POST /domains/{domain_id}/grids/exports/geotiff`.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to get the
    signed download URL.
    """
    return await _create_export(
        owner_id=request.state.id,
        domain_id=domain["id"],
        grid_ids=[grid_id],
        bands=body.bands,
        expiration_days=body.expiration_days,
        name=body.name,
        description=body.description,
        tags=body.tags,
    )
