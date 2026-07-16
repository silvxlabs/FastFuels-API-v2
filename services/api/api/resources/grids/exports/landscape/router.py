"""
api/v2/resources/grids/exports/landscape/router.py

Router for the landscape combined export endpoint.

URL pattern: POST /v2/domains/{domain_id}/grids/exports/landscape

Lifecycle endpoints (list, get, patch, delete) are at /v2/exports/.
"""

import uuid
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.exports.schema import Export
from api.resources.grids.exports.landscape.examples import (
    CREATE_LANDSCAPE_EXPORT_OPENAPI_EXAMPLES,
)
from api.resources.grids.exports.landscape.schema import LandscapeExportRequest
from api.resources.grids.exports.landscape.validators import (
    validate_landscape_request,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import EXPORTER_QUEUE, EXPORTER_SERVICE, EXPORTS_COLLECTION

router = APIRouter()


@router.post(
    "",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export terrain + fuel + canopy grids to a landscape GeoTIFF",
    responses=QUOTA_429_RESPONSE,
)
async def create_landscape_export(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        LandscapeExportRequest,
        Body(openapi_examples=CREATE_LANDSCAPE_EXPORT_OPENAPI_EXAMPLES),
    ],
):
    """Assemble terrain, surface fuel model, and canopy grids into an
    8-band LANDFIRE-style landscape GeoTIFF for operational fire behavior
    tools (FlamMap, IFTDSS, WFDSS).

    The output `landscape.tif` carries the standard LANDFIRE band order —
    elevation, slope, aspect, fuel model, canopy cover, canopy height,
    canopy base height, canopy bulk density — with LANDFIRE's int16 scaled
    encodings, embedded CRS, and per-band name/unit metadata. This is the
    format LANDFIRE distributes and IFTDSS accepts for upload.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to retrieve the
    signed download URL.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(EXPORTS_COLLECTION, request)

    source = await validate_landscape_request(body, owner_id, domain)

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
    register_dispatch(request, response, background_tasks)

    return Export(**export_data)
