"""
api/v2/resources/grids/exports/quicfire/router.py

Router for the QUIC-Fire combined export endpoint.

URL pattern: POST /v2/domains/{domain_id}/grids/exports/quicfire

Lifecycle endpoints (list, get, patch, delete) are at /v2/exports/.
"""

import uuid
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.exports.schema import Export
from api.resources.grids.exports.quicfire.examples import (
    CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES,
)
from api.resources.grids.exports.quicfire.schema import QuicfireExportRequest
from api.resources.grids.exports.quicfire.validators import (
    validate_quicfire_request,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import EXPORTER_QUEUE, EXPORTER_SERVICE, EXPORTS_COLLECTION

router = APIRouter()


@router.post(
    "",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export combined fuel + topography grids to QUIC-Fire format",
)
async def create_quicfire_export(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        QuicfireExportRequest,
        Body(openapi_examples=CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES),
    ],
):
    """Bundle surface fuel + canopy fuel + (optional) topography grids into a
    QUIC-Fire-loadable zip archive.

    The output zip contains `treesrhof.dat`, `treesmoist.dat`,
    `treesfueldepth.dat`, `metadata.json`, and `domain.geojson` always; plus
    `topo.dat` when a topography role is provided, plus `treesss.dat` when
    both canopy and surface SAVR roles are provided.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to retrieve the
    signed download URL.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    source = await validate_quicfire_request(body, owner_id, domain)

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
