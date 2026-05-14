"""
api/v2/resources/inventories/exports/router.py

Router for inventory export creation endpoints.

URL pattern: POST /domains/{domain_id}/inventories/{inventory_id}/exports/{format}

Lifecycle endpoints (list, get, patch, delete) are at /v2/exports/.
"""

import uuid
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.exports.schema import (
    Export,
    ExportInventoryRequest,
    InventoryExportFormat,
    InventoryExportSource,
)
from api.resources.inventories.exports.examples import (
    CREATE_INVENTORY_EXPORT_OPENAPI_EXAMPLES,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    EXPORTER_QUEUE,
    EXPORTER_SERVICE,
    EXPORTS_COLLECTION,
    INVENTORIES_COLLECTION,
)

router = APIRouter()


@router.post(
    "/{format}",
    response_model=Export,
    status_code=status.HTTP_201_CREATED,
    summary="Export an inventory",
)
async def create_inventory_export(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
    format: InventoryExportFormat,
    body: Annotated[
        ExportInventoryRequest,
        Body(openapi_examples=CREATE_INVENTORY_EXPORT_OPENAPI_EXAMPLES),
    ],
):
    """Export an inventory to the specified format.

    Supported formats: `parquet` (zipped), `csv`, `geojson`, `geopackage`.

    The inventory must belong to this domain and have status `completed`.
    If `columns` is specified, only those columns are included; otherwise
    all columns are exported.

    Returns an Export resource with status `pending`. Poll
    `GET /exports/{export_id}` until status is `completed` to get the
    signed download URL.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Validate inventory exists, is owned, belongs to domain, and is completed
    _, inventory_snapshot = await get_document_async(
        INVENTORIES_COLLECTION,
        inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = inventory_snapshot.to_dict()

    # Validate column subset if provided
    if body.columns is not None:
        available_keys = [col["key"] for col in inventory_data["columns"]]
        missing = [c for c in body.columns if c not in available_keys]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Columns not found in inventory: {missing}. "
                    f"Available columns: {available_keys}"
                ),
            )

    # Extract CRS from inventory georeference
    crs = None
    if inventory_data.get("georeference"):
        crs = inventory_data["georeference"].get("crs")

    # Build source metadata
    source = InventoryExportSource(
        name=format.value,
        inventory_id=inventory_id,
        columns=body.columns,
        crs=crs,
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
