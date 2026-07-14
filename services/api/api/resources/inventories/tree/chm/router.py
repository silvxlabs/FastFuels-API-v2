"""
api/v2/resources/inventories/tree/chm/router.py

Router for CHM extraction inventory creation.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.utils import validate_band_unit, validate_grid_has_band
from api.resources.inventories.schema import CHM_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.chm.examples import CREATE_CHM_OPENAPI_EXAMPLES
from api.resources.inventories.tree.chm.schema import (
    ChmInventorySource,
    CreateChmInventoryRequest,
)
from api.resources.inventories.utils import validate_feature_conditions
from api.resources.modifications import stringify_modification_coordinates
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    STANDGEN_QUEUE,
    STANDGEN_SERVICE,
)

router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION


@router.post(
    "",
    response_model=Inventory,
    status_code=status.HTTP_201_CREATED,
    summary="Create an inventory from a Canopy Height Model (CHM)",
    responses=QUOTA_429_RESPONSE,
)
async def create_chm_inventory(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateChmInventoryRequest,
        Body(openapi_examples=CREATE_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create CHM Extraction Inventory

    Extracts individual tree records from a Canopy Height Model (CHM) grid
    using a specified stem isolation algorithm.

    Currently supports two algorithms:
    1. **Local Maximum Filtering (LMF)**: Sweeps a fixed circular window across the CHM.
    2. **Variable Window Filtering (VWF)**: Sweeps a dynamic window that scales in size based on the height of the canopy, allowing for better detection of mixed stand structures.

    ## Request Body

    - **source_chm_grid_id**: (required) ID of a completed CHM grid.
    - **algorithm**: (optional) Configuration for the stem isolation algorithm. Must specify `"name": "lmf"` or `"name": "vwf"`. Defaults to LMF.
    - **type**: (optional) Entity type. Default: ``"tree"``.
    - **name**: (optional) Name for the inventory.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing inventories.

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The
    backend (Standgen) will process the extraction asynchronously and update
    status to ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_conditions(
        [*body.modifications, *body.treatments], owner_id, domain_id
    )

    # Validate source CHM grid exists, is owned, in this domain, and completed
    _, source_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_chm_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # Any completed grid with a 'chm' band qualifies — a canopy-provider grid
    # (Meta/NAIP/LANDFIRE) or a user-uploaded CHM (e.g. ALS/TLS-derived). The band
    # must be in meters: the stem-isolation algorithm compares pixel values to a
    # meter height threshold and sizes detection windows against the metric-CRS
    # resolution without ever converting units, so a non-meter CHM would run but
    # produce silently-wrong tree heights and windows.
    validate_grid_has_band(source_grid_data, body.source_chm_grid_id, "chm")
    validate_band_unit(source_grid_data, body.source_chm_grid_id, "chm", "m")

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = ChmInventorySource(
        source_chm_grid_id=body.source_chm_grid_id,
        source_chm_grid_checksum=source_grid_data.get("checksum"),
        algorithm=body.algorithm,
    )

    inventory_data = {
        "id": inventory_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": body.type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": stringify_modification_coordinates(
            [m.model_dump() for m in body.modifications]
        ),
        "treatments": stringify_modification_coordinates(
            [t.model_dump() for t in body.treatments]
        ),
        "columns": [c.model_dump() for c in CHM_INVENTORY_COLUMNS],
        "georeference": None,  # Will be set by standgen
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)
    register_dispatch(request, response, background_tasks)

    return Inventory(**inventory_data)
