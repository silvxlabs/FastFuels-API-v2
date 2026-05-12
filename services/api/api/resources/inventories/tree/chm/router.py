"""
api/v2/resources/inventories/tree/chm/router.py

Router for CHM extraction inventory creation.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.inventories.schema import BASE_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.chm.examples import CREATE_CHM_OPENAPI_EXAMPLES
from api.resources.inventories.tree.chm.schema import (
    ChmInventorySource,
    CreateChmInventoryRequest,
)
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
)
async def create_chm_inventory(
    request: Request,
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

    # Validate source CHM grid exists, is owned, in this domain, and completed
    _, source_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_chm_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # Verify it is a canopy grid (source.name == "canopy")
    grid_source = source_grid_data.get("source", {})
    if grid_source.get("name") != "canopy":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Grid '{body.source_chm_grid_id}' is not a canopy grid. "
                f"This endpoint requires a canopy grid (with a 'chm' band) as the source."
            ),
        )

    # Verify the grid contains the actual 'chm' band
    grid_bands = source_grid_data.get("bands", [])
    band_keys = [b["key"] if isinstance(b, dict) else b for b in grid_bands]

    if "chm" not in band_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Source grid is missing the required 'chm' band. "
                f"Available bands: {band_keys}. "
                f"This endpoint requires a grid containing a Canopy Height Model."
            ),
        )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = ChmInventorySource(
        source_chm_grid_id=body.source_chm_grid_id,
        algorithm=body.algorithm,
    )

    inventory_data = {
        "id": inventory_id,
        "domain_id": domain_id,
        "type": body.type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [m.model_dump() for m in body.modifications],
        "columns": [c.model_dump() for c in BASE_INVENTORY_COLUMNS],
        "georeference": None,  # Will be set by standgen
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)

    return Inventory(**inventory_data)
