"""
api/v2/resources/inventories/tree/pim/router.py

Router for PIM expansion inventory creation.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.inventories.schema import BASE_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.pim.examples import CREATE_PIM_OPENAPI_EXAMPLES
from api.resources.inventories.tree.pim.schema import (
    CreatePimInventoryRequest,
    PimInventorySource,
)
from api.resources.inventories.utils import (
    validate_feature_conditions,
    validate_inventory_wide_treatment_area,
)
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
    summary="Create an inventory from PIM expansion",
    responses=QUOTA_429_RESPONSE,
)
async def create_pim_inventory(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreatePimInventoryRequest,
        Body(openapi_examples=CREATE_PIM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create PIM Expansion Inventory

    Expands a Plot Imputation Map (PIM) grid into individual tree records
    with spatial coordinates.

    A PIM grid maps each 30m cell to an FIA plot ID. This endpoint takes
    that mapping and generates a full tree inventory using an inhomogeneous
    Poisson point process:

    1. Tree density (trees per area) is interpolated from plot-level data
       onto a sub-cell grid (15m resolution)
    2. Plot IDs are assigned to each sub-cell via nearest-neighbor
       interpolation (Voronoi tessellation)
    3. For each sub-cell, a Poisson-distributed random count of trees is
       drawn from the local density
    4. Trees are sampled from the assigned plot's tree list, weighted by
       trees-per-area (TPA)
    5. Each tree receives a random coordinate within its sub-cell

    The result is a spatially explicit tree inventory that preserves the
    species composition and size distributions of the FIA plots while
    producing realistic spatial patterns.

    The PIM endpoint is source-agnostic: it works the same regardless of
    whether the source grid is from TreeMap, BIGMAP, or FSE. The grid's
    own ``source`` field carries that lineage.

    ## Request Body

    - **source_pim_grid_id**: (required) ID of a completed PIM grid.
    - **seed**: (optional) Random seed for reproducibility. Generated
      randomly if omitted.
    - **point_process**: (optional) Spatial point process for coordinate
      assignment. Default: ``"inhomogeneous_poisson"``.
    - **type**: (optional) Entity type. Default: ``"tree"``.
    - **name**: (optional) Name for the inventory.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing inventories.

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The
    backend (Standgen) will process the expansion asynchronously and update
    status to ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_conditions(
        [*body.modifications, *body.treatments], owner_id, domain_id
    )
    validate_inventory_wide_treatment_area(domain, body.treatments)

    # Validate source PIM grid exists, is owned, in this domain, and completed
    _, source_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_pim_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # Verify it is a PIM grid (source.name == "pim")
    grid_source = source_grid_data.get("source", {})
    if grid_source.get("name") != "pim":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Grid '{body.source_pim_grid_id}' is not a PIM grid. "
                f"This endpoint requires a PIM grid as the source."
            ),
        )

    # Validate the grid contains the plot ID band needed for expansion
    product = grid_source.get("product", "treemap")
    required_bands = {"treemap": "tm_id"}
    required_band = required_bands.get(product)
    if required_band:
        grid_bands = source_grid_data.get("bands", [])
        band_keys = [b["key"] if isinstance(b, dict) else b for b in grid_bands]
        if required_band not in band_keys:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Source PIM grid is missing the required '{required_band}' band. "
                    f"Available bands: {band_keys}. "
                    f"Create a PIM grid that includes '{required_band}' for inventory expansion."
                ),
            )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = PimInventorySource(
        source_pim_grid_id=body.source_pim_grid_id,
        source_pim_grid_checksum=source_grid_data.get("checksum"),
        point_process=body.point_process,
        seed=body.seed,
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
        "columns": [c.model_dump() for c in BASE_INVENTORY_COLUMNS],
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)

    return Inventory(**inventory_data)
