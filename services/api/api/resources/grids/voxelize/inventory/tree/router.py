"""
api/v2/resources/grids/voxelize/inventory/tree/router.py

Router for voxelizing a tree inventory into a 3D tree fuel grid.

Takes a tree inventory resource and voxelizes it into a 3D grid of canopy
fuel properties using crown profile and biomass models from fastfuels-core.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.schema import Grid
from api.resources.grids.voxelize.inventory.tree.examples import (
    CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES,
)
from api.resources.grids.voxelize.inventory.tree.schema import (
    CreateTreeInventoryRequest,
    TreeInventorySource,
    build_tree_bands,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    TREEVOX_QUEUE,
    TREEVOX_SERVICE,
)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a 3D tree fuel grid from a tree inventory",
)
async def create_tree_inventory_grid(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateTreeInventoryRequest,
        Body(openapi_examples=CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Tree Inventory Grid

    Voxelizes a tree inventory into a 3D canopy fuel grid. Each tree's crown
    is discretized onto the voxel grid using a species-specific crown profile
    model, and per-voxel fuel properties (bulk density, moisture, SAV) are
    computed from biomass and moisture models.

    This is a 3D grid product — resampling and modifications are not
    supported. Apply modifications to the source inventory before voxelizing.

    ## Request Body

    - **source_inventory_id**: (required) ID of a completed tree inventory.
    - **resolution**: (optional) Voxel resolution in meters. Defaults to
      `{"horizontal": 2.0, "vertical": 1.0}`. All components must be positive.
    - **bands**: (optional) Which output bands to produce. Defaults to
      `["bulk_density.foliage.live"]`. Must be non-empty and contain no
      duplicates. Branchwood and fine bands are accepted by the API, but
      Treevox currently fails those jobs with a not-implemented processing
      error.
    - **crown_profile_model**: (optional) Crown geometry model. One of
      `purves` (default) or `beta`.
    - **biomass_source**: (optional) Biomass source and requested components. The
      default uses NSVB allometry for foliage. Inventory-column sources must
      provide per-tree kg values for each requested direct component.
    - **max_crown_radius_source**: (optional) Source of each tree's maximum
      crown radius. Defaults to the crown profile model's allometric value;
      pass `{"type": "inventory_column", "column": <name>}` to read a per-tree
      maximum radius (m) from an inventory column (e.g. derived from LiDAR).
      The crown profile model still controls the crown shape — only the peak
      radius is rescaled.
    - **moisture_model**: (optional) Live/dead fuel moisture configuration.
      Required shape: `{"live": {"method": "uniform", "value": <percent>}}`
      and/or `{"dead": {"method": "uniform", "value": <percent>}}`.
      Applied only when matching `fuel_moisture.*` bands are requested. Live
      defaults to 100%; dead defaults to 10%.
    - **name**, **description**, **tags**: (optional) Standard metadata.

    ## Response

    Returns the created Grid resource with status `"pending"` and
    `georeference: null`. The Treevox backend performs voxelization
    asynchronously and updates the grid to `"completed"` with a
    `Georeference3D` when done.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Validate source inventory: exists, owned, in this domain, and completed.
    _, inventory_snapshot = await get_document_async(
        INVENTORIES_COLLECTION,
        body.source_inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = inventory_snapshot.to_dict()

    # Must be a tree inventory — other entity types can't be voxelized this way.
    if inventory_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' has type "
                f"'{inventory_data.get('type')}'. This endpoint requires "
                f"a tree inventory."
            ),
        )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = TreeInventorySource(
        source_inventory_id=body.source_inventory_id,
        resolution=body.resolution,
        bands=body.bands,
        crown_profile_model=body.crown_profile_model,
        biomass_source=body.biomass_source,
        max_crown_radius_source=body.max_crown_radius_source,
        moisture_model=body.moisture_model,
        seed=body.seed,
    )
    bands = build_tree_bands(body.bands)

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(mode="json", exclude_none=True),
        # 3D grids do not support modifications — always empty.
        "modifications": [],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        # chunks is computed by Treevox once the 3D grid dimensions
        # are known from the domain bbox + resolution.
        "chunks": None,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Treevox for voxelization.
    await create_http_task_async(TREEVOX_QUEUE, TREEVOX_SERVICE, grid_id)

    return Grid(**grid_data)
