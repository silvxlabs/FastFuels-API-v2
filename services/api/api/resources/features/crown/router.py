"""
api/v2/resources/features/crown/router.py

Router for Crown feature creation from a CHM.
"""

import math
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    HTTPException,
    Request,
    Response,
    status,
)

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.crown_segmentation import MAX_CROWN_SEGMENTATION_RESOLUTION_M
from api.resources.features.crown.examples import CREATE_CROWN_OPENAPI_EXAMPLES
from api.resources.features.crown.schema import (
    ChmCrownSource,
    CreateChmCrownFeatureRequest,
)
from api.resources.features.schema import Feature, FeatureType
from api.resources.grids.utils import validate_band_unit, validate_grid_has_band
from api.resources.inventories.schema import TREE_ID_COLUMN
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    FEATURES_COLLECTION,
    FEATURES_QUEUE,
    FEATURES_SERVICE,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
)

router = APIRouter()

COLLECTION = FEATURES_COLLECTION


def validate_grid_resolution(grid_data: dict, grid_id: str, max_cell_size: float):
    """Reject a grid whose cells are coarser than ``max_cell_size`` metres."""
    transform = (grid_data.get("georeference") or {}).get("transform")
    if not transform:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Grid {grid_id} has no georeference.",
        )
    a, b, _, d, e, _ = transform[:6]
    cell_size = max(math.hypot(a, d), math.hypot(b, e))
    if cell_size > max_cell_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Grid {grid_id} has {cell_size:g} m cells. Crown segmentation "
                f"needs cells of {max_cell_size:g} m or finer."
            ),
        )


@router.post(
    "/chm",
    response_model=Feature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a crown feature from a CHM",
    responses=QUOTA_429_RESPONSE,
)
async def create_chm_crown_feature(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateChmCrownFeatureRequest,
        Body(openapi_examples=CREATE_CROWN_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create CHM Crown Feature

    Outlines each tree's crown by growing it on a Canopy Height Model (CHM)
    from the tree's position in an existing tree inventory.

    Each inventory tree seeds one crown in the CHM cell that contains its
    `x`, `y`. Crowns grow outward one ring of 4-connected cells at a time
    (Dalponte & Coomes, 2016) and never overlap. A tree gets no crown if it
    lies outside the CHM, sits on a no-data cell, or shares its cell with a
    taller tree (ties go to the lower `tree_id`).

    The completed feature holds one row per crown with columns `tree_id` (the
    seed tree's `tree_id`) and `geometry` (a `Polygon` in the domain CRS,
    traced along CHM cell edges). Read it from the feature's `/data`
    endpoints.

    ## Request Body

    - **source_inventory_id**: (required) ID of a completed tree inventory
      with a `tree_id` column.
    - **source_chm_grid_id**: (required) ID of a completed grid with a `chm`
      band in metres, at 2 m resolution or finer.
    - **crown_segmentation**: (optional) Segmentation settings. Defaults are
      used for any field left out.
    - **name**, **description**, **tags**: (optional) Standard metadata.

    ## Response

    Returns the created Feature resource with status `"pending"`. The backend
    worker segments the crowns asynchronously and updates status to
    `"completed"` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    _, inventory_snapshot = await get_document_async(
        INVENTORIES_COLLECTION,
        body.source_inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = inventory_snapshot.to_dict()
    if inventory_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' has type "
                f"'{inventory_data.get('type')}'. Crowns require a tree inventory."
            ),
        )
    column_keys = {
        c["key"] if isinstance(c, dict) else c
        for c in inventory_data.get("columns", [])
    }
    if TREE_ID_COLUMN.key not in column_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' has no "
                f"'{TREE_ID_COLUMN.key}' column."
            ),
        )

    _, grid_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_chm_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    grid_data = grid_snapshot.to_dict()
    validate_grid_has_band(grid_data, body.source_chm_grid_id, "chm")
    validate_band_unit(grid_data, body.source_chm_grid_id, "chm", "m")
    validate_grid_resolution(
        grid_data, body.source_chm_grid_id, MAX_CROWN_SEGMENTATION_RESOLUTION_M
    )

    feature_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = ChmCrownSource(
        source_inventory_id=body.source_inventory_id,
        source_inventory_checksum=inventory_data.get("checksum"),
        source_chm_grid_id=body.source_chm_grid_id,
        source_chm_grid_checksum=grid_data.get("checksum"),
        crown_segmentation=body.crown_segmentation,
    )

    feature_data = {
        "id": feature_id,
        "domain_id": domain_id,
        "type": FeatureType.crown.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(mode="json"),
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, feature_id, feature_data)

    await create_http_task_async(FEATURES_QUEUE, FEATURES_SERVICE, feature_id)
    register_dispatch(request, response, background_tasks)

    return Feature(**feature_data)
