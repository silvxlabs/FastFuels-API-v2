"""
api/v2/resources/grids/fccs/router.py

Router for FCCS grid product endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.fccs.examples import (
    CREATE_LANDFIRE_FCCS_OPENAPI_EXAMPLES,
)
from api.resources.grids.fccs.schema import (
    FCCS_BAND,
    CreateLandfireFccsRequest,
    LandfireFccsSource,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/landfire",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE FCCS",
)
async def create_landfire_fccs(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireFccsRequest,
        Body(openapi_examples=CREATE_LANDFIRE_FCCS_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE FCCS Grid

    Creates a grid with FCCS fuelbed IDs from LANDFIRE at 30m resolution.

    The grid contains a single categorical band (`fccs`) with fuel
    classification system fuelbed IDs (e.g., 26, 598, 34721).

    To convert fuelbed IDs to fuel parameters (fuel loads, SAV, depth),
    use the `/grids/lookup/fccs` endpoint.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2023".
    - **remove_bare_ground**: (optional) Remove bare ground cells (fuelbed ID 0),
                              replaced by neighboring majority. Default: False.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireFccsSource(
        version=body.version,
        remove_bare_ground=body.remove_bare_ground,
    )

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": dump_modifications_for_firestore(body.modifications),
        "bands": [FCCS_BAND.model_dump()],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
