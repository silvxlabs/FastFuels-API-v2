"""
api/v2/resources/grids/uniform/router.py

Router for uniform (constant-value) grid source endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.uniform.examples import CREATE_UNIFORM_OPENAPI_EXAMPLES
from api.resources.grids.uniform.schema import (
    CreateUniformRequest,
    UniformSource,
    build_uniform_bands,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a uniform (constant-value) grid",
)
async def create_uniform_grid(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateUniformRequest,
        Body(openapi_examples=CREATE_UNIFORM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Uniform Grid

    Creates a grid where every cell is filled with a constant value for each
    specified band. Useful for fuel moisture scenarios, constant fuel loads,
    and other spatially-uniform inputs.

    ## Request Body

    - **resolution**: (required) Grid resolution in meters (>= 1). No default
      since uniform grids have no "native resolution."
    - **bands**: (required) One or more bands, each with a key and value.
      Band keys must be unique.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Available Bands

    **Fuel moisture** (unit: %): `fuel_moisture.1hr`, `fuel_moisture.10hr`,
    `fuel_moisture.100hr`, `fuel_moisture.live_herb`, `fuel_moisture.live_woody`

    **Curing** (unit: %): `curing`

    **Fuel load** (unit: kg/m**2): `fuel_load.1hr`, `fuel_load.10hr`,
    `fuel_load.100hr`, `fuel_load.live_herb`, `fuel_load.live_woody`

    **Fuel depth** (unit: m): `fuel_depth`

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    generate the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = UniformSource(bands=body.bands, resolution=body.resolution)
    bands = build_uniform_bands(body.bands)

    grid_data = {
        "id": grid_id,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": [m.model_dump() for m in body.modifications],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
