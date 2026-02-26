"""
api/v2/resources/grids/topography/router.py

Router for Topography grid product endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.topography.examples import (
    CREATE_3DEP_TOPOGRAPHY_OPENAPI_EXAMPLES,
    CREATE_LANDFIRE_TOPOGRAPHY_OPENAPI_EXAMPLES,
)
from api.resources.grids.topography.schema import (
    CreateLandfireTopographyRequest,
    CreateThreeDepTopographyRequest,
    LandfireTopographySource,
    ThreeDepTopographySource,
    build_topography_bands,
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
    summary="Create a grid from LANDFIRE topographic data",
)
async def create_landfire_topography(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireTopographyRequest,
        Body(openapi_examples=CREATE_LANDFIRE_TOPOGRAPHY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE Topography Grid

    Creates a grid with topographic data from LANDFIRE at 30m resolution.

    Available bands:
    - **elevation**: meters above sea level
    - **slope**: degrees (0-90)
    - **aspect**: degrees clockwise from north (0-360)

    By default all three bands are included. Use the `bands` field to select
    a subset.

    ## Request Body

    - **bands**: (optional) Which bands to include. Default: all three.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2020".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireTopographySource(version=body.version, bands=body.bands)
    bands = build_topography_bands(body.bands)

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
        "chunk_shape": CHUNK_SHAPE,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)


@router.post(
    "/3dep",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from 3DEP topographic data",
)
async def create_3dep_topography(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateThreeDepTopographyRequest,
        Body(openapi_examples=CREATE_3DEP_TOPOGRAPHY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create 3DEP Topography Grid

    Creates a grid with topographic data from USGS 3DEP at selectable resolution.

    Available resolutions:
    - **1m**: Seamless 1-meter (S1M). Coverage varies by region; areas without
      S1M data will return a COVERAGE_ERROR.
    - **10m**: 1/3 arc-second seamless (default)
    - **30m**: 1 arc-second seamless

    Available bands:
    - **elevation**: meters above sea level (default)
    - **slope**: degrees (0-90)
    - **aspect**: degrees clockwise from north (0-360)

    Slope and aspect are computed locally from the DEM using Horn's method.

    ## Request Body

    - **resolution**: (optional) Resolution in meters: 1, 10, or 30. Default: 10.
    - **bands**: (optional) Which bands to include. Default: elevation only.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = ThreeDepTopographySource(resolution=body.resolution, bands=body.bands)
    bands = build_topography_bands(body.bands)

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
        "chunk_shape": CHUNK_SHAPE,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Griddle for processing
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
