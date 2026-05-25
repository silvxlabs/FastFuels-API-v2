"""
api/v2/resources/grids/topography/router.py

Router for Topography grid product endpoints.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Query, Request, status

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
    ThreeDepCoverageResponse,
    ThreeDepResolution,
    ThreeDepTopographySource,
    build_topography_bands,
)
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_target_grid_alignment,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.threedep import discover_s1m_tiles, discover_tiles_arc_second

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

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireTopographySource(
        version=body.version,
        bands=body.bands,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
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
        "modifications": dump_modifications_for_firestore(body.modifications),
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

    - **source_resolution**: (optional) Source product family in meters:
      1, 10, or 30. Default: 10. To change the *output* cell size, set
      ``alignment.resolution``.
    - **bands**: (optional) Which bands to include. Default: elevation only.
    - **alignment**: (optional) Output alignment target. See alignment docs.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = ThreeDepTopographySource(
        source_resolution=body.source_resolution,
        bands=body.bands,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
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
        "modifications": dump_modifications_for_firestore(body.modifications),
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


@router.get(
    "/3dep/coverage",
    response_model=ThreeDepCoverageResponse,
    summary="Check 3DEP tile coverage for a domain",
)
async def check_3dep_coverage(
    domain: VerifiedDomain,
    resolution: Annotated[
        ThreeDepResolution,
        Query(description="3DEP resolution in meters: 1, 10, or 30"),
    ] = ThreeDepResolution.one_meter,
):
    """
    # Check 3DEP Tile Coverage

    Immediate pre-flight check that reports which 3DEP tiles are available
    for the domain at the requested resolution. Use this before creating a
    3DEP grid to avoid waiting for async processing only to discover a
    coverage gap — especially useful for 1m (S1M) data where coverage is
    regional.

    ## Query Parameters

    - **resolution**: Resolution in meters: 1, 10, or 30. Default: 1.

    ## Response

    Returns tile availability, count, URLs, and (for 1m) acquisition dates.
    """
    roi = parse_domain_gdf(domain)

    if resolution in (10, 30):
        tile_urls = discover_tiles_arc_second(roi, resolution)
        acquisition_dates = None
    else:
        tile_urls, acquisition_dates = await asyncio.to_thread(discover_s1m_tiles, roi)
        acquisition_dates = acquisition_dates or None

    return ThreeDepCoverageResponse(
        resolution=resolution,
        available=len(tile_urls) > 0,
        tile_count=len(tile_urls),
        tiles=tile_urls,
        acquisition_dates=acquisition_dates,
    )
