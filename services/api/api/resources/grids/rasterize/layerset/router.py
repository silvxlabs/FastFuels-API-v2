"""
api/v2/resources/grids/rasterize/layerset/router.py

Router for the layerset rasterize grid product endpoint.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas
from api.resources.grids.rasterize.layerset.examples import (
    CREATE_LAYERSET_RASTERIZE_OPENAPI_EXAMPLES,
)
from api.resources.grids.rasterize.layerset.schema import (
    CreateLayersetRasterizeRequest,
    LayersetSource,
    build_layerset_bands,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_target_grid_alignment,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    FEATURES_COLLECTION,
    GRIDDLE_QUEUE,
    GRIDDLE_SERVICE,
    GRIDS_COLLECTION,
)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "/layerset",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid by rasterizing a layerset",
    responses=QUOTA_429_RESPONSE,
)
async def create_layerset_rasterize(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLayersetRasterizeRequest,
        Body(openapi_examples=CREATE_LAYERSET_RASTERIZE_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Layerset-Rasterized Grid

    Rasterizes a previously-uploaded fuelbed layerset into a grid aligned
    to the domain (default) or to a target grid.

    The `layerset_id` must reference a Feature uploaded for this domain via
    `POST /domains/{domain_id}/features/layerset` and owned by the caller.

    ## Request Body

    - **layerset_id**: (required) Feature ID of the layerset to rasterize.
    - **overlap_method**: (optional) Per-cell reduction when polygons of the
      same `fuel_type` overlap a single cell. One of `mean`, `max`, `min`.
      Default: `mean`. (Loading is always summed across overlapping polygons
      regardless of this setting.)
    - **alignment**: (optional) See alignment docs. Default: anchored to domain.
    - **extent_buffer_cells**: (optional) Buffer in result-grid cells around
      the domain extent. Cells inside the buffered extent that fall outside
      polygon coverage are populated with the rasterizer's fill value.
    - **name**, **description**, **tags**, **modifications**: standard grid metadata.

    ## Response

    Returns the created Grid resource with status `pending`. The backend
    fetches the layerset GeoJSON from GCS, rasterizes it, and updates the
    status to `completed` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # Validate alignment target grid (if any) — raises 404/422 inline.
    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    # Validate the referenced layerset exists, is owned by the caller, and
    # belongs to this domain. get_document_async raises 404 on any mismatch
    # (existence, owner, domain).
    await get_document_async(
        FEATURES_COLLECTION,
        body.layerset_id,
        owner_id=owner_id,
        domain_id=domain_id,
    )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LayersetSource(
        layerset_id=body.layerset_id,
        overlap_method=body.overlap_method,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment.model_dump(),
    )
    bands = build_layerset_bands()

    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
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
