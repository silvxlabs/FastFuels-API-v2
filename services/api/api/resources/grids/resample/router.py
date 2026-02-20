"""
api/v2/resources/grids/resample/router.py

Router for grid resampling endpoints.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.resources.grids.resample.examples import CREATE_RESAMPLE_OPENAPI_EXAMPLES
from api.resources.grids.resample.schema import (
    CreateResampleRequest,
    ResampleSource,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import validate_grid_has_georeference
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid by resampling an existing grid",
)
async def create_resample(
    request: Request,
    domain: VerifiedDomain,
    body: Annotated[
        CreateResampleRequest,
        Body(openapi_examples=CREATE_RESAMPLE_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Resampled Grid

    Resamples an existing grid to a new spatial resolution. This is the key
    operation for unifying grids at a common resolution (e.g., LANDFIRE 30m
    to 2m for QUIC-Fire input).

    The resampled grid propagates domain_id and bands from the source grid.

    ## Request Body

    - **source_grid_id**: (required) Grid to resample. Must have status
      "completed" and a georeference.
    - **resolution**: (required) Target resolution in meters. Minimum 1m.
      Contact the developers if you need sub-meter resolution.
    - **method**: (optional) Default resampling method. One of:
      `nearest`, `bilinear`, `cubic`, `cubic_spline`, `lanczos`,
      `average`, `mode`, `max`, `min`, `median`, `first_quartile`,
      `third_quartile`, `sum`, `root_mean_square`. Default: `bilinear`.
    - **method_overrides**: (optional) Per-band resampling method overrides
      keyed by band key. Useful for using nearest-neighbor on categorical
      bands while using bilinear on continuous bands.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Response

    Returns the created Grid with status "pending". The backend performs the
    resampling and updates status to "completed" when ready. The georeference
    will be null until processing completes, since the exact output shape
    depends on the resampling result.

    ## Notes

    - Domain is propagated from the source grid (resampling doesn't change
      geographic extent).
    - Bands are propagated from the source grid (resampling only changes
      resolution).
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Validate source grid: exists, owned, in this domain, and completed
    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # Validate source grid has a georeference
    validate_grid_has_georeference(source_grid_data, body.source_grid_id)

    # Validate method override keys exist in source bands
    source_band_keys = {b["key"] for b in source_grid_data.get("bands", [])}
    invalid_keys = set(body.method_overrides.keys()) - source_band_keys
    if invalid_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Method override keys not found in source grid bands: "
                f"{sorted(invalid_keys)}. "
                f"Available bands: {sorted(source_band_keys)}"
            ),
        )

    # Propagate bands from source grid
    bands = source_grid_data.get("bands", [])

    # Build the new grid document
    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = ResampleSource(
        source_grid_id=body.source_grid_id,
        target_resolution=body.resolution,
        method=body.method,
        method_overrides=body.method_overrides,
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
        "modifications": [m.model_dump() for m in body.modifications],
        "bands": bands,
        "georeference": None,
        "tags": body.tags,
        "chunk_shape": CHUNK_SHAPE,
        "owner_id": owner_id,
    }

    # Write the data to Firestore
    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue the griddle task
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)

    return Grid(**grid_data)
