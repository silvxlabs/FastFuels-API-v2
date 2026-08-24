"""
api/v2/resources/grids/solar/irradiance/leaflux/router.py

Router for producing a LeafLux irradiance grid from a source 3D fuel grid's
leaf_area_density band and a solar position.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Request,
    Response,
    status,
)

# get_document_async returns (DocumentReference, DocumentSnapshot); inspect
# snapshot.exists / snapshot.to_dict(). See tests/db/test_documents_*.
from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.schema import Grid
from api.resources.grids.utils import (
    validate_grid_dimensionality,
    validate_grid_has_band,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDS_COLLECTION, TREEVOX_QUEUE, TREEVOX_SERVICE
from lib.domain_utils import domain_centroid_lat_lon

from .examples import CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES
from .schema import (
    CreateLeafluxIrradianceRequest,
    IrradianceLeafluxSource,
    build_leaflux_bands,
)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

LEAF_AREA_DENSITY_KEY = "leaf_area_density"


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a 3D LeafLux irradiance grid from a source fuel grid",
    responses=QUOTA_429_RESPONSE,
)
async def create_leaflux_irradiance_grid(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLeafluxIrradianceRequest,
        Body(openapi_examples=CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES),
    ],
):
    """Create a pending irradiance Grid and dispatch it to Treevox for compute."""
    owner_id = request.state.id
    domain_id = domain["id"]

    # ***NOTE: quota enforcement + dispatch registration mirror the topography
    # and treevox routers; kept so this endpoint counts against create quotas.
    await enforce_create_quotas(COLLECTION, request)

    # Resolve lat/lon from the domain centroid when either is not supplied.
    latitude, longitude = domain_centroid_lat_lon(domain)

    # Check that grid exists, is owned, complete, and in domain
    _, source_snapshot = await get_document_async(
        collection=COLLECTION,
        document_id=body.source_grid_id,
        owner_id=owner_id,
        document_status="completed",
    )

    # Validate that we have LAD band
    grid_data = source_snapshot.to_dict()
    validate_grid_has_band(
        grid_data=grid_data, grid_id=body.source_grid_id, required=LEAF_AREA_DENSITY_KEY
    )

    # Validate that is 3D
    validate_grid_dimensionality(
        grid_data=grid_data, grid_id=body.source_grid_id, expected=3
    )

    if body.source_terrain_grid_id is not None:
        # Validate terrain grid exists, is owned, is completed
        _, terrain_source_snapshot = await get_document_async(
            collection=COLLECTION,
            document_id=body.source_terrain_grid_id,
            owner_id=owner_id,
            document_status="completed",
        )
        terrain_grid_data = terrain_source_snapshot.to_dict()

        # Validate we have elevation band
        validate_grid_has_band(
            grid_data=terrain_grid_data,
            grid_id=body.source_terrain_grid_id,
            required="elevation",
        )

        # Validate is 2D
        validate_grid_dimensionality(
            grid_data=terrain_grid_data,
            grid_id=body.source_terrain_grid_id,
            expected=2,
        )

    source = IrradianceLeafluxSource(
        source_grid_id=body.source_grid_id,
        source_grid_checksum=grid_data.get("checksum"),
        source_terrain_grid_id=body.source_terrain_grid_id,
        bands=body.bands,
        latitude=latitude,
        longitude=longitude,
        date_time=body.date_time,
        extinction_coefficient=body.extinction_coefficient,
    )
    bands = build_leaflux_bands(body.bands)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(mode="json", exclude_none=True),
        "modifications": [],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        "chunks": None,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    await create_http_task_async(TREEVOX_QUEUE, TREEVOX_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
