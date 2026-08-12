"""
api/v2/resources/grids/solar/irradiance/leaflux/router.py

Router for producing a LeafLux irradiance grid from a source 3D fuel grid's
leaf_area_density band and a solar position.
"""

import uuid
from datetime import datetime
from typing import Annotated

import geopandas as gpd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    HTTPException,
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
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDS_COLLECTION, TREEVOX_QUEUE, TREEVOX_SERVICE
from lib.domain_utils import parse_domain_gdf

from .examples import CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES
from .schema import (
    CreateLeafluxIrradianceRequest,
    IrradianceLeafluxSource,
    build_leaflux_bands,
)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

LEAF_AREA_DENSITY_KEY = "leaf_area_density"


def _domain_centroid_lat_lon(domain) -> tuple[float, float]:
    """Return (lat, lon) of the domain centroid in EPSG:4326."""
    gdf = parse_domain_gdf(domain)
    # TODO: assumes parse_domain_gdf returns a projected CRS
    centroid = gdf.geometry.union_all().centroid
    point = gpd.GeoSeries([centroid], crs=gdf.crs).to_crs("EPSG:4326").iloc[0]
    return float(point.y), float(point.x)


async def _get_validated_source_grid(
    source_grid_id: str, owner_id: str, domain_id: str
) -> Grid:
    _, snapshot = await get_document_async(COLLECTION, source_grid_id)
    if not snapshot.exists:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source grid {source_grid_id!r} does not exist.",
        )
    data = snapshot.to_dict()

    # Redundant with VerifiedDomain + the same-domain check below, but cheap.
    # Can be delegated to get_document_async(owner_id=...) once confirmed.
    if data.get("owner_id") != owner_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source grid {source_grid_id!r} is not owned by the caller.",
        )
    grid = Grid(**data)
    if grid.domain_id != domain_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source grid {source_grid_id!r} is not in this domain.",
        )

    # Check that grid is completed
    if grid.status != JobStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source grid {source_grid_id!r} is not completed.",
        )

    # Check for 3D grid
    if grid.georeference is None or len(grid.georeference.shape) != 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source grid {source_grid_id!r} must be a 3D grid.",
        )

    # Check that we have a LAD band
    if not any(b.key == LEAF_AREA_DENSITY_KEY for b in grid.bands):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Source grid {source_grid_id!r} has no {LEAF_AREA_DENSITY_KEY!r} "
                "band. Voxelize the inventory with that band first."
            ),
        )
    return grid


# TODO: are there any other checks needed here?
async def _validate_terrain_grid(
    terrain_grid_id: str, owner_id: str, domain_id: str
) -> None:
    _, snapshot = await get_document_async(COLLECTION, terrain_grid_id)
    if not snapshot.exists:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} does not exist.",
        )
    data = snapshot.to_dict()

    # Check owner
    if data.get("owner_id") != owner_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} is not owned by the caller.",
        )

    # Check domain
    grid = Grid(**data)
    if grid.domain_id != domain_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} is not in this domain.",
        )

    # Check that grid is completed
    if grid.status != JobStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} is not completed.",
        )

    # Check that is 2D
    if grid.georeference is None or len(grid.georeference.shape) != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} must be a 2D grid.",
        )

    # Check for elevation band
    if not any(b.key == "elevation" for b in grid.bands):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Terrain grid {terrain_grid_id!r} has no 'elevation' band.",
        )


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
    latitude, longitude = body.latitude, body.longitude
    if latitude is None or longitude is None:
        latitude, longitude = _domain_centroid_lat_lon(domain)

    source_grid = await _get_validated_source_grid(
        body.source_grid_id, owner_id, domain_id
    )

    if body.source_terrain_grid_id is not None:
        await _validate_terrain_grid(body.source_terrain_grid_id, owner_id, domain_id)

    source = IrradianceLeafluxSource(
        source_grid_id=body.source_grid_id,
        source_grid_checksum=source_grid.checksum,
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
