"""
api/v2/resources/grids/fuel_moisture/dead/fosberg/router.py

Router for the Fosberg 1-hour dead fuel moisture content grid. Derives a
single 2-D `fuel_moisture.dead.1hr` band in griddle from a topography grid
(slope + aspect) and a leaflux irradiance grid (surface shading).
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Request, Response, status

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.fuel_moisture.dead.fosberg.examples import (
    CREATE_FOSBERG_FUEL_MOISTURE_OPENAPI_EXAMPLES,
)
from api.resources.grids.fuel_moisture.dead.fosberg.schema import (
    DEAD_1HR_BAND,
    CreateFosbergFuelMoistureRequest,
    FosbergFuelMoistureSource,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    validate_grid_dimensionality,
    validate_grid_has_band,
    validate_grids_share_horizontal_lattice,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

SURFACE_IRRADIANCE_KEY = "irradiance.surface.relative"


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Fosberg 1-hour dead fuel moisture grid",
    responses=QUOTA_429_RESPONSE,
)
async def create_fosberg_fuel_moisture_grid(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateFosbergFuelMoistureRequest,
        Body(openapi_examples=CREATE_FOSBERG_FUEL_MOISTURE_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Fosberg 1-hour Dead Fuel Moisture Grid

    Creates a grid with a single continuous band, `fuel_moisture.dead.1hr`
    (percent), computed with the Fosberg & Deeming (1971) 1-hour dead fuel
    moisture model.

    The grid is derived from two completed source grids in the same domain:

    - **Topography** — supplies the `slope` and `aspect` bands (both degrees).
    - **Leaflux irradiance** — supplies `irradiance.surface.relative`, from
      which per-cell shading is derived as `1 - irradiance.surface.relative`.

    The remaining inputs are scalar weather/scenario parameters:
    `dry_bulb_temp` (°F), `relative_humidity` (%), `time` (local HHMM,
    0800-1959), `month`, and `elevation` (site position relative to the
    reference weather station).

    The output inherits the topography grid's domain, CRS, transform, and
    georeference. Keeping `time`/`month` consistent with the sun position that
    produced the irradiance grid is the caller's responsibility.

    ## Response

    Returns the created Grid resource with status "pending". The backend
    computes the moisture surface and updates status to "completed".
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # Topography grid: owned, completed, in this domain, 2D, slope + aspect.
    _, topo_snapshot = await get_document_async(
        collection=COLLECTION,
        document_id=body.source_topography_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    topo_grid = topo_snapshot.to_dict()
    validate_grid_dimensionality(
        grid_data=topo_grid, grid_id=body.source_topography_grid_id, expected=2
    )
    validate_grid_has_band(
        grid_data=topo_grid,
        grid_id=body.source_topography_grid_id,
        required=["slope", "aspect"],
    )

    # Irradiance grid: owned, completed, in this domain, surface band present.
    _, irr_snapshot = await get_document_async(
        collection=COLLECTION,
        document_id=body.source_irradiance_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    irr_grid = irr_snapshot.to_dict()
    validate_grid_has_band(
        grid_data=irr_grid,
        grid_id=body.source_irradiance_grid_id,
        required=SURFACE_IRRADIANCE_KEY,
    )

    # The moisture surface is derived cell-for-cell from both grids and inherits
    # the topography grid's georeference, so the irradiance grid must sit on the
    # same horizontal lattice (the irradiance grid may be 3D; only y/x matter).
    validate_grids_share_horizontal_lattice(
        reference_grid=topo_grid,
        candidate_grid=irr_grid,
    )

    source = FosbergFuelMoistureSource(
        source_topography_grid_id=body.source_topography_grid_id,
        source_topography_grid_checksum=topo_grid.get("checksum"),
        source_irradiance_grid_id=body.source_irradiance_grid_id,
        source_irradiance_grid_checksum=irr_grid.get("checksum"),
        dry_bulb_temp=body.dry_bulb_temp,
        relative_humidity=body.relative_humidity,
        time=body.time,
        month=body.month,
        elevation=body.elevation,
    )

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
        "source": source.model_dump(mode="json"),
        "modifications": [],
        "bands": [DEAD_1HR_BAND.model_dump()],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
