"""
api/v2/resources/grids/duet/router.py

Router for creating DUET surface fuel grids from a 3D tree grid.

DUET is a derived grid: it consumes a completed tree grid and produces 2D
surface bands. Treevox runs the DUET binary asynchronously.
"""

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
from api.resources.grids.duet.examples import CREATE_DUET_OPENAPI_EXAMPLES
from api.resources.grids.duet.schema import (
    DUET_REQUIRED_SOURCE_BANDS,
    CreateDuetRequest,
    DuetSource,
    build_duet_bands,
)
from api.resources.grids.schema import Grid
from api.resources.grids.utils import validate_grid_has_band
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import GRIDS_COLLECTION, TREEVOX_QUEUE, TREEVOX_SERVICE

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a surface fuel grid with DUET",
    responses=QUOTA_429_RESPONSE,
)
async def create_duet_grid(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateDuetRequest,
        Body(openapi_examples=CREATE_DUET_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create a DUET Surface Fuel Grid

    Runs DUET (Distribution of Understory using Elliptical Transport) over a 3D
    tree grid to produce 2D surface fuels. DUET drops leaf and needle litter
    from each tree's crown along wind-driven elliptical fall trajectories, then
    grows grass as a function of shade and litter cover — so litter accumulates
    under and downwind of crowns, and grass fills the gaps between them.

    ## What DUET does and does not give you

    DUET supplies the **spatial pattern** of surface fuels, keyed to real canopy
    structure. It does **not** supply physical magnitudes: raw DUET loadings are
    idiosyncratic to the model and should not be read as fuel loads or fed to a
    fire model as-is. Use `calibration` to impose magnitudes you trust — from
    field data, from the literature, or from an FBFM40 grid.

    ## Request Body

    - **source_grid_id**: (required) A completed 3D tree grid carrying the
      `bulk_density.foliage.live`, `spcd`, and `fuel_moisture.live` bands.
      Create one with `POST /grids/voxelize/inventory/tree`, requesting those
      three bands — `spcd` in particular is not voxelized by default.
    - **years_since_burn**: (required) Years of litter accumulation to simulate,
      1–100. DUET starts from the year of the last fire, with grass and litter
      consumed, so this is the stand's time since fire. It is the single most
      consequential parameter: a low value yields almost no litter because there
      has been no time for any to fall. It also drives runtime.
    - **wind_direction**: (optional) Degrees clockwise from north. Default 270.
    - **wind_variability**: (optional) Angular spread in degrees. Default 30.
    - **bands**: (optional) Output bands. Defaults to `fuel_load.grass` and
      `fuel_load.litter`. DUET separates fuels by type rather than size class,
      so bands are named for `grass`, `litter` (and its `litter.coniferous` /
      `litter.deciduous` parts), and `total`.
    - **calibration**: (optional) Per-parameter, per-fuel-type targets. Omit to
      store raw output.
    - **name**, **description**, **tags**: (optional) Standard metadata.

    ## Calibration

    Each of `fuel_load`, `fuel_depth`, and `fuel_moisture` is calibrated
    independently, and within each, per fuel type (`grass`, `coniferous`,
    `deciduous`, `litter`, or `all` — which is exclusive of the others). Methods:

    - `maxmin` — rescale to a target maximum and minimum. Best when fuel data
      are limited, or when their distribution does not resemble DUET's.
    - `meansd` — rescale to a target mean and standard deviation. Appropriate
      only when the targets come from a dataset large enough to approximate a
      normal distribution.
    - `constant` — assign a single value. Reasonable only when that is the only
      value available.

    Calibration rescales only cells that already carry fuel; cells DUET left
    empty stay empty. A consequence worth expecting: where cover is sparse, the
    domain-wide mean will sit well below a `meansd` target, because the target
    applies to the covered cells rather than to the domain.

    ## Response

    Returns the created Grid with status `"pending"` and `georeference: null`.
    Treevox runs DUET asynchronously and updates the grid to `"completed"` with
    a 2D `Georeference` when done.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # Validate source grid: exists, owned, in this domain, and completed.
    _, source_snapshot = await get_document_async(
        COLLECTION,
        body.source_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    source_grid_data = source_snapshot.to_dict()

    # DUET reads a 3D canopy. A 2D grid carries no crowns to drop litter from,
    # and would fail in the handler on an opaque shape mismatch.
    source_config = source_grid_data.get("source", {})
    if source_config.get("entity") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Grid '{body.source_grid_id}' is not a 3D tree grid. DUET "
                f"distributes litter from tree crowns, so its source must be a "
                f"tree grid created with POST /domains/{domain_id}/grids/"
                f"voxelize/inventory/tree."
            ),
        )

    # Reject early rather than dispatching a job that fails on an opaque read.
    # `spcd` is the band users most often lack: voxelize defaults to foliage
    # bulk density alone, so a grid built for any other purpose won't have it.
    validate_grid_has_band(
        source_grid_data, body.source_grid_id, list(DUET_REQUIRED_SOURCE_BANDS)
    )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = DuetSource(
        source_grid_id=body.source_grid_id,
        source_grid_checksum=source_grid_data.get("checksum"),
        years_since_burn=body.years_since_burn,
        wind_direction=body.wind_direction,
        wind_variability=body.wind_variability,
        bands=body.bands,
        calibration=body.calibration,
    )
    bands = build_duet_bands(body.bands)

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
        # Derived from a 3D grid, which does not support modifications; apply
        # them to the inventory before voxelizing.
        "modifications": [],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        # chunks is computed by Treevox once the 2D surface dimensions are
        # known from the source grid.
        "chunks": None,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    await create_http_task_async(TREEVOX_QUEUE, TREEVOX_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
