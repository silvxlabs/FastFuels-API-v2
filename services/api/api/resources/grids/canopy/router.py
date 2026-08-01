"""
api/v2/resources/grids/canopy/router.py

Router for canopy grid product endpoints.
"""

import math
import os
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
from api.resources.grids.canopy.examples import (
    CREATE_LANDFIRE_CANOPY_OPENAPI_EXAMPLES,
    CREATE_META_CHM_OPENAPI_EXAMPLES,
    CREATE_NAIP_CHM_OPENAPI_EXAMPLES,
    CREATE_POINT_CLOUD_CHM_OPENAPI_EXAMPLES,
)
from api.resources.grids.canopy.schema import (
    CreateLandfireCanopyRequest,
    CreateMetaChmRequest,
    CreateNaipChmRequest,
    CreatePointCloudChmRequest,
    LandfireCanopySource,
    MetaChmSource,
    NaipChmSource,
    PointCloudChmSource,
    build_chm_bands,
    build_landfire_canopy_bands,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import (
    dump_modifications_for_firestore,
    validate_feature_modifications,
    validate_target_grid_alignment,
)
from api.resources.point_clouds.schema import PointCloudType
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    GRIDDLE_QUEUE,
    GRIDDLE_SERVICE,
    GRIDS_COLLECTION,
    POINT_CLOUDS_COLLECTION,
)
from lib.domain_utils import parse_domain_gdf

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

# Cell size used when a point-cloud CHM request does not pin one. A point cloud
# has no native pixel size to inherit, so this source supplies its own default —
# and resolves it here, storing the result on the grid, so the worker reads a
# value rather than reapplying a default of its own. 1 m is what the downstream
# tree detection is tuned for.
DEFAULT_POINT_CLOUD_RESOLUTION_M = 1.0

# Rasterizing a point cloud holds the output grid in memory for the whole job,
# so grid size — not point count — is what bounds the work. Rejecting an
# oversized lattice here means a doomed request never becomes a failed grid.
# Matches the cap the landscape and QUIC-Fire exports already apply.
MAX_CHM_CELLS = int(os.getenv("CHM_MAX_CELLS", 200_000_000))


@router.post(
    "/meta",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from Meta CHM",
    responses=QUOTA_429_RESPONSE,
)
async def create_meta_chm(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateMetaChmRequest,
        Body(openapi_examples=CREATE_META_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Meta CHM Grid

    Creates a grid with canopy height data from Meta's global canopy height
    model at ~1m resolution.

    ## Request Body

    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) Meta CHM version. Default: "2".
        - **1**: Tolan, J. et al. (2024). Very high resolution canopy height maps from RGB imagery.
        - **2**: Brandt, J. et al. (2026). CHMv2: Improvements in Global Canopy Height Mapping using DINOv3.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = MetaChmSource(
        version=body.version,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
    bands = build_chm_bands()

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
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)


@router.post(
    "/naip",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from NAIP CHM",
    responses=QUOTA_429_RESPONSE,
)
async def create_naip_chm(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateNaipChmRequest,
        Body(openapi_examples=CREATE_NAIP_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create NAIP CHM Grid

    Creates a grid with canopy height data from the NAIP high-resolution
    canopy height model at ~0.6m resolution (CONUS).

    ## Request Body
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = NaipChmSource(
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
    bands = build_chm_bands()

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
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)


@router.post(
    "/landfire",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a grid from LANDFIRE canopy data",
    responses=QUOTA_429_RESPONSE,
)
async def create_landfire_canopy(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateLandfireCanopyRequest,
        Body(openapi_examples=CREATE_LANDFIRE_CANOPY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create LANDFIRE Canopy Grid

    Creates a grid with canopy fuel data from LANDFIRE at 30m resolution
    (CONUS).

    Available bands:
    - **chm**: canopy height in meters
    - **cbd**: canopy bulk density in kg/m**3
    - **cbh**: canopy base height in meters
    - **cc**:  canopy cover in percent (0-100)

    By default all four bands are included. Use the `bands` field to select
    a subset.

    ## Request Body

    - **bands**: (optional) Which bands to include. Default: all four.
    - **name**: (optional) Name for the grid.
    - **description**: (optional) Description.
    - **tags**: (optional) Tags for organizing grids.
    - **version**: (optional) LANDFIRE version. Default: "2024".

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    fetch the data and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = LandfireCanopySource(
        version=body.version,
        bands=body.bands,
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=body.alignment,
    )
    bands = build_landfire_canopy_bands(body.bands)

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
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)


def _validate_cell_count(domain: dict, resolution: float) -> None:
    """Reject a lattice too large to rasterize.

    Raises:
        HTTPException: 422 when the resolved grid exceeds ``MAX_CHM_CELLS``.
    """
    minx, miny, maxx, maxy = parse_domain_gdf(domain).total_bounds
    width = max(1, math.ceil((maxx - minx) / resolution))
    height = max(1, math.ceil((maxy - miny) / resolution))
    cells = width * height
    if cells > MAX_CHM_CELLS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"A {resolution} m canopy height model over this domain would "
                f"be {cells:,} cells (width={width:,}, height={height:,}), "
                f"exceeding the cap of {MAX_CHM_CELLS:,}. Use a coarser "
                f"`alignment.resolution` or a smaller domain."
            ),
        )


@router.post(
    "/point_cloud",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a CHM grid from a point cloud",
    responses=QUOTA_429_RESPONSE,
)
async def create_point_cloud_chm(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreatePointCloudChmRequest,
        Body(openapi_examples=CREATE_POINT_CLOUD_CHM_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create a CHM Grid from a Point Cloud

    Creates a grid with canopy height data rasterized from a point cloud. Each
    cell holds the greatest height above ground of any return that falls in it.

    The resulting grid carries the same `chm` band as the Meta, NAIP, and
    LANDFIRE canopy sources, so it can be used anywhere they can — including as
    the source for individual tree detection
    (`POST /domains/{domain_id}/inventories/tree/chm`).

    ## Ground

    Canopy height is height above ground, so the ground surface underneath
    matters. When the point cloud carries ASPRS ground classification
    (class 2), those returns define the ground. When it does not — an upload
    may carry no classification at all — the ground surface is derived from the
    data.

    Which path was taken, and how well the data constrained it, is recorded on
    the completed grid under `source.ground`. Derived ground is accurate in
    forested terrain and degrades over wide areas with no ground returns, such
    as large building footprints or very dense canopy; `source.ground.coverage`
    and `source.ground.max_ground_distance_m` are what reveal that.

    ## Request Body

    - **source_point_cloud_id**: The point cloud to rasterize. Must be airborne
      (`type: als`), `completed`, and in this domain.
    - **alignment**: (optional) Output lattice. `resolution` defaults to 1 m;
      unlike the raster-backed canopy sources there is no source pixel size to
      inherit. `target: "native"` is not supported — a point cloud has no pixel
      anchor to preserve.
    - **name**, **description**, **tags**: (optional) Metadata.

    ## Response

    Returns the created Grid resource with status "pending". The backend will
    build the grid and update status to "completed" when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # Exists, owned, in this domain, and finished — one call covers all four.
    _, point_cloud_snapshot = await get_document_async(
        POINT_CLOUDS_COLLECTION,
        body.source_point_cloud_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    point_cloud = point_cloud_snapshot.to_dict()

    # A terrestrial scan images a plot from within it, so it has no landscape
    # canopy surface to rasterize.
    if point_cloud.get("type") != PointCloudType.als.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Point cloud '{body.source_point_cloud_id}' has type "
                f"'{point_cloud.get('type')}'. A canopy height model requires "
                f"an airborne (als) point cloud."
            ),
        )

    if body.alignment.target == "native":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "alignment.target 'native' is not supported for a point cloud "
                "source: there is no source raster whose pixel anchor could be "
                "preserved. Use 'domain' or 'grid'."
            ),
        )

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)
    await validate_feature_modifications(body.modifications, owner_id, domain_id)

    # Resolve the default here rather than leaving it to the worker. For the
    # raster-backed sources `resolution: null` means "use the source's native
    # cell size", which the worker can only answer at fetch time. A point cloud
    # has no native cell size, so null has no meaning to carry forward — filling
    # it in now keeps the cap check and the job provably on the same number, and
    # records on the grid exactly what it was built at.
    alignment = body.alignment.model_copy(
        update={
            "resolution": body.alignment.resolution or DEFAULT_POINT_CLOUD_RESOLUTION_M
        }
    )
    _validate_cell_count(domain, alignment.resolution)

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = PointCloudChmSource(
        source_point_cloud_id=body.source_point_cloud_id,
        source_point_cloud_checksum=point_cloud.get("checksum"),
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=alignment,
    )
    bands = build_chm_bands()

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
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
