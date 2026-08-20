"""
api/v2/resources/grids/canopy/router.py

Router for canopy grid product endpoints.
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

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

# Cell size used when a domain-aligned point-cloud CHM request does not pin one.
# A point cloud has no native pixel size to inherit, so this source supplies its
# own default — and resolves it here, storing the result on the grid, so the
# worker reads a value rather than reapplying a default of its own. 1 m is what
# the downstream tree detection is tuned for.
DEFAULT_POINT_CLOUD_RESOLUTION_M = 1.0


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
    cell reduces the heights above ground of the returns that fall in it to one
    value — by default the greatest of them.

    ## Aggregation

    Which statistic that is matters more as cells grow, because a cell 1 m
    across describes a crown while a cell 30 m across describes a stand, and a
    maximum reports the tallest tree in either. Measured on one cloud
    rasterized twice, the same returns gave a mean height of 4.83 m at 1 m
    cells and 15.48 m at 10 m.

    `max` is the tallest return. `mean` averages every return, so a cell that
    is half canopy and half gap reads between the two. `median` and
    `percentile` take a rank, which is what keeps one surviving noise return
    from setting a cell on its own.

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
    as large building footprints or very dense canopy;
    `source.ground.ground_coverage` and `source.ground.max_ground_distance_m`
    are what reveal that.

    ## Request Body

    - **source_point_cloud_id**: The point cloud to rasterize. Must be airborne
      (`type: als`), `completed`, and in this domain.
    - **alignment**: (optional) Output lattice. Against the domain
      (`target: "domain"`, the default) `resolution` defaults to 1 m — unlike
      the raster-backed canopy sources there is no source pixel size to
      inherit. Against another grid (`target: "grid"`) omitting `resolution`
      matches that grid cell-for-cell, and the output covers the target's
      extent rather than the domain's; giving one keeps the target's origin at
      the new cell size. The target grid must be in this domain's CRS.
      `target: "native"` is not supported — a point cloud has no pixel anchor
      to preserve.
    - **spike_filter**: (optional) Removal of lone spurious returns. Under the
      default `max` aggregation a cell takes the tallest return in it, so one
      bad return — a bird, haze — sets the cell unless the cloud classified it
      as noise, and many clouds do not. Such a return leaves a shape real
      canopy cannot: a single cell towering over everything around it. Both
      fields are in meters, so they mean the same thing at any resolution. Send
      `null` to keep every return. A `mean`, `median`, or `percentile`
      aggregation already resists a lone return, so the filter matters most
      with `max`.
    - **aggregation**: (optional) The statistic above, as
      `{"method": "max" | "mean" | "median" | "percentile"}`. `percentile`
      carries the rank it takes, as in
      `{"method": "percentile", "percentile": 98}`. Defaults to `max`.
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

    # Against a grid, `resolution: null` already means something — match that
    # grid's lattice cell-for-cell — so it is left alone and the worker reads
    # the cell size off the target's transform. Against the domain there is
    # nothing for null to defer to: the raster-backed sources read it as "use
    # the source's native cell size", and a point cloud has none. Resolving it
    # here rather than in the worker keeps the default in one place and makes
    # the stored grid a record of exactly what it was built at.
    alignment = body.alignment
    if alignment.target != "grid" and alignment.resolution is None:
        alignment = alignment.model_copy(
            update={"resolution": DEFAULT_POINT_CLOUD_RESOLUTION_M}
        )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = PointCloudChmSource(
        source_point_cloud_id=body.source_point_cloud_id,
        source_point_cloud_checksum=point_cloud.get("checksum"),
        extent_buffer_cells=body.extent_buffer_cells,
        alignment=alignment,
        spike_filter=body.spike_filter,
        aggregation=body.aggregation,
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
