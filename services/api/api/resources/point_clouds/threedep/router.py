"""
api/v2/resources/point_clouds/threedep/router.py

Router for creating a point cloud from USGS 3DEP, and for checking coverage
before committing to one.

Both endpoints resolve which 3DEP acquisitions cover the domain, which is a
catalog lookup and some geometry — no point data is read here. The fetch itself
runs in the lakitu worker.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Request, Response
from fastapi import status as http_status

from api.db.documents import set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.point_clouds.schema import PointCloud, PointCloudType
from api.resources.point_clouds.threedep.examples import CREATE_3DEP_OPENAPI_EXAMPLES
from api.resources.point_clouds.threedep.schema import (
    CreateThreeDepPointCloudRequest,
    ThreeDepCoverageResponse,
    ThreeDepDatasetCoverage,
    ThreeDepPointCloudSource,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import LAKITU_QUEUE, LAKITU_SERVICE, POINT_CLOUDS_COLLECTION
from lib.domain_utils import parse_domain_gdf
from lib.entwine import (
    MAX_POINTS,
    DatasetNotFoundError,
    DatasetOutsideDomainError,
    EptCatalogError,
    EptSelection,
    search_3dep_ept,
    select_datasets,
)

router = APIRouter()

# The catalog's point counts are spread over an acquisition's whole published
# extent, including water and other gaps that hold no points, so an estimate
# derived from them runs low. The budget check pads for that; the worker
# re-checks against the exact per-node counts before reading anything.
POINT_ESTIMATE_SAFETY_FACTOR = 1.5


async def _resolve_coverage(domain: dict, pinned: list[str] | None) -> EptSelection:
    """Work out which 3DEP acquisitions cover a domain.

    Reads the acquisition catalog and runs the geometry, both blocking, so the
    whole thing is handed to a thread.

    Raises:
        HTTPException: 503 if the 3DEP catalog cannot be reached, 422 if a
            pinned acquisition is unusable for this domain.
    """
    roi = parse_domain_gdf(domain)

    def resolve() -> EptSelection:
        return select_datasets(roi, search_3dep_ept(roi), pinned=pinned)

    try:
        return await asyncio.to_thread(resolve)
    except EptCatalogError as e:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The USGS 3DEP lidar catalog is temporarily unavailable. "
                "Please try again shortly."
            ),
        ) from e
    except (DatasetNotFoundError, DatasetOutsideDomainError) as e:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e


@router.post(
    "",
    response_model=PointCloud,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a point cloud from USGS 3DEP",
    responses=QUOTA_429_RESPONSE,
)
async def create_3dep_point_cloud(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateThreeDepPointCloudRequest,
        Body(openapi_examples=CREATE_3DEP_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create a Point Cloud from USGS 3DEP

    Fetches public airborne lidar from the USGS 3D Elevation Program for this
    domain. The points are clipped to the domain, reprojected to the domain's
    coordinate reference system, and stored as a point cloud you can build on —
    most directly as a canopy height model, which in turn feeds a tree
    inventory.

    The point cloud is returned immediately with `status` = `pending` and is
    fetched in the background: `status` becomes `running`, then `completed` once
    the points are stored and `georeference` and `summary` are filled in — or
    `failed` if the fetch cannot be completed. Poll
    `GET /domains/{domain_id}/pointclouds/{id}` to follow progress.

    3DEP is airborne, so the resulting point cloud is always type `als`. There
    is no acquisition type to choose.

    ## Choosing acquisitions

    3DEP is published as separate surveys, which overlap and differ in age and
    point density. By default the backend prefers a single survey that covers
    the whole domain, and otherwise combines the fewest surveys that fill it —
    each additional survey introduces a seam between flights of different dates
    and densities. Pass `datasets` to pin the fetch to specific surveys
    instead; check the coverage endpoint first to see what is available.

    Survey boundaries are irregular, so a domain is often covered to
    99-point-something percent rather than exactly 100. Any coverage above zero
    produces a point cloud, and the fraction actually covered is recorded on the
    result as `source.coverage_fraction` — check it if a gap would matter, since
    `summary.density` is measured over the points that exist and looks healthy
    either way. Use the coverage endpoint to see the shortfall before creating
    anything.

    ## Request Body

    - **name**: (optional) Human-readable name.
    - **description**: (optional) Longer free-text description.
    - **tags**: (optional) Tags for organizing and filtering.
    - **datasets**: (optional) Acquisition names to read, in priority order.
      Omit to choose automatically.

    ## Coordinate reference system

    Points are reprojected to the domain's CRS. Only horizontal coordinates are
    transformed — elevations are stored exactly as USGS published them, never
    converted. `georeference.vertical_crs` records what they are measured from
    when the survey declares it, and is null when it does not.

    ## Error Responses

    - **422**: No 3DEP lidar covers this domain, a pinned acquisition is
      unknown or does not overlap the domain, or the fetch would exceed the
      point budget.
    - **429**: A quota was exceeded.
    - **503**: The USGS 3DEP catalog is temporarily unreachable.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(POINT_CLOUDS_COLLECTION, request)

    # Everything that can reject the request is resolved before the resource is
    # created, so a doomed fetch never leaves a failed point cloud behind.
    selection = await _resolve_coverage(domain, body.datasets)

    if not selection.datasets:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No USGS 3DEP lidar is available for this domain. Check "
                "coverage with GET /domains/{domain_id}/pointclouds/3dep/coverage."
            ),
        )

    estimate = selection.estimated_point_count
    if estimate * POINT_ESTIMATE_SAFETY_FACTOR > MAX_POINTS:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"This domain would return roughly {estimate:,} points, which "
                f"exceeds the {MAX_POINTS:,} point limit for a single fetch. "
                "Use a smaller domain."
            ),
        )

    point_cloud_id = uuid.uuid4().hex
    request_time = datetime.now(UTC)

    source = ThreeDepPointCloudSource(
        datasets=[d.name for d in selection.datasets],
        requested_datasets=body.datasets,
        coverage_fraction=selection.coverage_fraction,
        catalog_fetched_on=(
            selection.catalog_fetched_on.isoformat()
            if selection.catalog_fetched_on
            else None
        ),
    )

    point_cloud_data = {
        "id": point_cloud_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": PointCloudType.als.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "georeference": None,
        "summary": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }
    await set_document_async(POINT_CLOUDS_COLLECTION, point_cloud_id, point_cloud_data)

    await create_http_task_async(LAKITU_QUEUE, LAKITU_SERVICE, point_cloud_id)
    register_dispatch(request, response, background_tasks)

    return PointCloud(**point_cloud_data)


@router.get(
    "/coverage",
    response_model=ThreeDepCoverageResponse,
    summary="Check 3DEP lidar coverage for a domain",
)
async def check_3dep_point_cloud_coverage(domain: VerifiedDomain):
    """
    # Check 3DEP Lidar Coverage

    Immediate pre-flight check reporting which USGS 3DEP lidar surveys are
    available for this domain, how much of it they cover, and roughly how many
    points a fetch would return. Use it before creating a 3DEP point cloud to
    avoid waiting on a background job only to find a coverage gap — 3DEP is
    regional, and survey boundaries are irregular.

    This checks lidar point clouds. Elevation raster coverage is a separate
    product with its own check at
    `GET /domains/{domain_id}/grids/topography/3dep/coverage`.

    ## Response

    Reports whether any lidar is available, the fraction of the domain covered,
    the surveys that would be read with what each contributes, and the
    estimated point count against the per-fetch budget. `datasets[].name`
    values can be passed as `datasets` when creating the point cloud to pin the
    fetch.

    ## Error Responses

    - **503**: The USGS 3DEP catalog is temporarily unreachable.
    """
    selection = await _resolve_coverage(domain, None)
    estimate = selection.estimated_point_count

    return ThreeDepCoverageResponse(
        available=bool(selection.datasets),
        coverage_fraction=selection.coverage_fraction,
        datasets=[
            ThreeDepDatasetCoverage(
                name=d.name,
                url=d.url,
                contribution_fraction=d.contribution_fraction,
                estimated_density=d.estimated_density,
                estimated_points=d.estimated_points,
            )
            for d in selection.datasets
        ],
        estimated_point_count=estimate,
        point_budget=MAX_POINTS,
        exceeds_point_budget=estimate * POINT_ESTIMATE_SAFETY_FACTOR > MAX_POINTS,
    )
