"""
api/v2/resources/point_clouds/router.py

Router for the Point Cloud resource with standard CRUD endpoints.

Creation endpoints (uploading a file, fetching from USGS 3DEP) are added by
source-specific sub-routers in follow-on work; this module provides only the
list / get / update / delete surface. The router does Firestore and
GCS bookkeeping only — all point-cloud parsing happens in worker services, so
the API stays free of GDAL/PDAL at runtime.
"""

from datetime import datetime

from fastapi import APIRouter, Query, Request, status

from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.point_clouds.schema import (
    ListPointCloudsResponse,
    PointCloud,
    PointCloudSortField,
    PointCloudType,
    UpdatePointCloudRequestBody,
)
from api.resources.point_clouds.upload.router import router as upload_router
from api.schema import SortOrder
from lib.config import POINT_CLOUDS_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()

COLLECTION = POINT_CLOUDS_COLLECTION

# Source-specific creation sub-routers. The upload source creates a point cloud
# from a user-supplied file (#328); the 3dep source (#329) attaches similarly.
router.include_router(upload_router, prefix="/upload", tags=["Point Clouds - Upload"])


@wildcard_router.get(
    "",
    response_model=ListPointCloudsResponse,
    status_code=status.HTTP_200_OK,
    summary="List point clouds across all domains",
)
async def list_point_clouds_cross_domain(
    request: Request,
    page: int = Query(
        0,
        ge=0,
        description="The page number to retrieve (zero-indexed).",
    ),
    size: int = Query(
        100,
        ge=1,
        le=1000,
        description="The number of point clouds to retrieve per page.",
    ),
    sort_by: PointCloudSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: PointCloudType | None = Query(
        None,
        description="Filter point clouds by acquisition type (`als` or `tls`).",
    ),
    source: str | None = Query(
        None,
        description="Filter point clouds by source name (e.g., `3dep`, `upload`).",
    ),
    tag: str | None = Query(
        None,
        description="Filter point clouds that contain this tag.",
    ),
) -> ListPointCloudsResponse:
    """
    # List Point Clouds (All Domains)

    Retrieves a paginated list of every point cloud belonging to the
    authenticated user, across all of their domains.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **type**: (string, optional) Filter by acquisition type: `als` or `tls`.
    - **source**: (string, optional) Filter by source name (e.g., `3dep`, `upload`).
    - **tag**: (string, optional) Filter point clouds that contain this tag.

    ## Response

    Returns a paginated list of point clouds with metadata.
    """
    owner_id = request.state.id

    filters = {}
    if type:
        filters["type"] = type.value
    if source:
        filters["source.name"] = source

    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    documents, total_count = await list_documents_async(
        collection=COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters if filters else None,
        array_contains_filters=(
            array_contains_filters if array_contains_filters else None
        ),
    )

    point_clouds = [PointCloud(**doc.to_dict()) for doc in documents]

    return ListPointCloudsResponse(
        point_clouds=point_clouds,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "",
    response_model=ListPointCloudsResponse,
    status_code=status.HTTP_200_OK,
    summary="List point clouds in a domain",
)
async def list_point_clouds(
    request: Request,
    domain: VerifiedDomain,
    page: int = Query(
        0,
        ge=0,
        description="The page number to retrieve (zero-indexed).",
    ),
    size: int = Query(
        100,
        ge=1,
        le=1000,
        description="The number of point clouds to retrieve per page.",
    ),
    sort_by: PointCloudSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: PointCloudType | None = Query(
        None,
        description="Filter point clouds by acquisition type (`als` or `tls`).",
    ),
    source: str | None = Query(
        None,
        description="Filter point clouds by source name (e.g., `3dep`, `upload`).",
    ),
    tag: str | None = Query(
        None,
        description="Filter point clouds that contain this tag.",
    ),
) -> ListPointCloudsResponse:
    """
    # List Point Clouds (Domain)

    Retrieves a paginated list of the point clouds within a single domain
    belonging to the authenticated user.

    ## Path Parameters

    - **domain_id**: (string) The domain to list point clouds for.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **type**: (string, optional) Filter by acquisition type: `als` or `tls`.
    - **source**: (string, optional) Filter by source name (e.g., `3dep`, `upload`).
    - **tag**: (string, optional) Filter point clouds that contain this tag.

    ## Response

    Returns a paginated list of point clouds with metadata.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    filters = {"domain_id": domain_id}
    if type:
        filters["type"] = type.value
    if source:
        filters["source.name"] = source

    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    documents, total_count = await list_documents_async(
        collection=COLLECTION,
        owner_id=owner_id,
        page=page,
        size=size,
        sort_by=sort_by.value if sort_by else None,
        sort_order=sort_order.value if sort_order else None,
        filters=filters if filters else None,
        array_contains_filters=(
            array_contains_filters if array_contains_filters else None
        ),
    )

    point_clouds = [PointCloud(**doc.to_dict()) for doc in documents]

    return ListPointCloudsResponse(
        point_clouds=point_clouds,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{point_cloud_id}",
    response_model=PointCloud,
    status_code=status.HTTP_200_OK,
    summary="Get a point cloud by ID",
)
async def get_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
):
    """
    # Get Point Cloud

    Retrieves a specific point cloud resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The domain the point cloud belongs to.
    - **point_cloud_id**: (string) The unique 32-character hex identifier.

    ## Response

    Returns the point cloud resource.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, point_cloud_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    return PointCloud(**snapshot.to_dict())


@router.patch(
    "/{point_cloud_id}",
    response_model=PointCloud,
    status_code=status.HTTP_200_OK,
    summary="Update a point cloud",
)
async def update_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
    body: UpdatePointCloudRequestBody,
):
    """
    # Update Point Cloud

    Updates the metadata of an existing point cloud. Only the fields provided in
    the request body are modified.

    ## Path Parameters

    - **domain_id**: (string) The domain the point cloud belongs to.
    - **point_cloud_id**: (string) The unique identifier of the point cloud.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the point cloud.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## What Cannot Be Updated

    The following are immutable through this endpoint:

    - **id**, **domain_id**, **type**, **source**, **georeference**
    - **created_on** (creation timestamp is permanent)
    - **checksum** (changes only when the point cloud's content is rebuilt, never
      via metadata updates)

    The **modified_on** field is updated automatically.

    ## Response

    Returns the updated point cloud resource.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, point_cloud_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    point_cloud_data = snapshot.to_dict()

    update_data = body.model_dump(exclude_none=True)
    update_data["modified_on"] = datetime.now()

    await update_document_async(
        collection=COLLECTION,
        document_id=point_cloud_id,
        data=update_data,
    )

    point_cloud_data.update(update_data)

    return PointCloud(**point_cloud_data)


@router.delete(
    "/{point_cloud_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a point cloud",
)
async def delete_point_cloud(
    request: Request,
    domain: VerifiedDomain,
    point_cloud_id: str,
):
    """
    # Delete Point Cloud

    Permanently deletes a point cloud by its unique identifier, including the
    stored point data in GCS. This action cannot be undone.

    ## Path Parameters

    - **domain_id**: (string) The domain the point cloud belongs to.
    - **point_cloud_id**: (string) The unique identifier of the point cloud.

    ## Response

    Returns HTTP 204 No Content with an empty response body.

    ## Error Responses

    - **404 Not Found**: The point cloud does not exist or the user does not have access.
    """
    await get_document_async(
        COLLECTION, point_cloud_id, owner_id=request.state.id, domain_id=domain["id"]
    )

    await delete_document_async(
        collection=COLLECTION,
        document_id=point_cloud_id,
    )
