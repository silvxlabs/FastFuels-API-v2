"""
api/v2/resources/grids/router.py

Router for the Grid resource with standard CRUD endpoints.
Product-specific endpoints (FBFM40, Topography, etc.) are in their respective subdirectories.
"""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Query, Request, status

from api.db.blobs import delete_directory_safe
from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.grids.exports.router import router as grid_exports_router
from api.resources.grids.exports.router import (
    single_grid_router as single_grid_exports_router,
)
from api.resources.grids.fbfm40.router import router as fbfm40_router
from api.resources.grids.lookup.router import router as lookup_router
from api.resources.grids.resample.router import router as resample_router
from api.resources.grids.schema import (
    Grid,
    GridSortField,
    ListGridsResponse,
    UpdateGridRequestBody,
)
from api.resources.grids.topography.router import router as topography_router
from api.resources.grids.uniform.router import router as uniform_router
from api.schema import SortOrder
from lib.config import GRIDS_BUCKET, GRIDS_COLLECTION

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.get(
    "",
    response_model=ListGridsResponse,
    status_code=status.HTTP_200_OK,
    summary="List all grids",
)
async def list_grids(
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
        description="The number of grids to retrieve per page.",
    ),
    sort_by: GridSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    source: str | None = Query(
        None,
        description="Filter grids by source name (e.g., 'landfire', '3dep').",
    ),
    product: str | None = Query(
        None,
        description="Filter grids by source product (e.g., 'fbfm40', 'topography'). Requires source filter.",
    ),
    tag: str | None = Query(
        None,
        description="Filter grids that contain this tag.",
    ),
) -> ListGridsResponse:
    """
    # List Grids Endpoint

    Retrieves a paginated list of all grids within a domain belonging to the
    authenticated user.

    ## Path Parameters

    - **domain_id**: (string) The domain to list grids for.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **source**: (string, optional) Filter grids by source name (e.g., `landfire`, `3dep`).
    - **product**: (string, optional) Filter grids by source product (e.g., `fbfm40`, `topography`).
    - **tag**: (string, optional) Filter grids that contain this tag.

    ## Response

    Returns a paginated list of grids with metadata.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Build equality filters — domain_id is always present from URL
    filters = {"domain_id": domain_id}
    if source:
        filters["source.name"] = source
    if product:
        filters["source.product"] = product

    # Build array-contains filters
    array_contains_filters = {}
    if tag:
        array_contains_filters["tags"] = tag

    # Query Firestore
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

    # Convert Firestore documents to Grid models
    grids = [Grid(**doc.to_dict()) for doc in documents]

    return ListGridsResponse(
        grids=grids,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{grid_id}",
    response_model=Grid,
    status_code=status.HTTP_200_OK,
    summary="Get a grid by ID",
)
async def get_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
):
    """
    # Get Grid Endpoint

    Retrieves a specific grid resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique 32-character hex identifier of the grid.

    ## Response

    Returns the grid resource.

    ## Error Responses

    - **404 Not Found**: The grid does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    return Grid(**snapshot.to_dict())


@router.patch(
    "/{grid_id}",
    response_model=Grid,
    status_code=status.HTTP_200_OK,
    summary="Update a grid",
)
async def update_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    body: UpdateGridRequestBody,
):
    """
    # Update Grid Endpoint

    Updates the metadata of an existing grid resource. Only the fields provided
    in the request body will be modified.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the grid.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## What Cannot Be Updated

    The following fields are immutable:

    - **id**, **domain_id**, **source**, **modifications**, **bands**, **georeference**
    - **created_on** (creation timestamp is permanent)

    The **modified_on** field is automatically updated.

    ## Response

    Returns the updated grid resource.
    """
    _, snapshot = await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    grid_data = snapshot.to_dict()

    # Build update data from provided fields only
    update_data = body.model_dump(exclude_none=True)

    # Always update modified_on timestamp
    update_data["modified_on"] = datetime.now()

    # Perform the partial update
    await update_document_async(
        collection=COLLECTION,
        document_id=grid_id,
        data=update_data,
    )

    # Merge updates with existing data to return the full grid
    grid_data.update(update_data)

    return Grid(**grid_data)


@router.delete(
    "/{grid_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a grid",
)
async def delete_grid(
    request: Request,
    domain: VerifiedDomain,
    grid_id: str,
    background_tasks: BackgroundTasks,
):
    """
    # Delete Grid Endpoint

    Permanently deletes a grid resource by its unique identifier.
    This action cannot be undone.

    ## Path Parameters

    - **domain_id**: (string) The domain the grid belongs to.
    - **grid_id**: (string) The unique identifier of the grid.

    ## Response

    Returns HTTP 204 No Content with an empty response body.

    ## Error Responses

    - **404 Not Found**: The grid does not exist or the user does not have access.
    """
    await get_document_async(
        COLLECTION, grid_id, owner_id=request.state.id, domain_id=domain["id"]
    )

    await delete_document_async(
        collection=COLLECTION,
        document_id=grid_id,
    )

    background_tasks.add_task(delete_directory_safe, GRIDS_BUCKET, grid_id)


router.include_router(grid_exports_router, prefix="/exports", tags=["Grids - Exports"])
router.include_router(
    single_grid_exports_router, prefix="/{grid_id}/exports", tags=["Grids - Exports"]
)
router.include_router(fbfm40_router, prefix="/fbfm40", tags=["Grids - FBFM40"])
router.include_router(
    topography_router, prefix="/topography", tags=["Grids - Topography"]
)
router.include_router(lookup_router, prefix="/lookup", tags=["Grids - Lookup"])
router.include_router(resample_router, prefix="/resample", tags=["Grids - Resample"])
router.include_router(uniform_router, prefix="/uniform", tags=["Grids - Uniform"])
