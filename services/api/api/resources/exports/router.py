"""
api/v2/resources/exports/router.py

Router for the Export resource with top-level lifecycle CRUD endpoints.
Creation endpoints are domain-scoped and live under grids/exports/.
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
from api.resources.exports.schema import (
    Export,
    ExportSortField,
    ListExportsResponse,
    UpdateExportRequestBody,
)
from api.schema import SortOrder
from lib.config import EXPORTS_BUCKET, EXPORTS_COLLECTION

router = APIRouter()

COLLECTION = EXPORTS_COLLECTION


@router.get(
    "",
    response_model=ListExportsResponse,
    status_code=status.HTTP_200_OK,
    summary="List all exports",
)
async def list_exports(
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
        description="The number of exports to retrieve per page.",
    ),
    sort_by: ExportSortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    domain_id: str | None = Query(
        None,
        description="Filter exports by domain ID.",
    ),
    source_name: str | None = Query(
        None,
        description="Filter exports by source format (e.g., 'geotiff').",
    ),
    tag: str | None = Query(
        None,
        description="Filter exports that contain this tag.",
    ),
) -> ListExportsResponse:
    """
    # List Exports Endpoint

    Retrieves a paginated list of all exports belonging to the authenticated user.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **domain_id**: (string, optional) Filter by source domain.
    - **source_name**: (string, optional) Filter by export format (e.g., `geotiff`).
    - **tag**: (string, optional) Filter exports that contain this tag.

    ## Response

    Returns a paginated list of exports with metadata.
    """
    owner_id = request.state.id

    filters = {}
    if domain_id:
        filters["domain_id"] = domain_id
    if source_name:
        filters["source.name"] = source_name

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

    exports = [Export(**doc.to_dict()) for doc in documents]

    return ListExportsResponse(
        exports=exports,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{export_id}",
    response_model=Export,
    status_code=status.HTTP_200_OK,
    summary="Get an export by ID",
)
async def get_export(
    request: Request,
    export_id: str,
):
    """
    # Get Export Endpoint

    Retrieves a specific export resource by its unique identifier.
    When the export is completed, the response includes a signed_url.

    ## Path Parameters

    - **export_id**: (string) The unique identifier of the export.

    ## Response

    Returns the export resource.

    ## Error Responses

    - **404 Not Found**: The export does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, export_id, owner_id=request.state.id
    )
    return Export(**snapshot.to_dict())


@router.patch(
    "/{export_id}",
    response_model=Export,
    status_code=status.HTTP_200_OK,
    summary="Update an export",
)
async def update_export(
    request: Request,
    export_id: str,
    body: UpdateExportRequestBody,
):
    """
    # Update Export Endpoint

    Updates the metadata of an existing export resource. Only the fields provided
    in the request body will be modified.

    ## Path Parameters

    - **export_id**: (string) The unique identifier of the export.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the export.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## Response

    Returns the updated export resource.
    """
    _, snapshot = await get_document_async(
        COLLECTION, export_id, owner_id=request.state.id
    )
    export_data = snapshot.to_dict()

    update_data = body.model_dump(exclude_none=True)
    update_data["modified_on"] = datetime.now()

    await update_document_async(
        collection=COLLECTION,
        document_id=export_id,
        data=update_data,
    )

    export_data.update(update_data)

    return Export(**export_data)


@router.delete(
    "/{export_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an export",
)
async def delete_export(
    request: Request,
    export_id: str,
    background_tasks: BackgroundTasks,
):
    """
    # Delete Export Endpoint

    Permanently deletes an export resource by its unique identifier.
    This action cannot be undone.

    ## Path Parameters

    - **export_id**: (string) The unique identifier of the export.

    ## Response

    Returns HTTP 204 No Content with an empty response body.

    ## Error Responses

    - **404 Not Found**: The export does not exist or the user does not have access.
    """
    await get_document_async(COLLECTION, export_id, owner_id=request.state.id)

    await delete_document_async(
        collection=COLLECTION,
        document_id=export_id,
    )

    background_tasks.add_task(delete_directory_safe, EXPORTS_BUCKET, export_id)
