"""
api/v2/resources/inventories/router.py

Router for the Inventory resource with standard CRUD endpoints.
Algorithm-specific creation endpoints are in their respective subdirectories.
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
from api.resources.inventories.pim.router import router as pim_router
from api.resources.inventories.schema import (
    Inventory,
    InventorySortField,
    InventoryType,
    ListInventoriesResponse,
    UpdateInventoryRequestBody,
)
from api.schema import SortOrder
from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION

router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION


@router.get(
    "",
    response_model=ListInventoriesResponse,
    status_code=status.HTTP_200_OK,
    summary="List all inventories",
)
async def list_inventories(
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
        description="The number of inventories to retrieve per page.",
    ),
    sort_by: InventorySortField | None = Query(
        None,
        description="The field to sort results by.",
    ),
    sort_order: SortOrder | None = Query(
        None,
        description="The order to sort results (ascending or descending).",
    ),
    type: InventoryType | None = Query(
        None,
        description="Filter inventories by entity type (e.g., 'tree').",
    ),
    source: str | None = Query(
        None,
        description="Filter inventories by source name (e.g., 'pim').",
    ),
    tag: str | None = Query(
        None,
        description="Filter inventories that contain this tag.",
    ),
) -> ListInventoriesResponse:
    """
    # List Inventories Endpoint

    Retrieves a paginated list of all inventories within a domain belonging to
    the authenticated user.

    ## Path Parameters

    - **domain_id**: (string) The domain to list inventories for.

    ## Query Parameters

    - **page**: (integer, optional) Page number (zero-indexed). Default: 0.
    - **size**: (integer, optional) Items per page (1-1000). Default: 100.
    - **sort_by**: (string, optional) Field to sort by: `created_on`, `modified_on`, `name`.
    - **sort_order**: (string, optional) Sort direction: `ascending`, `descending`.
    - **type**: (string, optional) Filter by entity type (e.g., `tree`).
    - **source**: (string, optional) Filter by source name (e.g., `pim`).
    - **tag**: (string, optional) Filter inventories that contain this tag.

    ## Response

    Returns a paginated list of inventories with metadata.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    # Build equality filters
    filters = {"domain_id": domain_id}
    if type:
        filters["type"] = type.value
    if source:
        filters["source.name"] = source

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

    inventories = [Inventory(**doc.to_dict()) for doc in documents]

    return ListInventoriesResponse(
        inventories=inventories,
        current_page=page,
        page_size=size,
        total_items=total_count,
    )


@router.get(
    "/{inventory_id}",
    response_model=Inventory,
    status_code=status.HTTP_200_OK,
    summary="Get an inventory by ID",
)
async def get_inventory(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
):
    """
    # Get Inventory Endpoint

    Retrieves a specific inventory resource by its unique identifier.

    ## Path Parameters

    - **domain_id**: (string) The domain the inventory belongs to.
    - **inventory_id**: (string) The unique 32-character hex identifier of the inventory.

    ## Response

    Returns the inventory resource.

    ## Error Responses

    - **404 Not Found**: The inventory does not exist or the user does not have access.
    """
    _, snapshot = await get_document_async(
        COLLECTION, inventory_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    return Inventory(**snapshot.to_dict())


@router.patch(
    "/{inventory_id}",
    response_model=Inventory,
    status_code=status.HTTP_200_OK,
    summary="Update an inventory",
)
async def update_inventory(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
    body: UpdateInventoryRequestBody,
):
    """
    # Update Inventory Endpoint

    Updates the metadata of an existing inventory resource. Only the fields
    provided in the request body will be modified.

    ## Path Parameters

    - **domain_id**: (string) The domain the inventory belongs to.
    - **inventory_id**: (string) The unique identifier of the inventory.

    ## Request Body

    All fields are optional:

    - **name**: (string) New name for the inventory.
    - **description**: (string) New description.
    - **tags**: (array of strings) New tags (replaces existing).

    ## What Cannot Be Updated

    The following fields are immutable:

    - **id**, **domain_id**, **type**, **source**, **modifications**, **summary**, **georeference**
    - **created_on** (creation timestamp is permanent)

    The **modified_on** field is automatically updated.

    ## Response

    Returns the updated inventory resource.
    """
    _, snapshot = await get_document_async(
        COLLECTION, inventory_id, owner_id=request.state.id, domain_id=domain["id"]
    )
    inventory_data = snapshot.to_dict()

    update_data = body.model_dump(exclude_none=True)
    update_data["modified_on"] = datetime.now()

    await update_document_async(
        collection=COLLECTION,
        document_id=inventory_id,
        data=update_data,
    )

    inventory_data.update(update_data)

    return Inventory(**inventory_data)


@router.delete(
    "/{inventory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an inventory",
)
async def delete_inventory(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
    background_tasks: BackgroundTasks,
):
    """
    # Delete Inventory Endpoint

    Permanently deletes an inventory resource by its unique identifier.
    This action cannot be undone.

    ## Path Parameters

    - **domain_id**: (string) The domain the inventory belongs to.
    - **inventory_id**: (string) The unique identifier of the inventory.

    ## Response

    Returns HTTP 204 No Content with an empty response body.

    ## Error Responses

    - **404 Not Found**: The inventory does not exist or the user does not have access.
    """
    await get_document_async(
        COLLECTION, inventory_id, owner_id=request.state.id, domain_id=domain["id"]
    )

    await delete_document_async(
        collection=COLLECTION,
        document_id=inventory_id,
    )

    background_tasks.add_task(delete_directory_safe, INVENTORIES_BUCKET, inventory_id)


router.include_router(pim_router, prefix="/pim", tags=["Inventories - PIM"])
