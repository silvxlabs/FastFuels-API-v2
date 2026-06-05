"""
api/v2/resources/inventories/router.py

Router for the Inventory resource with standard CRUD endpoints.
Algorithm-specific creation endpoints are in their respective subdirectories.
"""

import io
from datetime import datetime

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)

from api.db.blobs import delete_directory_safe
from api.db.documents import (
    delete_document_async,
    get_document_async,
    list_documents_async,
    update_document_async,
)
from api.dependencies import VerifiedDomain
from api.resources.inventories.cache import get_inventory_metadata, read_partition
from api.resources.inventories.exports.router import router as exports_router
from api.resources.inventories.schema import (
    Inventory,
    InventoryDataFormat,
    InventoryDataMetadata,
    InventoryDataResponse,
    InventoryJsonOrientation,
    InventoryPartitionInfo,
    InventorySortField,
    InventoryType,
    ListInventoriesResponse,
    UpdateInventoryRequestBody,
)
from api.resources.inventories.tree.router import router as tree_router
from api.schema import SortOrder
from lib.config import INVENTORIES_BUCKET, INVENTORIES_COLLECTION

router = APIRouter()
wildcard_router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION


@wildcard_router.get(
    "",
    response_model=ListInventoriesResponse,
    status_code=status.HTTP_200_OK,
    summary="List inventories across all domains",
)
async def list_inventories_cross_domain(
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

    Retrieves a paginated list of all inventories across all domains belonging to
    the authenticated user.

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

    # Build equality filters
    filters = {}
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

    - **id**, **domain_id**, **type**, **source**, **modifications**, **georeference**
    - **created_on** (creation timestamp is permanent)
    - **checksum** (changes only when the inventory's content is rebuilt, never
      via metadata updates)

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


@router.get(
    "/{inventory_id}/data/metadata",
    response_model=InventoryDataMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get inventory data metadata",
)
async def get_inventory_data_metadata(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
):
    """
    # Get Inventory Data Metadata

    Returns partition count, total rows, per-partition row counts, and column
    names for a completed inventory. Reads only the `_metadata` file from GCS
    (cached after first access).

    ## Path Parameters

    - **domain_id**: The domain the inventory belongs to.
    - **inventory_id**: The unique identifier of the inventory.

    ## Response

    - **inventory_id**: The inventory ID.
    - **num_partitions**: Number of Parquet partitions.
    - **total_rows**: Total row count across all partitions.
    - **columns**: List of column names.
    - **partitions**: Per-partition index and row count.

    ## Error Responses

    - **404 Not Found**: Inventory does not exist or user does not have access.
    - **422 Unprocessable Entity**: Inventory is not completed, or metadata
      file is not available.
    """
    await get_document_async(
        COLLECTION,
        inventory_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )

    try:
        meta = await get_inventory_metadata(inventory_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Inventory data metadata not available. "
                "Re-create the inventory to enable data streaming."
            ),
        )

    return InventoryDataMetadata(
        inventory_id=inventory_id,
        num_partitions=meta.num_partitions,
        total_rows=meta.total_rows,
        columns=meta.columns,
        partitions=[
            InventoryPartitionInfo(index=p.index, num_rows=p.num_rows)
            for p in meta.partitions
        ],
    )


@router.get(
    "/{inventory_id}/data/{partition_index}",
    response_model=InventoryDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get inventory data for a partition",
)
async def get_inventory_data(
    request: Request,
    domain: VerifiedDomain,
    inventory_id: str,
    partition_index: int = Path(..., ge=0, description="Zero-based partition index."),
    data_format: InventoryDataFormat = Query(
        InventoryDataFormat.json, alias="format", description="Response format."
    ),
    json_orientation: InventoryJsonOrientation = Query(
        InventoryJsonOrientation.split,
        description="JSON orientation. Ignored for CSV.",
    ),
    columns: str | None = Query(None, description="Comma-separated column subset."),
):
    """
    # Get Inventory Data

    Reads a single partition of a completed inventory's Parquet data on GCS.
    Returns the tree records as JSON (split or records orientation) or CSV.

    ## Path Parameters

    - **domain_id**: The domain the inventory belongs to.
    - **inventory_id**: The unique identifier of the inventory.
    - **partition_index**: Zero-based partition index.

    ## Query Parameters

    - **format**: Response format: `json` (default) or `csv`.
    - **json_orientation**: JSON layout: `split` (default, compact) or
      `records` (self-describing). Ignored for CSV.
    - **columns**: Comma-separated column subset (default: all).

    ## Response

    **JSON split** (default): column names + 2D array of values.

    **JSON records**: list of row objects.

    **CSV**: `text/csv` body with metadata in response headers
    `X-Partition-Index`, `X-Row-Count`, `X-Total-Rows`, `X-Num-Partitions`.

    ## Error Responses

    - **404 Not Found**: Inventory does not exist or user does not have access.
    - **422 Unprocessable Entity**: Inventory not completed, partition index
      out of range, invalid column names, or metadata not available.
    """
    await get_document_async(
        COLLECTION,
        inventory_id,
        owner_id=request.state.id,
        domain_id=domain["id"],
        document_status="completed",
    )

    try:
        meta = await get_inventory_metadata(inventory_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Inventory data metadata not available. "
                "Re-create the inventory to enable data streaming."
            ),
        )

    if partition_index >= meta.num_partitions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Partition index {partition_index} out of range. "
                f"Inventory has {meta.num_partitions} partitions "
                f"(0-{meta.num_partitions - 1})."
            ),
        )

    requested_columns = None
    if columns:
        requested_columns = [c.strip() for c in columns.split(",")]
        missing = [c for c in requested_columns if c not in meta.columns]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(f"Columns not found: {missing}. Available: {meta.columns}"),
            )

    partition = meta.partitions[partition_index]
    df = await read_partition(inventory_id, partition.path, columns=requested_columns)

    col_names = list(df.columns)

    if data_format == InventoryDataFormat.csv:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={
                "X-Partition-Index": str(partition_index),
                "X-Row-Count": str(len(df)),
                "X-Total-Rows": str(meta.total_rows),
                "X-Num-Partitions": str(meta.num_partitions),
            },
        )

    if json_orientation == InventoryJsonOrientation.records:
        data = df.to_dict(orient="records")
    else:
        data = df.to_dict(orient="split")["data"]

    return InventoryDataResponse(
        partition=partition_index,
        num_rows=len(df),
        columns=col_names,
        data=data,
    )


router.include_router(tree_router, prefix="/tree")
# router.include_router(
#     modifications_router,
#     prefix="/{inventory_id}/modifications",
#     tags=["Inventories - Modifications"],
# )
router.include_router(
    exports_router,
    prefix="/{inventory_id}/exports",
    tags=["Inventories - Exports"],
)
