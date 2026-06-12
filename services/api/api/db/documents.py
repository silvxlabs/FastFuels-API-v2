"""
Async Firestore document operations for FastAPI routes.

Provides document retrieval and storage with ownership and status validation.
"""

import logging
from typing import Any, Literal

from fastapi import HTTPException, status
from google.api_core.exceptions import FailedPrecondition
from google.cloud import firestore
from google.cloud.firestore import (
    AsyncDocumentReference,
    DocumentSnapshot,
    FieldFilter,
)
from google.cloud.firestore import Query as FirestoreQuery

from lib.config import SUPPORT_EMAIL

logger = logging.getLogger(__name__)

firestore_client: firestore.AsyncClient = firestore.AsyncClient()


async def get_document_async(
    collection: str,
    document_id: str,
    owner_id: str | None = None,
    document_status: str | None = None,
    domain_id: str | None = None,
) -> tuple[AsyncDocumentReference, DocumentSnapshot]:
    """
    Retrieve a Firestore document with optional validation.

    Args:
        collection: The collection name (e.g., "domains-v2", "grids-v2").
        document_id: The document ID.
        owner_id: If provided, validates that the document's owner_id field
            matches this value.
        document_status: If provided, validates that the document's status field
            matches this value.
        domain_id: If provided, validates that the document's domain_id field
            matches this value.

    Returns:
        Tuple of (AsyncDocumentReference, DocumentSnapshot).

    Raises:
        HTTPException(404): If the document does not exist.
        HTTPException(404): If owner_id is provided and doesn't match
            (returns 404 to avoid leaking existence information).
        HTTPException(404): If domain_id is provided and doesn't match
            (the document doesn't exist within this domain scope).
        HTTPException(422): If status is provided and doesn't match.
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    document_snapshot = await document_ref.get()

    # Drop version information from collection name (e.g. "domains-v2" -> "domains")
    collection_name = collection.split("-")[0]

    if not document_snapshot.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {collection_name}/{document_id}",
        )

    document_data = document_snapshot.to_dict()

    if owner_id is not None:
        document_owner_id = document_data.get("owner_id")
        if document_owner_id is None or document_owner_id != owner_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {collection_name}/{document_id}",
            )

    if domain_id is not None:
        document_domain_id = document_data.get("domain_id")
        if document_domain_id is None or document_domain_id != domain_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {collection_name}/{document_id}",
            )

    if document_status is not None:
        actual_status = document_data.get("status")
        if actual_status != document_status:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"{collection_name}/{document_id} status is '{actual_status}', expected '{document_status}'.",
            )

    return document_ref, document_snapshot


async def set_document_async(
    collection: str,
    document_id: str,
    data: dict,
) -> AsyncDocumentReference:
    """
    Create or overwrite a Firestore document.

    Args:
        collection: The collection name (e.g., "domains-v2", "grids").
        document_id: The document ID.
        data: The document data to store.

    Returns:
        The AsyncDocumentReference for the created/updated document.

    Note:
        This function performs an upsert operation - it will create the document
        if it doesn't exist, or overwrite it if it does.
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    await document_ref.set(data)

    return document_ref


async def list_documents_async(
    collection: str,
    owner_id: str,
    page: int = 0,
    size: int = 100,
    sort_by: str | None = None,
    sort_order: Literal["ascending", "descending"] | None = None,
    filters: dict[str, Any] | None = None,
    array_contains_filters: dict[str, Any] | None = None,
) -> tuple[list[DocumentSnapshot], int]:
    """
    List documents from a Firestore collection filtered by owner.

    Retrieves a paginated list of documents belonging to a specific owner,
    with optional sorting and additional filters. Returns both the documents
    and total count.

    Args:
        collection: The collection name (e.g., "domains-v2", "grids-v2").
        owner_id: Filter documents by this owner_id value.
        page: Zero-indexed page number for pagination. Default: 0.
        size: Number of documents per page. Default: 100.
        sort_by: Field name to sort by (e.g., "created_on", "name").
            If None, no sorting is applied (Firestore default order).
        sort_order: Sort direction - "ascending" or "descending".
            Only used if sort_by is provided. Default: "descending".
        filters: Optional dict of field -> value pairs for equality filtering.
            Each filter is ANDed with the owner_id filter. For example,
            {"domain_id": "abc123", "status": "completed"} will filter for
            documents where domain_id == "abc123" AND status == "completed".
            Supports dot notation for nested fields (e.g., "source.name").
        array_contains_filters: Optional dict of field -> value pairs for
            array-contains filtering. Each filter checks if the array field
            contains the specified value. For example, {"tags": "surface-fuel"}
            will filter for documents where the tags array contains "surface-fuel".

    Returns:
        Tuple of (list of DocumentSnapshots, total_count).
        - documents: List of DocumentSnapshot objects for the requested page.
        - total_count: Total number of documents matching all filters
          (across all pages).

    Example:
        documents, total = await list_documents_async(
            collection="grids-v2",
            owner_id="user123",
            page=0,
            size=10,
            sort_by="created_on",
            sort_order="descending",
            filters={"domain_id": "domain-abc"},
        )
    """
    # Build base query with owner filter
    field_filter = FieldFilter("owner_id", "==", owner_id)
    query = firestore_client.collection(collection).where(filter=field_filter)

    # Apply additional equality filters if specified
    if filters:
        for field, value in filters.items():
            additional_filter = FieldFilter(field, "==", value)
            query = query.where(filter=additional_filter)

    # Apply array-contains filters if specified
    if array_contains_filters:
        for field, value in array_contains_filters.items():
            additional_filter = FieldFilter(field, "array_contains", value)
            query = query.where(filter=additional_filter)

    # Get total count for pagination metadata
    aggregation_query = query.count()
    aggregation_result = await aggregation_query.get()
    total_count = aggregation_result[0][0].value

    # Apply sorting if specified
    if sort_by:
        direction = (
            FirestoreQuery.ASCENDING
            if sort_order == "ascending"
            else FirestoreQuery.DESCENDING
        )
        query = query.order_by(sort_by, direction=direction)

    # Apply pagination
    query = query.offset(page * size).limit(size)

    # Execute query and return results
    try:
        documents = await query.get()
    except FailedPrecondition as exc:
        # Firestore raises FailedPrecondition when the query needs a composite
        # index that hasn't been provisioned. The exception message contains a
        # console link to create it.
        logger.error(
            "Firestore query on '%s' requires a missing composite index "
            "(sort_by=%s, sort_order=%s, filters=%s): %s",
            collection,
            sort_by,
            sort_order,
            filters,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "This list query is not supported yet: the database index for "
                f"sort_by={sort_by!r} with sort_order={sort_order or 'descending'!r} "
                "has not been provisioned. This is a server-side configuration "
                "issue, not a problem with your request. Please contact "
                f"{SUPPORT_EMAIL} so it can be fixed."
            ),
        ) from exc

    return list(documents), total_count


async def update_document_async(
    collection: str,
    document_id: str,
    data: dict,
) -> AsyncDocumentReference:
    """
    Update fields in an existing Firestore document.

    Performs a partial update (merge) on the document, only modifying the
    fields specified in the data dict. The document must exist.

    Args:
        collection: The collection name (e.g., "domains-v2", "grids-v2").
        document_id: The document ID.
        data: The fields to update. Only these fields will be modified;
            other existing fields remain unchanged.

    Returns:
        The AsyncDocumentReference for the updated document.

    Raises:
        google.cloud.exceptions.NotFound: If the document does not exist.

    Note:
        This function does not perform ownership validation. Callers should
        use get_document_async with owner_id validation before calling this
        function to ensure proper access control.

    Example:
        # First validate ownership
        ref, snapshot = await get_document_async(
            collection="domains-v2",
            document_id="abc123",
            owner_id="user-456",
        )
        # Then update
        await update_document_async(
            collection="domains-v2",
            document_id="abc123",
            data={"name": "New Name", "modified_on": datetime.now()},
        )
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    await document_ref.update(data)

    return document_ref


async def delete_document_async(
    collection: str,
    document_id: str,
) -> None:
    """
    Delete a Firestore document.

    Deletes the document from the specified collection. This operation is
    idempotent - deleting a non-existent document does not raise an error.

    Args:
        collection: The collection name (e.g., "domains-v2", "grids-v2").
        document_id: The document ID.

    Returns:
        None

    Note:
        This function does not perform ownership validation. Callers should
        use get_document_async with owner_id validation before calling this
        function to ensure proper access control.

    Example:
        # First validate ownership
        ref, snapshot = await get_document_async(
            collection="domains-v2",
            document_id="abc123",
            owner_id="user-456",
        )
        # Then delete
        await delete_document_async(
            collection="domains-v2",
            document_id="abc123",
        )
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    await document_ref.delete()
