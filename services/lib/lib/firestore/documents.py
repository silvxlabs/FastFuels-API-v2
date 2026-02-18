"""
Synchronous Firestore document operations.

Provides document CRUD with ownership validation for background jobs.
"""

from typing import Any, Literal

from google.api_core.exceptions import NotFound
from google.cloud import firestore
from google.cloud.firestore import DocumentReference, DocumentSnapshot, FieldFilter
from google.cloud.firestore import Query as FirestoreQuery

firestore_client: firestore.Client = firestore.Client()


class DocumentNotFoundError(Exception):
    """Raised when a requested document does not exist in Firestore."""

    pass


class UnauthorizedAccessError(Exception):
    """Raised when the caller does not have access to the requested document."""

    pass


def get_document(
    collection: str,
    document_id: str,
    owner_id: str | None = None,
    document_status: str | None = None,
) -> tuple[DocumentReference, DocumentSnapshot]:
    """
    Retrieve a Firestore document with optional validation.

    Args:
        collection: The collection name (e.g., "domains", "grids").
        document_id: The document ID.
        owner_id: If provided, validates that the document's owner_id field
            matches this value.
        document_status: If provided, validates that the document's status field
            matches this value.

    Returns:
        Tuple of (DocumentReference, DocumentSnapshot).

    Raises:
        DocumentNotFoundError: If the document does not exist.
        UnauthorizedAccessError: If owner_id is provided and doesn't match.
        ValueError: If document_status is provided and doesn't match.
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    document_snapshot = document_ref.get()

    if not document_snapshot.exists:
        raise DocumentNotFoundError(f"Document not found: {collection}/{document_id}")

    document_data = document_snapshot.to_dict()

    if owner_id is not None:
        document_owner_id = document_data.get("owner_id")
        if document_owner_id is None or document_owner_id != owner_id:
            raise UnauthorizedAccessError(
                f"Unauthorized access to document: {collection}/{document_id}"
            )

    if document_status is not None:
        actual_status = document_data.get("status")
        if actual_status != document_status:
            raise ValueError(
                f"Document status is '{actual_status}', expected '{document_status}'"
            )

    return document_ref, document_snapshot


def set_document(
    collection: str,
    document_id: str,
    data: dict,
) -> DocumentReference:
    """
    Create or overwrite a Firestore document.

    Args:
        collection: The collection name (e.g., "domains", "grids").
        document_id: The document ID.
        data: The document data to store.

    Returns:
        The DocumentReference for the created/updated document.

    Note:
        This function performs an upsert operation - it will create the document
        if it doesn't exist, or overwrite it if it does.
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    document_ref.set(data)
    return document_ref


def update_document(
    collection: str,
    document_id: str,
    data: dict,
) -> DocumentReference:
    """
    Update fields in an existing Firestore document.

    Performs a partial update (merge) on the document, only modifying the
    fields specified in the data dict. The document must exist.

    Args:
        collection: The collection name (e.g., "domains", "grids").
        document_id: The document ID.
        data: The fields to update. Only these fields will be modified;
            other existing fields remain unchanged.

    Returns:
        The DocumentReference for the updated document.

    Raises:
        DocumentNotFoundError: If the document does not exist. This can be
            used to detect cancellation (document deleted by user).

    Note:
        This function does not perform ownership validation. Callers should
        use get_document with owner_id validation before calling this
        function to ensure proper access control.

    Example:
        # Progress update with cancellation detection
        try:
            update_document("grids", grid_id, {"progress": {"message": "Processing..."}})
        except DocumentNotFoundError:
            # Document was deleted (user cancelled)
            raise CancelledException(f"Grid {grid_id} was cancelled")
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    try:
        document_ref.update(data)
    except NotFound:
        raise DocumentNotFoundError(f"Document not found: {collection}/{document_id}")
    return document_ref


def delete_document(
    collection: str,
    document_id: str,
) -> None:
    """
    Delete a Firestore document.

    Deletes the document from the specified collection. This operation is
    idempotent - deleting a non-existent document does not raise an error.

    Args:
        collection: The collection name (e.g., "domains", "grids").
        document_id: The document ID.

    Note:
        This function does not perform ownership validation. Callers should
        use get_document with owner_id validation before calling this
        function to ensure proper access control.
    """
    document_ref = firestore_client.collection(collection).document(document_id)
    document_ref.delete()


def list_documents(
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
        collection: The collection name (e.g., "domains", "grids").
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
            contains the specified value.

    Returns:
        Tuple of (list of DocumentSnapshots, total_count).
        - documents: List of DocumentSnapshot objects for the requested page.
        - total_count: Total number of documents matching all filters.
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
    aggregation_result = aggregation_query.get()
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
    documents = query.get()

    return list(documents), total_count
