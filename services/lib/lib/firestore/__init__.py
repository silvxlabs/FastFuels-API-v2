"""
lib.firestore - Synchronous Firestore operations

Provides document CRUD, ownership validation, and serialization utilities.
"""

from lib.firestore.documents import (
    DocumentNotFoundError,
    UnauthorizedAccessError,
    delete_document,
    firestore_client,
    get_document,
    list_documents,
    set_document,
    update_document,
)
from lib.firestore.serializers import (
    deserialize_coordinates,
    serialize_coordinates,
)

__all__ = [
    "firestore_client",
    "get_document",
    "set_document",
    "update_document",
    "delete_document",
    "list_documents",
    "DocumentNotFoundError",
    "UnauthorizedAccessError",
    "serialize_coordinates",
    "deserialize_coordinates",
]
