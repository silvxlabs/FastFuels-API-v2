# lib/v2/firestore - Synchronous Firestore Operations

This module provides synchronous Firestore client and operations for background jobs.

## Modules

- **documents.py** - Firestore client and document CRUD with validation
- **serializers.py** - Data serialization helpers for Firestore storage limitations

## Usage

```python
from lib.firestore.documents import (
    firestore_client,
    get_document,
    set_document,
    update_document,
    delete_document,
    list_documents,
    DocumentNotFoundError,
    UnauthorizedAccessError,
)

# Get a document with ownership check
ref, snapshot = get_document("domains", domain_id, owner_id=user_id)

# Get a document that must have completed status
ref, snapshot = get_document("grids", grid_id, owner_id=user_id, document_status="completed")

# Create or overwrite a document
set_document("grids", grid_id, data)

# Partial update (raises DocumentNotFoundError if deleted/cancelled)
update_document("grids", grid_id, {"progress": {"message": "Processing..."}})

# Delete a document
delete_document("grids", grid_id)

# List documents with pagination, sorting, and filters
docs, total = list_documents(
    "grids",
    owner_id=user_id,
    page=0,
    size=100,
    sort_by="created_on",
    sort_order="descending",
    filters={"domain_id": domain_id, "status": "completed"},
)
```

## Serialization

Firestore has limitations on deeply nested arrays. Use serializers for GeoJSON coordinates:

```python
from lib.firestore.serializers import serialize_coordinates, deserialize_coordinates

# Before storing in Firestore
data = serialize_coordinates(data)

# After reading from Firestore
data = deserialize_coordinates(data)
```
