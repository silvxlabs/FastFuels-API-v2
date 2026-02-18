# api/db - Async Database Operations for FastAPI

This module provides asynchronous database operations for use in FastAPI route handlers.

## Structure

```
db/
├── documents.py    # Async Firestore client and document operations
└── blobs.py        # Async GCSFS client and blob operations
```

## Usage

### Document Operations

```python
from api import get_document

# Get a document with ownership check
ref, snapshot = await get_document("domains-v2", domain_id, owner_id=user_id)
data = snapshot.to_dict()

# Get a document that must have completed status
ref, snapshot = await get_document("grids", grid_id, owner_id=user_id, document_status="completed")
```

### Blob Operations

```python
from api import delete_directory, check_exists

# Delete a directory in background
await delete_directory("bucket-name", f"{domain_id}/tree")

# Check if path exists
if await check_exists("bucket-name", f"{domain_id}/surface"):
    ...
```

## Relationship to lib/v2

This module provides **async** versions of operations in `lib/v2/`. The sync versions
in `lib/v2/` are used by background jobs; these async versions are for FastAPI routes.

Serialization utilities are shared - import from `lib.firestore.serializers`:

```python
from lib.firestore.serializers import serialize_coordinates, deserialize_coordinates
```

## Note on Signed URLs

Signed URL generation remains in `lib.gcs.signed_urls` (sync only). The Google Cloud
Storage library doesn't provide async URL signing. Call from async routes as:

```python
from lib.gcs import generate_upload_signed_url

# This is a sync call but fast enough (~10-50ms) to be acceptable
url = generate_upload_signed_url(bucket, path, max_size)
```

For high-throughput scenarios, consider `await asyncio.to_thread(generate_upload_signed_url, ...)`.
