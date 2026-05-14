# Uploader Service

Processes files uploaded directly to GCS by clients via signed URLs.

## Trigger

Unlike all other v2 services, uploader uses `FUNCTION_SIGNATURE_TYPE=cloudevent` and is
triggered by **Eventarc** — not Cloud Tasks. When a client PUTs a file to the uploads
bucket, GCS emits a `google.cloud.storage.object.v1.finalized` event, Eventarc delivers
it here.

## Object Path Convention

All files in the uploads bucket must follow this path structure:

```
{resource_type}/{resource_id}/{filename}
```

- `resource_type`: `inventories` or `grids`
- `resource_id`: the Firestore document ID created by the API before issuing the signed URL
- `filename`: the original filename (e.g. `trees.csv`, `fuel.tif`)

Objects that don't match this structure are silently ignored.

## Supported Resource Types

| Type | Handler | Status |
|---|---|---|
| `inventories` | `handlers/inventory.py` | Issue #214 |
| `grids` | `handlers/grid.py` | Issue #215 |

## Local Testing

Construct a `CloudEvent` object and call `process_upload` directly:

```python
from cloudevents.http import CloudEvent
from uploader.main import process_upload

event = CloudEvent(
    attributes={"type": "google.cloud.storage.object.v1.finalized", "source": "test"},
    data={"bucket": "my-uploads-bucket", "name": "inventories/abc123/trees.csv"},
)
process_upload(event)
```

## Eventarc Trigger Setup (post-deploy)

Run once per environment after the service first deploys:

```bash
gcloud eventarc triggers create uploader-v2-prod-trigger \
  --destination-run-service=uploader-v2-prod \
  --destination-run-region=us-west1 \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${UPLOADS_BUCKET}" \
  --location=us-west1 \
  --project=silvx-fastfuels
```
