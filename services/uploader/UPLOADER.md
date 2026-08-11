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

- `resource_type`: `inventories`, `grids`, or `pointclouds`
- `resource_id`: the Firestore document ID created by the API before issuing the signed URL
- `filename`: the original filename (e.g. `trees.csv`, `fuel.tif`)

Objects that don't match this structure are silently ignored.

## Supported Resource Types

| Type | Handler | Status |
|---|---|---|
| `inventories` | `handlers/inventory.py` | Issue #214 |
| `grids` | `handlers/grid.py` | Issue #215 |
| `pointclouds` | `handlers/point_cloud.py` | Issue #328 |

## Point Clouds: the shared dataset format

The point cloud handler stores uploads as a **partitioned Parquet dataset**
(`pointclouds/{id}/cloud.parquet/`, always in the domain CRS) — the same format
lakitu writes for 3DEP, defined once in `lib.pointcloud.schema`. A point cloud is
one resource type, and a reader should not have to ask where it came from.

This replaced plain LAZ. The reasoning for LAZ over COPC still holds and is why
COPC was never the answer either: every complete COPC writer needs the native
PDAL stack (conda-forge only, no pip wheels, removed from Debian in 2022), and
COPC octree builds need scratch at ~8× the compressed input, which on Cloud Run
is tmpfs and therefore RAM. Partitioned Parquet gets the spatial chunking and the
LOD pyramid that motivated COPC, in one streaming pass, with no native stack and
no scratch: partitions *are* the output.

**What the change costs.** Two things a LAZ rewrite preserved are gone, because
a fixed schema has nowhere to put them:

- **extra dimensions** — anything beyond position, intensity, classification and
  colour
- **gps_time** — 23% of the stored bytes, and nothing downstream reads it

Scaling is *not* lost: the dataset records its own scale in the manifest, so a
sub-millimetre terrestrial scan keeps its precision rather than being quantised
to the canonical millimetre. There is a test for exactly that.

**The server-side copy fast path is also gone.** An upload already in LAZ and
already in the domain CRS used to be copied without rewriting. Every upload is
now decoded and re-encoded. The cost is smaller than it sounds — the copy path
already decoded every point to census it, so what is added is the encode.


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
