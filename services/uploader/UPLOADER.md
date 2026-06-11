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

## Point Clouds: Why LAZ (not COPC), and Eventual COPC Support

The point cloud handler stores uploads as plain **LAZ** (`pointclouds/{id}/cloud.laz`,
always in the domain CRS), not as Cloud Optimized Point Cloud (COPC). This was a
deliberate trade, made after benchmarking every available COPC writer (June 2026):

- **Every complete, correct COPC writer needs the native PDAL stack.** PDAL is
  conda-forge-only (no pip wheels; removed from Debian in 2022), which would force a
  conda layer into this service's image, CI, and dev setup. The pure-Rust
  `copc-converter` was evaluated and is fast, but v0.11.0 silently zeroes all LAS 1.4
  flag bits (synthetic/key_point/withheld/overlap/scanner_channel/scan_direction/edge)
  — disqualifying data loss.
- **COPC octree builds need large seekable scratch.** Measured: untwine's temp dir
  peaks at ~8× the compressed input. On Cloud Run all local disk is tmpfs — it counts
  against instance RAM — so a 1 GiB upload would need a ~16 GB instance. The
  ephemeral-disk volume type that would fix this is a beta feature we chose not to
  build core architecture around.
- **LAZ does everything current consumers need.** The handler streams uploads from GCS
  with laspy (pure pip, `lazrs` Rust codec), reprojects via pyproj when needed, and
  rewrites through an in-memory buffer — no local disk at all, peak RAM bounded by the
  upload size cap, and the service keeps the standard `uv sync --frozen` setup.

**Eventual COPC support:** COPC is a strict superset of LAZ (a COPC file *is* valid
LAZ), so the upgrade is a lossless, in-place `cloud.laz` → COPC batch conversion that
existing consumers won't notice. When in-browser point cloud streaming lands on the
FastFuels-Web roadmap, run that conversion with **untwine** (benchmarked: 314M points
in 2m13s / 1.7 GB RAM, all attributes preserved, well-formed LOD octree) wherever the
native toolchain legitimately lives — e.g. griddle, which needs PDAL for #329's 3DEP
EPT fetch anyway — or with `copc-converter` if its flag-byte bug has been fixed
upstream. Re-evaluate the Cloud Run ephemeral-disk volume then; if it has reached GA
it provides the scratch space the build needs without the RAM cost.

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
