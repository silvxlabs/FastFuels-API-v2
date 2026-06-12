# Point Clouds Resource (v2)

Internal design notes for contributors maintaining the `point_clouds` resource. This is **not**
user-facing documentation — the user-facing surface is the OpenAPI spec generated from the route
docstrings and the `description` / `json_schema_extra` fields in
[`schema.py`](schema.py). Keep those authoritative and self-contained for SDK / external users.

## Overview

A **point cloud** is a collection of 3D points captured by a laser scanner. It is the one primitive
v1's monolithic `treecondenser` pipeline was missing in v2. With it, that monolith decomposes into a
composition of existing primitives:

```
ALS (3dep | upload) → CHM canopy grid (point_cloud source) → tree inventory (chm ITD) → grids / exports
```

Point clouds are **domain-scoped** and **multiple-per-domain** (unlike v1's one-per-domain,
ALS-only model). Each carries a `type` discriminator:

- `als` — **Airborne Laser Scanning** (aircraft/drone, top-down). Source: `upload` or `3dep`.
- `tls` — **Terrestrial Laser Scanning** (tripod, ground-up). Source: `upload` only — 3DEP is
  airborne and cannot produce terrestrial scans.

`type` is a plain top-level categorical field (not a shape-changing discriminated union): `als` and
`tls` share identical structure and are listed together, so a single resource with a filterable
`type` is the right model (cf. _API Design Patterns_ ch. 16, Polymorphism).

## Scope of this resource

This package ships the **CRUD framework only**:

- `schema.py` — `PointCloud`, `PointCloudType`, `PointCloudGeoreference`, the update request body,
  and the list response.
- `router.py` — list (cross-domain + domain-scoped), get, patch (metadata only), and delete (with
  async GCS cleanup).

**Creation endpoints are intentionally absent here** and arrive in follow-on work:

- **#328** — upload a point cloud (signed-URL ingest + uploader handler).
- **#329** — fetch an ALS point cloud from USGS 3DEP (EPT fetch + coverage pre-flight).
- **#330** — convert a point cloud to a CHM (a new `point_cloud` source on the existing
  `grids/canopy` grid — not a new resource).

A point cloud comes into existence via a creation router (the `upload` source, #328) or by being
seeded directly in Firestore (tests). The creation routers use a `source`-keyed sub-router
structure (`upload` / `3dep`).

## URL structure

```
GET    /domains/-/pointclouds                       # list across all domains (wildcard parent)
GET    /domains/{domain_id}/pointclouds             # list within a domain
GET    /domains/{domain_id}/pointclouds/{id}        # get
PATCH  /domains/{domain_id}/pointclouds/{id}        # update metadata only
DELETE /domains/{domain_id}/pointclouds/{id}        # delete (+ async GCS cleanup)
```

The path segment is the **collapsed single token `pointclouds`** (no underscore, no hyphen), matching
the existing single-token segments (`/grids`, `/inventories`) and v1's `pointclouds`. Internal Python
identifiers stay snake_case (package `point_clouds/`, `POINT_CLOUDS_*` constants) per PEP 8.

## Resource model

See `schema.py` for the authoritative definition. Notable fields:

- **`source: dict`** — provenance, always with a `name` discriminator (`upload` | `3dep`) plus
  source-specific parameters. Loosely typed here; the typed source models land with #328/#329 (same
  pattern as grids/inventories, whose `source` is also a `dict`).
- **`georeference: PointCloudGeoreference | None`** — `crs` plus a flat 3D bounding box
  `[min_x, min_y, min_z, max_x, max_y, max_z]`. `null` until the worker finishes ingesting. The
  bbox is a single-level array, so it needs **no** coordinate stringification (unlike domains'
  nested GeoJSON coordinates — Firestore rejects nested arrays). `crs` is **always the domain
  CRS**: unlike grids (where reprojection = resampling and a mismatch is rejected), point
  reprojection is an exact per-point transform, so the #328 upload worker reprojects mismatched
  uploads instead of rejecting them.
- **`summary: PointCloudSummary | None`** — per-cloud statistics (`point_count`, `point_classes` =
  ASPRS classes present, `density` = points/m²). `null` until the worker finishes ingesting. Nested in
  a `summary` sub-model mirroring the planned grid/inventory `summary` pattern (#257/#258). First
  populated by the **#328 upload handler** (the first worker that reads a cloud's bytes); #329/#330
  populate it the same way.

## Checksum & staleness

Follows the #304 pattern: `checksum` is an opaque `uuid4().hex` assigned at creation (not a hash),
changed whenever content is rebuilt, and **unaffected by metadata-only PATCH**. Derivatives capture
the value they were built from — e.g. the CHM grid in #330 stores `source_pointcloud_checksum` —
and staleness detection is user-space (compare stored vs. current; no stale flag/warn/block in the
API).

## Storage

- Firestore collection: `POINT_CLOUDS_COLLECTION` (default `pointclouds-v2`).
- GCS bucket: `POINT_CLOUDS_BUCKET`. Each point cloud owns the directory `{id}/` at the bucket root;
  `delete` operates on that whole directory via `delete_directory_safe` and is therefore
  format-agnostic.
- The **concrete object layout under `{id}/` and the file format are owned by the ingest workers**
  (#328/#329). This resource deliberately commits to no specific format (e.g. COPC vs. plain LAZ) —
  that decision belongs to whoever writes the bytes. The #328 upload worker stores `{id}/cloud.laz`
  (plain LAZ, domain CRS); see the uploader service README for why LAZ-not-COPC and the planned
  lossless LAZ → COPC upgrade path.

## Service boundary

The API must stay free of GDAL/PDAL at runtime (see the repo `CLAUDE.md`). This router does Firestore
+ (future) signed-URL + Cloud Tasks dispatch only; it imports just `api.db.*`, `api.dependencies`,
`api.schema`, and `lib.config`. All point-cloud parsing happens in workers. Do not import any `lib`
module that transitively pulls in `rasterio` / `affine` / `pdal`.

## Out of scope

TLS-specific fuel/tree reconstruction (plot-scale voxel fuels), point-cloud exports, and a direct
point-cloud → inventory shortcut (intentionally omitted in favor of composition through the CHM
grid).
