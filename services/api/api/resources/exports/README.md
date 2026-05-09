# V2 Export Resource

Exports let users download v2 data in usable formats (GeoTIFF, QUIC-Fire, etc.).

## Design Decisions

- **Format in URL**: `/exports/geotiff`, not a discriminator field in body
- **Two creation endpoints**: Domain-level for one or more grids, per-grid for single grid convenience
- **Lifecycle CRUD is top-level**: `GET /v2/exports/{id}`, `DELETE /v2/exports/{id}`
- **Exports survive domain deletion**: No cascade delete; exports are standalone artifacts
- **Separate exporter-v2 backend service**: Not griddle — different concerns (read + convert + upload vs fetch +
  compute + store)

## Endpoints

### Creation (domain-scoped, under grids)

| Endpoint                                                       | Body field for grid(s)    | Use case                                    |
|----------------------------------------------------------------|---------------------------|---------------------------------------------|
| `POST /v2/domains/{domain_id}/grids/exports/geotiff`           | `grid_ids: list[str]`     | Export one or more grids                    |
| `POST /v2/domains/{domain_id}/grids/{grid_id}/exports/geotiff` | *(none — grid_id in URL)* | Export a single grid                        |
| `POST /v2/domains/{domain_id}/grids/exports/quicfire`          | typed roles (see below)   | Combined surface + canopy fuel (+ terrain)  |

### Lifecycle (top-level)

```
GET    /v2/exports                     # List exports (filter by domain_id, source_name, tag)
GET    /v2/exports/{export_id}         # Get status + signed URL + curl
PATCH  /v2/exports/{export_id}         # Update name/description/tags
DELETE /v2/exports/{export_id}         # Delete export + GCS files
```

## Request Examples

### Export a single grid (per-grid endpoint)

```
POST /v2/domains/{domain_id}/grids/{grid_id}/exports/geotiff
```

```json
{
    "bands": ["fuel_load.1hr", "fuel_load.10hr"],
    "name": "Fuel loads only",
    "tags": ["surface-fuel"]
}
```

The body is optional — send `{}` to export all bands with no metadata.

### Export multiple grids (domain-level endpoint)

```
POST /v2/domains/{domain_id}/grids/exports/geotiff
```

```json
{
    "grid_ids": ["abc123def456", "ghi789jkl012"],
    "name": "Surface fuels + topography",
    "tags": ["combined"]
}
```

### QUIC-Fire combined export (typed roles)

```
POST /v2/domains/{domain_id}/grids/exports/quicfire
```

Each role is `{grid_id, band}`. Five roles are required (`canopy_bulk_density`,
`canopy_moisture`, `surface_fuel_load`, `surface_fuel_depth`, `surface_moisture`)
and three are optional (`topography`, `canopy_savr`, `surface_savr`). The SAVR
roles are paired — supply both or neither.

```json
{
    "canopy_bulk_density": {"grid_id": "tree_xyz",    "band": "bulk_density.foliage.live"},
    "canopy_moisture":     {"grid_id": "tree_xyz",    "band": "fuel_moisture.live"},
    "surface_fuel_load":   {"grid_id": "lookup_abc",  "band": "fuel_load.1hr"},
    "surface_fuel_depth":  {"grid_id": "lookup_abc",  "band": "fuel_depth"},
    "surface_moisture":    {"grid_id": "uniform_def", "band": "fuel_moisture.1hr"},
    "topography":          {"grid_id": "topo_xyz",    "band": "elevation"}
}
```

Validation (router pre-write):
- Every grid exists, is owned by the user, lives in this domain, has `status="completed"`.
- Every named band exists on its grid and carries the role-required unit
  (`canopy_bulk_density` → `kg/m³`, moisture → `%`, fuel_load → `kg/m²`,
  depth/elevation → `m`, SAVR → `m⁻¹`).
- Canopy roles are 3D, surface and topography roles are 2D.
- Every grid's cell size matches the canopy grid's `dx`. On mismatch the
  router returns 422 with a pointer to `POST .../grids/{id}/resample`; the
  exporter never resamples silently.

Output zip contains `treesrhof.dat`, `treesmoist.dat`, `treesfueldepth.dat`,
`metadata.json`, `domain.geojson`; plus `topo.dat` when `topography` is set,
plus `treesss.dat` when both SAVR roles are set.

## Export Schema

```python
class Export(BaseModel):
    id: str
    domain_id: str              # provenance, not lifecycle dependency
    name: str
    description: str
    tags: list[str]
    status: JobStatus           # pending -> running -> completed | failed
    progress: Optional[JobProgress]
    error: Optional[JobError]
    source: dict                # format-specific (same pattern as Grid.source)
    signed_url: Optional[str]   # signed GCS URL, populated on completion
    curl: Optional[str]         # curl command for download, populated on completion
    expires_on: Optional[datetime]
    created_on: datetime
    modified_on: datetime
    owner_id: str
```

## Source Schema

Each export format has its own source schema with `name` identifying the format:

### GeoTIFF

```json
{
    "source": {
        "name": "geotiff",
        "grid_ids": ["abc123"],
        "bands": ["fuel_load.1hr", "fuel_load.10hr"]
    }
}
```

- `grid_ids` is always a list, even for single-grid exports via the per-grid endpoint
- `bands` is optional — `null` means all bands from the grid(s)

### QUIC-Fire

```json
{
    "source": {
        "name": "quicfire",
        "domain_id": "abc123",
        "canopy_bulk_density": {"grid_id": "tree_xyz",    "band": "bulk_density.foliage.live"},
        "canopy_moisture":     {"grid_id": "tree_xyz",    "band": "fuel_moisture.live"},
        "canopy_savr":         null,
        "surface_fuel_load":   {"grid_id": "lookup_abc",  "band": "fuel_load.1hr"},
        "surface_fuel_depth":  {"grid_id": "lookup_abc",  "band": "fuel_depth"},
        "surface_moisture":    {"grid_id": "uniform_def", "band": "fuel_moisture.1hr"},
        "surface_savr":        null,
        "topography":          {"grid_id": "topo_xyz",    "band": "elevation"},
        "resolved": {
            "domain": {"crs": "...", "bbox": [...]},
            "fire_grid": {"nx": ..., "ny": ..., "nz": ..., "transform": [...], "z_origin": ..., "z_resolution": ..., "crs": "..."},
            "roles": {
                "canopy_bulk_density": {"grid_id": "tree_xyz", "band": "bulk_density.foliage.live", "unit": "kg/m³", "dimensionality": 3, "shape": [...], "transform": [...], "crs": "..."},
                "canopy_moisture":     {"...": "..."},
                "...": "..."
            }
        }
    }
}
```

The `resolved` block snapshots CRS / transform / shape / units per role at
request time so the exporter consumes pre-validated data and the export
remains reproducible if a source grid is later modified or deleted.

Forward extensibility for `nfuel>1` (when QUIC-Fire's multi-fuel-type
capability becomes relevant): each per-fuel-type role accepts
`FieldSource | dict[FuelType, FieldSource]` keyed by the canonical names
(`dead_thin`, `live_thin`, `dead_thick`, `unburnable`). Today's scalar
requests keep working unchanged.

## Lifecycle

1. User creates export via either creation endpoint
2. API validates each grid (exists, owned by user, in this domain, status `completed`)
3. API validates requested bands exist in the grid(s)
4. API creates Export doc in Firestore (status: `pending`)
5. API enqueues Cloud Task to `exporter-v2-queue`
6. exporter-v2 loads grid Zarr, converts to GeoTIFF, uploads to GCS
7. exporter-v2 generates a signed download URL and curl command
8. exporter-v2 updates Export doc (status: `completed`, `signed_url` and `curl` populated)
9. User polls `GET /exports/{id}` until `completed`, then downloads via `signed_url`

## No Cascade Delete

Exports are standalone artifacts. Deleting a domain does NOT delete its exports. The `domain_id` in the export is
provenance metadata, not a lifecycle dependency. This allows users to:

- Delete a domain while keeping generated exports
- Clean up input data without losing output files

## Backend Service

Exports are processed by **exporter-v2**, a separate Cloud Run service from griddle.
See [services/exporter-v2/README.md](../../../../../exporter-v2/README.md).

## Future Work

- **Multi-grid alignment**: Truth grid concept for CRS/resolution matching across grids
- **Signed URL regeneration**: Endpoint to regenerate expired signed URLs
- **FDS/FIRETEC formats**: Additional fire model input formats
- **QUIC-Fire `nfuel>1`**: Multi-fuel-type slabs in `treesrhof.dat` / `treesmoist.dat` / `treesfueldepth.dat` / `treesss.dat`, additive to today's per-role schema
