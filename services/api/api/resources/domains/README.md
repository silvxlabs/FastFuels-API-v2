# Domains Resource (v2)

This document describes the v2 implementation of the Domains resource for the FastFuels API.

## Overview

A Domain represents a geographic area of interest defined by GeoJSON geometry. Domains are the **parent container** for
grids, inventories, exports, and features — all child resources are accessed through domain-scoped URLs (e.g.,
`/v2/domains/{domain_id}/grids/...`). Domain deletion cascades to child resources when `force=true` is specified (per
AIP-135).

The v2 API uses [geojson-pydantic](https://github.com/developmentseed/geojson-pydantic) for GeoJSON validation,
providing RFC 7946 compliant models out of the box.

## Endpoints

The domains resource provides full CRUD operations plus listing with pagination.

### POST /v2/domains

Create a new domain from GeoJSON geometry.

**Key Points:**

- Input must be a **FeatureCollection** (not a single Feature like v1)
- Geographic CRS (e.g., WGS84) is automatically projected to UTM
- Projected CRS (e.g., EPSG:5070) is preserved as-is
- Maximum area: 16 square kilometers (validated against the working extent, which may be padded)
- Must be within CONUS (validated against the original input polygon)
- Optional `pad_to_resolution` (meters) snaps the working extent to a grid for cross-resolution alignment
- Optional `style` object for map rendering (auxiliary metadata; see [Style](#style) below)
- Returns **201 Created** with the full domain resource (two features + bbox + optional fields)

### GET /v2/domains

List all domains belonging to the authenticated user with pagination and sorting.

**Query Parameters:**

- `page` (int, default: 0) - Zero-indexed page number
- `size` (int, default: 100, max: 1000) - Items per page
- `sort_by` (enum) - Field to sort by: `created_on`, `modified_on`, `name`
- `sort_order` (enum) - Sort direction: `ascending`, `descending`

**Response:**

```json
{
  "domains": [
    ...
  ],
  "current_page": 0,
  "page_size": 100,
  "total_items": 42
}
```

### GET /v2/domains/{domain_id}

Retrieve a specific domain by ID.

**Key Points:**

- Returns **404 Not Found** for both missing documents and ownership mismatches (avoids leaking existence)
- Returns the full domain resource including geometry

### PATCH /v2/domains/{domain_id}

Update domain metadata (partial update).

**Updatable Fields:**

- `name` - Domain name
- `description` - Domain description
- `tags` - Array of tags (replaces existing)
- `style` - Visual style sub-fields (merged, not replaced — see [Style](#style))

**Immutable Fields:**

- `id`, `features`, `crs`, `created_on`

**Key Points:**

- Only provided fields are updated
- `style` uses **nested merge semantics**: only the sub-fields you supply are updated; unspecified sub-fields keep
  their current values
- `modified_on` is automatically updated
- Returns the full updated domain resource

### DELETE /v2/domains/{domain_id}

Permanently delete a domain.

**Query Parameters:**

- `force` (bool, default: false) — Force cascade delete of all child resources (grids, etc.). Without this, returns 412
  if child resources exist.

**Key Points:**

- Returns **204 No Content** on success
- Returns **404 Not Found** if domain doesn't exist or user doesn't own it
- Returns **412 Precondition Failed** if domain has child grids and `force` is not set (per AIP-135)
- With `force=true`: cascade-deletes all child grids via Firestore batch delete, then deletes the domain
- Deletion is permanent and cannot be undone

## Two-Feature Structure

Every domain stores **two named features** in its FeatureCollection:

1. **`name: "domain"`** — A polygon covering the working extent. This is the
   bounding box of the user's input, optionally snapped to `pad_to_resolution`.
   It is the authoritative spatial extent used by griddle, standgen, and the
   exporter. Every grid, inventory, and export derived from this domain shares
   this extent.

2. **`name: "input"`** — The user's original projected geometry, preserved for
   visualization, reference, and any future operations that need the actual
   shape (e.g., polygon clipping). One or more features depending on whether
   the input was a single polygon or multiple.

The standard GeoJSON `bbox` field is also populated and equals the bounds of
the "domain" feature. Downstream consumers can read either the field or
filter features by `properties.name`.

### Why two features?

This makes the domain self-documenting at the data level. A reader inspecting
the JSON sees both the working extent and the input polygon as first-class
GeoJSON features that any viewer can render. There is no custom field whose
meaning must be looked up in documentation.

The "domain" feature's geometry always contains the "input" feature's geometry
by construction, so `gdf.total_bounds` of the parsed FeatureCollection equals
the working extent — downstream handlers that already use `total_bounds` get
the correct answer without modification.

## Style

Domains carry an optional `style` object describing how a client should render them on a map. The field lives at the
top of the `Domain` resource (not on individual features), since the only consumer — the webapp — applies one style per
domain rather than picking different colors for the working-extent and input features.

### Sub-fields

| Field            | Type    | Constraints   | Notes                                      |
|------------------|---------|---------------|--------------------------------------------|
| `stroke_color`   | string  | ≤ 64 chars    | any renderer-supported format (hex, named, `rgb()`, ...) |
| `stroke_opacity` | number  | `0 ≤ x ≤ 1`   | `1.0` = fully opaque                       |
| `stroke_width`   | number  | `x ≥ 0`       | pixels                                     |
| `fill_color`     | string  | ≤ 64 chars    | any renderer-supported format              |
| `fill_opacity`   | number  | `0 ≤ x ≤ 1`   |                                            |

All sub-fields are optional. Color strings are not format-validated (the API has no opinion about which color syntax
your renderer accepts) — only a defensive 64-character cap. Out-of-range opacities or negative widths return **422**.

### Merge semantics on PATCH

`PATCH /v2/domains/{id}` with `{"style": {"fill_color": "#abcdef"}}` updates **only** `fill_color`. All other style
sub-fields keep their current values. To reset a sub-field, supply the new value explicitly. There is no way to delete
an individual sub-field via PATCH (clients that want to "clear" a color should overwrite it with whatever default they
prefer).

### Why no per-feature endpoint?

v1 exposed `PATCH /v1/domains/{id}/features/{feature_name}/style`, letting clients style the `domain` and `input`
features separately. We deliberately did not port this shape: the webapp tracks one color per domain, and the
`domain`/`input` split is a server-side modeling artifact (working extent vs. original polygon) that doesn't show up in
the user's mental model. Putting the style on the resource root keeps the API surface flat and matches how clients
actually use it. If a future UI needs per-feature styling, it can be added as an additive change without breaking this
design.

## pad_to_resolution

`pad_to_resolution` is an optional float (meters) on domain creation. When
set, the bounding box of the projected input polygon is snapped outward
(`floor` for mins, `ceil` for maxs) to the nearest multiple of this value
before being stored as the "domain" feature.

### When to use it

For compositional workflows where multiple grids at different resolutions
need to share an extent. Example: a 2m QUIC-Fire export combining a 5m
topography grid, a 2m voxelized inventory, and a 30m LANDFIRE grid resampled
to 2m. Without alignment, these resources have different footprints. With
`pad_to_resolution: 2` (or any divisor of 2 — 2, 6, 10, 30), every resource
on the domain inherits the same padded extent and the export grid is
unambiguous.

## File Structure

```
api/
├── db/
│   └── documents.py       # Async Firestore operations (shared by all resources)
└── resources/domains/
    ├── __init__.py
    ├── router.py          # FastAPI route handlers with endpoint documentation
    ├── schema.py          # Pydantic models with Firestore serialization
    ├── validate.py        # Modular validation functions
    ├── examples.py        # OpenAPI documentation examples (tested in CI)
    └── README.md          # This file

api/data/
└── conus_4326.geojson     # CONUS boundary for validation

tests/
├── conftest.py            # Test client and fixtures
├── db/
│   ├── test_documents_unit.py         # Unit tests for db functions (mocked Firestore)
│   └── test_documents_integration.py  # Integration tests (real Firestore)
└── resources/domains/
    ├── test_validate.py   # Unit tests for validation functions
    ├── test_router.py     # Integration tests (calls live API)
    └── data/              # Test GeoJSON files
        ├── blue_mountain_feature_4326.geojson
        ├── point.geojson
        ├── polygon_in_alaska.geojson
        ├── polygon_in_italy.geojson
        ├── polygon_utm.geojson
        └── saint_mary_5070.geojson
```

## Validation

Domain validation is implemented in `validate.py` as modular, independently testable functions:

| Function                        | Purpose                                   | Error Code |
|---------------------------------|-------------------------------------------|------------|
| `validate_crs()`                | Validates CRS is a valid authority string | 422        |
| `validate_geometry_has_area()`  | Ensures geometry has non-zero area        | 422        |
| `validate_area_within_limits()` | Checks area < 16 sq km                    | 422        |
| `validate_within_conus()`       | Verifies geometry is in CONUS             | 422        |
| `estimate_utm_crs()`            | Determines appropriate UTM zone           | 422        |

The main `validate_domain()` function orchestrates all validations and returns a `DomainValidationResult`.

## Database Operations

The `api/db/documents.py` module provides shared async Firestore operations used by all v2 resources:

| Function                  | Purpose                                   | Returns                    |
|---------------------------|-------------------------------------------|----------------------------|
| `get_document_async()`    | Retrieve with ownership/status validation | `(ref, snapshot)`          |
| `set_document_async()`    | Create or overwrite document              | `ref`                      |
| `list_documents_async()`  | Paginated list with sorting               | `(documents, total_count)` |
| `update_document_async()` | Partial update (merge)                    | `ref`                      |
| `delete_document_async()` | Delete document                           | `None`                     |

**Ownership Pattern:**

- All documents store an `owner_id` field
- `get_document_async()` validates ownership and returns 404 for both missing and unauthorized
- Callers must validate ownership via `get_document_async()` before calling update/delete

## Firestore Serialization

GeoJSON coordinates are deeply nested arrays which Firestore doesn't support. The `Domain` model automatically:

- Stringifies coordinates when writing (`model_dump(context={'for_firestore': True})`)
- Parses stringified coordinates when reading (automatic in `@model_validator`)

## Examples

The `examples.py` file contains example request bodies used in:

1. OpenAPI/Swagger documentation
2. Integration tests (ensuring examples always work)

All examples are FeatureCollections covering different CRS formats:

- WGS84 (default and explicit)
- EPSG:5070 (CONUS Albers)
- UTM (EPSG:32611)

## Testing Strategy

Tests are organized by type and scope:

| Test File                       | Type        | Scope                               | Fixture Scope |
|---------------------------------|-------------|-------------------------------------|---------------|
| `test_documents_unit.py`        | Unit        | DB operations with mocked Firestore | Function      |
| `test_documents_integration.py` | Integration | DB operations with real Firestore   | Session       |
| `test_validate.py`              | Unit        | Validation functions                | Function      |
| `test_router.py`                | Integration | Full HTTP request/response          | Session*      |

*Exception: Delete tests use function-scoped fixtures since the resource is consumed during the test.

**Key Testing Patterns:**

- Session-scoped fixtures for test data that persists across tests (better performance)
- Function-scoped fixtures when the test consumes/modifies the resource destructively
- Fixtures include cleanup logic to handle both success and failure cases
- Integration tests use `pytest-asyncio` with `loop_scope="session"`

See `tests/README.md` for the complete testing protocol.

## Migration from v1

| Aspect          | v1                                                         | v2                                                               |
|-----------------|------------------------------------------------------------|------------------------------------------------------------------|
| Input types     | Feature or FeatureCollection                               | **FeatureCollection only**                                       |
| GeoJSON models  | Custom implementation                                      | geojson-pydantic                                                 |
| Serialization   | Manual function calls                                      | Automatic via model decorators                                   |
| Validation      | Mixed in router/utils                                      | Modular validate.py                                              |
| Local CRS       | Supported                                                  | **Not supported**                                                |
| Tests           | Mixed concerns                                             | Unit (validate/db) + Integration (router/db)                     |
| CRUD operations | Create, Get, Delete                                        | **Create, Get, Update, Delete, List**                            |
| List endpoint   | None                                                       | **Paginated with sorting**                                       |
| DB operations   | Inline in router                                           | **Shared documents.py module**                                   |
| Stored features | "domain" (padded bbox) + "input" (original polygon)        | **Same: "domain" + "input"** (explicit `properties.name`)        |
| Working extent  | `domain.horizontalResolution` mandatory; bbox padded to it | **Optional `pad_to_resolution`**; resolution moved to grids      |
| bbox field      | Not populated                                              | **Populated** (standard GeoJSON, equals "domain" feature bounds) |
