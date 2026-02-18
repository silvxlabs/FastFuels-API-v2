# Domains Resource (v2)

This document describes the v2 implementation of the Domains resource for the FastFuels API.

## Overview

A Domain represents a geographic area of interest defined by GeoJSON geometry. Domains are the **parent container** for grids, inventories, exports, and features — all child resources are accessed through domain-scoped URLs (e.g., `/v2/domains/{domain_id}/grids/...`). Domain deletion cascades to child resources when `force=true` is specified (per AIP-135).

The v2 API uses [geojson-pydantic](https://github.com/developmentseed/geojson-pydantic) for GeoJSON validation, providing RFC 7946 compliant models out of the box.

## Endpoints

The domains resource provides full CRUD operations plus listing with pagination.

### POST /v2/domains

Create a new domain from GeoJSON geometry.

**Key Points:**
- Input must be a **FeatureCollection** (not a single Feature like v1)
- Geographic CRS (e.g., WGS84) is automatically projected to UTM
- Projected CRS (e.g., EPSG:5070) is preserved as-is
- Maximum area: 16 square kilometers
- Must be within CONUS
- Returns **201 Created** with the full domain resource

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
  "domains": [...],
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

**Immutable Fields:**
- `id`, `features`, `crs`, `created_on`

**Key Points:**
- Only provided fields are updated
- `modified_on` is automatically updated
- Returns the full updated domain resource

### DELETE /v2/domains/{domain_id}

Permanently delete a domain.

**Query Parameters:**
- `force` (bool, default: false) — Force cascade delete of all child resources (grids, etc.). Without this, returns 412 if child resources exist.

**Key Points:**
- Returns **204 No Content** on success
- Returns **404 Not Found** if domain doesn't exist or user doesn't own it
- Returns **412 Precondition Failed** if domain has child grids and `force` is not set (per AIP-135)
- With `force=true`: cascade-deletes all child grids via Firestore batch delete, then deletes the domain
- Deletion is permanent and cannot be undone

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

| Function | Purpose | Error Code |
|----------|---------|------------|
| `validate_crs()` | Validates CRS is a valid authority string | 422 |
| `validate_geometry_has_area()` | Ensures geometry has non-zero area | 422 |
| `validate_area_within_limits()` | Checks area < 16 sq km | 422 |
| `validate_within_conus()` | Verifies geometry is in CONUS | 422 |
| `estimate_utm_crs()` | Determines appropriate UTM zone | 422 |

The main `validate_domain()` function orchestrates all validations and returns a `DomainValidationResult`.

## Database Operations

The `api/db/documents.py` module provides shared async Firestore operations used by all v2 resources:

| Function | Purpose | Returns |
|----------|---------|---------|
| `get_document_async()` | Retrieve with ownership/status validation | `(ref, snapshot)` |
| `set_document_async()` | Create or overwrite document | `ref` |
| `list_documents_async()` | Paginated list with sorting | `(documents, total_count)` |
| `update_document_async()` | Partial update (merge) | `ref` |
| `delete_document_async()` | Delete document | `None` |

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

| Test File | Type | Scope | Fixture Scope |
|-----------|------|-------|---------------|
| `test_documents_unit.py` | Unit | DB operations with mocked Firestore | Function |
| `test_documents_integration.py` | Integration | DB operations with real Firestore | Session |
| `test_validate.py` | Unit | Validation functions | Function |
| `test_router.py` | Integration | Full HTTP request/response | Session* |

*Exception: Delete tests use function-scoped fixtures since the resource is consumed during the test.

**Key Testing Patterns:**
- Session-scoped fixtures for test data that persists across tests (better performance)
- Function-scoped fixtures when the test consumes/modifies the resource destructively
- Fixtures include cleanup logic to handle both success and failure cases
- Integration tests use `pytest-asyncio` with `loop_scope="session"`

See `tests/README.md` for the complete testing protocol.

## Migration from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Input types | Feature or FeatureCollection | **FeatureCollection only** |
| GeoJSON models | Custom implementation | geojson-pydantic |
| Serialization | Manual function calls | Automatic via model decorators |
| Validation | Mixed in router/utils | Modular validate.py |
| Local CRS | Supported | **Not supported** |
| Tests | Mixed concerns | Unit (validate/db) + Integration (router/db) |
| CRUD operations | Create, Get, Delete | **Create, Get, Update, Delete, List** |
| List endpoint | None | **Paginated with sorting** |
| DB operations | Inline in router | **Shared documents.py module** |
