# V2 API Testing Guide

This document codifies the testing practices for the v2 API. Use this as context when writing new tests to ensure consistency across the codebase.

**Reference Implementation:** The domains resource tests serve as the reference implementation for these patterns. See `tests/resources/domains/test_router.py` and `tests/db/test_documents_*.py` for complete examples.

## Directory Structure

```
tests/
├── conftest.py                    # Session-scoped HTTP client + firestore_client
├── fixtures.py                    # Shared factory functions (make_domain_data, make_grid_data)
├── db/
│   ├── test_documents_unit.py     # Unit tests with mocked Firestore
│   └── test_documents_integration.py  # Integration tests with real Firestore
└── resources/
    ├── domains/
    │   ├── data/                   # GeoJSON test fixtures
    │   ├── test_router.py          # Endpoint integration tests
    │   ├── test_schema.py          # Pydantic model unit tests
    │   └── test_validate.py        # Validation function unit tests
    └── grids/
        ├── test_router.py          # CRUD endpoint tests (/grids, /grids/{id})
        ├── fbfm40/
        │   ├── test_router.py      # FBFM40 endpoint tests
        │   └── test_schema.py      # FBFM40 schema unit tests
        └── topography/
            ├── test_router.py      # Topography endpoint tests
            └── test_schema.py      # Topography schema unit tests
```

**Important:** Test directory structure mirrors code structure. Product-specific endpoints (FBFM40, Topography, etc.) have their own test subdirectories matching `api/resources/grids/{product}/`.

## Test Categories

### Unit Tests
- Test individual functions/classes in isolation
- Use mocks for external dependencies (Firestore, HTTP clients)
- Fast, deterministic, no network/database access
- File naming: `test_<module>_unit.py` or `test_<module>.py` for pure logic modules

### Integration Tests
- Test endpoints or database operations with real services
- Use session-scoped fixtures to minimize setup/teardown
- File naming: `test_<module>_integration.py` or `test_router.py` for endpoints

## Layered Testing Strategy

The v2 API uses a layered testing strategy where **exhaustive edge case testing happens at the database layer**, allowing router/endpoint tests to focus on integration concerns.

### Philosophy

```
┌─────────────────────────────────────────────────────────────────────┐
│  Router Tests (test_router.py)                                      │
│  - Happy paths with documented examples                             │
│  - HTTP-specific concerns (status codes, response structure)        │
│  - Relies on db layer tests for edge case coverage                  │
│  - Exception: List endpoint query params (see below)                │
└─────────────────────────────────────────────────────────────────────┘
                              │ calls
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Database Layer (db/documents.py)                                   │
│  - Unit tests: Every edge case, mocked Firestore                    │
│  - Integration tests: Real Firestore, verify behavior               │
│  - Exhaustive coverage of: owner validation, status validation,     │
│    not-found handling, validation order, pagination, sorting, etc.  │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Structure?

1. **Drill into edge cases once** - The `db.documents` module handles all document CRUD operations. By exhaustively testing every edge case there (owner mismatch → 404, status mismatch → 422, validation order, etc.), we don't need to re-test these scenarios in every router that uses these functions.

2. **Router tests verify integration** - Router tests confirm that endpoints correctly call the db layer and return appropriate HTTP responses. They don't need to test every permutation of owner/status/not-found because those are covered at the db layer.

3. **Intentional redundancy is acceptable** - Some overlap between layers is fine. For example, both db and router tests verify that owner mismatch returns 404. This redundancy catches integration bugs (e.g., router passing wrong owner_id). Just be intentional about what you're testing.

### Exception: List Endpoint Query Parameters

**List endpoints MUST test filter and sorting query parameters at the router level**, even though the db layer tests sorting/pagination.

**Why?** Firestore requires a **composite index** for each unique combination of:
- Filter field (e.g., `owner_id`)
- Sort field (e.g., `name`, `created_on`)
- Sort direction (ascending/descending)

If an index doesn't exist, the query fails at runtime. Router integration tests against real Firestore catch missing indexes that unit tests cannot.

```python
class TestListDomains:
    # These tests verify Firestore indexes exist for each query pattern
    def test_list_sorting_by_name_ascending(self, client, domains_for_listing):
        response = client.get(f"{self.route}?sort_by=name&sort_order=ascending")
        assert response.status_code == 200

    def test_list_sorting_by_name_descending(self, client, domains_for_listing):
        response = client.get(f"{self.route}?sort_by=name&sort_order=descending")
        assert response.status_code == 200

    def test_list_sorting_by_created_on(self, client, domains_for_listing):
        response = client.get(f"{self.route}?sort_by=created_on&sort_order=descending")
        assert response.status_code == 200
```

### What Goes Where?

| Test Type | Location | What to Test |
|-----------|----------|--------------|
| Owner validation edge cases | `test_documents_unit.py` | Mocked: owner match, mismatch, missing field |
| Owner validation behavior | `test_documents_integration.py` | Real Firestore: confirms behavior |
| Status validation edge cases | `test_documents_unit.py` | Mocked: status match, mismatch, validation order |
| Pagination math | `test_documents_unit.py` | Offset calculations for page/size |
| Sorting direction | `test_documents_unit.py` | Ascending vs descending |
| **List query params** | `test_router.py` | **Real Firestore with indexes** |
| Endpoint returns 404 for wrong owner | `test_router.py` | Confirms router passes owner_id correctly |
| Endpoint returns 200 for happy path | `test_router.py` | Confirms endpoint works end-to-end |
| Validation errors (422) | `test_router.py` | Confirms validation functions are called |

## Core Principles

### 1. Test Independence
Tests must not depend on each other. Each test should set up its own state and clean up after itself.

**Bad:**
```python
# GET tests depend on POST tests creating data
def test_create_domain(self, client):
    response = client.post("/domains", json=data)
    CREATED_ID = response.json()["id"]  # Shared state!

def test_get_domain(self, client):
    response = client.get(f"/domains/{CREATED_ID}")  # Depends on previous test
```

**Good:**
```python
@pytest.fixture(scope="session")
def domain_in_firestore(firestore_client):
    """Create test data directly in Firestore, independent of API."""
    domain_data = make_domain_data()
    doc_ref = firestore_client.collection("domains-v2").document(domain_data["id"])
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()

def test_get_domain(self, client, domain_in_firestore):
    response = client.get(f"/domains/{domain_in_firestore['id']}")
    assert response.status_code == 200
```

### 2. Shared Fixtures Module

Place reusable factory functions in `tests/fixtures.py` to avoid duplication across test files.

```python
# tests/fixtures.py
"""Shared factory functions for v2 tests."""

import uuid
from datetime import datetime

def make_domain_data(
    owner_id: str = "test-owner",
    name: str = "Test Domain",
    ...
) -> dict:
    """Factory function to create domain data as stored in Firestore."""
    return {...}

def make_grid_data(
    domain_id: str,
    owner_id: str = "test-owner",
    ...
) -> dict:
    """Factory function to create grid data as stored in Firestore."""
    return {...}
```

Import in test files:
```python
from tests.v2.fixtures import make_domain_data, make_grid_data
```

### 3. Factory Functions Over Redundant Fixtures
Use factory functions to create test data with sensible defaults and optional overrides.

**Bad:**
```python
@pytest.fixture
def domain_with_owner_a():
    return {"id": uuid.uuid4().hex, "owner_id": "owner-a", "name": "Test", ...}

@pytest.fixture
def domain_with_owner_b():
    return {"id": uuid.uuid4().hex, "owner_id": "owner-b", "name": "Test", ...}
```

**Good:**
```python
from tests.v2.fixtures import make_domain_data

@pytest.fixture
def domain_with_different_owner(firestore_client):
    domain_data = make_domain_data(owner_id="different-owner")
    ...
```

### 4. Helper Functions for Test Data
Create helper functions for common test data transformations.

```python
def make_feature_collection(features: list[dict], crs: str = "EPSG:4326") -> dict:
    """Create a GeoJSON FeatureCollection dict."""
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs}},
    }

def load_geojson(path: Path, crs: str | None = None) -> dict:
    """Load a GeoJSON file and wrap as FeatureCollection if needed."""
    ...
```

### 5. Sync for HTTP, Async for Database Integration Tests

**HTTP/Router Tests:** Use synchronous HTTP clients. The async machinery adds complexity without providing concurrency benefits (tests run sequentially).

```python
# conftest.py
from httpx import Client  # Sync, not AsyncClient

@pytest.fixture(scope="session")
def client():
    with Client(base_url=TEST_URL_V2, headers=HEADERS) as client:
        yield client

# test_router.py - uses sync Firestore for fixture setup
from google.cloud import firestore  # Sync Client

@pytest.fixture(scope="session")
def firestore_client():
    return firestore.Client(database=V2_FIRESTORE_DATABASE_ID)
```

**Database Integration Tests:** Use async Firestore client with `pytest-asyncio` to test the actual async functions.

```python
# test_documents_integration.py
import pytest_asyncio
from google.cloud.firestore import AsyncClient

@pytest_asyncio.fixture(loop_scope="session")
async def firestore_client():
    """Async Firestore client for database integration tests."""
    return AsyncClient(database=V2_FIRESTORE_DATABASE_ID)

class TestGetDocumentAsync:
    async def test_returns_document_when_exists(self, firestore_client, test_document):
        ref, snapshot = await get_document_async(
            collection=TEST_COLLECTION,
            document_id=test_document["doc_id"],
        )
        assert snapshot.exists
```

Note: Use `loop_scope="session"` with `pytest_asyncio.fixture` to share the event loop across session-scoped async fixtures.

### 6. Test What Could Fail
Focus tests on code paths that could realistically fail or have failed before.

**Validation tests should cover:**
- Valid inputs (happy path)
- Each validation rule's failure case
- Edge cases (empty, null, boundary values)
- Error message content

```python
def test_zero_area_raises_422(self, zero_area_geojson):
    with pytest.raises(HTTPException) as exc:
        validate_domain(zero_area_geojson)

    assert exc.value.status_code == 422
    assert "area greater than zero" in exc.value.detail
```

### 7. Verify Error Details, Not Just Status Codes
Assert on error message content to ensure helpful error messages.

```python
def test_owner_validation_mismatch_raises_404(self, ...):
    with pytest.raises(HTTPException) as exc:
        await get_document_async(collection, doc_id, owner_id="wrong")

    assert exc.value.status_code == 404
    assert "Document not found" in exc.value.detail  # Verify message
```

### 8. Document Test Intent
Use descriptive test names and docstrings explaining what behavior is being tested.

```python
def test_owner_validation_mismatch_raises_404(self, ...):
    """Test that owner_id mismatch raises 404 (not 403) for security.

    Returns 404 to avoid leaking document existence information.
    """
```

## Test File Organization

### Class-Based Organization
Group related tests into classes for better organization and shared setup.

```python
class TestValidateDomain:
    """Integration tests for the validate_domain function."""

    def test_valid_geographic_crs_succeeds(self, valid_polygon_geojson):
        ...

    def test_invalid_crs_raises_422(self, invalid_crs_geojson):
        ...

class TestValidateDomainRealData:
    """Tests using real GeoJSON files from the data directory."""

    def test_blue_mountain_succeeds(self, blue_mountain_geojson):
        ...
```

### Section Comments
Use section comments to organize large test files.

```python
# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_polygon_geojson():
    ...

# =============================================================================
# parse_geojson_to_gdf Tests
# =============================================================================

class TestParseGeojsonToGdf:
    ...
```

## Fixture Patterns

### Session-Scoped for Expensive Resources
Use session scope for resources that are expensive to create.

```python
@pytest.fixture(scope="session")
def firestore_client():
    """Session-scoped Firestore client."""
    return firestore.Client(database=V2_FIRESTORE_DATABASE_ID)

@pytest.fixture(scope="session")
def domain_in_firestore(firestore_client):
    """Create once, use across all tests, cleanup at end."""
    ...
    yield domain_data
    doc_ref.delete()  # Cleanup
```

### Function-Scoped for Isolation
Use function scope when tests need fresh data or might modify fixtures.

```python
@pytest.fixture
def valid_polygon_geojson():
    """Fresh fixture for each test."""
    return make_feature_collection(...)
```

### Function-Scoped for Destructive Operations
**Critical:** Use function scope when the test consumes or destroys the resource (e.g., delete tests).

```python
@pytest.fixture(scope="function")  # NOT session!
def domain_for_delete(self, firestore_client):
    """Create a domain that will be deleted during the test.

    Function-scoped because each delete test needs its own domain.
    """
    domain_data = make_domain_data(name="Domain to Delete")
    doc_ref = firestore_client.collection("domains-v2").document(domain_data["id"])
    doc_ref.set(domain_data)
    yield domain_data
    # Cleanup handles both success (already deleted) and failure (still exists)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.delete()

def test_delete_domain(self, client, domain_for_delete):
    response = client.delete(f"/domains/{domain_for_delete['id']}")
    assert response.status_code == 204

def test_delete_twice_returns_404(self, client, domain_for_delete):
    # First delete succeeds
    client.delete(f"/domains/{domain_for_delete['id']}")
    # Second delete returns 404
    response = client.delete(f"/domains/{domain_for_delete['id']}")
    assert response.status_code == 404
```

If you use session scope for delete test fixtures, the first test deletes the resource and all subsequent tests fail.

### Fixtures From Files
Load real-world test data from files in the `data/` directory.

```python
@pytest.fixture
def blue_mountain_geojson():
    """Blue Mountain polygon in Montana (valid, CONUS, WGS84)."""
    return load_geojson(TEST_DATA_DIR / "blue_mountain_feature_4326.geojson")
```

## Test Data Directory

Store GeoJSON and other test data files in a `data/` subdirectory:

```
tests/resources/domains/data/
├── blue_mountain_feature_4326.geojson  # Valid polygon, CONUS, WGS84
├── point.geojson                       # Zero area (validation failure)
├── polygon_in_alaska.geojson           # Outside CONUS (validation failure)
├── polygon_in_italy.geojson            # Outside CONUS (validation failure)
├── polygon_utm.geojson                 # Already projected (UTM)
└── saint_mary_5070.geojson             # Oversized (>16 sq km)
```

Name files descriptively to indicate their test purpose.

## Unit Test Patterns

### Mocking External Services
Use `unittest.mock` for unit tests that need to isolate from external services.

```python
@pytest.fixture
def mock_document_snapshot():
    """Factory fixture for creating mock document snapshots."""
    def _create_snapshot(exists: bool = True, data: dict | None = None):
        snapshot = MagicMock()
        snapshot.exists = exists
        snapshot.to_dict.return_value = data if data else {}
        return snapshot
    return _create_snapshot
```

### Testing Pydantic Models
Test serialization, deserialization, defaults, and validation.

```python
class TestDomainFirestoreSerialization:
    def test_model_dump_firestore_context_stringifies_coordinates(self, ...):
        domain = Domain(**sample_domain_data)
        dumped = domain.model_dump(context={"for_firestore": True})
        coords = dumped["features"][0]["geometry"]["coordinates"]

        assert isinstance(coords, str)  # Stringified for Firestore
```

### Round-Trip Tests
Test that data survives serialization/deserialization cycles.

```python
def test_serialize_deserialize_round_trip(self, sample_domain_data):
    original = Domain(**sample_domain_data)
    firestore_data = original.model_dump(context={"for_firestore": True})
    restored = Domain(**firestore_data)

    assert restored.id == original.id
    assert restored.name == original.name
```

## Integration Test Patterns

### Testing Endpoints with Examples
Verify that documented API examples actually work.

```python
@pytest.mark.parametrize("example_name,example_value", ALL_EXAMPLE_VALUES)
def test_example_creates_domain(self, client, example_name, example_value):
    """Each documented example should successfully create a domain."""
    response = client.post("/domains", json=example_value)

    assert response.status_code == 201, (
        f"Example '{example_name}' failed: {response.json()}"
    )
```

### Testing Error Responses
Verify both status codes and error message content.

```python
def test_outside_conus_returns_422(self, client):
    response = client.post("/domains", json=alaska_request)

    assert response.status_code == 422
    assert "within CONUS" in response.json()["detail"]
```

### Cleanup in Fixtures
Always clean up test data created in fixtures.

```python
@pytest.fixture(scope="session")
def domain_in_firestore(firestore_client):
    domain_data = make_domain_data()
    doc_ref = firestore_client.collection("domains-v2").document(domain_data["id"])
    doc_ref.set(domain_data)
    yield domain_data
    doc_ref.delete()  # Always cleanup
```

### Multiple Fixtures for List Tests
Create multiple resources to properly test list, pagination, and sorting.

```python
@pytest.fixture(scope="session")
def multiple_domains_for_list(firestore_client):
    """Create multiple domains with different names and timestamps for list tests."""
    domains = []
    for i, name in enumerate(["Alpha", "Beta", "Gamma"]):
        domain_data = make_domain_data(name=name)
        doc_ref = firestore_client.collection("domains-v2").document(domain_data["id"])
        doc_ref.set(domain_data)
        domains.append({"data": domain_data, "ref": doc_ref})
        time.sleep(0.1)  # Ensure different created_on timestamps

    yield [d["data"] for d in domains]

    for d in domains:
        d["ref"].delete()

class TestListDomains:
    def test_returns_all_domains(self, client, multiple_domains_for_list):
        response = client.get("/domains")
        assert response.json()["total_items"] >= len(multiple_domains_for_list)

    def test_sort_by_name_ascending(self, client, multiple_domains_for_list):
        response = client.get("/domains?sort_by=name&sort_order=ascending")
        names = [d["name"] for d in response.json()["domains"]]
        assert names == sorted(names)
```

## What to Test

### For Validation Functions
1. Each validation passing (happy path)
2. Each validation failing with correct HTTP status
3. Error message content is helpful
4. Edge cases (empty, null, boundary values)
5. Geographic to projected CRS transformation

### For Pydantic Models
1. Default values
2. Required field validation
3. Serialization (model_dump)
4. Deserialization from various formats
5. Round-trip integrity

### For Endpoints

**Create (POST):**
1. Happy path with documented examples
2. Validation error responses (422)
3. Response structure matches schema
4. Resource persists in database

**Get (GET /{id}):**
1. Returns existing resource
2. Returns 404 for non-existent resource
3. Returns 404 for wrong owner (not 403)
4. Response structure matches schema

**List (GET):**
1. Returns empty list when no resources
2. Returns all resources for owner
3. Excludes resources from other owners
4. Pagination works (page, size parameters)
5. Sorting works (sort_by, sort_order parameters)
6. Total count is accurate
7. Response structure matches schema

**Update (PATCH /{id}):**
1. Updates provided fields only
2. Preserves non-provided fields
3. Updates modified_on timestamp
4. Returns 404 for non-existent resource
5. Returns 404 for wrong owner
6. Returns full updated resource

**Delete (DELETE /{id}):**
1. Returns 204 No Content on success
2. Returns 404 for non-existent resource
3. Returns 404 for wrong owner
4. Resource no longer retrievable after delete
5. Resource no longer appears in list
6. Delete twice returns 404 second time
7. Response body is empty

### For Database Functions
1. Document found/not found
2. Owner validation (returns 404, not 403)
3. Status validation
4. Validation order (owner checked before status)
5. Data integrity through read/write cycles
6. List pagination (page, size, offset calculations)
7. List sorting (ascending, descending, by different fields)
8. List total count accuracy
9. Update partial merge behavior
10. Delete idempotency (deleting non-existent document doesn't error)
