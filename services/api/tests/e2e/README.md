# API E2E Tests — Static Fixture Generation

## Purpose

This module generates static test data in GCS that backend services (griddle,
exporter) use as fixtures for their own integration tests. It exercises the full
API -> Cloud Tasks -> griddle pipeline to produce completed grids, then copies
the output zarr to well-known static paths and saves JSON templates.

## Prerequisites

- API server must be running and accessible (default: `http://127.0.0.1:8080`)
- Griddle must be running/accessible (Cloud Tasks dispatches to griddle)
- Environment variables (same as regular API tests):
  - `TEST_API_KEY` — valid API key for the test environment
  - `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service account key
  - `GCP_PROJECT`, `GCP_REGION`
  - `GRIDS_BUCKET`, `EXPORTS_BUCKET`, `RASTERS_BUCKET`

## How to run

```bash
cd services/api
uv run pytest tests/e2e/ -v --log-cli-level=INFO
```

## Domains

Each domain gets its own session-scoped fixture in `conftest.py` and its own
block of tests in `test_create_fixtures.py`. Static fixture names include the
domain abbreviation (e.g., `static-test-blue-mtn-landfire-fbfm40`).

To add a new domain:
1. Add a session-scoped fixture in `conftest.py` (like `blue_mountain_domain`)
2. Add test functions that use that domain fixture

## How to add a new static fixture

1. Add a test function in `test_create_fixtures.py` that calls
   `create_static_fixture()` with the appropriate endpoint, request body, and
   `static_name`
2. Run the e2e tests to generate the zarr in GCS and the JSON template
3. Commit the generated JSON template from `services/griddle/tests/data/grids/`

## Chained fixtures

Some fixtures depend on other static fixtures as source grids. For example, a
resampled FBFM40 grid at 2m needs the LANDFIRE FBFM40 static fixture to exist.

Any `static-test-*` values in the request body are **automatically detected**
and temporarily registered in Firestore so the API's source grid validation
passes. Use `@pytest.mark.dependency` to enforce ordering:

```python
@pytest.mark.dependency()
def test_create_blue_mtn_landfire_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/landfire/fbfm40",
        body={},
        static_name="static-test-blue-mtn-landfire-fbfm40",
    )

@pytest.mark.dependency(depends=["test_create_blue_mtn_landfire_fbfm40"])
def test_create_blue_mtn_fbfm40_2m(
    create_static_fixture, client, blue_mountain_domain
):
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/resample",
        body={
            "source_grid_id": "static-test-blue-mtn-landfire-fbfm40",
            "resolution": 2,
            "method": "nearest",
        },
        static_name="static-test-blue-mtn-fbfm40-2m",
    )
```

`@pytest.mark.dependency` ensures base fixtures run first. If a dependency
fails, dependent tests are automatically skipped.

## Where outputs go

- **Zarr data**: `gs://{GRIDS_BUCKET}/static-test-{domain}-{name}/`
- **JSON template**: `services/griddle/tests/data/grids/static-test-{domain}-{name}.json`

## When to regenerate

- When a handler's output format changes
- When test domain geometry changes
- When adding new static fixtures
