# API E2E Tests

## Purpose

This module runs each pipeline end to end: API → Cloud Tasks → worker → stored
data. Every test creates a grid, inventory, feature or point cloud through the
deployed API and checks that it completes.

The same tests also produce the static test data in GCS (`static-test-*`) that
backend services (griddle, treevox, exporter, standgen) read in their own
integration tests. They publish it only when asked to (see
[Publishing static fixtures](#publishing-static-fixtures)).

## Prerequisites

- The API and the workers it dispatches to must be running and reachable
  (default API URL: `http://127.0.0.1:8080`).
- Environment variables (same as the other API tests):
  - `TEST_API_URL`, `TEST_API_KEY`
  - `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT`, `GCP_REGION`
  - the bucket variables (`GRIDS_BUCKET`, `INVENTORIES_BUCKET`, `FEATURES_BUCKET`,
    `POINT_CLOUDS_BUCKET`, …)

## Running

```bash
cd services/api
uv run pytest tests/e2e/ -v --log-cli-level=INFO
```

This is the mode CI runs on every push. It writes nothing shared: each domain
fixture is session-scoped, every resource is created in it, and the domain is
force-deleted at session end, which removes the resources too. Overlapping runs
don't interfere with each other or with other services' tests.

## Chained tests

Some tests use the output of another as their source: PIM treemap → PIM
inventory → tree voxels, FBFM40 → lookup, 3DEP point cloud → CHM → inventory.

A test names its sources by static name, both in the request body and in
`dependencies`, keyed by resource type. The factory replaces each one with the
resource the upstream test created earlier in the same session, so the chain
runs entirely on this session's data. Use `@pytest.mark.dependency` so the
upstream test runs first; if it fails, the chained test is skipped.

```python
@pytest.mark.dependency()
def test_create_blue_mtn_landfire_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/fbfm40/landfire",
        body={"alignment": {"target": "domain", "resolution": 2, "method": "nearest"}},
        static_name="static-test-blue-mtn-landfire-fbfm40",
    )


@pytest.mark.dependency(depends=["test_create_blue_mtn_landfire_fbfm40"])
def test_create_blue_mtn_lookup_fbfm40(
    create_static_fixture, client, blue_mountain_domain
):
    create_static_fixture(
        client=client,
        domain_id=blue_mountain_domain["id"],
        endpoint="/grids/lookup/fbfm40",
        body={
            "source_grid_id": "static-test-blue-mtn-landfire-fbfm40",
            "bands": ["fuel_load.fine_dead", "fuel_depth", "savr.fine_dead"],
        },
        static_name="static-test-blue-mtn-lookup-fbfm40",
        dependencies={"grids": ["static-test-blue-mtn-landfire-fbfm40"]},
    )
```

A chained test needs its upstream test in the same pytest session; selecting it
alone (for example with `-k`) skips it.

## Publishing static fixtures

To regenerate the static fixtures, run the suite with `E2E_PUBLISH_FIXTURES=1`:

```bash
cd services/api
E2E_PUBLISH_FIXTURES=1 uv run pytest tests/e2e/ -v --log-cli-level=INFO
```

After each resource completes, the suite also:

- copies its data to the static path (see [Where outputs go](#where-outputs-go)),
  overwriting each object in place so the path never disappears for readers (#311);
- writes its JSON template under `services/lib/tests/shared_data/`. Source IDs
  that point at this session's resources are written as their static names, so
  templates reference the published fixtures.

Commit the regenerated templates in the same PR as the change that needed them,
so the GCS data and the templates change together. Publishing overwrites data
other services' integration tests read, so don't publish while their pipelines
are running.

Regenerate when:

- a handler's output format changes;
- test domain geometry changes;
- adding a new static fixture.

## Adding a fixture

1. Add a test in `test_create_fixtures.py` that calls the factory for its
   resource type (`create_static_fixture`, `create_static_inventory_fixture`,
   `create_static_feature_fixture` or `create_static_point_cloud_fixture`) with
   the endpoint, request body and `static_name`.
2. Run the suite with `E2E_PUBLISH_FIXTURES=1` to publish it.
3. Commit the new JSON template.

To add a domain, add a session-scoped fixture in `conftest.py` like
`blue_mountain_domain`, and tests that use it.

## Where outputs go

| Type | Data | Template |
|---|---|---|
| Grid | `gs://{GRIDS_BUCKET}/static-test-*/` (zarr) | `shared_data/grids/` |
| Inventory | `gs://{INVENTORIES_BUCKET}/static-test-*/` (parquet) | `shared_data/inventories/` |
| Point cloud | `gs://{POINT_CLOUDS_BUCKET}/static-test-*/` | `shared_data/pointclouds/` |
| Feature | `gs://{FEATURES_BUCKET}/static-test-*.parquet` | `shared_data/features/` |
