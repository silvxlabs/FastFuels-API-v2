# Exporter Integration Tests

## Purpose

Integration tests that run the full exporter pipeline (Firestore -> handler ->
export file) and verify output correctness. Tests hit real GCS/Firestore, so they
require valid credentials.

## Test Data

Three sources of test data:

- **Export templates**: `tests/data/exports/` (e.g., `geotiff.json`, `parquet.json`)
- **Grid templates**: `services/lib/tests/static_data/grids/` (shared format,
  e.g., `static-test-blue-mtn-landfire-fbfm40.json`)
- **Inventory templates**: `services/lib/tests/static_data/inventories/` (shared format,
  e.g., `static-test-blue-mtn-pim-inventory.json`)

Static grid data lives in GCS at `gs://{GRIDS_BUCKET}/static-test-*` and static
inventory data at `gs://{INVENTORIES_BUCKET}/static-test-*`. Both are created by
`services/api/tests/e2e/`. Never deleted by integration test cleanup.

## How to add tests for a new export format

1. Create an export JSON template in `tests/data/exports/` (e.g., `quicfire.json`)
2. Write tests using `exporter_runner` with `source_grid` (for grid exports) or
   `inventory_exporter_runner` with `source_inventory` (for inventory exports):

```python
# Grid-based export
@pytest.mark.parametrize("source_grid", ["static-test-blue-mtn-landfire-fbfm40"], indirect=True)
def test_quicfire_export(exporter_runner, source_grid):
    export = exporter_runner(source_grid, "quicfire.json")
    # verify output files

# Inventory-based export
@pytest.mark.parametrize("source_inventory", ["static-test-blue-mtn-pim-inventory"], indirect=True)
def test_parquet_export(inventory_exporter_runner, source_inventory):
    export = inventory_exporter_runner(source_inventory, "parquet.json")
    # verify output files
```

3. If the handler needs a new type of source data, add a static fixture via the
   API e2e module first (`services/api/tests/e2e/`)

## Running

Locally:
```bash
cd services/exporter
DEPLOYMENT_ENV=local uv run pytest tests/integration/ -v
```

Against deployed environment:
```bash
DEPLOYMENT_ENV=prod uv run pytest tests/integration/ -v
```

## Static Test Data

Static fixtures live in GCS and should never be deleted by integration test cleanup:

- **Grids**: `gs://{GRIDS_BUCKET}/static-test-*` (zarr directories)
- **Inventories**: `gs://{INVENTORIES_BUCKET}/static-test-*` (partitioned parquet)

Both are created by the API e2e module (`services/api/tests/e2e/`). Regenerate when:

- A handler's output format changes
- Test domain geometry changes
- Adding new static fixtures
