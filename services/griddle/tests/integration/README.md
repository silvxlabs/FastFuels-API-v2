# Griddle Integration Tests

## Purpose

Integration tests that run the full griddle pipeline (Firestore -> handler ->
zarr) and verify output correctness. Tests hit real GCS/Firestore and may
fetch remote data (LANDFIRE COGs), so they require valid credentials.

## Test Categories

### Source handlers (LANDFIRE, uniform)

Use `griddle_runner(domain_file, grid_file)` directly. These handlers fetch
data from external sources or generate it from parameters.

```python
def test_something(griddle_runner):
    ds = griddle_runner("blue_mtn.json", "landfire_fbfm40.json")
    assert "fbfm" in ds.data_vars
```

### Transform handlers (lookup, resample)

Require a completed source grid as input. Use the `source_grid` fixture
(copies static data from GCS per-test) combined with `source_overrides`:

```python
@pytest.mark.parametrize("source_grid", ["static-test-landfire-fbfm40"], indirect=True)
def test_lookup(griddle_runner, source_grid):
    ds = griddle_runner(
        "blue_mtn.json", "lookup_fbfm40.json",
        source_overrides={"source_grid_id": source_grid},
    )
```

Static test data lives in GCS at `gs://{GRIDS_BUCKET}/static-test-*` and is
created by `services/api/tests/e2e/`. Never deleted by integration test cleanup.

## Test Data

- **Domain GeoJSONs**: `tests/data/domains/` (e.g., `blue_mtn.json`)
- **Grid JSON templates**: `tests/data/grids/` (e.g., `landfire_fbfm40.json`,
  `lookup_fbfm40.json`)
- **Static zarr data**: `gs://{GRIDS_BUCKET}/static-test-*` (in GCS)

## How to add a regression test for a bug report

1. Save the domain GeoJSON to `tests/data/domains/bug_123.json`
2. Create or reuse a grid JSON in `tests/data/grids/`
3. Write a test:
   ```python
   def test_bug_123(griddle_runner):
       ds = griddle_runner("bug_123.json", "landfire_fbfm40.json")
       # assert the bug is fixed
   ```

## How to add tests for a new handler

1. Create a grid JSON template in `tests/data/grids/`
2. If the handler needs a source grid, add a static fixture via the API e2e
   module first (`services/api/tests/e2e/`)
3. Write a test using `griddle_runner` (with `source_overrides` if needed)

## Running

Locally:
```bash
cd services/griddle
DEPLOYMENT_ENV=local uv run pytest tests/integration/ -v
```

Against deployed environment:
```bash
DEPLOYMENT_ENV=prod uv run pytest tests/integration/ -n auto -v
```

## Static Test Data

Static fixtures live in GCS at `gs://{GRIDS_BUCKET}/static-test-*`. They are
created by the API e2e module (`services/api/tests/e2e/`) and should never be
deleted by integration test cleanup. Regenerate when:

- A handler's output format changes
- Test domain geometry changes
- Adding new static fixtures
