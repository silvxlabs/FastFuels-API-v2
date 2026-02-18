# Griddle

Backend service for processing grid resources. Receives tasks from Cloud Tasks, fetches data from external sources (LANDFIRE COGs, etc.), and writes Zarr stores to GCS.

## Architecture

```
griddle/
├── main.py             # functions-framework entry point
├── dispatch.py         # Route requests to handlers by grid type
├── storage.py          # Zarr read/write, GCS cleanup
└── handlers/           # Grid type handlers (landfire, resample, lookup, uniform)
```

Deploys as Cloud Run service `griddle-v2` using `functions-framework`.

## Local Development

```bash
uv sync
uv run functions-framework --target=process_grid_request --signature-type=http
```

Or process a specific grid directly:

```bash
GRID_ID=abc123 uv run python -m griddle.main
```

### Environment Variables

Requires `GCP_PROJECT`, `GCP_REGION`, `GRIDS_BUCKET`, and `RASTERS_BUCKET`. See `.env.example` in the repo root.

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Docker

Build context is the repo root:

```bash
docker build -f services/griddle/Dockerfile -t griddle-v2:latest .
```

## Cloud Logging

```
resource.type = "cloud_run_revision"
resource.labels.service_name = "griddle-v2"
jsonPayload.grid_id = "your-grid-id-here"
```

See [GRIDDLE.md](../../docs/GRIDDLE.md) for detailed design documentation.
