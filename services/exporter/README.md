# Exporter

Backend service for processing export tasks. Receives tasks from Cloud Tasks, loads grid data from Zarr stores, converts to output formats (GeoTIFF, etc.), and writes results to GCS.

## Architecture

```
exporter/
├── main.py             # functions-framework entry point
├── dispatch.py         # Route requests to handlers by export type
├── storage.py          # Zarr read, GCS cleanup, signed URLs
├── errors.py           # ProcessingError definitions
└── handlers/           # Export format handlers (geotiff)
```

Deploys as Cloud Run service `exporter-v2` using `functions-framework`.

## Local Development

```bash
uv sync
uv run functions-framework --target=process_export_request --signature-type=http
```

Or process a specific export directly:

```bash
EXPORT_ID=abc123 uv run python -m exporter.main
```

### Environment Variables

Requires `GCP_PROJECT`, `GCP_REGION`, `GRIDS_BUCKET`, and `EXPORTS_BUCKET`. See `.env.example` in the repo root.

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Docker

Build context is the repo root:

```bash
docker build -f services/exporter/Dockerfile -t exporter-v2:latest .
```

## Cloud Logging

```
resource.type = "cloud_run_revision"
resource.labels.service_name = "exporter-v2"
jsonPayload.export_id = "your-export-id-here"
```
