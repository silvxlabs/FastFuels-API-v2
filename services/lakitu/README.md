# Lakitu

Point cloud production service for FastFuels API v2. Lakitu builds point clouds
that have to be *fetched* — currently USGS 3DEP airborne lidar.

Uploaded point clouds are a different path: they arrive as a GCS object and are
ingested by the **uploader** service on a storage event. Lakitu is triggered by
Cloud Tasks, from the API.

See [LAKITU.md](LAKITU.md) for design notes.

## Running locally

Requires a `.env` at the repo root with `GCP_PROJECT`, `GCP_REGION`,
`POINT_CLOUDS_BUCKET`, `RASTERS_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS`.

Process an existing pending point cloud by id:

```bash
POINT_CLOUD_ID=<id> uv run python lakitu/main.py
```

Or run the function server and post to it the way Cloud Tasks would:

```bash
uv run functions-framework --target process_point_cloud_request \
  --source lakitu/main.py --port 8082
curl -X POST localhost:8082 -H 'Content-Type: application/json' -d '{"id": "<id>"}'
```

## Tests

```bash
uv run pytest tests/ --ignore=tests/integration -v   # offline
uv run pytest tests/integration -v                   # live 3DEP + GCP
```

Integration tests read the real USGS archive and take a couple of minutes.

## Maintenance

The 3DEP acquisition catalog is mirrored to GCS so a user-facing endpoint does
not depend on a GitHub raw URL. Refresh it when USGS publishes new surveys:

```bash
uv run python scripts/refresh_ept_catalog.py
```

## Deploying

Pushes to `main` deploy `lakitu-v2-prod` via `.github/workflows/lakitu.yml`.
Unlike the older services, the workflow sets memory, CPU, timeout, and
concurrency explicitly — see LAKITU.md for why.

The `lakitu-v2-queue` Cloud Tasks queue is a one-time manual resource:

```bash
gcloud tasks queues create lakitu-v2-queue --location=us-west1 \
  --max-attempts=2 --max-concurrent-dispatches=10
```

`--max-attempts=2` matters because `main.py` fails the resource on any retry
rather than re-running an hour-long fetch, and the concurrency cap matches the
service's `--max-instances`.
