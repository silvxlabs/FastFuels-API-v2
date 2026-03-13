# API

FastAPI REST API for FastFuels v2. Serves as the public interface for managing domains, grids, and exports.

## Architecture

```
api/
├── app.py              # FastAPI app with CORS and router setup
├── main.py             # Entrypoint (imports app)
├── auth.py             # API key authentication
├── tasks.py            # Cloud Tasks dispatch
├── dependencies.py     # Shared FastAPI dependencies
├── db/                 # Database layer (Firestore, GCS)
└── resources/          # Resource routers (domains, grids, exports, keys, applications)
```

Deploys as Cloud Run service `api-v2`.

## Local Development

```bash
uv sync
uv run python -m api.main
```

The server starts at `http://127.0.0.1:8080`. Append `/docs` for the interactive OpenAPI documentation.

### Environment Variables

Requires `GCP_PROJECT`, `GCP_REGION`, and bucket variables. See `.env.example` in the repo root. Also requires GCP authentication:

```bash
gcloud auth application-default login
```
**Windows:** Pass the path to your `.env` file explicitly when **running or testing**
```bash
uv run --env-file ..\..\.env python -m api.main
```

## Testing

**Note:** Before running tests, ensure `INFRA_ENV=prod` is set in your `.env` file.
```bash
uv run python -m pytest tests/ -v
```

Integration and router tests require the local API server to be running. See [tests/readme.md](tests/README.md) for the full testing guide.

## Docker

Build context is the repo root:

```bash
docker build -f services/api/Dockerfile -t api-v2:latest .
```
