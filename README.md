# FastFuels API v2

A standalone API for generating high-resolution 3D fuel inputs for physics-based wildfire simulation models (QUIC-Fire, FIRETEC, FDS).

## Repository Structure

```
services/
├── api/          # FastAPI REST API (Cloud Run)
├── griddle/      # Grid processing service (Cloud Run)
├── exporter/     # Export processing service (Cloud Run)
├── standgen/     # Tree inventory service (Cloud Run)
├── lib/          # Shared library code
docs/             # Design and planning documents
```

Each service is an independent Python project with its own `pyproject.toml`, `uv.lock`, and `Dockerfile`.

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. Each service manages its own dependencies independently.

### Prerequisites

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Working with a service

From any service directory (e.g. `services/api/`):

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run a command in the service's virtual environment
uv run python -m api.main

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --group dev <package>

# Update the lockfile after editing pyproject.toml
uv lock
```

### How it works

- `pyproject.toml` declares direct dependencies with minimum versions.
- `uv.lock` pins the full dependency tree for reproducible installs. This file is committed to the repo.
- `uv sync --frozen` installs from the lockfile exactly (used in Dockerfiles).
- Each service has its own `.venv/` directory (git-ignored).

### Shared library

`services/lib/` contains shared code (Firestore client, GCS operations, config). It is not installed as a package — instead, each service copies it into its container at build time and adds it to the Python path:

```dockerfile
COPY services/lib/lib /api/lib
```

## Configuration

Infrastructure values are loaded from environment variables. See `.env.example` for the full list.

Required:
- `GCP_PROJECT` — Google Cloud project ID
- `GCP_REGION` — Google Cloud region
- `GRIDS_BUCKET`, `EXPORTS_BUCKET`, `RASTERS_BUCKET` — GCS bucket names

## Deployment

Each service deploys as a Cloud Run service via GitHub Actions. Pushes to `main` deploy to production, pushes to `dev` deploy to the dev environment. See `.github/workflows/` for details.
