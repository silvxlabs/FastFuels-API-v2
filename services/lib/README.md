# lib — Shared library for FastFuels API v2

A pip-installable package with **optional dependency extras** so each service declares only what it needs.

## Installation

Add lib to your service's `pyproject.toml` with the extras you need:

```toml
[project]
dependencies = [
    "lib[firestore,gcs,zarr]",
]

[tool.uv.sources]
lib = { path = "../lib" }
```

Then run `uv sync` in your service directory.

## Available Extras

| Extra | Deps | Used by |
|-------|------|---------|
| `firestore` | google-cloud-firestore | griddle, exporter |
| `gcs` | gcsfs, google-cloud-storage | griddle, exporter |
| `zarr` | gcsfs, xarray, zarr | griddle, exporter |
| `raster` | rioxarray, rasterio, geopandas, numpy, xarray | griddle |

Zero extras (config only): API

## Modules

| Module | Extra required | Description |
|--------|---------------|-------------|
| `lib.config` | none | Central configuration (env vars, constants) |
| `lib.firestore` | `firestore` | Firestore document operations (sync) |
| `lib.gcs` | `gcs` | GCS blob/directory operations, signed URLs |
| `lib.zarr` | `zarr` | Zarr save/load to GCS |
| `lib.raster` | `raster` | Raster connections and utilities |
| `lib.grid` | none | Grid utilities |

## Configuration (`lib.config`)

All infrastructure values come from environment variables with safe defaults.

**Required** (app fails fast if missing):
| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | GCP project ID |
| `GCP_REGION` | GCP region (e.g. `us-west1`) |

**Optional** (safe defaults):
| Variable | Default | Description |
|----------|---------|-------------|
| `GRIDS_BUCKET` | `""` | GCS bucket for grid data |
| `EXPORTS_BUCKET` | `""` | GCS bucket for export data |
| `RASTERS_BUCKET` | `""` | GCS bucket for raster data |
| `DOMAINS_COLLECTION` | `domains-v2` | Firestore collection for domains |
| `GRIDS_COLLECTION` | `grids-v2` | Firestore collection for grids |
| `EXPORTS_COLLECTION` | `exports-v2` | Firestore collection for exports |
| `KEYS_COLLECTION` | `keys-v2` | Firestore collection for API keys |
| `APPLICATIONS_COLLECTION` | `applications-v2` | Firestore collection for applications |
| `GRIDDLE_QUEUE` | `griddle-queue` | Cloud Tasks queue for griddle |
| `EXPORTER_QUEUE` | `exporter-queue` | Cloud Tasks queue for exporter |
| `GRIDDLE_SERVICE` | `griddle-v2` | Cloud Run service name for griddle |
| `EXPORTER_SERVICE` | `exporter-v2` | Cloud Run service name for exporter |
| `DEPLOYMENT_ENV` | `local` | Deployment environment |

See `.env.example` in the repo root for the full list.

## Sync vs Async

This library provides **synchronous** implementations for use in backend services (Cloud Run jobs, CLI tools). For async implementations used by FastAPI routes, see `services/api/api/db/`.

## Docker

In Dockerfiles, lib is copied as a sibling directory and installed via `uv sync`:

```dockerfile
COPY services/lib/ /app/lib/
COPY services/<service>/ /app/<service>/
WORKDIR /app/<service>
RUN uv sync --frozen
```

The `[tool.uv.sources]` path (`../lib`) resolves to `/app/lib/` inside the container.
