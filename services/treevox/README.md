# Treevox

Cloud Run service that voxelizes tree inventories into 3D canopy fuel grids.

Triggered by Cloud Tasks with `{"id": grid_id}`; reads the grid document and tree inventory from GCS/Firestore, voxelizes each tree into a chunked 3D zarr store, and marks the grid completed.

## Env vars

See `.env.example` at the repo root. Key vars: `GCP_PROJECT`, `GCP_REGION`, `GRIDS_BUCKET`, `INVENTORIES_BUCKET`, `GRIDS_COLLECTION`.

## Local development

```bash
cd services/treevox
uv sync
GRID_ID=<grid_id> uv run treevox/main.py
```

## Deployment

Cloud Run service with **minimum 4 GB memory** (multiprocessing workers × fastfuels-core reference data). Smaller containers will OOM.

## Tests

```bash
uv run --active pytest -m "not integration" -v
uv run --active pytest tests/integration -v -m integration  # requires GCP
```
