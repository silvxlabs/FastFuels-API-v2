# Contributing to FastFuels API v2

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- GCP credentials for integration tests (see `.env.example`)

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in values
3. Install dependencies for the service you're working on:

```bash
cd services/<service>
uv sync
```

## Pre-commit

This project uses [pre-commit](https://pre-commit.com/) to run code quality checks before each commit. Install it and set up the hooks:

```bash
pip install pre-commit
pre-commit install
```

The hooks are defined in `.pre-commit-config.yaml` and include:

- **[Ruff](https://docs.astral.sh/ruff/)** — linting and formatting (replaces flake8, black, and isort)
- **[Gitleaks](https://github.com/gitleaks/gitleaks)** — secret detection
- **General checks** — YAML/TOML validation, trailing whitespace, private key detection

Hooks run automatically on `git commit`. To run them manually on all files:

```bash
pre-commit run --all-files
```

To update hooks to their latest versions:

```bash
pre-commit autoupdate
```

## Shared Library (`services/lib`)

The `lib` package is a pip-installable local package with optional dependency extras. Services declare only the extras they need:

```toml
# In a service's pyproject.toml
dependencies = [
    "lib[firestore,gcs,zarr]",
]

[tool.uv.sources]
lib = { path = "../lib" }
```

Available extras: `firestore`, `gcs`, `zarr`, `raster`

See `services/lib/README.md` for details.

## Running Tests

Each service has its own test suite. From a service directory:

```bash
uv run pytest tests/ -v
```

API tests require additional environment variables — see `services/api/tests/README.md`.

## Docker Builds

Build context is the repository root:

```bash
docker build -f services/<service>/Dockerfile .
```

## Coding Conventions

- Use direct imports, not `__init__.py` re-exports
- Import config from `lib.config`, not from `lib` directly
- No decorative comment banners
- See `CLAUDE.md` for full guidelines
