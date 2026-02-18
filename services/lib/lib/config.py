"""
Central configuration module.

All infrastructure-specific values are loaded from environment variables.
Required values fail fast on import if missing; optional values have safe defaults.

A .env file at the repository root is loaded automatically via python-dotenv.
"""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Required — app fails fast if missing
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_REGION = os.environ["GCP_REGION"]

# Firestore collections
DOMAINS_COLLECTION = os.getenv("DOMAINS_COLLECTION", "domains-v2")
GRIDS_COLLECTION = os.getenv("GRIDS_COLLECTION", "grids-v2")
EXPORTS_COLLECTION = os.getenv("EXPORTS_COLLECTION", "exports-v2")
KEYS_COLLECTION = os.getenv("KEYS_COLLECTION", "keys-v2")
APPLICATIONS_COLLECTION = os.getenv("APPLICATIONS_COLLECTION", "applications-v2")

# GCS buckets — required
GRIDS_BUCKET = os.environ["GRIDS_BUCKET"]
EXPORTS_BUCKET = os.environ["EXPORTS_BUCKET"]
RASTERS_BUCKET = os.environ["RASTERS_BUCKET"]

# Cloud Tasks
GRIDDLE_QUEUE = os.getenv("GRIDDLE_QUEUE", "griddle-queue")
EXPORTER_QUEUE = os.getenv("EXPORTER_QUEUE", "exporter-queue")
GRIDDLE_SERVICE = os.getenv("GRIDDLE_SERVICE", "griddle-v2")
EXPORTER_SERVICE = os.getenv("EXPORTER_SERVICE", "exporter-v2")

# CORS origins — comma-separated list from env var
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(",")
    if origin.strip()
]

# API URL — for self-referencing responses
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")

# Deployment environment
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "local")

# Dev mode — disabled by default; must be explicitly enabled
FASTFUELS_DEV_MODE = os.getenv("FASTFUELS_DEV_MODE", "false").lower() in ("true", "1")
DEV_API_KEY = os.getenv("DEV_API_KEY", "")
DEV_OWNER_ID = os.getenv("DEV_OWNER_ID", "")
