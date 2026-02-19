"""
Central configuration module.

All infrastructure-specific values are loaded from environment variables.
Required values fail fast on import if missing; optional values have safe defaults.

A .env file at the repository root is loaded automatically via python-dotenv.
"""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Required — credentials and project identity
GCP_PROJECT = os.environ["GCP_PROJECT"]

# Non-sensitive config with safe defaults
GCP_REGION = os.getenv("GCP_REGION", "us-west1")
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "local")

# Bucket names
GRIDS_BUCKET = os.getenv("GRIDS_BUCKET", "placeholder-bucket-name")
EXPORTS_BUCKET = os.getenv("EXPORTS_BUCKET", "placeholder-bucket-name")
RASTERS_BUCKET = os.getenv("RASTERS_BUCKET", "placeholder-bucket-name")
TEST_BUCKET = os.getenv("TEST_BUCKET", "placeholder-bucket-name")

# Collection names
DOMAINS_COLLECTION = os.getenv("DOMAINS_COLLECTION", "domains-v2")
GRIDS_COLLECTION = os.getenv("GRIDS_COLLECTION", "grids-v2")
EXPORTS_COLLECTION = os.getenv("EXPORTS_COLLECTION", "exports-v2")
KEYS_COLLECTION = os.getenv("KEYS_COLLECTION", "keys-v2")
APPLICATIONS_COLLECTION = os.getenv("APPLICATIONS_COLLECTION", "applications-v2")

# Queue names
GRIDDLE_QUEUE = os.getenv("GRIDDLE_QUEUE", "griddle-v2-queue")
EXPORTER_QUEUE = os.getenv("EXPORTER_QUEUE", "exporter-v2-queue")

# Service names
GRIDDLE_SERVICE = os.getenv("GRIDDLE_SERVICE", "griddle-v2")
EXPORTER_SERVICE = os.getenv("EXPORTER_SERVICE", "exporter-v2")

# Dev mode — disabled by default; must be explicitly enabled
FASTFUELS_DEV_MODE = os.getenv("FASTFUELS_DEV_MODE", "false").lower() in ("true", "1")
DEV_API_KEY = os.getenv("DEV_API_KEY", "")
DEV_OWNER_ID = os.getenv("DEV_OWNER_ID", "")
