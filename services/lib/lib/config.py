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
INFRA_ENV = os.getenv("INFRA_ENV", "dev")

# Bucket names
GRIDS_BUCKET = os.getenv("GRIDS_BUCKET", "placeholder-bucket-name")
EXPORTS_BUCKET = os.getenv("EXPORTS_BUCKET", "placeholder-bucket-name")
RASTERS_BUCKET = os.getenv("RASTERS_BUCKET", "placeholder-bucket-name")
INVENTORIES_BUCKET = os.getenv("INVENTORIES_BUCKET", "placeholder-bucket-name")
FEATURES_BUCKET = os.getenv("FEATURES_BUCKET", "placeholder-bucket-name")
OSM_BUCKET = os.getenv("OSM_BUCKET", "placeholder-bucket-name")
TABLES_BUCKET = os.getenv("TABLES_BUCKET", "placeholder-bucket-name")
TEST_BUCKET = os.getenv("TEST_BUCKET", "placeholder-bucket-name")
UPLOADS_BUCKET = os.getenv("UPLOADS_BUCKET", "placeholder-bucket-name")
POINT_CLOUDS_BUCKET = os.getenv("POINT_CLOUDS_BUCKET", "placeholder-bucket-name")

# Collection names
DOMAINS_COLLECTION = os.getenv("DOMAINS_COLLECTION", "domains-v2")
GRIDS_COLLECTION = os.getenv("GRIDS_COLLECTION", "grids-v2")
EXPORTS_COLLECTION = os.getenv("EXPORTS_COLLECTION", "exports-v2")
INVENTORIES_COLLECTION = os.getenv("INVENTORIES_COLLECTION", "inventories-v2")
FEATURES_COLLECTION = os.getenv("FEATURES_COLLECTION", "features-v2")
KEYS_COLLECTION = os.getenv("KEYS_COLLECTION", "keys-v2")
APPLICATIONS_COLLECTION = os.getenv("APPLICATIONS_COLLECTION", "applications-v2")
POINT_CLOUDS_COLLECTION = os.getenv("POINT_CLOUDS_COLLECTION", "pointclouds-v2")

# Queue names
GRIDDLE_QUEUE = os.getenv("GRIDDLE_QUEUE", "griddle-v2-queue")
EXPORTER_QUEUE = os.getenv("EXPORTER_QUEUE", "exporter-v2-queue")
STANDGEN_QUEUE = os.getenv("STANDGEN_QUEUE", "standgen-v2-queue")
FEATURES_QUEUE = os.getenv("FEATURES_QUEUE", "etcher-v2-queue")
TREEVOX_QUEUE = os.getenv("TREEVOX_QUEUE", "treevox-v2-queue")


# Service names
GRIDDLE_SERVICE = os.getenv("GRIDDLE_SERVICE", f"griddle-v2-{INFRA_ENV}")
EXPORTER_SERVICE = os.getenv("EXPORTER_SERVICE", f"exporter-v2-{INFRA_ENV}")
STANDGEN_SERVICE = os.getenv("STANDGEN_SERVICE", f"standgen-v2-{INFRA_ENV}")
FEATURES_SERVICE = os.getenv("FEATURES_SERVICE", f"etcher-v2-{INFRA_ENV}")
TREEVOX_SERVICE = os.getenv("TREEVOX_SERVICE", f"treevox-v2-{INFRA_ENV}")
UPLOADER_SERVICE = os.getenv("UPLOADER_SERVICE", f"uploader-v2-{INFRA_ENV}")

# Support contact surfaced in user-facing error messages.
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support.fastfuels@silvxlabs.com")
