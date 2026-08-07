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

# GCS path to the DUET binary. It is a restricted third-party artifact fetched
# at runtime (never committed); treevox downloads it from here on first use. The
# real value is set via env in the treevox service and local .env, not here.
DUET_BINARY_GCS = os.getenv("DUET_BINARY_GCS", "gs://placeholder-bucket/duet.exe")

# Collection names
DOMAINS_COLLECTION = os.getenv("DOMAINS_COLLECTION", "domains-v2")
GRIDS_COLLECTION = os.getenv("GRIDS_COLLECTION", "grids-v2")
EXPORTS_COLLECTION = os.getenv("EXPORTS_COLLECTION", "exports-v2")
INVENTORIES_COLLECTION = os.getenv("INVENTORIES_COLLECTION", "inventories-v2")
FEATURES_COLLECTION = os.getenv("FEATURES_COLLECTION", "features-v2")
KEYS_COLLECTION = os.getenv("KEYS_COLLECTION", "keys-v2")
APPLICATIONS_COLLECTION = os.getenv("APPLICATIONS_COLLECTION", "applications-v2")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users-v2")
POINT_CLOUDS_COLLECTION = os.getenv("POINT_CLOUDS_COLLECTION", "pointclouds-v2")
CREATE_BUDGETS_COLLECTION = os.getenv("CREATE_BUDGETS_COLLECTION", "create-budgets-v2")

# Queue names
GRIDDLE_QUEUE = os.getenv("GRIDDLE_QUEUE", "griddle-v2-queue")
EXPORTER_QUEUE = os.getenv("EXPORTER_QUEUE", "exporter-v2-queue")
STANDGEN_QUEUE = os.getenv("STANDGEN_QUEUE", "standgen-v2-queue")
FEATURES_QUEUE = os.getenv("FEATURES_QUEUE", "etcher-v2-queue")
TREEVOX_QUEUE = os.getenv("TREEVOX_QUEUE", "treevox-v2-queue")
LAKITU_QUEUE = os.getenv("LAKITU_QUEUE", "lakitu-v2-queue")


# Service names
GRIDDLE_SERVICE = os.getenv("GRIDDLE_SERVICE", f"griddle-v2-{INFRA_ENV}")
EXPORTER_SERVICE = os.getenv("EXPORTER_SERVICE", f"exporter-v2-{INFRA_ENV}")
STANDGEN_SERVICE = os.getenv("STANDGEN_SERVICE", f"standgen-v2-{INFRA_ENV}")
FEATURES_SERVICE = os.getenv("FEATURES_SERVICE", f"etcher-v2-{INFRA_ENV}")
TREEVOX_SERVICE = os.getenv("TREEVOX_SERVICE", f"treevox-v2-{INFRA_ENV}")
UPLOADER_SERVICE = os.getenv("UPLOADER_SERVICE", f"uploader-v2-{INFRA_ENV}")
LAKITU_SERVICE = os.getenv("LAKITU_SERVICE", f"lakitu-v2-{INFRA_ENV}")

# Point-cloud write concurrency. These track the Cloud Run vCPU allocation
# rather than taste: worker counts are sharply peaked at the core count, and
# oversubscribing measured 2.4x the CPU for byte-identical output.
# os.cpu_count() cannot be used -- it reports host cores, not the quota.
LAKITU_WRITE_WORKERS = int(os.getenv("LAKITU_WRITE_WORKERS", 8))
LAKITU_WRITE_QUEUE_DEPTH = int(os.getenv("LAKITU_WRITE_QUEUE_DEPTH", 4))
LAKITU_CHAIN_WORKERS = int(os.getenv("LAKITU_CHAIN_WORKERS", 6))
LAKITU_DOWNLOAD_WORKERS = int(os.getenv("LAKITU_DOWNLOAD_WORKERS", 32))

# How much routed-but-unwritten point data the parent holds. Under the tile
# schedule this is a backstop rather than the flush trigger -- every time it
# fires it splits a tile that was going to be written whole -- so it is sized to
# the peak the schedule actually needs, not to a memory floor. 512 MiB was
# measured at 64 km2: it costs 15% wall and 1.2 GB of RSS over 192 MiB, and
# takes the output from 1,069 files to 449. See writer.BUFFER_BUDGET.
LAKITU_BUFFER_BUDGET_MB = int(os.getenv("LAKITU_BUFFER_BUDGET_MB", 512))

# Flush each tile when its last node has been routed, rather than evicting the
# largest tile under buffer pressure. Off keeps the eviction-only behaviour.
LAKITU_TILE_SCHEDULE = os.getenv("LAKITU_TILE_SCHEDULE", "1") == "1"

# Dask worker threads for griddle's blocked point-cloud work. Set explicitly for
# the same reason the lakitu worker counts are: dask sizes its pool from
# `os.cpu_count()`, which under Cloud Run reports the host's cores rather than
# the vCPU quota. Each thread holds a block's points, so peak memory is threads
# x block area -- a 64 km2 CHM at 1 m OOMed an 8 GiB / 2 vCPU container on the
# default and completed in 666 s at 2.
GRIDDLE_DASK_WORKERS = int(os.getenv("GRIDDLE_DASK_WORKERS", 2))

# Worker processes for griddle's point-cloud block reads. Processes rather than
# threads because that read is throughput-bound inside one interpreter, not
# latency-bound: at 8 vCPU with the dataset shared, raising dask threads 8 -> 16
# -> 32 moved the 64 km2 ground read 99.4 s -> 102.6 s -> 103.0 s while
# per-thread wait grew linearly and CPU stayed near 1.6 cores. gcsfs is
# instance-cached, so every block shares one asyncio event loop, and pyarrow
# reaches it through a handler that takes the GIL.
#
# Each worker is a fresh interpreter holding one block's points, so this trades
# resident memory for read throughput -- size it to the vCPU allocation, as
# LAKITU_WRITE_WORKERS is, and not above it.
GRIDDLE_READ_WORKERS = int(os.getenv("GRIDDLE_READ_WORKERS", 2))

# Support contact surfaced in user-facing error messages.
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support.fastfuels@silvxlabs.com")
