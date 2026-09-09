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

# Arrow sizes its scanner pool from the host CPU count, which Cloud Run exposes
# rather than the container's vCPU quota (5 reported in the 2-vCPU API
# container). Keep one tile scan inside the allocated cores; concurrent HTTP
# requests already provide the outer parallelism.
POINT_CLOUD_READ_THREADS = int(os.getenv("POINT_CLOUD_READ_THREADS", 2))

# Point-cloud decode (chain) and encode (write) concurrency. The two pools run
# concurrently in a pipeline, so their processes together exceed the vCPU count
# on purpose: 6 chain + 8 write on an 8-vCPU container is the measured-best
# balance, not an oversight. Sizing either down to make the total fit the cores
# is slower -- it starves the pipeline, leaving the parent waiting on decode
# (chain 4 / write 4 measured slower than the default). Each pool still has its
# own peak: raising the write pool alone past the core count measured 2.4x the
# CPU for byte-identical output. os.cpu_count() cannot be used to set these --
# under Cloud Run it reports host cores, not the quota.
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
# MEMORY, not vCPU, is what bounds this. Each worker is a fresh interpreter
# holding a whole block's points, and the height pass reads every surface class,
# so a worker needs 2-4 GiB. Measured at 64 km2 / 1 m, a worker dies and the job
# fails whenever the container cannot feed them all:
#
#   2 workers /  8 GiB  ok, 298 s      4 workers /  8 GiB  FAILS
#   4 workers / 16 GiB  ok, 266 s      8 workers /  8 GiB  FAILS
#   8 workers / 32 GiB  ok, 251 s
#
# So raise this only together with the memory limit. Raising it to match a vCPU
# bump alone is the failing column above. It buys little anyway: the ground read
# is already saturated at 2 workers (107 s, 112 s, 117 s at 2, 4 and 8) and only
# the height pass scales, so 4x the workers and 4x the memory is 1.19x.
GRIDDLE_READ_WORKERS = int(os.getenv("GRIDDLE_READ_WORKERS", 2))

# Support contact surfaced in user-facing error messages.
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support.fastfuels@silvxlabs.com")
