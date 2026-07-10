"""Standgen-local configuration.

GDAM service settings are read here rather than from ``lib.config``: editing
``services/lib`` is on every service's CI/CD ``paths:`` trigger and redeploys the
entire pipeline, and only standgen talks to GDAM. The defaults below are the
production values; override any of them via environment variables (wire them into
``.github/workflows/standgen.yml`` if a non-default is needed in deployment).

Handlers should reference these as ``config.GDAM_*`` (not import-bind the values) so
tests can monkeypatch them.
"""

import os

# Public GDAM allometry inference API. Override via GDAM_API_URL.
GDAM_API_URL = os.getenv(
    "GDAM_API_URL", "https://gdam-api-v2-782971006568.us-west1.run.app"
)

# Target trees per dask partition, i.e. per /predict/batch request. The handler
# repartitions the source inventory to this size so each GDAM call (and each
# partition held in memory) is bounded. 20k sits at ~97% of GDAM's per-instance
# throughput ceiling (a benchmark showed the fixed per-request overhead amortizes
# out by ~20k, plateauing near 54k trees/s at cpu=2/concurrency=8) while p50
# latency (~3s) stays far under GDAM_REQUEST_TIMEOUT_S.
GDAM_BATCH_SIZE = int(os.getenv("GDAM_BATCH_SIZE", "20000"))

# Per-request timeout (seconds) on outbound GDAM calls. Must stay under Cloud Run's
# request timeout so the handler returns before the container is SIGTERM'd.
GDAM_REQUEST_TIMEOUT_S = float(os.getenv("GDAM_REQUEST_TIMEOUT_S", "120"))

# Attempts for a single GDAM /predict/batch call. Attempts after the first retry
# only transient transport errors (a dropped/reset connection) with backoff; a
# non-2xx status or a read/write timeout is terminal.
GDAM_MAX_ATTEMPTS = int(os.getenv("GDAM_MAX_ATTEMPTS", "3"))
