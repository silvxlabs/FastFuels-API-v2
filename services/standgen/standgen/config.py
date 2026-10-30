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

# Public GDAM allometry inference API (behind the FastFuels load balancer); the
# handler appends /predict/batch. Override via GDAM_API_URL.
GDAM_API_URL = os.getenv("GDAM_API_URL", "https://api.fastfuels.silvxlabs.com/gdam")

# Target trees per dask partition, i.e. per /predict/batch request. The handler
# repartitions the source inventory to this size so each GDAM call (and each
# partition held in memory) is bounded, and stays under GDAM's 50,000-tree limit.
# Measured against the public URL with the earlier flow-matching model
# v2_west.1, a 20k-tree batch takes ~10s warm (~24s cold), far under
# GDAM_REQUEST_TIMEOUT_S; 2k trees take ~2s, so the per-tree cost falls with
# batch size. Not yet re-measured with the pinned model (``GDAM_MODEL``).
GDAM_BATCH_SIZE = int(os.getenv("GDAM_BATCH_SIZE", "20000"))

# Per-request timeout (seconds) on outbound GDAM calls. Must stay under Cloud Run's
# request timeout so the handler returns before the container is SIGTERM'd.
GDAM_REQUEST_TIMEOUT_S = float(os.getenv("GDAM_REQUEST_TIMEOUT_S", "120"))

# Attempts for a single GDAM /predict/batch call. Attempts after the first retry
# only transient errors (a dropped/reset connection, 429, 502/503/504) with
# backoff; any other non-2xx status or a read/write timeout is terminal.
GDAM_MAX_ATTEMPTS = int(os.getenv("GDAM_MAX_ATTEMPTS", "3"))
