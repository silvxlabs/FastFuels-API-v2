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

# Trees sent per /predict/batch request.
GDAM_BATCH_SIZE = int(os.getenv("GDAM_BATCH_SIZE", "5000"))

# Per-request timeout (seconds) on outbound GDAM calls. Must stay under Cloud Run's
# request timeout so the handler returns before the container is SIGTERM'd.
GDAM_REQUEST_TIMEOUT_S = float(os.getenv("GDAM_REQUEST_TIMEOUT_S", "120"))
