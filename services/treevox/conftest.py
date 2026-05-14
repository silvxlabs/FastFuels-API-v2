"""Pytest bootstrap — runs before any test module or child conftest is loaded.

Sets gRPC env vars here so that by the time test modules import google-cloud
/ gcsfs / grpcio, the gRPC shared library sees `GRPC_VERBOSITY=ERROR` and
skips the atfork-handler spam that `multiprocessing.Pool(spawn)` triggers.
The production Cloud Run path gets the same env vars from the Dockerfile.

Keeping the env-var setting out of individual modules (treevox/main.py,
tests/integration/conftest.py) lets those files keep strict ruff E402
enforcement — regressions in import ordering will fail lint.
"""

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
