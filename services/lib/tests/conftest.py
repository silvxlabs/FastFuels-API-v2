"""Pytest bootstrap — runs before any test module is imported.

`lib.config` reads ``GCP_PROJECT`` from the environment at import time and has no
default for it, so any test touching a module that imports config — the point
cloud writer, for one — fails on collection unless it is set first. Setting it
here rather than inside the test modules keeps those files under strict ruff E402
enforcement, the same reason treevox has a root conftest.

``setdefault``, so a real value from the repo's `.env` always wins: nothing here
reaches GCP, but a test that did should talk to the configured project.
"""

import os

os.environ.setdefault("GCP_PROJECT", "test-project")
