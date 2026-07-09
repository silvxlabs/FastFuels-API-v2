"""walle test setup.

Importing walle.cleanup constructs lib's Firestore client at import time, which
needs GCP credentials. Load the repo-root ``.env`` (the same file the other
services use) so ``uv run pytest`` works locally; CI provides the credentials in
the environment.
"""

from pathlib import Path

from dotenv import load_dotenv

# services/walle/tests/conftest.py -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")
