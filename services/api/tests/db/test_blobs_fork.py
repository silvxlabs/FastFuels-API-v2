"""
Fork-safety integration test for api.db.blobs (#702).

gunicorn ``--preload`` imports api.db.blobs in the master and forks the
workers. The first GCS call in a forked worker must succeed. Runs against real
GCS using the test bucket (TEST_BUCKET env var).

The import-then-fork runs in a fresh interpreter: forking the multi-threaded
pytest process itself can crash the child.
"""

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

from lib.config import TEST_BUCKET

API_ROOT = Path(__file__).resolve().parents[2]

FORK_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import os
    import sys

    from api.db import blobs

    bucket, path = sys.argv[1], sys.argv[2]
    pid = os.fork()
    if pid == 0:
        try:
            asyncio.run(blobs.check_exists(bucket, path))
            print("ok", flush=True)
        except BaseException as e:
            print(f"{type(e).__name__}: {e}", flush=True)
        os._exit(0)
    os.waitpid(pid, 0)
    """
)


def test_first_call_after_fork_succeeds():
    path = f"blobs/{uuid.uuid4().hex}/nonexistent"

    result = subprocess.run(
        [sys.executable, "-c", FORK_SCRIPT, TEST_BUCKET, path],
        cwd=API_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.stdout.strip() == "ok", result.stdout + result.stderr
