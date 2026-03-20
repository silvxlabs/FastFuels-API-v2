"""Shared fixtures for all exporter tests."""

import asyncio
import sys

import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(autouse=True, scope="session")
def _cleanup_gcsfs_sessions():
    """Cleanly shut down gcsfs sessions after all tests complete.

    gcsfs registers a weakref finalizer that calls close_session() when
    instances are garbage collected. During Python's atexit phase, this
    finalizer tries to close the aiohttp session via fsspec's IO thread
    loop, but the session's internal Futures are bound to a different loop,
    producing a RuntimeError.

    Fix: stop the fsspec IO thread loop before clearing the instance cache.
    This forces the finalizer into the synchronous force_close path
    (connector._close()), which doesn't involve cross-loop Future issues.
    """
    yield

    import fsspec.asyn as fasyn
    import gcsfs

    loop = fasyn.loop[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        thread = fasyn.iothread[0]
        if thread is not None:
            thread.join(timeout=5)
        fasyn.loop[0] = None
        fasyn.iothread[0] = None

    gcsfs.GCSFileSystem.clear_instance_cache()
