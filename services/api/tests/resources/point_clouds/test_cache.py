"""
Tests for api/v2/resources/point_clouds/cache.py GCS access.

They pin the storage access pattern that keeps the point-cloud data endpoints
off the grpcio poller race (#265, #703): reads must go through a gcsfs client
bound to the worker's running event loop, never the default sync client whose
own ``fsspecIO`` loop thread floods the logs with ``BlockingIOError`` on every
later Firestore RPC.
"""

import asyncio
import inspect
import subprocess
import sys
from uuid import uuid4

import gcsfs
import pytest
from api.resources.point_clouds import cache as point_clouds_cache
from api.resources.point_clouds import utils as point_clouds_utils

from lib.pointcloud import reader

pytestmark = pytest.mark.anyio

STATIC_POINT_CLOUD_ID = "static-test-blackfoot-3dep"


async def test_open_passes_gcsfs_bound_to_running_loop(monkeypatch):
    """Regression for #703: no ``filesystem`` falls back to the sync client."""
    captured = {}

    def fake_open(prefix, filesystem=None):
        captured["filesystem"] = filesystem
        return "storage"

    def no_default_client():
        raise AssertionError("point-cloud reads must not use get_gcsfs_client")

    monkeypatch.setattr(point_clouds_cache, "open_point_cloud", fake_open)
    monkeypatch.setattr(reader, "get_gcsfs_client", no_default_client)

    storage = await point_clouds_cache.get_point_cloud_storage(
        f"test-{uuid4().hex}", "checksum"
    )

    assert storage == "storage"
    assert isinstance(captured["filesystem"], gcsfs.GCSFileSystem)
    assert captured["filesystem"].loop is asyncio.get_running_loop()


async def test_client_is_reused_on_the_same_loop():
    assert (
        point_clouds_cache._gcsfs_on_running_loop()
        is point_clouds_cache._gcsfs_on_running_loop()
    )


def test_client_not_built_at_import():
    """Nothing may be built in the ``--preload`` gunicorn master (#702)."""
    code = "from api.resources.point_clouds import cache; assert cache._gcsfs is None"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("module", [point_clouds_cache, point_clouds_utils])
def test_module_does_not_reference_default_sync_client(module):
    source = inspect.getsource(module)
    assert "get_gcsfs_client" not in source


async def test_reading_a_cloud_does_not_start_fsspec_loop_thread():
    """Open a real cloud and read a tile; no ``fsspecIO`` thread may appear.

    Runs a fresh interpreter so a thread started by another test cannot mask
    or fake the result. Reads the static test point cloud from GCS.
    """
    code = f"""
import asyncio, threading
from api.resources.point_clouds import cache

async def main():
    storage = await cache.get_point_cloud_storage({STATIC_POINT_CLOUD_ID!r}, "t")
    tile = storage.tiles[0]
    await cache.get_point_cloud_tile(
        storage, tile["tile_x"], tile["tile_y"], ["X", "Y", "Z"], None, 0
    )
    print(sorted(t.name for t in threading.enumerate()))

asyncio.run(main())
"""
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "fsspecIO" not in result.stdout, result.stdout
