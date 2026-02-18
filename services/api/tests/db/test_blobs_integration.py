"""
Integration tests for api.db.blobs module.

Tests GCS blob operations against real GCS using a dedicated test bucket.

Prerequisite: The test bucket (TEST_BUCKET env var) must exist.
"""

import os
import uuid

import gcsfs
import pytest
import pytest_asyncio
from api.db.blobs import (
    check_exists,
    delete_directory,
    delete_directory_safe,
    delete_file,
    download_file,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

TEST_BUCKET = os.environ.get("TEST_BUCKET", "test-bucket")


@pytest_asyncio.fixture(loop_scope="session")
async def gcs():
    """Session-scoped async gcsfs client for test setup/teardown."""
    return gcsfs.GCSFileSystem(asynchronous=True)


@pytest_asyncio.fixture(loop_scope="session")
async def test_prefix(gcs):
    """Create a unique test prefix and seed file, clean up after."""
    prefix = f"blobs/{uuid.uuid4().hex}"
    await gcs._pipe_file(f"{TEST_BUCKET}/{prefix}/seed.txt", b"test data")
    yield prefix
    try:
        await gcs._rm(f"{TEST_BUCKET}/{prefix}", recursive=True)
    except FileNotFoundError:
        pass


class TestDeleteDirectory:
    async def test_deletes_existing_directory(self, gcs, test_prefix):
        subdir = f"{test_prefix}/del_dir_{uuid.uuid4().hex}"
        await gcs._pipe_file(f"{TEST_BUCKET}/{subdir}/a.txt", b"a")
        await gcs._pipe_file(f"{TEST_BUCKET}/{subdir}/b.txt", b"b")

        await delete_directory(TEST_BUCKET, subdir)

        assert await check_exists(TEST_BUCKET, subdir) is False

    async def test_nonexistent_directory_is_noop(self):
        fake_path = f"blobs/{uuid.uuid4().hex}/nonexistent"
        await delete_directory(TEST_BUCKET, fake_path)


class TestDeleteFile:
    async def test_deletes_existing_file(self, gcs, test_prefix):
        file_path = f"{test_prefix}/del_file_{uuid.uuid4().hex}.txt"
        await gcs._pipe_file(f"{TEST_BUCKET}/{file_path}", b"delete me")

        await delete_file(TEST_BUCKET, file_path)

        assert await check_exists(TEST_BUCKET, file_path) is False


class TestCheckExists:
    async def test_existing_path_returns_true(self, test_prefix):
        result = await check_exists(TEST_BUCKET, f"{test_prefix}/seed.txt")
        assert result is True

    async def test_nonexistent_path_returns_false(self):
        result = await check_exists(TEST_BUCKET, f"blobs/{uuid.uuid4().hex}/nope")
        assert result is False


class TestDownloadFile:
    async def test_downloads_to_local(self, gcs, test_prefix, tmp_path):
        file_path = f"{test_prefix}/download_{uuid.uuid4().hex}.txt"
        content = b"hello from gcs"
        await gcs._pipe_file(f"{TEST_BUCKET}/{file_path}", content)

        local_file = str(tmp_path / "downloaded.txt")
        await download_file(f"{TEST_BUCKET}/{file_path}", local_file)

        with open(local_file, "rb") as f:
            assert f.read() == content

    async def test_downloads_with_gs_prefix(self, gcs, test_prefix, tmp_path):
        file_path = f"{test_prefix}/download_gs_{uuid.uuid4().hex}.txt"
        content = b"gs prefix test"
        await gcs._pipe_file(f"{TEST_BUCKET}/{file_path}", content)

        local_file = str(tmp_path / "downloaded_gs.txt")
        await download_file(f"gs://{TEST_BUCKET}/{file_path}", local_file)

        with open(local_file, "rb") as f:
            assert f.read() == content


class TestDeleteDirectorySafe:
    async def test_deletes_existing_directory(self, gcs, test_prefix):
        subdir = f"{test_prefix}/safe_del_{uuid.uuid4().hex}"
        await gcs._pipe_file(f"{TEST_BUCKET}/{subdir}/file.txt", b"data")

        await delete_directory_safe(TEST_BUCKET, subdir)

        assert await check_exists(TEST_BUCKET, subdir) is False

    async def test_nonexistent_is_silent(self):
        fake_path = f"blobs/{uuid.uuid4().hex}/nonexistent"
        await delete_directory_safe(TEST_BUCKET, fake_path)
