"""
Tests for api.tasks module.

Unit tests use mocked Cloud Run and Cloud Tasks clients.
Integration tests run against live GCP infrastructure and require
gcloud auth application-default login.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.config import GRIDDLE_QUEUE, GRIDDLE_SERVICE

pytestmark = pytest.mark.anyio


def _clear_cache(tasks_module):
    """Clear ring LRU cache for all known service keys."""
    for service in ["griddle", "test-service", "cache-test"]:
        try:
            tasks_module._get_service_url.delete(service)
        except Exception:
            pass


@pytest.fixture
def tasks_module():
    """Import tasks module with mocked Cloud Run client."""
    with patch("google.cloud.run_v2.ServicesAsyncClient"):
        from api import tasks

        _clear_cache(tasks)
        yield tasks
        _clear_cache(tasks)


@pytest.fixture
def mock_service_url(tasks_module):
    """Set up Cloud Run mock to return a URL for griddle."""
    mock_svc = MagicMock(uri="https://griddle.example.com")
    mock_client = MagicMock()
    mock_client.get_service = AsyncMock(return_value=mock_svc)

    with patch("api.tasks.run_v2.ServicesAsyncClient", return_value=mock_client):
        yield


class TestGetServiceUrl:
    """Tests for _get_service_url function."""

    async def test_reads_url_from_cloud_run(self, tasks_module):
        """Reads URI from Cloud Run service."""
        mock_svc = MagicMock(uri="https://griddle.example.com")
        mock_client = MagicMock()
        mock_client.get_service = AsyncMock(return_value=mock_svc)

        with patch("api.tasks.run_v2.ServicesAsyncClient", return_value=mock_client):
            url = await tasks_module._get_service_url("griddle")

        assert url == "https://griddle.example.com"
        mock_client.get_service.assert_called_once()

    async def test_caches_result(self, tasks_module):
        """Second call returns cached result without hitting Cloud Run."""
        mock_svc = MagicMock(uri="https://cache-test.example.com")
        mock_client = MagicMock()
        mock_client.get_service = AsyncMock(return_value=mock_svc)

        with patch("api.tasks.run_v2.ServicesAsyncClient", return_value=mock_client):
            await tasks_module._get_service_url("cache-test")
            url = await tasks_module._get_service_url("cache-test")

        assert url == "https://cache-test.example.com"
        mock_client.get_service.assert_called_once()

    async def test_missing_service_raises(self, tasks_module):
        """Raises NotFound when service does not exist."""
        from google.api_core.exceptions import NotFound

        mock_client = MagicMock()
        mock_client.get_service = AsyncMock(side_effect=NotFound("not found"))

        with patch("api.tasks.run_v2.ServicesAsyncClient", return_value=mock_client):
            with pytest.raises(NotFound):
                await tasks_module._get_service_url("nonexistent")


class TestCreateHttpTaskAsync:
    """Tests for create_http_task_async function."""

    @patch("api.tasks.tasks_v2.CloudTasksAsyncClient")
    async def test_creates_task(
        self, mock_client_class, tasks_module, mock_service_url
    ):
        """create_http_task_async sends task to queue."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = "projects/p/locations/l/queues/q/tasks/t"
        mock_result = MagicMock()
        mock_client.create_task = AsyncMock(return_value=mock_result)

        result = await tasks_module.create_http_task_async(
            GRIDDLE_QUEUE, GRIDDLE_SERVICE, "test-123"
        )

        assert result == mock_result
        mock_client.create_task.assert_called_once()

    @patch("api.tasks.tasks_v2.CloudTasksAsyncClient")
    async def test_returns_none_on_already_exists(
        self, mock_client_class, tasks_module, mock_service_url
    ):
        """create_http_task_async returns None when task already exists."""
        from google.api_core.exceptions import AlreadyExists

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = "projects/p/locations/l/queues/q/tasks/t"
        mock_client.create_task = AsyncMock(side_effect=AlreadyExists("task exists"))

        result = await tasks_module.create_http_task_async(
            GRIDDLE_QUEUE, GRIDDLE_SERVICE, "test-123"
        )

        assert result is None

    @patch("api.tasks.tasks_v2.CloudTasksAsyncClient")
    async def test_task_id_used_in_task_path(
        self, mock_client_class, tasks_module, mock_service_url
    ):
        """Task name is derived from the task_id parameter."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = (
            "projects/p/locations/l/queues/q/tasks/my-grid-id"
        )
        mock_client.create_task = AsyncMock(return_value=MagicMock())

        await tasks_module.create_http_task_async(
            GRIDDLE_QUEUE, GRIDDLE_SERVICE, "my-grid-id"
        )

        mock_client.task_path.assert_called_once()
        call_args = mock_client.task_path.call_args
        assert call_args[0][3] == "my-grid-id"

    @patch("api.tasks.tasks_v2.CloudTasksAsyncClient")
    async def test_payload_contains_id(
        self, mock_client_class, tasks_module, mock_service_url
    ):
        """Task HTTP body contains {"id": task_id}."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = "projects/p/locations/l/queues/q/tasks/abc"
        mock_client.create_task = AsyncMock(return_value=MagicMock())

        await tasks_module.create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, "abc")

        call_kwargs = mock_client.create_task.call_args[1]
        body = call_kwargs["task"].http_request.body
        assert b'"id": "abc"' in body


class TestCreateHttpTaskAsyncIntegration:
    """Integration tests for create_http_task_async against live test-queue.

    Requires gcloud auth application-default login.
    """

    async def test_creates_task_successfully(self):
        """Create a task on the live test-queue."""
        from api.tasks import create_http_task_async

        task_id = uuid.uuid4().hex

        result = await create_http_task_async("test-queue", "test-service", task_id)

        assert result is not None
        assert task_id in result.name

    async def test_deduplication_returns_none(self):
        """Second call with same task_id returns None."""
        from api.tasks import create_http_task_async

        task_id = uuid.uuid4().hex

        result1 = await create_http_task_async("test-queue", "test-service", task_id)
        assert result1 is not None

        result2 = await create_http_task_async("test-queue", "test-service", task_id)
        assert result2 is None

    async def test_different_task_ids_create_separate_tasks(self):
        """Different task_ids create separate tasks."""
        from api.tasks import create_http_task_async

        id1 = uuid.uuid4().hex
        id2 = uuid.uuid4().hex

        result1 = await create_http_task_async("test-queue", "test-service", id1)
        result2 = await create_http_task_async("test-queue", "test-service", id2)

        assert result1 is not None
        assert result2 is not None
        assert result1.name != result2.name
