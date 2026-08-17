"""
Unit tests for griddle.tasks module.

All against mocked Cloud Run / Cloud Tasks clients -- no live GCP calls.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from google.cloud import tasks_v2
from griddle import tasks

from lib.config import GCP_PROJECT, GCP_REGION, GRIDDLE_QUEUE, GRIDDLE_SERVICE


class TestServiceUrl:
    """Tests for _service_url."""

    def test_reads_uri_from_cloud_run(self):
        """Resolves a service's URI via the Cloud Run Admin API."""
        tasks._service_url.cache_clear()
        mock_svc = MagicMock(uri="https://griddle.example.com")
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get_service.return_value = mock_svc

        with patch("griddle.tasks.run_v2.ServicesClient", return_value=mock_client):
            url = tasks._service_url("griddle-v2-prod")

        assert url == "https://griddle.example.com"
        mock_client.get_service.assert_called_once()

    def test_caches_result_per_service(self):
        """A second call for the same service doesn't re-query Cloud Run."""
        tasks._service_url.cache_clear()
        mock_svc = MagicMock(uri="https://cache-test.example.com")
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get_service.return_value = mock_svc

        with patch("griddle.tasks.run_v2.ServicesClient", return_value=mock_client):
            tasks._service_url("cache-test")
            url = tasks._service_url("cache-test")

        assert url == "https://cache-test.example.com"
        mock_client.get_service.assert_called_once()


class TestEnqueueDelayedTask:
    """Tests for enqueue_delayed_task."""

    def _mock_tasks_client(self):
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"
        mock_client.task_path.return_value = "projects/p/locations/l/queues/q/tasks/t"
        return mock_client

    @patch("griddle.tasks._service_url", return_value="https://griddle.example.com")
    def test_creates_task_on_the_given_queue(self, mock_url):
        mock_client = self._mock_tasks_client()
        with patch("griddle.tasks.tasks_v2.CloudTasksClient", return_value=mock_client):
            tasks.enqueue_delayed_task(
                "grid-1", 10, queue="test-queue", service="test-service"
            )

        mock_client.queue_path.assert_called_once()
        assert mock_client.queue_path.call_args[0][2] == "test-queue"
        mock_client.create_task.assert_called_once()

    @patch("griddle.tasks._service_url", return_value="https://griddle.example.com")
    def test_defaults_to_griddles_own_queue_and_service(self, mock_url):
        mock_client = self._mock_tasks_client()
        with patch("griddle.tasks.tasks_v2.CloudTasksClient", return_value=mock_client):
            tasks.enqueue_delayed_task("grid-1", 10)

        assert mock_client.queue_path.call_args[0][2] == GRIDDLE_QUEUE
        mock_url.assert_called_once_with(GRIDDLE_SERVICE)

    @patch("griddle.tasks._service_url", return_value="https://griddle.example.com")
    def test_task_name_is_unique_per_call(self, mock_url):
        """Task names must be unique -- the original task was named `grid_id`
        and Cloud Tasks tombstones completed names, so reusing it would
        silently no-op."""
        mock_client = self._mock_tasks_client()
        with patch("griddle.tasks.tasks_v2.CloudTasksClient", return_value=mock_client):
            tasks.enqueue_delayed_task("grid-1", 10)
            tasks.enqueue_delayed_task("grid-1", 10)

        first_name = mock_client.task_path.call_args_list[0][0][3]
        second_name = mock_client.task_path.call_args_list[1][0][3]
        assert first_name != second_name
        assert first_name.startswith("grid-1-")

    @patch("griddle.tasks._service_url", return_value="https://griddle.example.com")
    def test_body_contains_grid_id(self, mock_url):
        mock_client = self._mock_tasks_client()
        with patch("griddle.tasks.tasks_v2.CloudTasksClient", return_value=mock_client):
            tasks.enqueue_delayed_task("grid-1", 10)

        task = mock_client.create_task.call_args[1]["task"]
        assert b'"id": "grid-1"' in task.http_request.body

    @patch("griddle.tasks._service_url", return_value="https://griddle.example.com")
    def test_schedule_time_is_in_the_future(self, mock_url):
        mock_client = self._mock_tasks_client()
        with patch("griddle.tasks.tasks_v2.CloudTasksClient", return_value=mock_client):
            tasks.enqueue_delayed_task("grid-1", 60)

        task = mock_client.create_task.call_args[1]["task"]
        assert task.schedule_time > datetime.now(UTC)


class TestEnqueueDelayedTaskIntegration:
    """Live tests for enqueue_delayed_task -- schedules a reminder
    with Google's task queue and checks it landed.

    Requires gcloud auth application-default login. Targets the same
    "test-queue"/"test-service" api's own live-queue tests use, never
    griddle's real production queue/service.
    """

    def test_creates_task_on_live_queue(self):
        """Create a real task on the live test-queue and confirm it landed."""
        grid_id = uuid.uuid4().hex

        tasks.enqueue_delayed_task(
            grid_id, delay_seconds=60, queue="test-queue", service="test-service"
        )

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(GCP_PROJECT, GCP_REGION, "test-queue")
        task_names = [t.name for t in client.list_tasks(parent=parent)]

        assert any(grid_id in name for name in task_names)
