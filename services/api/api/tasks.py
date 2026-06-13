"""
Async Cloud Tasks operations for FastAPI routes.
"""

import json

from google.api_core.exceptions import AlreadyExists
from google.cloud import run_v2, tasks_v2
from google.cloud.tasks_v2 import HttpMethod, Task
from ring import lru

from lib.config import GCP_PROJECT, GCP_REGION


@lru(force_asyncio=True)
async def _get_service_url(service: str) -> str:
    """Get Cloud Run service URL.

    Uses the Cloud Run async API to look up the service URI and caches the
    result using ring LRU.

    Returns:
        The service URL.

    Raises:
        google.api_core.exceptions.NotFound: If the service does not exist.
    """
    name = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{service}"
    async with run_v2.ServicesAsyncClient() as client:
        svc = await client.get_service(name=name)
    return svc.uri


async def create_http_task_async(
    queue: str,
    service: str,
    task_id: str,
    task_name: str | None = None,
) -> Task | None:
    """Enqueue an HTTP POST task asynchronously. Returns None if task already exists.

    Args:
        queue: Cloud Tasks queue name.
        service: Cloud Run service the task targets.
        task_id: Resource ID sent in the task body as {"id": task_id}.
        task_name: Optional Cloud Tasks task name. Defaults to task_id, which
            deduplicates create requests — but completed task names are
            tombstoned, so re-processing the same resource later (e.g. an
            in-place modification) must pass a unique name here or the task
            is silently dropped as AlreadyExists.
    """
    url = await _get_service_url(service)

    async with tasks_v2.CloudTasksAsyncClient() as client:
        parent = client.queue_path(GCP_PROJECT, GCP_REGION, queue)

        task = Task(
            name=client.task_path(GCP_PROJECT, GCP_REGION, queue, task_name or task_id),
            http_request={
                "http_method": HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": task_id}).encode(),
            },
        )

        try:
            return await client.create_task(parent=parent, task=task)
        except AlreadyExists:
            return None
