"""
Async Cloud Tasks operations for FastAPI routes.
"""

import json
from datetime import timedelta

from google.api_core.exceptions import AlreadyExists
from google.cloud import run_v2, tasks_v2
from google.cloud.tasks_v2 import HttpMethod, Task
from ring import lru

from lib.config import GCP_PROJECT, GCP_REGION

# Cloud Tasks accepts a dispatch deadline between 15 seconds and 30 minutes for
# HTTP targets, and uses 10 minutes when none is given.
MIN_DISPATCH_DEADLINE = timedelta(seconds=15)
MAX_DISPATCH_DEADLINE = timedelta(minutes=30)
DEFAULT_DISPATCH_DEADLINE = timedelta(minutes=10)


@lru(force_asyncio=True)
async def _get_service_target(service: str) -> tuple[str, timedelta]:
    """Get a Cloud Run service's URL and the deadline tasks targeting it get.

    The deadline is read from the service rather than configured alongside it.
    Cloud Tasks abandons a request once its own deadline passes and hands the
    task back to the queue for retry, so a deadline shorter than the service's
    request timeout turns a slow-but-successful job into a retried failure.
    Deriving it means raising a service's timeout is one change, not two that
    silently have to agree.

    Both values are cached together for the process lifetime, so a timeout
    changed on a deployed service is picked up on the next revision of this one.

    Returns:
        Tuple of (service URL, dispatch deadline clamped to Cloud Tasks' range).

    Raises:
        google.api_core.exceptions.NotFound: If the service does not exist.
    """
    name = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{service}"
    async with run_v2.ServicesAsyncClient() as client:
        svc = await client.get_service(name=name)

    timeout = svc.template.timeout or DEFAULT_DISPATCH_DEADLINE
    deadline = max(MIN_DISPATCH_DEADLINE, min(MAX_DISPATCH_DEADLINE, timeout))
    return svc.uri, deadline


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
    url, dispatch_deadline = await _get_service_target(service)

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
            dispatch_deadline=dispatch_deadline,
        )

        try:
            return await client.create_task(parent=parent, task=task)
        except AlreadyExists:
            return None
