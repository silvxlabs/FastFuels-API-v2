"""
Async Cloud Tasks operations for FastAPI routes.
"""

import json
import logging
from contextvars import ContextVar
from datetime import timedelta

from google.api_core.exceptions import AlreadyExists
from google.cloud import run_v2, tasks_v2
from google.cloud.tasks_v2 import HttpMethod, Task
from ring import lru
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from lib.config import GCP_PROJECT, GCP_REGION

logger = logging.getLogger(__name__)

# Cloud Tasks accepts a dispatch deadline between 15 seconds and 30 minutes for
# HTTP targets, and uses 10 minutes when none is given.
MIN_DISPATCH_DEADLINE = timedelta(seconds=15)
MAX_DISPATCH_DEADLINE = timedelta(minutes=30)
DEFAULT_DISPATCH_DEADLINE = timedelta(minutes=10)

# A request carrying this header creates its resource but queues no worker
# task, leaving it pending. Used by the API test suite (#680).
SKIP_DISPATCH_HEADER = "X-FastFuels-Skip-Dispatch"
skip_dispatch: ContextVar[bool] = ContextVar("skip_dispatch", default=False)


class SkipDispatchMiddleware:
    """Set ``skip_dispatch`` for requests carrying ``SKIP_DISPATCH_HEADER``.

    Pure ASGI so the var is set in the same context the endpoint runs in.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        value = Headers(scope=scope).get(SKIP_DISPATCH_HEADER, "")
        token = skip_dispatch.set(value.strip().lower() in ("1", "true", "yes"))
        try:
            await self.app(scope, receive, send)
        finally:
            skip_dispatch.reset(token)


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
    """Enqueue an HTTP POST task asynchronously. Returns None if task already
    exists, or if dispatch is skipped for this request (see SKIP_DISPATCH_HEADER).

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
    if skip_dispatch.get():
        logger.info(
            "Skipping dispatch (%s): queue=%s service=%s id=%s task=%s",
            SKIP_DISPATCH_HEADER,
            queue,
            service,
            task_id,
            task_name or task_id,
        )
        return None

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
