"""
Griddle's self-re-enqueue for multi-step (LFPS-style) processing.

Cloud Tasks invokes Griddle with exactly one grid per HTTP request (see
main.py). A handler that isn't done after one invocation persists its own
state to Firestore, enqueues a delayed follow-up task for the same grid,
and raises ``lib.errors.ProcessingDeferred`` so main.py stops without
marking the grid complete or failed. This module is the enqueue half of
that loop -- the synchronous counterpart to ``api.tasks``, which only ever
enqueues *other* services from the (async) API.
"""

import json
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from uuid import uuid4

from google.cloud import run_v2, tasks_v2
from google.cloud.tasks_v2 import HttpMethod, Task

from lib.config import GCP_PROJECT, GCP_REGION, GRIDDLE_QUEUE, GRIDDLE_SERVICE


@lru_cache
def _service_url(service: str) -> str:
    """A Cloud Run service's URL, resolved once per (process, service) pair.

    Mirrors ``api.tasks._get_service_target`` but synchronous -- griddle is
    a sync ``functions_framework`` app, not an asyncio one -- and skips the
    per-service dispatch-deadline lookup, since griddle only ever targets
    itself (or a test double of itself) rather than dispatching to others.
    """
    name = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/services/{service}"
    with run_v2.ServicesClient() as client:
        return client.get_service(name=name).uri


def enqueue_delayed_task(
    grid_id: str,
    delay_seconds: float,
    queue: str = GRIDDLE_QUEUE,
    service: str = GRIDDLE_SERVICE,
) -> None:
    """Enqueue a follow-up Griddle task for ``grid_id`` after ``delay_seconds``.

    ``queue``/``service`` default to griddle's own, overridable so tests can
    target a test queue/service instead of live production infrastructure.

    The task name must be unique per attempt: the grid's original task was
    named ``grid_id`` and Cloud Tasks tombstones completed task names, so
    reusing it here would silently no-op as already-existing (the same trap
    ``api.tasks.create_http_task_async``'s docstring already flags).
    """
    with tasks_v2.CloudTasksClient() as client:
        parent = client.queue_path(GCP_PROJECT, GCP_REGION, queue)
        task_name = client.task_path(
            GCP_PROJECT, GCP_REGION, queue, f"{grid_id}-{uuid4().hex}"
        )

        task = Task(
            name=task_name,
            http_request={
                "http_method": HttpMethod.POST,
                "url": _service_url(service),
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"id": grid_id}).encode(),
            },
            schedule_time=datetime.now(UTC) + timedelta(seconds=delay_seconds),
        )
        client.create_task(parent=parent, task=task)
