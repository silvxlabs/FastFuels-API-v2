"""
Unit tests for the CORS expose_headers allowlist in api.app.

A response header a browser client cannot read is invisible: cross-origin JS
only sees CORS-safelisted headers plus whatever `expose_headers` lists. These
tests bind the allowlist to the code that actually emits the headers, so adding
a header to a response without exposing it fails here rather than silently
degrading the webapp (#402). No server or Firestore required.
"""

from datetime import UTC, datetime, timedelta

import pytest
from api.app import EXPOSE_HEADERS
from api.quota import _raise_quota_exceeded, _ratelimit_headers
from api.resources.grids.router import _chunk_metadata_headers
from api.resources.grids.schema import GridDataChunkMetadata
from fastapi import HTTPException


def test_no_duplicate_or_empty_entries():
    assert len(EXPOSE_HEADERS) == len(set(EXPOSE_HEADERS))
    assert all(h.strip() for h in EXPOSE_HEADERS)


def test_grid_chunk_metadata_headers_are_exposed():
    """The 3D case emits every chunk-metadata header, including the two that
    are omitted for 2D grids."""
    meta = GridDataChunkMetadata(
        index=0,
        shape=(2, 3, 4),
        offset=(0, 0, 0),
        transform=(2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0),
        z_origin=0.0,
        z_resolution=1.0,
    )
    emitted = _chunk_metadata_headers(meta)
    assert {"X-Data-Z-Origin", "X-Data-Z-Resolution"} <= set(emitted)
    assert set(emitted) <= set(EXPOSE_HEADERS)


def test_ratelimit_headers_are_exposed():
    emitted = _ratelimit_headers(
        "max_weekly_grid_dispatches",
        remaining=0,
        limit=10,
        reset_at=datetime.now(UTC) + timedelta(days=1),
    )
    assert set(emitted) <= set(EXPOSE_HEADERS)


def test_active_job_429_retry_after_is_exposed():
    """The active-job 429 carries Retry-After so browser clients can offer a
    timed retry; unreadable cross-origin until it is exposed (#402)."""
    with pytest.raises(HTTPException) as exc:
        _raise_quota_exceeded(
            message="Too many active jobs.",
            quota="max_active_grid_jobs",
            current=5,
            limit=5,
            retry_after=True,
        )
    assert "Retry-After" in exc.value.headers
    assert set(exc.value.headers) <= set(EXPOSE_HEADERS)
