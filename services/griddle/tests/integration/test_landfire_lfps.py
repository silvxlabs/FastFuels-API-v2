"""
Integration test for seasonal LANDFIRE FBFM40 via LFPS.

Submits a real LFPS job, polls until it finishes, downloads, and runs it
through the normal fetch/align pipeline -- verifying our code's
assumptions about LFPS's real request/response shape still hold.
"""

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

import geopandas as gpd
import pytest
from griddle.handlers.landfire_lfps import (
    _LFPS_POLL_DELAY_SECONDS,
    _fetch_downloaded_fbfm40,
    _lfps_aoi_bbox,
    _lfps_fbfm40_layer_name,
)

from lib.config import GRIDS_COLLECTION
from lib.firestore.documents import delete_document, get_document, set_document
from lib.landfire import (
    LfpsJobFailedError,
    download,
    list_products,
    poll_status,
    submit_job,
)
from lib.testing import SHARED_TEST_DOMAINS_DIR

_MAX_WAIT_SECONDS = 300  # generous cap; our jobs typically finish in 12-30s


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    """Blue Mountain (used elsewhere in these tests) has no LFPS Seasonal
    Fuels coverage, so this domain is reused here instead."""
    with open(SHARED_TEST_DOMAINS_DIR / "threedep_ept_seam.json") as f:
        domain = json.load(f)
    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


def _assert_valid_data(ds, band, min_valid_frac=0.95):
    """Assert that a band has enough valid (non-nodata) pixels.

    Returns the array of valid values for further assertions.
    """
    values = ds[band].values.ravel()
    nodata = ds[band].rio.nodata
    valid_mask = values != nodata
    valid_frac = valid_mask.sum() / len(values)
    assert valid_frac >= min_valid_frac, (
        f"{band}: valid fraction {valid_frac:.3f} < {min_valid_frac}"
    )
    return values[valid_mask]


def test_seasonal_fbfm40(roi):
    """Submit a real seasonal FBFM40 job, wait for it, and fetch the result."""
    # Discover a currently-live season/version rather than hardcoding one,
    # since LFPS's Seasonal Fuels catalog rotates over time. Skip (not fail)
    # if nothing seasonal happens to be live right now.
    product = next(
        (
            p
            for p in list_products()
            if p.acronym.upper() == "FBFM40" and p.season is not None
        ),
        None,
    )
    if product is None:
        pytest.skip("No seasonal FBFM40 product currently live in LFPS's catalog.")
    layer = _lfps_fbfm40_layer_name(product.version.removeprefix("LF"), product.season)
    aoi = _lfps_aoi_bbox(roi, {"target": "domain"}, None, extent_buffer_cells=0)

    job = submit_job([layer], aoi)
    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    try:
        while job.status != "Succeeded":
            if time.monotonic() > deadline:
                pytest.fail(
                    f"LFPS job {job.job_id} did not finish within "
                    f"{_MAX_WAIT_SECONDS}s (status={job.status})"
                )
            time.sleep(_LFPS_POLL_DELAY_SECONDS)
            job = poll_status(job.job_id)
    except LfpsJobFailedError as e:
        pytest.fail(f"LFPS job failed: {e}")

    zip_bytes = download(job)
    ds = _fetch_downloaded_fbfm40(
        zip_bytes,
        roi,
        "test-grid-id",
        remove_non_burnable=None,
        extent_buffer_cells=0,
        alignment={"target": "domain"},
        target_grid_doc=None,
    )

    assert "fbfm" in ds.data_vars
    assert ds["fbfm"].dims == ("y", "x")
    assert ds["fbfm"].rio.nodata is not None
    assert ds.rio.height > 0
    assert ds.rio.width > 0

    fbfm_valid = _assert_valid_data(ds, "fbfm")
    assert fbfm_valid.max() <= 204  # matches test_landfire.py's FBFM40 range check


def test_datetime_survives_firestore_roundtrip():
    """Write a timestamp to Firestore, read it back, and check the elapsed
    time can still be computed correctly.

    process_lfps_fbfm40 relies on this: it writes down when an LFPS job
    started, then on a later call reads that back to decide whether the
    job has been running too long.
    """
    doc_id = f"test-{uuid4().hex}"
    set_document(GRIDS_COLLECTION, doc_id, {"submitted_at": datetime.now(UTC)})
    try:
        _, snapshot = get_document(GRIDS_COLLECTION, doc_id)
        submitted_at = snapshot.to_dict()["submitted_at"]
        elapsed = (datetime.now(UTC) - submitted_at).total_seconds()
        assert 0 <= elapsed < 5
    finally:
        delete_document(GRIDS_COLLECTION, doc_id)
