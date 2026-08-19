"""
Integration test for seasonal LANDFIRE FBFM40 via LFPS.

Submits a real LFPS job, waits for it, downloads, and runs it through the
normal fetch/align pipeline -- verifying our code's assumptions about
LFPS's real request/response shape still hold.
"""

import json

import geopandas as gpd
import pytest
from griddle.handlers import landfire

from lib.landfire import list_products
from lib.testing import SHARED_TEST_DOMAINS_DIR


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    """Blue Mountain (used elsewhere in these tests) has no LFPS Seasonal
    Fuels coverage, so this domain (near Kingman, AZ) is used here instead."""
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

    ds = landfire.fetch_fbfm40(
        roi,
        version=product.version.removeprefix("LF"),
        season=product.season,
        progress=lambda *a, **k: None,
    )

    assert "fbfm" in ds.data_vars
    assert ds["fbfm"].dims == ("y", "x")
    assert ds["fbfm"].rio.nodata is not None
    assert ds.rio.height > 0
    assert ds.rio.width > 0

    fbfm_valid = _assert_valid_data(ds, "fbfm")
    assert fbfm_valid.max() <= 204  # matches test_landfire.py's FBFM40 range check
