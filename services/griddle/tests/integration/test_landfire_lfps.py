"""
Integration test for seasonal LANDFIRE FBFM40 via LFPS.

Submits a real LFPS job, waits for it, downloads, and runs it through the
normal fetch/align pipeline -- verifying our code's assumptions about
LFPS's real request/response shape still hold.
"""

import json
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers import landfire, landfire_lfps

from lib.landfire import LANDFIRE_VERSIONS, list_products
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


def test_configured_seasonal_version_still_live():
    """Fails once LANDFIRE moves seasonal fuels past our configured
    version -- signals LANDFIRE_VERSIONS["fbfm40"]["lfps_available"]
    needs updating, rather than silently going stale."""
    version = LANDFIRE_VERSIONS["fbfm40"]["lfps_available"][0]
    live = any(
        p.acronym.upper() == "FBFM40"
        and p.season is not None
        and p.version == f"LF{version}"
        for p in list_products()
    )
    assert live, (
        f"No live seasonal FBFM40 product for version {version} -- LANDFIRE may "
        'have moved on; update LANDFIRE_VERSIONS["fbfm40"]["lfps_available"].'
    )


@pytest.mark.parametrize(
    ("product", "fetch_fn", "band"),
    [
        ("annual_disturbance", landfire.fetch_annual_disturbance, "annual_disturbance"),
        ("fbfm13", landfire.fetch_fbfm13, "fbfm13"),
        ("fbfm40", landfire.fetch_fbfm40, "fbfm"),
        ("fccs", landfire.fetch_fccs, "fccs"),
    ],
)
def test_annual_lfps_available_version(roi, product, fetch_fn, band):
    """Submit a real annual (non-seasonal) LFPS job for each product's
    configured lfps_available version -- confirms LANDFIRE Product Service
    is actually serving what our config says it should."""
    lfps_versions = LANDFIRE_VERSIONS[product].get("lfps_available", [])
    if not lfps_versions:
        pytest.skip(f"No lfps_available version configured for {product}.")

    ds = fetch_fn(roi, version=lfps_versions[0], progress=lambda *a, **k: None)

    assert band in ds.data_vars
    assert ds[band].dims == ("y", "x")
    assert ds[band].rio.nodata is not None
    assert ds.rio.height > 0
    assert ds.rio.width > 0


def test_multi_layer_job_returns_bands_in_submission_order(roi):
    """Submit one real job for several layers -- the combined-mode disturbance
    inputs -- and confirm LFPS hands back one multi-band file whose bands match
    the submitted order. fetch_lfps_multi raises LFPS_BAND_ORDER_MISMATCH if the
    band names LFPS tags don't, so getting a path back is the check; reading
    every band confirms they come out on one grid."""
    layers = {
        product: LANDFIRE_VERSIONS[product]["default"]
        for product in ("fdist", "fvt", "bps")
    }
    alignment = {"target": "domain"}

    with landfire_lfps.fetch_lfps_multi(
        roi, layers, alignment, None, 0, lambda *a, **k: None
    ) as path:
        with rasterio.open(path) as src:
            assert src.count == len(layers)

        shapes = {
            landfire._fetch_landfire_raster(
                roi, path, 0, alignment, None, is_categorical=True, band=band
            ).shape
            for band in range(1, len(layers) + 1)
        }

    assert len(shapes) == 1


@pytest.fixture
def disturbance_roi() -> gpd.GeoDataFrame:
    """The NAIP two-tile domain: the only shared test domain with disturbance in
    LANDFIRE's 2025 data (insect and mechanical treatments)."""
    with open(SHARED_TEST_DOMAINS_DIR / "naip_chm_2_tiles.json") as f:
        domain = json.load(f)
    crs = domain["crs"]["properties"]["name"]
    return gpd.GeoDataFrame.from_features(domain["features"], crs=crs)


def test_fbfm40_disturbed_combined_mode(disturbance_roi):
    """2025 LDist plus 2024 FDist update the 2024 FBFM40."""
    plan = {
        "source": "landfire",
        "mode": "ldist_and_last_year_fdist",
        "ldist_version": "2025",
        "fdist_version": "2024",
    }

    ds = landfire.fetch_disturbed_fuel(
        disturbance_roi, "fbfm40", "2024", plan, lambda *a, **k: None
    )
    plain = landfire.fetch_fbfm40(disturbance_roi, lambda *a, **k: None, version="2024")

    assert list(ds.data_vars) == ["fbfm"]
    assert ds["fbfm"].dims == ("y", "x")
    assert ds["fbfm"].dtype == plain["fbfm"].dtype
    assert ds["fbfm"].rio.nodata is not None
    # Same grid as the plain fetch of the same product, so the two line up.
    # The values aren't compared: this domain's disturbance is low-severity
    # insects and mechanical treatment, which the rules map to the fuel model
    # the pixels already have.
    assert ds["fbfm"].shape == plain["fbfm"].shape
    assert np.array_equal(ds.x, plain.x) and np.array_equal(ds.y, plain.y)


def test_fccs_disturbed_fdist_mode(disturbance_roi):
    """The 2025 FDist alone updates the 2023 FCCS."""
    plan = {"source": "landfire", "mode": "fdist", "fdist_version": "2025"}

    ds = landfire.fetch_disturbed_fuel(
        disturbance_roi, "fccs", "2023", plan, lambda *a, **k: None
    )
    plain = landfire.fetch_fccs(disturbance_roi, lambda *a, **k: None, version="2023")

    assert list(ds.data_vars) == ["fccs"]
    assert ds["fccs"].dims == ("y", "x")
    assert ds["fccs"].dtype == plain["fccs"].dtype
    assert ds["fccs"].rio.nodata is not None
    assert ds["fccs"].shape == plain["fccs"].shape
    assert np.array_equal(ds.x, plain.x) and np.array_equal(ds.y, plain.y)
    assert (ds["fccs"].values != plain["fccs"].values).any()


def _user_fdist_grid(plain: xr.Dataset, code: int) -> xr.DataArray:
    """A user FDist grid with `code` everywhere, on the plain fetch's lattice."""
    return xr.full_like(plain["fbfm"], code, dtype=np.int16)


def test_fbfm40_disturbed_user_fdist_grid(disturbance_roi):
    """A user's FDist grid alone updates the 2024 FBFM40 on LANDFIRE's real grid."""
    plain = landfire.fetch_fbfm40(disturbance_roi, lambda *a, **k: None, version="2024")
    plan = {
        "source": "grid",
        "grid_id": "g1",
        "checksum": "abc",
        "band": "fdist",
        "include_landfire": False,
    }

    # 131 is a high-severity fire in the mapping year, which LANDFIRE's rules map to
    # a different fuel model for most vegetation.
    with patch.object(landfire, "load_band", return_value=_user_fdist_grid(plain, 131)):
        ds = landfire.fetch_disturbed_fuel(
            disturbance_roi, "fbfm40", "2024", plan, lambda *a, **k: None
        )

    assert list(ds.data_vars) == ["fbfm"]
    assert ds["fbfm"].dims == ("y", "x")
    assert ds["fbfm"].dtype == plain["fbfm"].dtype
    assert ds["fbfm"].shape == plain["fbfm"].shape
    assert np.array_equal(ds.x, plain.x) and np.array_equal(ds.y, plain.y)
    assert (ds["fbfm"].values != plain["fbfm"].values).any()


def test_fbfm40_disturbed_user_polygons(disturbance_roi):
    """A user's disturbance polygons, given in lat/lon, update the 2024 FBFM40."""
    plain = landfire.fetch_fbfm40(disturbance_roi, lambda *a, **k: None, version="2024")
    plan = {"source": "feature", "feature_id": "f1", "include_landfire": False}

    # The whole domain in one polygon, reprojected to lat/lon so the burn also
    # has to reproject onto the real grid.
    polygons = gpd.GeoDataFrame(
        {"fdist": [131]},
        geometry=disturbance_roi.geometry.values[:1],
        crs=disturbance_roi.crs,
    ).to_crs("EPSG:4326")

    with patch.object(landfire, "_load_disturbance_polygons", return_value=polygons):
        ds = landfire.fetch_disturbed_fuel(
            disturbance_roi, "fbfm40", "2024", plan, lambda *a, **k: None
        )

    assert list(ds.data_vars) == ["fbfm"]
    assert ds["fbfm"].dims == ("y", "x")
    assert ds["fbfm"].shape == plain["fbfm"].shape
    assert np.array_equal(ds.x, plain.x) and np.array_equal(ds.y, plain.y)
    assert (ds["fbfm"].values != plain["fbfm"].values).any()
