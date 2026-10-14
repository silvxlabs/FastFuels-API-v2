"""
Tests for the layerset rasterization handler.

Unit-level only — no GCP required. The Parquet fetch is patched at
``griddle.handlers.layerset.gpd.read_parquet`` and alignment resolution is
patched at ``griddle.handlers.layerset.resolve_alignment_destination``
so the tests can pin a known output shape regardless of the
``domain_gdf`` fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import zarr
from affine import Affine
from fastfuels_core.layersets import rasterize_layerset
from griddle.handlers.layerset import (
    OVERLAP_METHODS,
    fetch_layerset,
    flatten_layerset,
)

from lib.errors import ProcessingError
from lib.layerset_bands import LAYERSET_PROPERTIES
from lib.testing import SHARED_TEST_FEATURES_DIR
from lib.units import validate_unit
from lib.zarr_utils import load_zarr, save_zarr

# Shared example layerset (also used by the integration tests). Polygon shapes
# derived from a Lubrecht site layerset, translated into the bounds of the
# Blackfoot example domain so the example pairs cleanly with that domain in
# both unit and integration tests. Lives in lib/tests/shared_data/features/
# alongside every other reusable fixture in the repo.
# Fixture is stored as GeoJSON (diffable, reviewable text) per the testing
# protocol. The handler reads GeoParquet from GCS in production; tests
# convert in-memory and either patch the read_parquet call (unit tests)
# or upload as Parquet to a temp GCS path (integration tests).
FIXTURE_PATH = SHARED_TEST_FEATURES_DIR / "blackfoot_example_layerset.geojson"

# Known alignment destination used to exercise the post-rasterize reprojection
# branch. Pinned via monkeypatching resolve_alignment_destination so the test
# is independent of the resampler's lattice math.
_KNOWN_DEST = {
    "destination_crs": "EPSG:32612",
    "destination_transform": Affine(10.0, 0.0, 312500.0, 0.0, -10.0, 5195400.0),
    "destination_shape": (50, 80),  # (ny, nx) — coarse, deliberately small
}


def _patch_dest(monkeypatch):
    """Helper: pin resolve_alignment_destination to _KNOWN_DEST for the test."""
    monkeypatch.setattr(
        "griddle.handlers.layerset.resolve_alignment_destination",
        lambda **kwargs: _KNOWN_DEST,
    )


def _patch_read_parquet_with_fixture(fixture_path: Path = FIXTURE_PATH):
    """Patch ``gpd.read_parquet`` at the handler's import site to return our fixture.

    The patch short-circuits the GCS fetch — we load the local GeoJSON
    fixture into a GeoDataFrame (matching the shape the production
    ``read_parquet`` call would return) and feed that into the rest of
    the handler. The full path GeoDataFrame → real ``rasterize_layerset``
    is still exercised end-to-end.
    """
    real_gdf = gpd.read_file(fixture_path)
    return patch(
        "griddle.handlers.layerset.gpd.read_parquet",
        return_value=real_gdf,
    )


class TestOverlapMethods:
    def test_keys_match_api_enum(self):
        """OVERLAP_METHODS keys mirror the API-side OverlapMethod enum.

        If the enum gains or loses a value, this assertion fails and we
        update the dict to match (or vice-versa).
        """
        assert set(OVERLAP_METHODS) == {"mean", "max", "min"}

    @pytest.mark.parametrize("method", ["mean", "max", "min"])
    def test_callable_reduces_array(self, method):
        """Each overlap callable reduces a 1-D array without raising."""
        result = OVERLAP_METHODS[method](np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, (int, float, np.floating, np.integer))


class TestFixture:
    """Confirms the fixture loads as a flat GeoDataFrame with the expected columns.

    These assertions defend the contract that the fixture produces exactly
    the input columns expected by ``fastfuels_core.rasterize_layerset``.
    The handler relies on the loaded GeoDataFrame shape directly — no
    flattening step in between.
    """

    def test_fixture_loads_as_flat_gdf(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        assert isinstance(gdf, gpd.GeoDataFrame)
        # Lubrecht fixture: 7 features spanning {shrub, herb, litter} × multiple fuelbeds
        assert len(gdf) == 7
        # CRS is honoured from the GeoJSON's crs block
        assert gdf.crs is not None
        assert "32612" in str(gdf.crs)

    def test_fixture_carries_rasterizer_columns(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        required = {
            "fuel_type",
            "fuel_loading",
            "fuel_height",
            "percent_cover",
            "distribution",
        }
        assert required.issubset(gdf.columns)


class TestRealRasterizeLayerset:
    """Smoke-test the real ``fastfuels_core.layersets.rasterize_layerset`` end-to-end.

    Protects against silent drift in the published library: variable layout,
    dim ordering, and band coord values are all assumptions baked into
    ``flatten_layerset``.
    """

    def test_output_shape_matches_published_contract(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        ds = rasterize_layerset(gdf, resolution=10.0)

        # One variable per unique fuel_type, in input order
        assert set(ds.data_vars) == {"shrub", "herb", "litter"}

        # Each variable has dims (band, y, x) with the documented 5-band coord
        expected_bands = [
            "loading",
            "height",
            "live_fuel_moisture",
            "dead_fuel_moisture",
            "heat_of_combustion",
        ]
        for name in ds.data_vars:
            da = ds[name]
            assert da.dims == ("band", "y", "x")
            assert da.dtype == np.dtype("float32")
            assert list(da.coords["band"].values) == expected_bands

        # CRS is honoured from the input gdf (UTM 12N)
        assert "32612" in str(ds.rio.crs)


class TestLayersetBandUnits:
    """Drift guard for ``lib.layerset_bands.LAYERSET_PROPERTIES``: every unit is
    canonical, and every rasterizer property has a definition."""

    def test_all_units_are_canonical(self):
        for defn in LAYERSET_PROPERTIES.values():
            validate_unit(defn["unit"])

    def test_all_defs_have_name_and_description(self):
        for defn in LAYERSET_PROPERTIES.values():
            assert defn["name"]
            assert defn["description"]

    def test_every_rasterizer_property_is_defined(self):
        ds = rasterize_layerset(gpd.read_file(FIXTURE_PATH), resolution=10.0)
        for name in ds.data_vars:
            assert set(ds[name].coords["band"].values) == set(LAYERSET_PROPERTIES)


def _layer_ds(fuel_types: list[str]) -> xr.Dataset:
    """A rasterizer-shaped Dataset: one (band, y, x) variable per fuel type."""
    props = list(LAYERSET_PROPERTIES)
    data = {
        ft: xr.DataArray(
            np.full((len(props), 2, 2), i + 1, dtype=np.float32),
            dims=("band", "y", "x"),
            coords={"band": props, "y": [1.0, 0.0], "x": [0.0, 1.0]},
        )
        for i, ft in enumerate(fuel_types)
    }
    return xr.Dataset(data).rio.write_crs("EPSG:32612")


class TestFlattenLayerset:
    """``flatten_layerset`` stores one variable per band, named by its key."""

    def test_bands_match_flat_dataset(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        ds = rasterize_layerset(gdf, resolution=10.0)

        flat, bands = flatten_layerset(ds)
        # 3 fuel types x 5 properties = 15 bands, one 2D variable each
        assert len(bands) == 15
        assert [b["index"] for b in bands] == list(range(15))
        assert [b["key"] for b in bands] == list(flat.data_vars)
        for key in flat.data_vars:
            assert flat[key].dims == ("y", "x")
        assert {b["type"] for b in bands} == {"continuous"}

        unit_for = {b["key"]: b["unit"] for b in bands}
        assert unit_for["fuel_load.shrub"] == "kg/m**2"
        assert unit_for["fuel_depth.herb"] == "m"
        assert unit_for["fuel_moisture.live.shrub"] == "%"
        assert unit_for["fuel_moisture.dead.litter"] == "%"
        assert unit_for["heat_content.litter"] == "kJ/kg"

        name_for = {b["key"]: b["name"] for b in bands}
        assert name_for["fuel_load.shrub"] == "Shrub Fuel Load"
        assert name_for["heat_content.litter"] == "Litter Heat Content"
        desc_for = {b["key"]: b["description"] for b in bands}
        assert "fuel type: shrub" in desc_for["fuel_load.shrub"]

        # Values and spatial metadata survive the split.
        np.testing.assert_array_equal(
            flat["fuel_load.litter"].values, ds["litter"].sel(band="loading").values
        )
        assert flat.rio.crs == ds.rio.crs
        assert np.isnan(flat["fuel_load.litter"].rio.nodata)

    def test_fuel_types_are_sanitized(self):
        flat, bands = flatten_layerset(_layer_ds(["Shrub Layer", "TL.3"]))
        keys = {b["key"] for b in bands}
        assert "fuel_load.shrub_layer" in keys
        assert "fuel_depth.tl_3" in keys
        assert "Shrub Layer" in bands[0]["description"]

    def test_fuel_types_colliding_after_sanitizing_rejected(self):
        with pytest.raises(ProcessingError) as exc:
            flatten_layerset(_layer_ds(["Shrub", "shrub "]))
        assert exc.value.code == "INVALID_FUEL_TYPE"

    def test_fuel_type_without_key_safe_characters_rejected(self):
        with pytest.raises(ProcessingError) as exc:
            flatten_layerset(_layer_ds(["---"]))
        assert exc.value.code == "INVALID_FUEL_TYPE"

    def test_flat_bands_readable_by_key_from_zarr(self, tmp_path):
        """New layerset grids are read the way the API chunk endpoint
        (zarr group ``getitem(band_key)``) and the QUIC-Fire / Landscape
        exporters (``load_zarr(...)[band]``) read every other grid."""
        flat, bands = flatten_layerset(_layer_ds(["shrub", "litter"]))
        path = str(tmp_path / "grid.zarr")
        save_zarr(path, flat, chunk_shape=(512, 512))

        group = zarr.open_group(path, mode="r")
        loaded = load_zarr(path)
        for band in bands:
            key = band["key"]
            np.testing.assert_array_equal(group[key][:], flat[key].values)
            arr = loaded[key].transpose("y", "x").values
            np.testing.assert_array_equal(arr, flat[key].values)


class TestFetchLayerset:
    def test_native_alignment_returns_unprojected_output(self, monkeypatch):
        """alignment.target='native' skips post-process reprojection."""
        domain_gdf = MagicMock()
        progress = MagicMock()

        # If the handler accidentally reprojected, it would call
        # resolve_alignment_destination — fail loudly if so.
        def _should_not_be_called(**kwargs):
            raise AssertionError(
                "resolve_alignment_destination called for native alignment"
            )

        monkeypatch.setattr(
            "griddle.handlers.layerset.resolve_alignment_destination",
            _should_not_be_called,
        )

        with _patch_read_parquet_with_fixture():
            ds = fetch_layerset(
                domain_gdf=domain_gdf,
                layerset_id="abc123",
                domain_id="dom-xyz",
                overlap_method="mean",
                progress=progress,
                alignment={"target": "native"},
                target_grid_doc=None,
            )

        # Native output: one variable per fuel_type, dims (band, y, x).
        assert set(ds.data_vars) == {"shrub", "herb", "litter"}
        for name in ds.data_vars:
            assert ds[name].dims == ("band", "y", "x")

    def test_random_distributions_are_reproducible(self):
        """Repeated runs on the same layerset give identical rasters."""

        def _run():
            with _patch_read_parquet_with_fixture():
                return fetch_layerset(
                    domain_gdf=MagicMock(),
                    layerset_id="abc123",
                    domain_id="dom-xyz",
                    overlap_method="mean",
                    progress=MagicMock(),
                    alignment={"target": "native"},
                    target_grid_doc=None,
                )

        first, second = _run(), _run()
        for name in first.data_vars:
            np.testing.assert_array_equal(first[name].values, second[name].values)

    def test_domain_alignment_reprojects_to_destination(self, monkeypatch):
        """alignment.target='domain' triggers per-variable post-process reproject."""
        _patch_dest(monkeypatch)
        domain_gdf = MagicMock()
        progress = MagicMock()

        with _patch_read_parquet_with_fixture():
            ds = fetch_layerset(
                domain_gdf=domain_gdf,
                layerset_id="abc123",
                domain_id="dom-xyz",
                overlap_method="mean",
                progress=progress,
                alignment={"target": "domain"},
                target_grid_doc=None,
            )

        # All variables reprojected to the pinned destination shape
        ny, nx = _KNOWN_DEST["destination_shape"]
        for name in ds.data_vars:
            da = ds[name]
            # Per-variable y/x match destination; band dim untouched (5 bands)
            assert da.sizes["y"] == ny
            assert da.sizes["x"] == nx
            assert da.sizes["band"] == 5

        # CRS written through rio
        assert str(ds.rio.crs) == _KNOWN_DEST["destination_crs"]

    def test_unknown_overlap_method_raises_processing_error(self):
        """An overlap_method outside OVERLAP_METHODS raises a structured error."""
        with pytest.raises(ProcessingError) as exc_info:
            fetch_layerset(
                domain_gdf=MagicMock(),
                layerset_id="abc",
                domain_id="dom",
                overlap_method="bogus",
                progress=MagicMock(),
            )
        assert exc_info.value.code == "UNKNOWN_OVERLAP_METHOD"

    def test_missing_layerset_raises_layerset_not_found(self):
        """A FileNotFoundError from gpd.read_parquet is translated to LAYERSET_NOT_FOUND."""

        def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError("simulated missing object in GCS")

        with patch(
            "griddle.handlers.layerset.gpd.read_parquet", side_effect=_raise_fnf
        ):
            with pytest.raises(ProcessingError) as exc_info:
                fetch_layerset(
                    domain_gdf=MagicMock(),
                    layerset_id="missing-id",
                    domain_id="dom",
                    overlap_method="mean",
                    progress=MagicMock(),
                    alignment={"target": "native"},
                )
        assert exc_info.value.code == "LAYERSET_NOT_FOUND"
