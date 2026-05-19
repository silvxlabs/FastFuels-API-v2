"""
Tests for the layerset rasterization handler.

Unit-level only — no GCP required. The GeoJSON fetch is patched at
``griddle.handlers.layerset.gpd.read_file`` and alignment resolution is
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
from affine import Affine
from fastfuels_core.layersets import rasterize_layerset
from griddle.handlers.layerset import (
    OVERLAP_METHODS,
    build_layerset_bands,
    fetch_layerset,
)

from lib.errors import ProcessingError
from lib.testing import SHARED_TEST_FEATURES_DIR

# Shared example layerset (also used by the integration tests). Polygon shapes
# derived from a Lubrecht site layerset, translated into the bounds of the
# Blackfoot example domain so the example pairs cleanly with that domain in
# both unit and integration tests. Lives in lib/tests/shared_data/features/
# alongside every other reusable fixture in the repo.
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


def _patch_read_file_with_fixture(fixture_path: Path = FIXTURE_PATH):
    """Patch ``gpd.read_file`` at the handler's import site to return our fixture.

    The patch only short-circuits the GCS fetch; the real geopandas still loads
    the local fixture, so the full path GeoJSON → flat GeoDataFrame → real
    ``rasterize_layerset`` is exercised.
    """
    real_gdf = gpd.read_file(fixture_path)
    return patch(
        "griddle.handlers.layerset.gpd.read_file",
        return_value=real_gdf,
    )


class TestOverlapMethods:
    def test_keys_match_api_enum(self):
        """OVERLAP_METHODS keys mirror the API-side OverlapMethod enum.

        If the enum gains or loses a value, this assertion fails and we
        update the dict to match (or vice-versa).
        """
        assert set(OVERLAP_METHODS) == {"mean", "max", "min", "sum", "first"}

    @pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "first"])
    def test_callable_reduces_array(self, method):
        """Each overlap callable reduces a 1-D array without raising."""
        result = OVERLAP_METHODS[method](np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, (int, float, np.floating, np.integer))


class TestFixtureGeoJson:
    """Confirms the fixture loads as a flat GeoDataFrame with the expected columns.

    These assertions defend the contract that ``gpd.read_file`` over a flat
    layerset GeoJSON produces exactly the input columns expected by
    ``fastfuels_core.rasterize_layerset``. The handler relies on this directly
    — no flattening step in between.
    """

    def test_fixture_loads_as_flat_gdf(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        assert isinstance(gdf, gpd.GeoDataFrame)
        # Lubrecht fixture: 7 features spanning {shrub, herb, litter} × multiple fuelbeds
        assert len(gdf) == 7
        # CRS is honoured from the GeoJSON's own crs block
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
    ``build_layerset_bands``.
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


class TestBuildLayersetBands:
    """Tests for ``build_layerset_bands`` — derives the Grid's bands field
    from the rasterized output Dataset."""

    def test_bands_match_dataset_layout(self):
        gdf = gpd.read_file(FIXTURE_PATH)
        ds = rasterize_layerset(gdf, resolution=10.0)

        bands = build_layerset_bands(ds)
        # 3 variables × 5 bands = 15 entries
        assert len(bands) == 15

        # Indices are contiguous from 0
        assert [b["index"] for b in bands] == list(range(15))

        # Every band is continuous
        assert {b["type"] for b in bands} == {"continuous"}

        # Keys are "<var>.<band>"; spot-check a few
        keys = [b["key"] for b in bands]
        assert "shrub.loading" in keys
        assert "herb.height" in keys
        assert "litter.heat_of_combustion" in keys

        # Units track the rasterizer's per-band documentation
        unit_for = {b["key"]: b["unit"] for b in bands}
        assert unit_for["shrub.loading"] == "kg/m²"
        assert unit_for["shrub.height"] == "m"
        assert unit_for["shrub.live_fuel_moisture"] == "%"
        assert unit_for["shrub.heat_of_combustion"] == "kJ/kg"


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

        with _patch_read_file_with_fixture():
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

    def test_domain_alignment_reprojects_to_destination(self, monkeypatch):
        """alignment.target='domain' triggers per-variable post-process reproject."""
        _patch_dest(monkeypatch)
        domain_gdf = MagicMock()
        progress = MagicMock()

        with _patch_read_file_with_fixture():
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
        """A FileNotFoundError from gpd.read_file is translated to LAYERSET_NOT_FOUND."""

        def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError("simulated missing object in GCS")

        with patch("griddle.handlers.layerset.gpd.read_file", side_effect=_raise_fnf):
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
