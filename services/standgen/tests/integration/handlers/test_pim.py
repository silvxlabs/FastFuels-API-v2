"""
Integration tests for PIM inventory expansion.

Tests the full standgen pipeline: Firestore setup -> process_inventory_request ->
verify GCS parquet output + Firestore georeference. Uses the Blue Mountain
domain (~1 sq km in Montana) with a static PIM TreeMap grid.

These tests hit real TreeMap tree tables in GCS, read real PIM grids, and write
real parquet to GCS/Firestore, so they require valid credentials.

Note: The static PIM grid is sparse (only a few valid plot pixels) because the
Blue Mountain domain sits on the boundary of UTM zones 11/12, so most TreeMap
pixels are NaN. Tests that need tree data use a minimum count guard. The pipeline
correctness (columns, structure, status transitions) is verified regardless.
"""

import dask.dataframe as dd
import pytest
from standgen.columns import BASE_COLUMNS

from lib.config import INVENTORIES_BUCKET

STATIC_PIM_GRID = "static-test-blue-mtn-pim-treemap"


class TestPimPipeline:
    """Full PIM pipeline: domain + PIM grid -> tree inventory in GCS."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_pipeline_completes(self, standgen_runner, source_pim_grid):
        """PIM expansion completes successfully with georeference."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        assert inventory["georeference"] is not None
        assert "crs" in inventory["georeference"]
        assert "bounds" in inventory["georeference"]

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_deterministic_tree_count(self, standgen_runner, source_pim_grid):
        """Same seed produces the same tree count (deterministic via SeedSequence)."""
        inventory_1 = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )
        path_1 = f"gs://{INVENTORIES_BUCKET}/{inventory_1['id']}"
        count_1 = len(dd.read_parquet(path_1))

        inventory_2 = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )
        path_2 = f"gs://{INVENTORIES_BUCKET}/{inventory_2['id']}"
        count_2 = len(dd.read_parquet(path_2))

        assert count_1 == count_2, (
            f"Expected deterministic tree count but got {count_1} vs {count_2}"
        )

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_different_seed_different_count(self, standgen_runner, source_pim_grid):
        """Different seeds produce different tree arrangements."""
        inventory_1 = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )
        path_1 = f"gs://{INVENTORIES_BUCKET}/{inventory_1['id']}"
        count_1 = len(dd.read_parquet(path_1))

        inventory_2 = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
            source_overrides={"seed": 99},
        )
        path_2 = f"gs://{INVENTORIES_BUCKET}/{inventory_2['id']}"
        count_2 = len(dd.read_parquet(path_2))

        # With a sparse grid both seeds may produce 0 trees.
        # Only assert difference when the grid has enough data.
        if count_1 == 0 and count_2 == 0:
            pytest.skip(
                "Grid too sparse to test seed variation (both runs produced 0 trees)"
            )

        assert count_1 != count_2


class TestParquetOutput:
    """Verify the parquet files written to GCS have correct schema and values."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_parquet_has_correct_columns(self, standgen_runner, source_pim_grid):
        """Output parquet should have exactly the base columns."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        path = f"gs://{INVENTORIES_BUCKET}/{inventory['id']}"
        ddf = dd.read_parquet(path)

        assert sorted(ddf.columns.tolist()) == sorted(BASE_COLUMNS)

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_parquet_values_are_sensible(self, standgen_runner, source_pim_grid):
        """Tree attribute values should be within physically reasonable ranges."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        path = f"gs://{INVENTORIES_BUCKET}/{inventory['id']}"
        df = dd.read_parquet(path).compute()

        if len(df) == 0:
            pytest.skip("No trees generated (sparse grid); skipping value validation")

        # DBH: 0-300 cm (reasonable range for any tree)
        assert df["dbh"].min() > 0
        assert df["dbh"].max() < 300

        # Height: 0-100 m
        assert df["height"].min() > 0
        assert df["height"].max() < 100

        # Crown ratio: 0-1 (fraction)
        assert df["crown_ratio"].min() >= 0
        assert df["crown_ratio"].max() <= 1

        # FIA species code: positive integers
        assert (df["fia_species_code"] > 0).all()

        # FIA status code: 1=live, 2=dead, 3=missing
        assert df["fia_status_code"].isin([1, 2, 3]).all()

        # No NaN values in any column
        assert not df.isna().any().any(), f"Found NaN values: {df.isna().sum()}"

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_tree_coordinates_within_domain(self, standgen_runner, source_pim_grid):
        """All tree coordinates should be within or near the domain bounds."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        path = f"gs://{INVENTORIES_BUCKET}/{inventory['id']}"
        df = dd.read_parquet(path).compute()

        if len(df) == 0:
            pytest.skip(
                "No trees generated (sparse grid); skipping coordinate validation"
            )

        geo = inventory["georeference"]
        bounds = geo["bounds"]  # [minx, miny, maxx, maxy]

        # Trees should be within domain bounds (with some tolerance for
        # pixel-edge placement — one 30m pixel buffer)
        buffer = 30.0
        assert df["x"].min() >= bounds[0] - buffer
        assert df["y"].min() >= bounds[1] - buffer
        assert df["x"].max() <= bounds[2] + buffer
        assert df["y"].max() <= bounds[3] + buffer


class TestGeoreference:
    """Verify georeference output."""

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_georeference_structure(self, standgen_runner, source_pim_grid):
        """Georeference should have CRS and bounds."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        geo = inventory["georeference"]
        assert "crs" in geo
        assert "bounds" in geo
        assert len(geo["bounds"]) == 4

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_georeference_crs_is_utm(self, standgen_runner, source_pim_grid):
        """Blue Mountain domain should produce a UTM CRS."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        crs = inventory["georeference"]["crs"]
        assert "utm" in crs.lower() or "326" in crs

    @pytest.mark.parametrize("source_pim_grid", [STATIC_PIM_GRID], indirect=True)
    def test_georeference_bounds_nonzero(self, standgen_runner, source_pim_grid):
        """Bounds should have positive extent."""
        inventory = standgen_runner(
            "blue_mtn.json",
            "pim_treemap.json",
            source_pim_grid_id=source_pim_grid,
        )

        bounds = inventory["georeference"]["bounds"]
        x_extent = bounds[2] - bounds[0]
        y_extent = bounds[3] - bounds[1]
        assert x_extent > 100, f"X extent too small: {x_extent}"
        assert y_extent > 100, f"Y extent too small: {y_extent}"
