"""Unit tests for treevox.voxelize — pure compute layer.

All tests mock fastfuels_core so they run without heavy data loading.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastfuels_core.voxelization import (
    compute_crown_probability_field,
    sample_occupancy,
    sample_occupied_cells,
)
from treevox import voxelize

# Fixtures


def fake_domain(minx=0.0, miny=0.0, maxx=100.0, maxy=100.0, crs="EPSG:32610"):
    """Minimal domain stand-in with `total_bounds` and `crs`."""
    return SimpleNamespace(
        total_bounds=np.array([minx, miny, maxx, maxy]),
        crs=crs,
    )


def fake_tree_df(
    n: int = 3,
    species=131,
    status=1,
    dbh=20.0,
    height=15.0,
    crown_ratio=0.4,
    xs=None,
    ys=None,
) -> pd.DataFrame:
    xs = xs if xs is not None else np.linspace(10, 90, n)
    ys = ys if ys is not None else np.linspace(10, 90, n)
    return pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "fia_species_code": [species] * n,
            "fia_status_code": [status] * n,
            "dbh": [dbh] * n,
            "height": [height] * n,
            "crown_ratio": [crown_ratio] * n,
        }
    )


def base_source_config():
    return {
        "resolution": (1.0, 1.0, 1.0),
        "crown_profile_model": "purves",
        "biomass_source": {
            "type": "allometry",
            "equations": "nsvb",
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        },
        "moisture_model": {"live": {"method": "uniform", "value": 100.0}},
    }


# compute_grid_dimensions


class TestComputeGridDimensions:
    def test_resolution_snap_no_extra_padding(self):
        """Domain 0..100 at 1m resolution: bounds already aligned, no extra padding."""
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 100, 100),
            fake_tree_df(height=10.0),
            (1.0, 1.0, 1.0),
        )
        assert dims["nx"] == 100
        assert dims["ny"] == 100
        assert dims["nz"] == 10
        assert dims["x_origin"] == 0.0
        assert dims["y_origin"] == 100.0

    def test_non_aligned_bounds_snap_outward(self):
        """Domain 0.5..99.5 at 1m resolution snaps to 0..100."""
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0.5, 0.5, 99.5, 99.5),
            fake_tree_df(height=5.0),
            (1.0, 1.0, 1.0),
        )
        assert dims["x_origin"] == 0.0
        assert dims["y_origin"] == 100.0
        assert dims["nx"] == 100
        assert dims["ny"] == 100

    def test_coarser_resolution_snap(self):
        """Domain 0..100 at 3m resolution snaps to 0..102."""
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 100, 100),
            fake_tree_df(height=9.0),
            (3.0, 3.0, 3.0),
        )
        assert dims["x_origin"] == 0.0
        assert dims["y_origin"] == 102.0
        assert dims["nx"] == 34
        assert dims["ny"] == 34
        assert dims["nz"] == 3

    def test_empty_df_uses_vr_as_max_height(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 10, 10), pd.DataFrame(), (1.0, 1.0, 2.0)
        )
        assert dims["nz"] == 1

    def test_anisotropic_resolution_rejected(self):
        with pytest.raises(voxelize.InvalidResolutionError, match="Anisotropic"):
            voxelize.compute_grid_dimensions(
                fake_domain(0, 0, 10, 10),
                fake_tree_df(),
                (1.0, 2.0, 1.0),
            )

    def test_zero_resolution_rejected(self):
        with pytest.raises(voxelize.InvalidResolutionError):
            voxelize.compute_grid_dimensions(
                fake_domain(0, 0, 10, 10), fake_tree_df(), (0.0, 0.0, 1.0)
            )

    def test_coord_arrays_are_cell_centers(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 10, 10), fake_tree_df(height=5.0), (1.0, 1.0, 1.0)
        )
        assert dims["x_coords"][0] == pytest.approx(0.5)
        assert dims["y_coords"][0] == pytest.approx(9.5)
        assert dims["z_coords"][0] == pytest.approx(0.5)

    def test_returns_georeference_fields(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 10, 20), fake_tree_df(height=5.0), (1.0, 1.0, 1.0)
        )
        assert dims["z_origin"] == 0.0
        assert dims["vr"] == 1.0
        assert len(dims["transform"]) == 6
        assert dims["crs"] == "EPSG:32610"


# build_tree


class TestBuildTree:
    def _row(self, **overrides):
        data = {
            "fia_species_code": 131,
            "fia_status_code": 1,
            "dbh": 20.0,
            "height": 15.0,
            "crown_ratio": 0.4,
            "x": 5.0,
            "y": 5.0,
        }
        data.update(overrides)
        return pd.Series(data)

    def test_maps_v2_columns_to_tree_kwargs(self):
        tree = voxelize.build_tree(self._row(), base_source_config())
        assert tree.species_code == 131
        assert tree.status_code == 1
        assert tree.diameter == 20.0
        assert tree.height == 15.0
        assert tree.crown_ratio == 0.4
        assert tree.x == 5.0
        assert tree.y == 5.0

    @pytest.mark.parametrize("model", ["purves", "beta"])
    def test_crown_profile_model_is_honored(self, model):
        cfg = base_source_config()
        cfg["crown_profile_model"] = model
        tree = voxelize.build_tree(self._row(), cfg)
        assert tree._crown_profile_model_type == model

    @pytest.mark.parametrize(
        "equations,ff_model",
        [("nsvb", "NSVB"), ("jenkins", "jenkins")],
    )
    def test_biomass_equation_mapping(self, equations, ff_model):
        cfg = base_source_config()
        cfg["biomass_source"]["equations"] = equations
        row = self._row()
        tree = voxelize.build_tree(row, cfg)
        assert tree._biomass_allometry_model_type == ff_model

    def test_inventory_biomass_reads_column_as_crown_fuel_load(self):
        cfg = base_source_config()
        cfg["biomass_source"] = {
            "type": "inventory_columns",
            "columns": {"foliage": {"column": "my_load", "unit": "kg"}},
            "components": ["foliage"],
        }
        row = self._row(my_load=42.0)
        tree = voxelize.build_tree(row, cfg)
        assert tree._crown_fuel_load_override == 42.0

    def test_non_inventory_does_not_read_column(self):
        cfg = base_source_config()  # nsvb
        row = self._row()
        tree = voxelize.build_tree(row, cfg)
        assert tree._crown_fuel_load_override is None

    def test_inventory_crown_radius_reads_column_as_max_crown_radius(self):
        cfg = base_source_config()
        cfg["max_crown_radius_source"] = {
            "type": "inventory_column",
            "column": "lidar_max_radius",
            "unit": "m",
        }
        row = self._row(lidar_max_radius=3.5)
        tree = voxelize.build_tree(row, cfg)
        assert tree._max_crown_radius_override == 3.5

    def test_default_crown_radius_source_does_not_set_override(self):
        cfg = base_source_config()  # no max_crown_radius_source
        row = self._row()
        tree = voxelize.build_tree(row, cfg)
        assert tree._max_crown_radius_override is None

    def test_allometry_crown_radius_source_does_not_set_override(self):
        cfg = base_source_config()
        cfg["max_crown_radius_source"] = {"type": "allometry"}
        row = self._row()
        tree = voxelize.build_tree(row, cfg)
        assert tree._max_crown_radius_override is None


class TestBiomassComponentDistribution:
    def test_foliage_calls_fastfuels_distribution(self):
        class FakeVT:
            def distribute_biomass(self):
                return np.ones((1, 2, 2), dtype="float32")

        out = voxelize.distribute_component_biomass(FakeVT(), "foliage")
        assert out.shape == (1, 2, 2)

    @pytest.mark.parametrize("component", ["branchwood", "fine"])
    def test_non_foliage_components_raise_not_implemented(self, component):
        class FakeVT:
            def distribute_biomass(self):
                raise AssertionError("should not call foliage distribution")

        with pytest.raises(NotImplementedError, match=component):
            voxelize.distribute_component_biomass(FakeVT(), component)

    @pytest.mark.parametrize(
        "band_key,component",
        [
            ("bulk_density.foliage.live", "foliage"),
            ("bulk_density.foliage.dead", "foliage"),
            ("bulk_density.branchwood.live", "branchwood"),
            ("bulk_density.branchwood.dead", "branchwood"),
            ("bulk_density.fine.live", "fine"),
            ("bulk_density.fine.dead", "fine"),
        ],
    )
    def test_component_selection_from_output_band(self, band_key, component):
        cfg = base_source_config()
        cfg["bands"] = [band_key]
        cfg["biomass_source"]["components"] = []
        assert voxelize.biomass_component_to_distribute(cfg) == component

    @pytest.mark.parametrize("component", ["branchwood", "fine"])
    def test_component_selection_from_source_components(self, component):
        cfg = base_source_config()
        cfg["biomass_source"]["components"] = [component]
        assert voxelize.biomass_component_to_distribute(cfg) == component


# compute_cache_keys


class TestComputeCacheKeys:
    def test_identical_trees_share_key(self):
        df = fake_tree_df(n=5, species=131, dbh=20.0, height=15.0, crown_ratio=0.4)
        keys = voxelize.compute_cache_keys(df)
        assert keys.nunique() == 1

    def test_different_species_different_keys(self):
        df = pd.DataFrame(
            {
                "fia_species_code": [131, 202],
                "fia_status_code": [1, 1],
                "dbh": [20.0, 20.0],
                "height": [15.0, 15.0],
                "crown_ratio": [0.4, 0.4],
                "x": [1, 2],
                "y": [1, 2],
            }
        )
        keys = voxelize.compute_cache_keys(df)
        assert keys.nunique() == 2

    def test_bin_boundary_equality(self):
        """height=1.0 and 1.9 land in the same bin at HEIGHT_BIN_M=1."""
        df = fake_tree_df(n=2, height=1.0)
        df.loc[1, "height"] = 1.9
        keys = voxelize.compute_cache_keys(df)
        assert keys.nunique() == 1

    def test_bin_boundary_differentiation(self):
        """height=1.0 vs 2.0 land in different bins."""
        df = fake_tree_df(n=2, height=1.0)
        df.loc[1, "height"] = 2.0
        keys = voxelize.compute_cache_keys(df)
        assert keys.nunique() == 2

    def test_inventory_foliage_biomass_splits_same_morphology(self):
        df = fake_tree_df(n=2, species=131, dbh=20.0, height=15.0, crown_ratio=0.4)
        df["my_fuel_load"] = [10.0, 25.0]
        cfg = base_source_config()
        cfg["biomass_source"] = {
            "type": "inventory_columns",
            "columns": {"foliage": {"column": "my_fuel_load", "unit": "kg"}},
            "components": ["foliage"],
        }

        keys = voxelize.compute_cache_keys(df, cfg)

        assert keys.nunique() == 2

    def test_allometry_ignores_unconfigured_biomass_columns(self):
        df = fake_tree_df(n=2, species=131, dbh=20.0, height=15.0, crown_ratio=0.4)
        df["my_fuel_load"] = [10.0, 25.0]

        keys = voxelize.compute_cache_keys(df, base_source_config())

        assert keys.nunique() == 1

    def test_inventory_max_crown_radius_splits_same_morphology(self):
        df = fake_tree_df(n=2, species=131, dbh=20.0, height=15.0, crown_ratio=0.4)
        df["lidar_max_radius"] = [2.5, 4.0]
        cfg = base_source_config()
        cfg["max_crown_radius_source"] = {
            "type": "inventory_column",
            "column": "lidar_max_radius",
            "unit": "m",
        }

        keys = voxelize.compute_cache_keys(df, cfg)

        assert keys.nunique() == 2

    def test_allometry_crown_radius_source_does_not_split(self):
        df = fake_tree_df(n=2, species=131, dbh=20.0, height=15.0, crown_ratio=0.4)
        df["lidar_max_radius"] = [2.5, 4.0]
        cfg = base_source_config()
        cfg["max_crown_radius_source"] = {"type": "allometry"}

        keys = voxelize.compute_cache_keys(df, cfg)

        assert keys.nunique() == 1


# calculate_arrays_to_cache


class TestCalculateArraysToCache:
    def test_zero_voxels_returns_one(self):
        assert voxelize.calculate_arrays_to_cache(0, 10) == 1

    def test_capped_by_tree_frequency(self):
        assert voxelize.calculate_arrays_to_cache(10000, 2) == 2

    def test_capped_by_max_cache(self):
        assert voxelize.calculate_arrays_to_cache(10**8, 10**6, max_cache=100) == 100

    def test_scales_with_voxels(self):
        a = voxelize.calculate_arrays_to_cache(10, 1000)
        b = voxelize.calculate_arrays_to_cache(1000, 1000)
        assert b > a


# assign_trees_to_chunks


class TestAssignTreesToChunks:
    def test_chunks_are_correct(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 100, 100), fake_tree_df(height=10.0), (1.0, 1.0, 1.0)
        )
        df = fake_tree_df(
            n=4,
            xs=[5.0, 55.0, 55.0, 5.0],
            ys=[5.0, 55.0, 5.0, 55.0],
        )
        out = voxelize.assign_trees_to_chunks(
            df,
            dims["x_origin"],
            dims["y_origin"],
            dims["hr"],
            dims["nx"],
            dims["ny"],
            chunk_xy=50,
        )
        assert "row_chunk" in out.columns
        assert "col_chunk" in out.columns
        assert len(out) == 4

    def test_sort_puts_tallest_last_within_chunk(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 100, 100), fake_tree_df(height=30.0), (1.0, 1.0, 1.0)
        )
        df = fake_tree_df(
            n=3,
            xs=[10, 20, 30],
            ys=[10, 20, 30],
        )
        df["height"] = [30.0, 10.0, 20.0]
        out = voxelize.assign_trees_to_chunks(
            df,
            dims["x_origin"],
            dims["y_origin"],
            dims["hr"],
            dims["nx"],
            dims["ny"],
            chunk_xy=200,
        )
        assert list(out["height"]) == [10.0, 20.0, 30.0]


# batch_union_slices and chunk_slice


class TestBatchUnionSlices:
    def test_single_chunk_includes_halo(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 2000, 2000), fake_tree_df(height=10.0), (1.0, 1.0, 1.0)
        )
        y, x = voxelize.batch_union_slices(
            [(1, 1)], dims["ny"], dims["nx"], chunk_xy=100, overlap_cells=10
        )
        assert y.start == 90
        assert y.stop == 210
        assert x.start == 90
        assert x.stop == 210

    def test_clamped_at_grid_edge(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 100, 100), fake_tree_df(height=10.0), (1.0, 1.0, 1.0)
        )
        y, x = voxelize.batch_union_slices(
            [(0, 0)], dims["ny"], dims["nx"], chunk_xy=50, overlap_cells=10
        )
        assert y.start == 0
        assert x.start == 0

    def test_union_of_multiple_chunks(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 2000, 2000), fake_tree_df(height=10.0), (1.0, 1.0, 1.0)
        )
        y, x = voxelize.batch_union_slices(
            [(0, 0), (1, 1)], dims["ny"], dims["nx"], chunk_xy=100, overlap_cells=10
        )
        assert y.start == 0
        assert y.stop == 210
        assert x.start == 0
        assert x.stop == 210

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            voxelize.batch_union_slices([], 100, 100, chunk_xy=50)


class TestChunkSlice:
    def test_matches_union_of_one(self):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 2000, 2000), fake_tree_df(height=10.0), (1.0, 1.0, 1.0)
        )
        ny, nx = dims["ny"], dims["nx"]
        s_y, s_x = voxelize.chunk_slice((1, 1), ny, nx, chunk_xy=100, overlap_cells=10)
        u_y, u_x = voxelize.batch_union_slices(
            [(1, 1)], ny, nx, chunk_xy=100, overlap_cells=10
        )
        assert (s_y.start, s_y.stop) == (u_y.start, u_y.stop)
        assert (s_x.start, s_x.stop) == (u_x.start, u_x.stop)


# build_chunk_cache (mocked fastfuels-core)


class TestBuildChunkCache:
    def test_empty_df_returns_empty_cache(self):
        empty = pd.DataFrame(
            {
                "fia_species_code": [],
                "fia_status_code": [],
                "dbh": [],
                "height": [],
                "crown_ratio": [],
                "x": [],
                "y": [],
                "_cache_key": [],
            }
        )
        result = voxelize.build_chunk_cache(
            empty, 1.0, 1.0, base_source_config(), np.random.default_rng(0)
        )
        assert result == {}

    def test_one_entry_per_cache_key(self, monkeypatch):
        canopy = np.ones((2, 3, 3))

        monkeypatch.setattr(
            voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
        )
        monkeypatch.setattr(
            voxelize,
            "compute_crown_probability_field",
            lambda m, **kw: (m, int(np.count_nonzero(m))),
        )
        monkeypatch.setattr(
            voxelize, "sample_occupancy", lambda m, field, n, **kw: m.copy()
        )

        class FakeVT:
            def __init__(self, tree, mask, hr, vr):
                self.mask = mask

            def distribute_biomass(self):
                return self.mask * 1.0

        monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)

        df = pd.DataFrame(
            {
                "fia_species_code": [131, 131, 202],
                "fia_status_code": [1, 1, 1],
                "dbh": [20.0, 20.0, 20.0],
                "height": [15.0, 15.0, 15.0],
                "crown_ratio": [0.4, 0.4, 0.4],
                "x": [1, 2, 3],
                "y": [1, 2, 3],
            }
        )
        df["_cache_key"] = voxelize.compute_cache_keys(df)

        cache = voxelize.build_chunk_cache(
            df, 1.0, 1.0, base_source_config(), np.random.default_rng(0)
        )

        assert len(cache) == df["_cache_key"].nunique()
        for _, entry in cache.items():
            assert isinstance(entry, voxelize.CacheEntry)
            assert len(entry.biomass_arrays) >= 1
            assert all(a.shape == canopy.shape for a in entry.biomass_arrays)
            # bin-representative attrs populated from the first-row Tree
            assert entry.species_code in {131, 202}
            assert entry.foliage_sav > 0
            assert entry.crown_base_height >= 0

    def test_biomass_column_flows_to_tree_crown_fuel_load(self, monkeypatch):
        """Inventory foliage column flows to Tree's crown_fuel_load.

        `build_tree` alone reading the column is covered by TestBuildTree; this
        asserts build_chunk_cache plumbs the source config through so the
        bin-representative Tree gets the right crown_fuel_load at cache-build
        time. Regression guard for wiring bugs between the orchestrator, cache
        builder, and build_tree.
        """
        canopy = np.ones((2, 3, 3))
        monkeypatch.setattr(
            voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
        )
        monkeypatch.setattr(
            voxelize,
            "compute_crown_probability_field",
            lambda m, **kw: (m, int(np.count_nonzero(m))),
        )
        monkeypatch.setattr(
            voxelize, "sample_occupancy", lambda m, field, n, **kw: m.copy()
        )

        class FakeVT:
            def __init__(self, tree, mask, hr, vr):
                self.tree = tree
                self.mask = mask

            def distribute_biomass(self):
                return self.mask * 1.0

        monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)

        built_trees: list = []
        real_build_tree = voxelize.build_tree

        def spying_build_tree(row, source_config):
            tree = real_build_tree(row, source_config)
            built_trees.append(tree)
            return tree

        monkeypatch.setattr(voxelize, "build_tree", spying_build_tree)

        df = pd.DataFrame(
            {
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.4],
                "x": [1.0],
                "y": [1.0],
                "my_fuel_load": [42.0],
            }
        )
        df["_cache_key"] = voxelize.compute_cache_keys(df)

        cfg = base_source_config()
        cfg["biomass_source"] = {
            "type": "inventory_columns",
            "columns": {"foliage": {"column": "my_fuel_load", "unit": "kg"}},
            "components": ["foliage"],
        }

        voxelize.build_chunk_cache(df, 1.0, 1.0, cfg, np.random.default_rng(0))

        assert len(built_trees) == 1
        assert built_trees[0]._crown_fuel_load_override == 42.0

    def test_inventory_biomass_column_is_preserved_per_cache_key(self, monkeypatch):
        canopy = np.ones((2, 3, 3))
        monkeypatch.setattr(
            voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
        )
        monkeypatch.setattr(
            voxelize,
            "compute_crown_probability_field",
            lambda m, **kw: (m, int(np.count_nonzero(m))),
        )
        monkeypatch.setattr(
            voxelize, "sample_occupancy", lambda m, field, n, **kw: m.copy()
        )

        class FakeVT:
            def __init__(self, tree, mask, hr, vr):
                self.tree = tree
                self.mask = mask

            def distribute_biomass(self):
                return self.mask * self.tree._crown_fuel_load_override

        monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)

        cfg = base_source_config()
        cfg["biomass_source"] = {
            "type": "inventory_columns",
            "columns": {"foliage": {"column": "my_fuel_load", "unit": "kg"}},
            "components": ["foliage"],
        }
        df = pd.DataFrame(
            {
                "fia_species_code": [131, 131],
                "fia_status_code": [1, 1],
                "dbh": [20.0, 20.0],
                "height": [15.0, 15.0],
                "crown_ratio": [0.4, 0.4],
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "my_fuel_load": [10.0, 25.0],
            }
        )
        df["_cache_key"] = voxelize.compute_cache_keys(df, cfg)

        cache = voxelize.build_chunk_cache(df, 1.0, 1.0, cfg, np.random.default_rng(0))
        cached_values = sorted(
            float(entry.biomass_arrays[0].max()) for entry in cache.values()
        )

        assert len(cache) == 2
        assert cached_values == [10.0, 25.0]

    @pytest.mark.parametrize("component", ["branchwood", "fine"])
    def test_non_foliage_component_failure_is_not_swallowed(
        self, monkeypatch, component
    ):
        canopy = np.ones((2, 3, 3))
        monkeypatch.setattr(
            voxelize, "discretize_crown_profile", lambda *a, **kw: canopy.copy()
        )
        monkeypatch.setattr(
            voxelize,
            "compute_crown_probability_field",
            lambda m, **kw: (m, int(np.count_nonzero(m))),
        )
        monkeypatch.setattr(
            voxelize, "sample_occupancy", lambda m, field, n, **kw: m.copy()
        )

        class FakeVT:
            def __init__(self, tree, mask, hr, vr):
                self.mask = mask

        monkeypatch.setattr(voxelize, "VoxelizedTree", FakeVT)

        df = pd.DataFrame(
            {
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.4],
                "x": [1.0],
                "y": [1.0],
            }
        )
        df["_cache_key"] = voxelize.compute_cache_keys(df)

        cfg = base_source_config()
        cfg["biomass_source"]["components"] = [component]
        if component == "fine":
            cfg["biomass_source"]["fine"] = {
                "recipe": "foliage_plus_branchwood_fraction",
                "branchwood_fraction": 0.1,
            }

        with pytest.raises(NotImplementedError, match=component):
            voxelize.build_chunk_cache(df, 1.0, 1.0, cfg, np.random.default_rng(0))


# field hoist equivalence (#399)


class TestFieldHoistEquivalence:
    """The hoisted seam must be byte-identical to the one-shot wrapper (#399).

    build_chunk_cache computes the crown-probability field once per bin and
    draws each realization via `sample_occupancy`; this must reproduce exactly
    what the old per-realization `sample_occupied_cells` produced for the same
    seed. Runs against real fastfuels-core (no mocks) so it guards the
    equivalence treevox now relies on for its per-bin EDT hoist.
    """

    def test_sample_occupancy_matches_wrapper_on_real_crown(self):
        tree = voxelize.build_tree(
            pd.Series(
                {
                    "fia_species_code": 122,
                    "fia_status_code": 1,
                    "dbh": 25.0,
                    "height": 18.0,
                    "crown_ratio": 0.5,
                    "x": 5.0,
                    "y": 5.0,
                }
            ),
            base_source_config(),
        )
        mask = voxelize.discretize_crown_profile(tree, 0.5, 0.5)
        assert np.count_nonzero(mask) > 0

        field, n = compute_crown_probability_field(mask, alpha=0.5, beta=0.5)
        for seed in (1, 7, 42, 2**31 - 2):
            hoisted = sample_occupancy(mask, field, n, seed=seed)
            wrapper = sample_occupied_cells(mask, alpha=0.5, beta=0.5, seed=seed)
            assert np.array_equal(hoisted, wrapper)


# voxelize_chunk (mocked cache)


class TestVoxelizeChunk:
    def _dims_and_buffers(
        self,
        keys=(
            "volume_fraction",
            "bulk_density.foliage.live",
            "spcd",
            "tree_id",
            "savr.foliage",
            "fuel_moisture.live",
        ),
    ):
        dims = voxelize.compute_grid_dimensions(
            fake_domain(0, 0, 20, 20), fake_tree_df(height=30.0), (1.0, 1.0, 1.0)
        )
        buffers = {}
        from treevox.storage import BAND_SPECS

        for k in keys:
            dtype, fill = BAND_SPECS[k]
            buffers[k] = np.full(
                (dims["nz"], dims["ny"], dims["nx"]), fill, dtype=dtype
            )
        return dims, buffers

    def _one_tree_df(self, species=131, x=15, y=15, height=3.0, tree_id=0):
        df = pd.DataFrame(
            {
                "fia_species_code": [species],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [height],
                "crown_ratio": [0.5],
                "x": [x],
                "y": [y],
                "tree_id": [tree_id],
            }
        )
        df["_cache_key"] = 0
        return df

    def _deterministic_cache(
        self,
        shape=(2, 3, 3),
        value=1.0,
        crown_base_height=1.5,
        foliage_sav=1600.0,
        specific_leaf_area=5.0,
        species_code=131,
    ):
        return {
            0: voxelize.CacheEntry(
                biomass_arrays=[np.full(shape, value, dtype="float32")],
                crown_base_height=crown_base_height,
                foliage_sav=foliage_sav,
                specific_leaf_area=specific_leaf_area,
                species_code=species_code,
            )
        }

    def test_single_tree_populates_all_requested_bands(self):
        dims, buffers = self._dims_and_buffers()
        df = self._one_tree_df()
        cache = self._deterministic_cache()
        voxelize.voxelize_chunk(
            df,
            buffers,
            cache,
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].sum() > 0
        assert buffers["bulk_density.foliage.live"].sum() > 0
        assert (buffers["spcd"] == 131).any()
        assert (buffers["tree_id"] == 0).any()
        assert buffers["fuel_moisture.live"].max() == 100.0
        assert buffers["savr.foliage"].max() > 0

    def test_overwrite_bands_take_taller_trees_value(self):
        """Trees are sorted height-ASC before dispatch; last writer = tallest.

        Two species → two cache_keys (as `compute_cache_keys` would produce),
        each with its own species_code in the cache entry.
        """
        dims, buffers = self._dims_and_buffers(keys=("spcd", "tree_id"))
        df = pd.DataFrame(
            {
                "fia_species_code": [131, 202],
                "fia_status_code": [1, 1],
                "dbh": [20.0, 20.0],
                "height": [5.0, 10.0],
                "crown_ratio": [0.5, 0.5],
                "x": [15.0, 15.0],
                "y": [15.0, 15.0],
                "tree_id": [0, 1],
                "_cache_key": [0, 1],
            }
        )
        arr = np.ones((2, 3, 3), dtype="float32")
        cache = {
            0: voxelize.CacheEntry(
                biomass_arrays=[arr],
                crown_base_height=1.5,
                foliage_sav=1600.0,
                specific_leaf_area=5.0,
                species_code=131,
            ),
            1: voxelize.CacheEntry(
                biomass_arrays=[arr],
                crown_base_height=1.5,
                foliage_sav=1600.0,
                specific_leaf_area=5.0,
                species_code=202,
            ),
        }
        voxelize.voxelize_chunk(
            df,
            buffers,
            cache,
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert (buffers["spcd"] == 202).any()
        assert (buffers["tree_id"] == 1).any()

    def test_accumulative_bands_sum_across_trees(self):
        dims, buffers = self._dims_and_buffers(
            keys=("volume_fraction", "bulk_density.foliage.live")
        )
        df = pd.DataFrame(
            {
                "fia_species_code": [131, 131],
                "fia_status_code": [1, 1],
                "dbh": [20.0, 20.0],
                "height": [5.0, 5.0],
                "crown_ratio": [0.5, 0.5],
                "x": [15.0, 15.0],
                "y": [15.0, 15.0],
                "tree_id": [0, 1],
                "_cache_key": [0, 0],
            }
        )
        cache = self._deterministic_cache(shape=(2, 3, 3), value=1.0)
        voxelize.voxelize_chunk(
            df,
            buffers,
            cache,
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].max() == pytest.approx(2.0)
        assert buffers["bulk_density.foliage.live"].max() == pytest.approx(2.0)

    def test_component_state_splits_voxelized_density_bands(self):
        dims, buffers = self._dims_and_buffers(
            keys=("bulk_density.foliage.live", "bulk_density.foliage.dead")
        )
        df = self._one_tree_df()
        cfg = base_source_config()
        cfg["biomass_source"]["component_states"] = {
            "foliage": {"live": 0.25, "dead": 0.75}
        }
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(value=2.0),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            cfg,
            np.random.default_rng(0),
        )

        assert buffers["bulk_density.foliage.live"].max() == pytest.approx(0.5)
        assert buffers["bulk_density.foliage.dead"].max() == pytest.approx(1.5)

    def test_subset_of_bands_only(self):
        dims, buffers = self._dims_and_buffers(keys=("volume_fraction",))
        df = self._one_tree_df()
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].sum() > 0
        assert list(buffers.keys()) == ["volume_fraction"]

    def test_fuel_moisture_uses_source_config_value(self):
        dims, buffers = self._dims_and_buffers(keys=("fuel_moisture.live",))
        df = self._one_tree_df()
        cfg = base_source_config()
        cfg["moisture_model"] = {"live": {"method": "uniform", "value": 75.0}}
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            cfg,
            np.random.default_rng(0),
        )
        assert buffers["fuel_moisture.live"].max() == 75.0

    def test_fuel_moisture_dead_uses_source_config_value(self):
        dims, buffers = self._dims_and_buffers(keys=("fuel_moisture.dead",))
        df = self._one_tree_df()
        cfg = base_source_config()
        cfg["moisture_model"] = {"dead": {"method": "uniform", "value": 9.0}}
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            cfg,
            np.random.default_rng(0),
        )
        assert buffers["fuel_moisture.dead"].max() == 9.0

    def test_empty_trees_leaves_buffers_untouched(self):
        dims, buffers = self._dims_and_buffers(keys=("volume_fraction",))
        voxelize.voxelize_chunk(
            pd.DataFrame(
                {
                    "fia_species_code": [],
                    "fia_status_code": [],
                    "dbh": [],
                    "height": [],
                    "crown_ratio": [],
                    "x": [],
                    "y": [],
                    "tree_id": [],
                    "_cache_key": [],
                }
            ),
            buffers,
            {},
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].sum() == 0

    def test_tree_outside_chunk_is_skipped(self):
        dims, buffers = self._dims_and_buffers(keys=("volume_fraction",))
        df = self._one_tree_df(x=1000.0, y=1000.0)
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].sum() == 0

    def test_tree_near_edge_gets_clipped(self):
        dims, buffers = self._dims_and_buffers(keys=("volume_fraction",))
        df = self._one_tree_df(x=dims["x_origin"] + 1.0, y=dims["y_origin"] - 1.0)
        voxelize.voxelize_chunk(
            df,
            buffers,
            self._deterministic_cache(shape=(2, 3, 3)),
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )
        assert buffers["volume_fraction"].sum() > 0

    def test_lad_is_computed(self):
        dims, buffers = self._dims_and_buffers(
            keys=("volume_fraction", "bulk_density.foliage.live", "leaf_area_density")
        )
        df = self._one_tree_df()
        cache = self._deterministic_cache(
            shape=(2, 3, 3), value=5.0, specific_leaf_area=5.0
        )

        voxelize.voxelize_chunk(
            df,
            buffers,
            cache,
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )

        assert buffers["bulk_density.foliage.live"].sum() > 0
        assert buffers["leaf_area_density"].min() >= 0
        assert buffers["leaf_area_density"].max() == pytest.approx(25.0)

    def test_lad_is_not_computed(self):
        dims, buffers = self._dims_and_buffers(
            keys=("volume_fraction", "bulk_density.foliage.live")
        )
        df = self._one_tree_df()
        cache = self._deterministic_cache(
            shape=(2, 3, 3), value=5.0, specific_leaf_area=5.0
        )

        voxelize.voxelize_chunk(
            df,
            buffers,
            cache,
            0,
            0,
            dims["hr"],
            dims["vr"],
            dims["x_origin"],
            dims["y_origin"],
            base_source_config(),
            np.random.default_rng(0),
        )

        assert buffers["bulk_density.foliage.live"].sum() > 0
        # LAD is skipped entirely when the band isn't requested, so no
        # buffer is allocated for it.
        assert "leaf_area_density" not in buffers


# Geometric placement helpers


class TestTreeCellIndices:
    def test_stem_at_origin_maps_to_cell_zero(self):
        assert voxelize._tree_cell_indices(
            x=0.0, y=100.0, x_origin=0.0, y_origin=100.0, hr=1.0
        ) == (0, 0)

    def test_cell_east_of_origin(self):
        assert voxelize._tree_cell_indices(
            x=1.5, y=100.0, x_origin=0.0, y_origin=100.0, hr=1.0
        ) == (1, 0)

    def test_cell_south_of_origin(self):
        """y_origin is the NORTH edge; rows increase southward."""
        assert voxelize._tree_cell_indices(
            x=0.0, y=98.5, x_origin=0.0, y_origin=100.0, hr=1.0
        ) == (0, 1)

    def test_stem_on_boundary_lands_in_higher_index_cell(self):
        """Floor rounds exactly-on-boundary stems into the east/south cell."""
        assert voxelize._tree_cell_indices(
            x=1.0, y=99.0, x_origin=0.0, y_origin=100.0, hr=1.0
        ) == (1, 1)

    def test_coarser_resolution(self):
        assert voxelize._tree_cell_indices(
            x=6.0, y=94.0, x_origin=0.0, y_origin=100.0, hr=3.0
        ) == (2, 2)

    def test_stem_west_of_origin_returns_negative(self):
        """Clamping is the caller's job — the helper is pure coord math."""
        col, row = voxelize._tree_cell_indices(
            x=-0.5, y=100.0, x_origin=0.0, y_origin=100.0, hr=1.0
        )
        assert col == -1
        assert row == 0

    def test_stem_north_of_origin_returns_negative_row(self):
        _, row = voxelize._tree_cell_indices(
            x=0.0, y=100.5, x_origin=0.0, y_origin=100.0, hr=1.0
        )
        assert row == -1


class TestClip1D:
    def test_fully_inside(self):
        buf, src = voxelize._clip_1d(start=2, span=3, dim=10)
        assert (buf.start, buf.stop) == (2, 5)
        assert (src.start, src.stop) == (0, 3)

    def test_overhang_left(self):
        buf, src = voxelize._clip_1d(start=-2, span=5, dim=10)
        assert (buf.start, buf.stop) == (0, 3)
        assert (src.start, src.stop) == (2, 5)

    def test_overhang_right(self):
        buf, src = voxelize._clip_1d(start=8, span=5, dim=10)
        assert (buf.start, buf.stop) == (8, 10)
        assert (src.start, src.stop) == (0, 2)

    def test_fully_outside_left(self):
        assert voxelize._clip_1d(start=-5, span=3, dim=10) is None

    def test_fully_outside_right(self):
        assert voxelize._clip_1d(start=10, span=3, dim=10) is None

    def test_touching_zero_edge_exactly(self):
        """span=5 starting at -5 → end=0 → fully outside (half-open)."""
        assert voxelize._clip_1d(start=-5, span=5, dim=10) is None

    def test_touching_dim_edge_exactly(self):
        """start=dim is fully outside; start=dim-1 keeps one cell."""
        assert voxelize._clip_1d(start=10, span=5, dim=10) is None
        buf, src = voxelize._clip_1d(start=9, span=5, dim=10)
        assert (buf.start, buf.stop) == (9, 10)
        assert (src.start, src.stop) == (0, 1)

    def test_span_exceeds_dim_both_sides(self):
        """span=20 starting at -5 against dim=10 → [0, 10) from src[5:15)."""
        buf, src = voxelize._clip_1d(start=-5, span=20, dim=10)
        assert (buf.start, buf.stop) == (0, 10)
        assert (src.start, src.stop) == (5, 15)

    def test_slices_preserve_array_write_equivalence(self):
        """buf[buf_slice] = src[src_slice] yields the expected buffer state."""
        buf = np.zeros(10, dtype="int32")
        src = np.arange(5, dtype="int32")  # [0, 1, 2, 3, 4]
        result = voxelize._clip_1d(start=-2, span=5, dim=10)
        assert result is not None
        buf_slice, src_slice = result
        buf[buf_slice] = src[src_slice]
        # src[2:5] = [2, 3, 4] lands in buf[0:3].
        assert list(buf) == [2, 3, 4, 0, 0, 0, 0, 0, 0, 0]


class TestPlaceBiomass:
    def _args(self, **overrides):
        defaults = dict(
            abs_col=10,
            abs_row=10,
            chunk_x_start=0,
            chunk_y_start=0,
            crown_base_height=3.0,
            biomass_shape=(4, 3, 3),  # (nz, ny, nx)
            buffer_shape=(10, 20, 20),
            vr=1.0,
        )
        defaults.update(overrides)
        return defaults

    def test_centered_fully_inside(self):
        placement = voxelize._place_biomass(**self._args())
        assert placement is not None
        buf_slices, src_slices = placement
        # crown_base_height=3.0 / vr=1.0 → z starts at 3, span 4 → buf z=[3,7)
        assert (buf_slices[0].start, buf_slices[0].stop) == (3, 7)
        # row_cell=10, b_ny=3, ny//2=1 → y_start=9, span 3 → buf y=[9,12)
        assert (buf_slices[1].start, buf_slices[1].stop) == (9, 12)
        # col_cell=10, b_nx=3, nx//2=1 → x_start=9, span 3 → buf x=[9,12)
        assert (buf_slices[2].start, buf_slices[2].stop) == (9, 12)
        # fully inside → src slices are full-span
        assert all(s.start == 0 for s in src_slices)

    def test_overhang_north(self):
        """abs_row near 0 → crown overhangs the north (y=0) edge."""
        placement = voxelize._place_biomass(
            **self._args(abs_row=0, biomass_shape=(4, 5, 5))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        # row_cell=0, b_ny=5, ny//2=2 → y_start=-2 → buf y=[0,3), src y=[2,5)
        assert (buf_slices[1].start, buf_slices[1].stop) == (0, 3)
        assert (src_slices[1].start, src_slices[1].stop) == (2, 5)

    def test_overhang_south(self):
        placement = voxelize._place_biomass(
            **self._args(abs_row=19, biomass_shape=(4, 5, 5), buffer_shape=(10, 20, 20))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        # row_cell=19, b_ny=5, ny//2=2 → y_start=17, y_end=22 → buf y=[17,20), src y=[0,3)
        assert (buf_slices[1].start, buf_slices[1].stop) == (17, 20)
        assert (src_slices[1].start, src_slices[1].stop) == (0, 3)

    def test_overhang_west(self):
        placement = voxelize._place_biomass(
            **self._args(abs_col=0, biomass_shape=(4, 5, 5))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        assert (buf_slices[2].start, buf_slices[2].stop) == (0, 3)
        assert (src_slices[2].start, src_slices[2].stop) == (2, 5)

    def test_overhang_east(self):
        placement = voxelize._place_biomass(
            **self._args(abs_col=19, biomass_shape=(4, 5, 5))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        assert (buf_slices[2].start, buf_slices[2].stop) == (17, 20)
        assert (src_slices[2].start, src_slices[2].stop) == (0, 3)

    def test_overhang_top(self):
        """Crown extends above nz: z_end > nz."""
        placement = voxelize._place_biomass(
            **self._args(crown_base_height=8.0, biomass_shape=(4, 3, 3))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        # z_start=8, span=4 → z_end=12 > nz=10 → buf z=[8,10), src z=[0,2)
        assert (buf_slices[0].start, buf_slices[0].stop) == (8, 10)
        assert (src_slices[0].start, src_slices[0].stop) == (0, 2)

    def test_overhang_bottom_negative_crown_base(self):
        """Crown bottom negative (shouldn't happen physically, but test math)."""
        placement = voxelize._place_biomass(
            **self._args(crown_base_height=-2.0, biomass_shape=(4, 3, 3))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        # z_start=-2, span=4 → z_end=2 → buf z=[0,2), src z=[2,4)
        assert (buf_slices[0].start, buf_slices[0].stop) == (0, 2)
        assert (src_slices[0].start, src_slices[0].stop) == (2, 4)

    def test_corner_overhang_two_faces(self):
        """NW corner: overhangs both north AND west simultaneously."""
        placement = voxelize._place_biomass(
            **self._args(abs_row=0, abs_col=0, biomass_shape=(4, 5, 5))
        )
        assert placement is not None
        buf_slices, src_slices = placement
        assert (buf_slices[1].start, buf_slices[1].stop) == (0, 3)
        assert (src_slices[1].start, src_slices[1].stop) == (2, 5)
        assert (buf_slices[2].start, buf_slices[2].stop) == (0, 3)
        assert (src_slices[2].start, src_slices[2].stop) == (2, 5)

    def test_fully_outside_returns_none(self):
        """Stem far east of buffer → None."""
        placement = voxelize._place_biomass(**self._args(abs_col=100))
        assert placement is None

    def test_fully_below_buffer_returns_none(self):
        """Crown entirely above nz → None."""
        placement = voxelize._place_biomass(**self._args(crown_base_height=100.0))
        assert placement is None

    def test_chunk_offset_translates_coords(self):
        """Non-zero chunk_{x,y}_start shifts stem into local frame."""
        placement = voxelize._place_biomass(
            **self._args(
                abs_col=110,
                abs_row=110,
                chunk_x_start=100,
                chunk_y_start=100,
                biomass_shape=(4, 3, 3),
            )
        )
        assert placement is not None
        buf_slices, _ = placement
        # local col = 110-100 = 10 → x_start = 10 - 1 = 9
        assert (buf_slices[1].start, buf_slices[1].stop) == (9, 12)
        assert (buf_slices[2].start, buf_slices[2].stop) == (9, 12)

    def test_vr_scales_z_placement(self):
        """crown_base_height / vr → z_start."""
        placement = voxelize._place_biomass(
            **self._args(crown_base_height=6.0, vr=2.0, biomass_shape=(2, 3, 3))
        )
        assert placement is not None
        buf_slices, _ = placement
        # z_start = int(6.0 / 2.0) = 3
        assert buf_slices[0].start == 3


class TestApplyBands:
    def _buffers(self, keys, shape=(4, 4, 4)):
        from treevox.storage import BAND_SPECS

        return {
            k: np.full(shape, BAND_SPECS[k][1], dtype=BAND_SPECS[k][0]) for k in keys
        }

    def _full_slice(self, shape):
        return tuple(slice(0, n) for n in shape)

    def _moisture(self):
        return {"live": 100.0, "dead": 10.0}

    def _component_state(self):
        return {"live": 1.0, "dead": 0.0}

    def test_volume_fraction_accumulates_from_mask(self):
        shape = (2, 3, 3)
        bufs = self._buffers(("volume_fraction",), shape=shape)
        biomass = np.ones(shape, dtype="float32")
        lad = np.ones(shape, dtype="float32")
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            species_code=131,
            foliage_sav=2000.0,
            tree_id=7,
            moisture_values=self._moisture(),
            component_state=self._component_state(),
        )
        assert bufs["volume_fraction"].sum() == biomass.size

    def test_volume_fraction_zero_biomass_does_nothing(self):
        shape = (2, 3, 3)
        bufs = self._buffers(("volume_fraction",), shape=shape)
        biomass = np.zeros(shape, dtype="float32")
        lad = np.zeros(shape, dtype="float32")
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            131,
            2000.0,
            0,
            self._moisture(),
            self._component_state(),
        )
        assert bufs["volume_fraction"].sum() == 0

    def test_bulk_density_sums_biomass(self):
        shape = (2, 3, 3)
        bufs = self._buffers(("bulk_density.foliage.live",), shape=shape)
        biomass = np.full(shape, 0.5, dtype="float32")
        lad = np.ones(shape, dtype="float32")
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            131,
            2000.0,
            0,
            self._moisture(),
            self._component_state(),
        )
        assert bufs["bulk_density.foliage.live"].sum() == pytest.approx(
            0.5 * biomass.size
        )

    def test_bulk_density_splits_live_dead_biomass(self):
        shape = (2, 3, 3)
        bufs = self._buffers(
            ("bulk_density.foliage.live", "bulk_density.foliage.dead"), shape=shape
        )
        biomass = np.full(shape, 2.0, dtype="float32")
        lad = np.full(shape, 2.0, dtype="float32")
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            131,
            2000.0,
            0,
            self._moisture(),
            {"live": 0.25, "dead": 0.75},
        )

        assert bufs["bulk_density.foliage.live"].sum() == pytest.approx(
            0.5 * biomass.size
        )
        assert bufs["bulk_density.foliage.dead"].sum() == pytest.approx(
            1.5 * biomass.size
        )

    def test_overwrite_bands_written_where_mask_nonzero(self):
        shape = (2, 3, 3)
        bufs = self._buffers(
            ("savr.foliage", "fuel_moisture.live", "spcd", "tree_id"), shape=shape
        )
        biomass = np.zeros(shape, dtype="float32")
        biomass[0, 1, 1] = 1.0  # single voxel

        lad = np.zeros(shape, dtype="float32")
        lad[0, 1, 1] = 1.0  # single voxel

        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            species_code=131,
            foliage_sav=2000.0,
            tree_id=7,
            moisture_values=self._moisture(),
            component_state=self._component_state(),
        )
        assert bufs["savr.foliage"][0, 1, 1] == 2000.0
        assert bufs["fuel_moisture.live"][0, 1, 1] == 100.0
        assert bufs["spcd"][0, 1, 1] == 131
        assert bufs["tree_id"][0, 1, 1] == 7
        # fill values preserved elsewhere
        assert bufs["spcd"].sum() == 131
        assert (bufs["tree_id"] == -1).sum() == biomass.size - 1

    def test_overwrite_second_tree_replaces_first(self):
        """Last-writer-wins semantics for overwrite bands."""
        shape = (1, 2, 2)
        bufs = self._buffers(("spcd", "tree_id"), shape=shape)
        biomass = np.ones(shape, dtype="float32")
        lad = np.ones(shape, dtype="float32")

        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            131,
            2000.0,
            0,
            self._moisture(),
            self._component_state(),
        )
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            202,
            2500.0,
            1,
            self._moisture(),
            self._component_state(),
        )
        assert (bufs["spcd"] == 202).all()
        assert (bufs["tree_id"] == 1).all()

    def test_accumulate_second_tree_sums_with_first(self):
        shape = (1, 2, 2)
        bufs = self._buffers(
            ("volume_fraction", "bulk_density.foliage.live"), shape=shape
        )
        biomass = np.ones(shape, dtype="float32")
        lad = np.ones(shape, dtype="float32")

        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            131,
            2000.0,
            0,
            self._moisture(),
            self._component_state(),
        )
        voxelize._apply_bands(
            bufs,
            self._full_slice(shape),
            biomass,
            lad,
            202,
            2500.0,
            1,
            self._moisture(),
            self._component_state(),
        )
        assert (bufs["volume_fraction"] == 2.0).all()
        assert (bufs["bulk_density.foliage.live"] == 2.0).all()

    def test_subset_of_bands_only_touches_those(self):
        """A caller requesting just volume_fraction shouldn't error on missing keys."""
        shape = (1, 2, 2)
        bufs = self._buffers(("volume_fraction",), shape=shape)
        biomass = np.ones(shape, dtype="float32")
        lad = np.ones(shape, dtype="float32")
        voxelize._apply_bands(
            buffers=bufs,
            buf_slices=self._full_slice(shape),
            biomass_clip=biomass,
            lad_clip=lad,
            species_code=131,
            foliage_sav=2000.0,
            tree_id=0,
            moisture_values=self._moisture(),
            component_state=self._component_state(),
        )
        assert list(bufs.keys()) == ["volume_fraction"]
        assert bufs["volume_fraction"].sum() == biomass.size

    def test_partial_slice_only_writes_within(self):
        """buf_slices narrower than buf shape → only that region is touched."""
        bufs = self._buffers(("spcd",), shape=(2, 4, 4))
        biomass = np.ones((2, 2, 2), dtype="float32")
        lad = np.ones((2, 2, 2), dtype="float32")
        sub = (slice(0, 2), slice(1, 3), slice(1, 3))

        voxelize._apply_bands(
            bufs,
            sub,
            biomass,
            lad_clip=lad,
            species_code=131,
            foliage_sav=0,
            tree_id=0,
            moisture_values={},
            component_state=self._component_state(),
        )
        # Inside sub: written to 131.
        assert (bufs["spcd"][sub] == 131).all()
        # Outside sub: untouched (fill value 0 per BAND_SPECS).
        mask = np.ones(bufs["spcd"].shape, dtype=bool)
        mask[sub] = False
        assert (bufs["spcd"][mask] == 0).all()
