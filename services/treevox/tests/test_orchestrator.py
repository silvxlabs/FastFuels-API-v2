"""Unit tests for treevox.orchestrator — dispatch + voxelize_inventory + stages.

Each stage of `voxelize_inventory` is testable in isolation thanks to the
decomposition (_load_inventory_dataframe, _plan_grid_layout,
_prepare_tree_chunks, _build_payloads, _process_batch, _run_voxelization_batches,
_build_voxelization_result). Full-flow tests additionally verify the stages
wire together correctly with all I/O + mp mocked out.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from treevox import orchestrator
from treevox.errors import ProcessingError
from treevox.orchestrator import (
    DEFAULT_MAX_WORKERS,
    GridLayout,
    VoxelizationResult,
    _pick_worker_count,
    dispatch_handler,
    voxelize_inventory,
)

# Fixtures


def _fake_domain():
    return SimpleNamespace(
        total_bounds=np.array([0.0, 0.0, 100.0, 100.0]),
        crs="EPSG:32610",
    )


def _base_grid(bands=None):
    return {
        "id": "g1",
        "domain_id": "d1",
        "source": {
            "name": "inventory",
            "source_inventory_id": "inv1",
            "resolution": (1.0, 1.0, 1.0),
            "crown_profile_model": "purves",
            "biomass_model": "nsvb",
            "biomass_column": None,
            "moisture_model": {"method": "uniform", "live": 100.0},
        },
        "bands": bands or [{"key": "volume_fraction"}],
    }


def _sample_df(n=1, height=5.0):
    return pd.DataFrame(
        {
            "x": [50.0] * n,
            "y": [50.0] * n,
            "fia_species_code": [131] * n,
            "fia_status_code": [1] * n,
            "dbh": [20.0] * n,
            "height": [height] * n,
            "crown_ratio": [0.5] * n,
        }
    )


# dispatch_handler


class TestDispatchHandler:
    @patch("treevox.orchestrator.voxelize_inventory")
    def test_inventory_routes_to_voxelize(self, mock_voxelize):
        mock_voxelize.return_value = "result"
        grid = {"source": {"name": "inventory"}}
        result = dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert result == "result"
        mock_voxelize.assert_called_once()

    def test_unknown_source_raises_processing_error(self):
        grid = {"source": {"name": "lidar"}}
        with pytest.raises(ProcessingError) as exc:
            dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert exc.value.code == "UNKNOWN_SOURCE"


# _pick_worker_count


class TestPickWorkerCount:
    def test_returns_at_least_one(self):
        assert _pick_worker_count() >= 1

    def test_capped_at_default_max(self, monkeypatch):
        # Pretend we have many CPUs and tons of memory.
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False
        )

        class FakeFile:
            def __enter__(self):
                return iter(["MemAvailable: 10485760 kB\n"])  # 10 GB

            def __exit__(self, *a):
                pass

        monkeypatch.setattr("builtins.open", lambda *a, **kw: FakeFile())
        assert _pick_worker_count() <= DEFAULT_MAX_WORKERS


# Individual stage tests — these are the main payoff of the decomposition:
# each stage is a small, pure-ish function we can exercise directly.


class TestLoadInventoryDataframe:
    @patch("treevox.orchestrator.download_inventory")
    def test_filters_to_live_and_assigns_tree_ids(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "fia_species_code": [131, 131],
                "fia_status_code": [1, 2],  # one live, one dead
                "dbh": [20.0, 20.0],
                "height": [15.0, 15.0],
                "crown_ratio": [0.4, 0.4],
            }
        )
        source = {"source_inventory_id": "inv1"}
        df = orchestrator._load_inventory_dataframe(source, lambda *a, **k: None)
        assert len(df) == 1
        assert list(df["tree_id"]) == [0]

    @patch("treevox.orchestrator.download_inventory")
    def test_empty_after_filter_raises_empty_inventory(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "fia_species_code": [131],
                "fia_status_code": [2],  # dead → filtered out
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.4],
            }
        )
        with pytest.raises(ProcessingError) as exc:
            orchestrator._load_inventory_dataframe(
                {"source_inventory_id": "inv1"}, lambda *a, **k: None
            )
        assert exc.value.code == "EMPTY_INVENTORY"


class TestPlanGridLayout:
    def test_builds_chunk_locations_in_block_order(self):
        layout = orchestrator._plan_grid_layout(
            _base_grid(), _fake_domain(), _sample_df(height=5.0)
        )
        assert isinstance(layout, GridLayout)
        assert layout.chunk_xy > 0
        assert layout.chunk_shape == (
            layout.dims["nz"],
            layout.chunk_xy,
            layout.chunk_xy,
        )
        assert all(isinstance(loc, tuple) for loc in layout.chunk_locations)

    def test_chunk_xy_clamped_to_grid_when_domain_tiny(self):
        """A small domain with a larger nominal chunk should clamp to nx/ny."""
        tiny_domain = SimpleNamespace(
            total_bounds=np.array([0.0, 0.0, 10.0, 10.0]),
            crs="EPSG:32610",
        )
        layout = orchestrator._plan_grid_layout(
            _base_grid(), tiny_domain, _sample_df(height=5.0)
        )
        assert layout.chunk_xy <= layout.dims["nx"]
        assert layout.chunk_xy <= layout.dims["ny"]

    def test_invalid_resolution_maps_to_processing_error(self):
        grid = _base_grid()
        grid["source"]["resolution"] = (0.0, 0.0, 1.0)
        with pytest.raises(ProcessingError) as exc:
            orchestrator._plan_grid_layout(grid, _fake_domain(), _sample_df())
        assert exc.value.code == "INVALID_RESOLUTION"


class TestPrepareTreeChunks:
    def test_attaches_cache_key_and_chunk_columns(self):
        df = _sample_df(n=3, height=10.0)
        layout = orchestrator._plan_grid_layout(_base_grid(), _fake_domain(), df)
        out = orchestrator._prepare_tree_chunks(df, layout)
        assert "_cache_key" in out.columns
        assert "row_chunk" in out.columns
        assert "col_chunk" in out.columns
        assert len(out) == 3

    def test_does_not_mutate_input(self):
        df = _sample_df(n=2)
        layout = orchestrator._plan_grid_layout(_base_grid(), _fake_domain(), df)
        _ = orchestrator._prepare_tree_chunks(df, layout)
        assert "_cache_key" not in df.columns


class TestBuildVoxelizationResult:
    def test_shape_and_georeference_populated(self):
        df = _sample_df(height=5.0)
        layout = orchestrator._plan_grid_layout(_base_grid(), _fake_domain(), df)
        result = orchestrator._build_voxelization_result(layout, "gs://x/g1")
        assert isinstance(result, VoxelizationResult)
        assert result.gcs_path == "gs://x/g1"
        assert result.georeference["shape"] == [
            layout.dims["nz"],
            layout.dims["ny"],
            layout.dims["nx"],
        ]
        assert result.chunk_shape == list(layout.chunk_shape)


class TestBuildPayloads:
    def test_payload_per_chunk_with_expected_keys(self):
        df = _sample_df(n=4, height=5.0)
        grid = _base_grid()
        layout = orchestrator._plan_grid_layout(grid, _fake_domain(), df)
        df_prepared = orchestrator._prepare_tree_chunks(df, layout)

        batch = layout.chunk_locations[:1]
        dims = layout.dims
        union_y, union_x = (
            slice(0, dims["ny"]),
            slice(0, dims["nx"]),
        )
        union_ds = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((dims["nz"], dims["ny"], dims["nx"]), dtype="float32"),
                )
            }
        )

        payloads = orchestrator._build_payloads(
            batch, union_ds, union_y, union_x, df_prepared, layout, grid["source"], "g1"
        )
        assert len(payloads) == 1
        p = payloads[0]
        for key in (
            "chunk_location",
            "buffers",
            "trees",
            "hr",
            "vr",
            "x_origin",
            "y_origin",
            "source_config",
            "chunk_y_start",
            "chunk_x_start",
            "y_slice",
            "x_slice",
            "rng_seed",
        ):
            assert key in p, f"missing key {key}"
        assert "volume_fraction" in p["buffers"]


# Full-flow integration tests for voxelize_inventory with mocks.


class TestVoxelizeInventoryFlow:
    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.download_inventory")
    def test_happy_path_calls_expected_stages(
        self,
        mock_download,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_download.return_value = _sample_df(height=5.0)
        ds = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )
        mock_read.return_value = ds
        mock_merge.return_value = ds

        with patch("treevox.orchestrator.multiprocessing.get_context") as mock_get_ctx:
            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False

            def fake_map(fn, payloads):
                return [
                    {
                        "chunk_location": p["chunk_location"],
                        "buffers": p["buffers"],
                        "y_slice": p["y_slice"],
                        "x_slice": p["x_slice"],
                    }
                    for p in payloads
                ]

            fake_pool.map.side_effect = fake_map
            fake_ctx = MagicMock()
            fake_ctx.Pool.return_value = fake_pool
            mock_get_ctx.return_value = fake_ctx

            progress_calls = []

            def progress(msg, pct=None):
                progress_calls.append((msg, pct))

            result = voxelize_inventory(_base_grid(), _fake_domain(), progress)

        assert isinstance(result, VoxelizationResult)
        assert len(result.chunk_shape) == 3
        assert mock_init.call_count == 1
        mock_consolidate.assert_called_once()
        msgs = [m for m, _ in progress_calls]
        assert any("Loading" in m for m in msgs)
        assert any("Initializing" in m for m in msgs)
        assert any("Finalizing" in m for m in msgs)

    @patch("treevox.orchestrator.download_inventory")
    def test_empty_inventory_raises(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "fia_species_code": [131],
                "fia_status_code": [2],  # dead → filtered out
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.4],
            }
        )
        with pytest.raises(ProcessingError) as exc:
            voxelize_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)
        assert exc.value.code == "EMPTY_INVENTORY"

    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.download_inventory")
    def test_worker_error_surfaces_as_voxelization_failed(
        self,
        mock_download,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_download.return_value = _sample_df(height=5.0)
        mock_read.return_value = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )

        with patch("treevox.orchestrator.multiprocessing.get_context") as mock_get_ctx:
            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False
            fake_pool.map.return_value = [{"chunk_location": (0, 0), "error": "boom"}]
            fake_ctx = MagicMock()
            fake_ctx.Pool.return_value = fake_pool
            mock_get_ctx.return_value = fake_ctx

            with pytest.raises(ProcessingError) as exc:
                voxelize_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)
        assert exc.value.code == "VOXELIZATION_FAILED"


class TestPersistentPool:
    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.download_inventory")
    def test_pool_instantiated_exactly_once(
        self,
        mock_download,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        """A job with multiple batches must create Pool only once."""
        mock_download.return_value = _sample_df(height=5.0)
        mock_read.return_value = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )
        mock_merge.side_effect = lambda union_ds, *a, **kw: union_ds

        with (
            patch("treevox.orchestrator.voxelize.CHUNK_LENGTH_METERS", 20),
            patch("treevox.orchestrator.multiprocessing.get_context") as mock_get_ctx,
        ):
            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False

            def fake_map(fn, payloads):
                return [
                    {
                        "chunk_location": p["chunk_location"],
                        "buffers": p["buffers"],
                        "y_slice": p["y_slice"],
                        "x_slice": p["x_slice"],
                    }
                    for p in payloads
                ]

            fake_pool.map.side_effect = fake_map
            fake_ctx = MagicMock()
            fake_ctx.Pool.return_value = fake_pool
            mock_get_ctx.return_value = fake_ctx

            voxelize_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)

            assert fake_ctx.Pool.call_count == 1
            assert fake_pool.map.call_count > 1
