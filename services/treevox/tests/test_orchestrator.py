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
from treevox._worker import run as worker_run
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


def _base_grid(bands=None, seed=42):
    return {
        "id": "g1",
        "domain_id": "d1",
        "source": {
            "operation": "voxelize",
            "input": "inventory",
            "entity": "tree",
            "source_inventory_id": "inv1",
            "resolution": (1.0, 1.0, 1.0),
            "crown_profile_model": "purves",
            "biomass_source": {
                "type": "allometry",
                "equations": "nsvb",
                "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "moisture_model": {"live": {"method": "uniform", "value": 100.0}},
            "seed": seed,
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
        grid = {
            "source": {
                "operation": "voxelize",
                "input": "inventory",
                "entity": "tree",
            }
        }
        result = dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert result == "result"
        mock_voxelize.assert_called_once()

    def test_unknown_source_raises_processing_error(self):
        grid = {
            "source": {
                "operation": "voxelize",
                "input": "inventory",
                "entity": "lidar",
            }
        }
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
    """`read_inventory` pushes `fia_status_code == 1` into the parquet read,
    so the mocked return values here already contain only live trees."""

    @patch("treevox.orchestrator.read_inventory")
    def test_drops_null_rows_and_assigns_tree_ids(self, mock_read):
        mock_read.return_value = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "fia_species_code": [131, 131],
                "fia_status_code": [1, 1],
                "dbh": [20.0, None],  # second row dropped by drop_null_rows
                "height": [15.0, 15.0],
                "crown_ratio": [0.4, 0.4],
            }
        )
        source = _base_grid()["source"]
        df = orchestrator._load_inventory_dataframe(source, lambda *a, **k: None)
        assert len(df) == 1
        assert list(df["tree_id"]) == [0]

    @patch("treevox.orchestrator.read_inventory")
    def test_empty_after_filter_raises_empty_inventory(self, mock_read):
        mock_read.return_value = pd.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [None],  # the only row is null → drops to empty
                "height": [15.0],
                "crown_ratio": [0.4],
            }
        )
        with pytest.raises(ProcessingError) as exc:
            orchestrator._load_inventory_dataframe(
                _base_grid()["source"], lambda *a, **k: None
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

        chunk_indices = df_prepared.groupby(
            ["row_chunk", "col_chunk"], sort=False
        ).indices
        payloads = orchestrator._build_payloads(
            batch,
            union_ds,
            union_y,
            union_x,
            df_prepared,
            chunk_indices,
            layout,
            grid["source"],
            "g1",
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


# _chunk_relative_slices, _materialize_chunk_buffer, _chunk_rng_seed


class TestChunkRelativeSlices:
    def test_single_chunk_union_matches_chunk(self):
        """Union equals chunk halo → rel slices start at 0."""
        rel_y, rel_x = orchestrator._chunk_relative_slices(
            slice(990, 2010),
            slice(990, 2010),
            slice(990, 2010),
            slice(990, 2010),
        )
        assert (rel_y.start, rel_y.stop) == (0, 1020)
        assert (rel_x.start, rel_x.stop) == (0, 1020)

    def test_second_chunk_offset_from_union_origin(self):
        """A chunk whose halo starts 1000 cells into the union → rel.start=1000."""
        rel_y, rel_x = orchestrator._chunk_relative_slices(
            chunk_y=slice(1000, 2010),
            chunk_x=slice(0, 1010),
            union_y=slice(0, 2010),
            union_x=slice(0, 1010),
        )
        assert (rel_y.start, rel_y.stop) == (1000, 2010)
        assert (rel_x.start, rel_x.stop) == (0, 1010)

    def test_chunk_span_preserved(self):
        """rel.stop - rel.start == chunk.stop - chunk.start on each axis."""
        rel_y, rel_x = orchestrator._chunk_relative_slices(
            slice(500, 750),
            slice(100, 200),
            slice(0, 1000),
            slice(0, 500),
        )
        assert rel_y.stop - rel_y.start == 250
        assert rel_x.stop - rel_x.start == 100

    def test_chunk_y_before_union_raises(self):
        with pytest.raises(ProcessingError) as exc:
            orchestrator._chunk_relative_slices(
                slice(-5, 100),
                slice(0, 100),
                slice(0, 200),
                slice(0, 200),
            )
        assert exc.value.code == "BATCH_SLICE_MISMATCH"
        assert "y-slice" in exc.value.message

    def test_chunk_y_extends_past_union_raises(self):
        with pytest.raises(ProcessingError) as exc:
            orchestrator._chunk_relative_slices(
                slice(0, 300),
                slice(0, 100),
                slice(0, 200),
                slice(0, 200),
            )
        assert exc.value.code == "BATCH_SLICE_MISMATCH"

    def test_chunk_x_out_of_bounds_raises(self):
        """Containment is enforced on x-axis too."""
        with pytest.raises(ProcessingError) as exc:
            orchestrator._chunk_relative_slices(
                slice(0, 100),
                slice(-1, 100),
                slice(0, 200),
                slice(0, 200),
            )
        assert exc.value.code == "BATCH_SLICE_MISMATCH"
        assert "x-slice" in exc.value.message


class TestMaterializeChunkBuffer:
    def _union(self, shape=(2, 20, 20), keys=("volume_fraction", "tree_id")):
        data_vars = {}
        for k in keys:
            dtype, fill = orchestrator.storage.BAND_SPECS[k]
            data_vars[k] = (("z", "y", "x"), np.full(shape, fill, dtype=dtype))
        return xr.Dataset(data_vars)

    def test_shape_match_returns_coerced_copy(self):
        """Slice matches expected_shape → plain copy with band dtype."""
        union = self._union()
        buf = orchestrator._materialize_chunk_buffer(
            union,
            "volume_fraction",
            rel_y=slice(5, 15),
            rel_x=slice(5, 15),
            expected_shape=(2, 10, 10),
        )
        assert buf.shape == (2, 10, 10)
        assert buf.dtype == np.float32
        # Independent buffer — mutations do not propagate back to union.
        buf[0, 0, 0] = 1.0
        assert union["volume_fraction"].values[0, 5, 5] == 0.0

    def test_band_dtype_takes_precedence(self):
        """Output dtype comes from BAND_SPECS, not from the union variable."""
        union = xr.Dataset(
            {
                "tree_id": (
                    ("z", "y", "x"),
                    np.full((2, 10, 10), -1, dtype="int64"),  # mismatched dtype
                )
            }
        )
        buf = orchestrator._materialize_chunk_buffer(
            union,
            "tree_id",
            rel_y=slice(0, 10),
            rel_x=slice(0, 10),
            expected_shape=(2, 10, 10),
        )
        assert buf.dtype == np.int32
        assert (buf == -1).all()

    def test_fill_values_preserved_on_slice(self):
        """tree_id cells carry fill=-1 after slice/copy."""
        union = self._union(keys=("tree_id",))
        buf = orchestrator._materialize_chunk_buffer(
            union,
            "tree_id",
            rel_y=slice(0, 5),
            rel_x=slice(0, 5),
            expected_shape=(2, 5, 5),
        )
        assert (buf == -1).all()

    def test_smaller_slice_pads_with_fill_and_warns(self):
        """Union slice smaller than expected → trailing cells filled, warning logged.

        Uses a direct logger mock rather than caplog because the treevox
        package logger sets `propagate=False` in main.py, so warnings don't
        bubble to pytest's root-level caplog handler once main is imported.
        """
        union = self._union(shape=(2, 10, 10), keys=("volume_fraction",))
        with patch.object(orchestrator, "logger") as mock_logger:
            buf = orchestrator._materialize_chunk_buffer(
                union,
                "volume_fraction",
                rel_y=slice(0, 10),
                rel_x=slice(0, 10),
                expected_shape=(2, 12, 12),
            )
        assert buf.shape == (2, 12, 12)
        # Leading cells copied; trailing cells filled with 0.0.
        assert (buf[:, :10, :10] == 0.0).all()
        assert (buf[:, 10:, :] == 0.0).all()
        mock_logger.warning.assert_called_once()
        assert "smaller than expected" in mock_logger.warning.call_args[0][0]

    def test_padding_uses_band_specific_fill(self):
        """tree_id pads with -1, not 0."""
        union = self._union(shape=(2, 10, 10), keys=("tree_id",))
        buf = orchestrator._materialize_chunk_buffer(
            union,
            "tree_id",
            rel_y=slice(0, 10),
            rel_x=slice(0, 10),
            expected_shape=(2, 12, 12),
        )
        assert (buf[:, 10:, :] == -1).all()
        assert (buf[:, :, 10:] == -1).all()

    def test_larger_slice_raises_union_shape_mismatch(self):
        """Union slice larger than expected → refuse to truncate."""
        union = self._union(shape=(2, 20, 20), keys=("volume_fraction",))
        with pytest.raises(ProcessingError) as exc:
            orchestrator._materialize_chunk_buffer(
                union,
                "volume_fraction",
                rel_y=slice(0, 15),
                rel_x=slice(0, 15),
                expected_shape=(2, 10, 10),
            )
        assert exc.value.code == "UNION_SHAPE_MISMATCH"
        assert "refusing to truncate" in exc.value.message


class TestChunkRngSeed:
    def test_same_inputs_produce_same_seed(self):
        a = orchestrator._chunk_rng_seed(42, 3, 7)
        b = orchestrator._chunk_rng_seed(42, 3, 7)
        assert a == b

    def test_different_row_col_produce_different_seeds(self):
        base = orchestrator._chunk_rng_seed(42, 0, 0)
        assert base != orchestrator._chunk_rng_seed(42, 0, 1)
        assert base != orchestrator._chunk_rng_seed(42, 1, 0)

    def test_different_base_seed_produces_different_seed(self):
        """Same (row, col), different base seeds → different chunk seeds."""
        assert orchestrator._chunk_rng_seed(42, 0, 0) != (
            orchestrator._chunk_rng_seed(43, 0, 0)
        )

    def test_seed_is_uint32_range(self):
        for r, c in [(0, 0), (999, 999), (-1, -1)]:
            seed = orchestrator._chunk_rng_seed(12345, r, c)
            assert 0 <= seed < 2**32

    def test_deterministic_across_python_processes(self):
        """CRC32-based seed must NOT depend on PYTHONHASHSEED.

        Spawns a fresh Python process with a random hash seed and checks
        that it computes the same value. Guards against regressing to
        `hash()`-based derivation.
        """
        import subprocess
        import sys

        expected = orchestrator._chunk_rng_seed(42, 3, 7)
        script = (
            "from treevox.orchestrator import _chunk_rng_seed; "
            "print(_chunk_rng_seed(42, 3, 7))"
        )
        env = dict(os.environ, PYTHONHASHSEED="random")
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert int(out.stdout.strip()) == expected

    def test_feeds_default_rng_without_error(self):
        """The seed is accepted by `np.random.default_rng`."""
        seed = orchestrator._chunk_rng_seed(42, 3, 7)
        rng = np.random.default_rng(seed)
        assert rng.random() is not None


class TestResolveBaseSeed:
    def test_uses_source_seed_when_present(self):
        source = {"seed": 12345}
        assert orchestrator._resolve_base_seed(source, "grid-abc") == 12345

    def test_accepts_large_seed_values(self):
        """API allows any int; _resolve_base_seed must pass it through as int."""
        source = {"seed": 999_999_999}
        assert orchestrator._resolve_base_seed(source, "grid-abc") == 999_999_999

    def test_coerces_to_int(self):
        """String seed (e.g. from a JSON-without-type-coercion source) → int."""
        source = {"seed": "42"}
        assert orchestrator._resolve_base_seed(source, "grid-abc") == 42

    def test_missing_seed_falls_back_to_grid_id_hash(self):
        """Legacy grids pre-dating the seed field still work."""
        with patch.object(orchestrator, "logger") as mock_logger:
            result = orchestrator._resolve_base_seed({}, "legacy-grid-id")
        expected = orchestrator.zlib.crc32(b"legacy-grid-id")
        assert result == expected
        mock_logger.warning.assert_called_once()
        assert "no `seed` field" in mock_logger.warning.call_args[0][0]

    def test_null_seed_also_falls_back(self):
        """Explicit null (JSON → Python None) takes the fallback path, not a crash."""
        with patch.object(orchestrator, "logger"):
            result = orchestrator._resolve_base_seed({"seed": None}, "legacy-grid-id")
        assert result == orchestrator.zlib.crc32(b"legacy-grid-id")

    def test_fallback_is_deterministic_for_same_grid_id(self):
        """Legacy grids still re-run reproducibly against their own grid_id."""
        with patch.object(orchestrator, "logger"):
            a = orchestrator._resolve_base_seed({}, "g1")
            b = orchestrator._resolve_base_seed({}, "g1")
        assert a == b


# _build_payloads (integration with helpers)


class TestBuildPayloadsZeroTrees:
    def test_chunk_with_no_trees_gets_empty_frame_with_schema(self):
        """groupby().indices omits empty chunks → df.iloc[0:0] preserves columns."""
        df = _sample_df(n=1, height=5.0)
        grid = _base_grid()
        layout = orchestrator._plan_grid_layout(grid, _fake_domain(), df)
        # Mirror _load_inventory_dataframe's tree_id assignment so the test
        # frame has the same schema real workers receive.
        df = orchestrator.assign_tree_ids(df)
        df_prepared = orchestrator._prepare_tree_chunks(df, layout)

        batch = [layout.chunk_locations[0]]
        dims = layout.dims
        chunk_y, chunk_x = orchestrator.voxelize.chunk_slice(
            batch[0],
            dims["ny"],
            dims["nx"],
            layout.chunk_xy,
            overlap_cells=orchestrator.voxelize.OVERLAP_CELLS,
        )
        union_ds = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros(
                        (
                            dims["nz"],
                            chunk_y.stop - chunk_y.start,
                            chunk_x.stop - chunk_x.start,
                        ),
                        dtype="float32",
                    ),
                )
            }
        )
        # Empty chunk_indices dict — simulates a chunk whose (row, col) has no
        # trees, which is the `chunk_indices.get(...) is None` branch.
        chunk_indices: dict = {}

        payloads = orchestrator._build_payloads(
            batch,
            union_ds,
            chunk_y,
            chunk_x,
            df_prepared,
            chunk_indices,
            layout,
            grid["source"],
            "g1",
        )
        assert len(payloads) == 1
        assert len(payloads[0]["trees"]) == 0
        for col in ("fia_species_code", "dbh", "height", "tree_id", "_cache_key"):
            assert col in payloads[0]["trees"].columns


# Full-flow integration tests for voxelize_inventory with mocks.


class TestVoxelizeInventoryFlow:
    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.read_inventory")
    def test_happy_path_calls_expected_stages(
        self,
        mock_read_inv,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_read_inv.return_value = _sample_df(height=5.0)
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
        # init_store writes consolidated metadata directly (via
        # to_zarr(consolidated=True)); no separate end-of-job call.
        mock_consolidate.assert_not_called()
        msgs = [m for m, _ in progress_calls]
        assert any("Loading" in m for m in msgs)
        assert any("Initializing" in m for m in msgs)
        assert any("Finalizing" in m for m in msgs)

    @patch("treevox.orchestrator.read_inventory")
    def test_empty_inventory_raises(self, mock_read_inv):
        mock_read_inv.return_value = pd.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [None],  # null → dropped by drop_null_rows → empty
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
    @patch("treevox.orchestrator.read_inventory")
    def test_worker_error_surfaces_as_voxelization_failed(
        self,
        mock_read_inv,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_read_inv.return_value = _sample_df(height=5.0)
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

    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.read_inventory")
    def test_worker_not_implemented_error_preserves_code(
        self,
        mock_read_inv,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_read_inv.return_value = _sample_df(height=5.0)
        mock_read.return_value = xr.Dataset(
            {
                "bulk_density.fine.live": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )

        with patch("treevox.orchestrator.multiprocessing.get_context") as mock_get_ctx:
            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False
            fake_pool.map.return_value = [
                {
                    "chunk_location": (0, 0),
                    "error_code": "BIOMASS_COMPONENT_NOT_IMPLEMENTED",
                    "error_message": "Treevox does not yet support fine biomass distribution.",
                    "error": "NotImplementedError: fine",
                }
            ]
            fake_ctx = MagicMock()
            fake_ctx.Pool.return_value = fake_pool
            mock_get_ctx.return_value = fake_ctx

            grid = _base_grid(bands=[{"key": "bulk_density.fine.live"}])
            with pytest.raises(ProcessingError) as exc:
                voxelize_inventory(grid, _fake_domain(), lambda *a, **k: None)

        assert exc.value.code == "BIOMASS_COMPONENT_NOT_IMPLEMENTED"
        assert "fine" in exc.value.message


class TestPersistentPool:
    @patch("treevox.orchestrator.storage.consolidate_metadata")
    @patch("treevox.orchestrator.storage.write_union")
    @patch("treevox.orchestrator.storage.masked_merge")
    @patch("treevox.orchestrator.storage.read_union")
    @patch("treevox.orchestrator.storage.init_store")
    @patch("treevox.orchestrator.read_inventory")
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
            patch("treevox.orchestrator.voxelize.CHUNK_SIZE_HORIZONTAL", 20),
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


# End-to-end halo + masked_merge exercise for one batch.
#
# Runs the per-batch pipeline _build_payloads → _worker.run → masked_merge
# in-process (no Pool, no GCS). This is the main architectural invariant
# of v2 — trees near a chunk seam must render into the neighbor's halo
# and the merge must stitch those halo cells back into the union so the
# eventual zarr write covers both chunks. Unit-level because pure compute.


class TestHaloMergeAcrossChunks:
    def _patch_fastfuels(self, monkeypatch, biomass_shape=(2, 5, 5), biomass_value=1.0):
        """Replace fastfuels-core with deterministic stand-ins.

        Same pattern as TestBuildChunkCache — we care about the halo
        plumbing, not fastfuels internals.
        """
        from treevox import voxelize as vmod

        mask = np.ones(biomass_shape, dtype="float32")

        def fake_build_tree(row, source_config):
            return SimpleNamespace(
                crown_base_height=1.0,
                foliage_sav=2000.0,
                specific_leaf_area=5.0,
                species_code=131,
            )

        class FakeVT:
            def __init__(self, tree, sampled, hr, vr):
                self._mask = sampled

            def distribute_biomass(self):
                return self._mask * biomass_value

        monkeypatch.setattr(vmod, "build_tree", fake_build_tree)
        monkeypatch.setattr(
            vmod, "discretize_crown_profile", lambda *a, **kw: mask.copy()
        )
        monkeypatch.setattr(vmod, "sample_occupied_cells", lambda m, **kw: m.copy())
        monkeypatch.setattr(vmod, "VoxelizedTree", FakeVT)

    def _build_union(self, layout, batch, keys):
        """Build an all-fill union Dataset covering the batch's halo."""
        from treevox import storage

        union_y, union_x = orchestrator.voxelize.batch_union_slices(
            batch,
            layout.dims["ny"],
            layout.dims["nx"],
            layout.chunk_xy,
            overlap_cells=orchestrator.voxelize.OVERLAP_CELLS,
        )
        span_y = union_y.stop - union_y.start
        span_x = union_x.stop - union_x.start
        data_vars = {}
        for k in keys:
            dtype, fill = storage.BAND_SPECS[k]
            data_vars[k] = (
                ("z", "y", "x"),
                np.full((layout.dims["nz"], span_y, span_x), fill, dtype=dtype),
            )
        return xr.Dataset(data_vars), union_y, union_x

    def _tiny_domain(self, nx=40, ny=20):
        return SimpleNamespace(
            total_bounds=np.array([0.0, 0.0, float(nx), float(ny)]),
            crs="EPSG:32610",
        )

    def _grid(self, bands, seed=42):
        return {
            "id": "g-halo",
            "domain_id": "d1",
            "source": {
                "name": "inventory",
                "source_inventory_id": "inv1",
                "resolution": (1.0, 1.0, 1.0),
                "crown_profile_model": "purves",
                "biomass_source": {
                    "type": "allometry",
                    "equations": "nsvb",
                    "components": ["foliage"],
                    "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
                },
                "moisture_model": {"live": {"method": "uniform", "value": 100.0}},
                "seed": seed,
            },
            "bands": [{"key": k} for k in bands],
        }

    def _force_chunk_xy(self, monkeypatch, chunk_xy=20):
        """Shrink CHUNK_SIZE_HORIZONTAL so the planner picks the size we want.

        `_plan_grid_layout` clamps chunk_xy to min(CHUNK_SIZE_HORIZONTAL, nx, ny).
        Setting it to 20 → two chunks on a 40-wide grid.
        """
        monkeypatch.setattr(orchestrator.voxelize, "CHUNK_SIZE_HORIZONTAL", chunk_xy)

    def test_tree_at_seam_writes_into_neighbor_halo_and_merges(self, monkeypatch):
        """Tree stem in chunk (0,0), crown spills east across the seam at col=20.

        With biomass_shape=(2, 5, 5), a stem at col=19 fills cols [17, 22).
        Cols [20, 22) lie in chunk (0,1)'s native territory but are written
        by chunk (0,0)'s halo-extended buffer. After masked_merge, the union
        must carry biomass continuously across the seam.
        """
        self._patch_fastfuels(monkeypatch, biomass_shape=(2, 5, 5), biomass_value=1.0)
        self._force_chunk_xy(monkeypatch, chunk_xy=20)

        # One tree at x=19.5 → cell col=19 → col_chunk=0 (west of seam).
        df = pd.DataFrame(
            {
                "x": [19.5],
                "y": [10.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [5.0],
                "crown_ratio": [0.5],
            }
        )
        grid = self._grid(bands=["volume_fraction"])
        layout = orchestrator._plan_grid_layout(grid, self._tiny_domain(), df)
        assert layout.chunk_xy == 20
        assert len(layout.chunk_locations) == 2  # (0,0) and (0,1)

        df = orchestrator.assign_tree_ids(df)
        df = orchestrator._prepare_tree_chunks(df, layout)
        assert df["col_chunk"].iloc[0] == 0  # stem firmly in chunk (0,0)

        batch = [(0, 0), (0, 1)]
        union_ds, union_y, union_x = self._build_union(
            layout, batch, ["volume_fraction"]
        )
        chunk_indices = df.groupby(["row_chunk", "col_chunk"], sort=False).indices

        payloads = orchestrator._build_payloads(
            batch,
            union_ds,
            union_y,
            union_x,
            df,
            chunk_indices,
            layout,
            grid["source"],
            grid["id"],
        )
        assert len(payloads) == 2
        payload_by_loc = {p["chunk_location"]: p for p in payloads}
        # Chunk (0,0) gets the tree; chunk (0,1) gets nothing.
        assert len(payload_by_loc[(0, 0)]["trees"]) == 1
        assert len(payload_by_loc[(0, 1)]["trees"]) == 0

        results = [worker_run(p) for p in payloads]
        assert all("error" not in r for r in results), [
            r.get("error") for r in results if "error" in r
        ]

        # Chunk (0,0)'s buffer: cols 17..22 in absolute coords. chunk_x_start=0
        # (no western halo at the grid edge), so buffer-local cols match absolute.
        vf_00 = results[0]["buffers"]["volume_fraction"]
        # Writes into the native region.
        assert vf_00[:, :, 17:20].sum() > 0
        # Writes into the east halo (absolute cols 20, 21 lie in chunk (0,1)'s
        # native territory but we expect them here thanks to the halo).
        assert vf_00[:, :, 20:22].sum() > 0, (
            "Tree's crown did not render into chunk (0,0)'s east halo — "
            "either the halo isn't being honored or _place_biomass clipped it."
        )

        # Chunk (0,1)'s buffer is empty because it had no trees assigned.
        vf_01 = results[1]["buffers"]["volume_fraction"]
        assert vf_01.sum() == 0

        # Masked merge stitches chunk (0,0)'s halo cells into the union.
        merged = orchestrator.storage.masked_merge(union_ds, results, union_y, union_x)
        vf_merged = merged["volume_fraction"].values
        # Continuous biomass across the seam at col=20 — no zero column.
        for col in range(17, 22):
            assert vf_merged[:, :, col].sum() > 0, (
                f"Column {col} is zero in the merged union — the halo/merge "
                f"path dropped cells that chunk (0,0) wrote."
            )
        # And only those columns were touched.
        assert vf_merged[:, :, :17].sum() == 0
        assert vf_merged[:, :, 22:].sum() == 0

    def test_halo_carries_overwrite_band_across_seam(self, monkeypatch):
        """Same geometry, checking an overwrite band (spcd).

        Catches any regression where overwrite-band halo cells get lost by a
        mask that only triggers on accumulate-band values.
        """
        self._patch_fastfuels(monkeypatch, biomass_shape=(2, 5, 5), biomass_value=1.0)
        self._force_chunk_xy(monkeypatch, chunk_xy=20)

        df = pd.DataFrame(
            {
                "x": [19.5],
                "y": [10.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [5.0],
                "crown_ratio": [0.5],
            }
        )
        grid = self._grid(bands=["spcd"])
        layout = orchestrator._plan_grid_layout(grid, self._tiny_domain(), df)
        df = orchestrator.assign_tree_ids(df)
        df = orchestrator._prepare_tree_chunks(df, layout)

        batch = [(0, 0), (0, 1)]
        union_ds, union_y, union_x = self._build_union(layout, batch, ["spcd"])
        chunk_indices = df.groupby(["row_chunk", "col_chunk"], sort=False).indices
        payloads = orchestrator._build_payloads(
            batch,
            union_ds,
            union_y,
            union_x,
            df,
            chunk_indices,
            layout,
            grid["source"],
            grid["id"],
        )
        results = [worker_run(p) for p in payloads]
        merged = orchestrator.storage.masked_merge(union_ds, results, union_y, union_x)
        spcd = merged["spcd"].values
        # Species code present on both sides of the seam.
        assert (spcd[:, :, 17:20] == 131).any()
        assert (spcd[:, :, 20:22] == 131).any()
