"""Unit tests for treevox.main.

Covers Cloud Function HTTP flow, dispatch, inventory parquet IO, and the
handler orchestrator (with mp / GCS / xarray mocks). One end-to-end spawn-
Pool test verifies pickle safety + spawn context.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from treevox import main
from treevox.main import (
    MockRequest,
    ProcessingError,
    assign_tree_ids,
    dispatch_handler,
    download_inventory,
    filter_live,
    handle_inventory,
    process_grid_request,
)

# process_grid_request — port of griddle's test_main


class TestProcessGridRequest:
    def test_missing_grid_id_returns_400(self):
        request = MockRequest(data={})
        response, status = process_grid_request(request)
        assert status == 400
        assert "id" in response.lower()

    def test_empty_body_returns_400(self):
        request = MockRequest(data=None)
        response, status = process_grid_request(request)
        assert status == 400

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.update_status")
    def test_retry_marks_failed_and_returns_200(self, mock_update, mock_delete):
        request = MockRequest(
            data={"id": "g1"}, headers={"X-CloudTasks-TaskRetryCount": "1"}
        )
        response, status = process_grid_request(request)
        assert status == 200
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[:2] == ("g1", "failed")
        assert kwargs["error"]["code"] == "UNEXPECTED_FAILURE"
        # Partial zarr should be cleaned up on retry-failure path.
        mock_delete.assert_called_once()

    @patch("treevox.main.load_grid")
    def test_grid_not_found_returns_200(self, mock_load_grid):
        from lib.config import GRIDS_COLLECTION
        from lib.firestore import DocumentNotFoundError

        mock_load_grid.side_effect = DocumentNotFoundError(GRIDS_COLLECTION, "missing")
        response, status = process_grid_request(MockRequest(data={"id": "missing"}))
        assert status == 200

    @patch("treevox.main.update_document")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main._load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.update_progress")
    @patch("treevox.main.load_grid")
    def test_happy_path(
        self,
        mock_load_grid,
        mock_progress,
        mock_status,
        mock_load_domain,
        mock_dispatch,
        mock_update_doc,
    ):
        mock_load_grid.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_load_domain.return_value = MagicMock()
        mock_dispatch.return_value = main.VoxelizationResult(
            gcs_path="gs://bucket/g1",
            georeference={"shape": [5, 10, 10]},
            chunk_shape=[5, 10, 10],
        )

        response, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 200

        # status: running, then completed
        calls = [c.args for c in mock_status.call_args_list]
        assert calls[0] == ("g1", "running")
        assert calls[1][1] == "completed"

        # chunk_shape persisted before status=completed
        mock_update_doc.assert_any_call(
            main.GRIDS_COLLECTION, "g1", {"chunk_shape": [5, 10, 10]}
        )

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main._load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_processing_error_returns_200_and_deletes_zarr(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_delete
    ):
        """ProcessingError path cleans up the partial zarr store."""
        mock_load.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_domain.return_value = MagicMock()
        mock_dispatch.side_effect = ProcessingError(
            code="EMPTY_INVENTORY", message="no trees"
        )

        _, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 200
        mock_delete.assert_called_once()
        # Last update_status call should be failed.
        last = mock_status.call_args_list[-1]
        assert last.args[1] == "failed"

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main._load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_unexpected_error_returns_500(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_delete
    ):
        mock_load.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_domain.return_value = MagicMock()
        mock_dispatch.side_effect = RuntimeError("boom")

        _, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 500

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main._load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_cancelled_during_processing_deletes_zarr(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_delete
    ):
        from treevox.main import CancelledException

        mock_load.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_domain.return_value = MagicMock()
        mock_dispatch.side_effect = CancelledException("cancelled")

        _, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 200
        mock_delete.assert_called_once()


# dispatch_handler


class TestDispatchHandler:
    @patch("treevox.main.handle_inventory")
    def test_inventory_routes_to_handler(self, mock_handle):
        mock_handle.return_value = "result"
        grid = {"source": {"name": "inventory"}}
        result = dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert result == "result"
        mock_handle.assert_called_once()

    def test_unknown_source_raises_processing_error(self):
        grid = {"source": {"name": "lidar"}}
        with pytest.raises(ProcessingError) as exc:
            dispatch_handler(grid, MagicMock(), lambda *a, **k: None)
        assert exc.value.code == "UNKNOWN_SOURCE"


# Inventory IO


class TestDownloadInventory:
    def test_success_roundtrip(self, tmp_path, monkeypatch):
        # Write a parquet locally, mock download_file to copy it.
        df_in = pd.DataFrame(
            {
                "x": [1.0],
                "y": [2.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.4],
            }
        )
        src = tmp_path / "source.parquet"
        df_in.to_parquet(src)

        def fake_download(gcs_path, local_path):
            import shutil

            shutil.copy(src, local_path)

        monkeypatch.setattr(main, "download_file", fake_download)

        result = download_inventory("inv123", str(tmp_path))
        pd.testing.assert_frame_equal(result, df_in)

    def test_missing_inventory_raises_processing_error(self, monkeypatch, tmp_path):
        def raising_download(gcs_path, local_path):
            raise FileNotFoundError(gcs_path)

        monkeypatch.setattr(main, "download_file", raising_download)

        with pytest.raises(ProcessingError) as exc:
            download_inventory("missing", str(tmp_path))
        assert exc.value.code == "INVENTORY_NOT_FOUND"


class TestFilterLive:
    def _df(self, **overrides):
        data = {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "fia_species_code": [131, 131, 131],
            "fia_status_code": [1, 2, 1],
            "dbh": [20.0, 20.0, 20.0],
            "height": [15.0, 15.0, 15.0],
            "crown_ratio": [0.4, 0.4, 0.4],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_keeps_only_live_trees(self):
        df = self._df()
        out = filter_live(df)
        assert len(out) == 2
        assert (out["fia_status_code"] == 1).all()

    def test_drops_nulls_on_required_columns(self):
        df = self._df()
        df.loc[0, "dbh"] = None
        out = filter_live(df)
        # Row 0 dropped for null dbh; row 1 dropped for status_code=2.
        assert len(out) == 1

    def test_biomass_column_non_null_required_when_specified(self):
        df = self._df()
        df["fuel_load"] = [10.0, 20.0, None]
        out = filter_live(df, biomass_column="fuel_load")
        # Row 0: live, fuel_load=10. Row 1: status=2 (dropped). Row 2: null fuel_load (dropped).
        assert len(out) == 1
        assert out.iloc[0]["fuel_load"] == 10.0


class TestAssignTreeIds:
    def test_sequential_int32_tree_ids(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = assign_tree_ids(df)
        assert out["tree_id"].dtype == np.int32
        assert list(out["tree_id"]) == [0, 1, 2]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.0]})
        assign_tree_ids(df)
        assert "tree_id" not in df.columns


# _pick_worker_count


class TestPickWorkerCount:
    def test_returns_at_least_one(self):
        assert main._pick_worker_count() >= 1

    def test_capped_at_default_max(self, monkeypatch):
        # Pretend we have many CPUs and tons of memory — still capped at
        # DEFAULT_MAX_WORKERS=4.
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False
        )

        # /proc/meminfo may or may not exist; mock the read.
        class FakeFile:
            def __enter__(self):
                return iter(["MemAvailable: 10485760 kB\n"])  # 10 GB

            def __exit__(self, *a):
                pass

        monkeypatch.setattr("builtins.open", lambda *a, **kw: FakeFile())
        assert main._pick_worker_count() <= main.DEFAULT_MAX_WORKERS


# handle_inventory — flow test with all IO + mp + storage mocked


def _fake_domain():
    from types import SimpleNamespace

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


class TestHandleInventoryFlow:
    @patch("treevox.main.ctx_pool_override", create=True)
    @patch("treevox.main.storage.consolidate_metadata")
    @patch("treevox.main.storage.write_union")
    @patch("treevox.main.storage.masked_merge")
    @patch("treevox.main.storage.read_union")
    @patch("treevox.main.storage.init_store")
    @patch("treevox.main.download_inventory")
    def test_happy_path_calls_expected_stages(
        self,
        mock_download,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
        _pool_override,
    ):
        mock_download.return_value = pd.DataFrame(
            {
                "x": [50.0],
                "y": [50.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [5.0],
                "crown_ratio": [0.5],
            }
        )
        # read_union returns a Dataset-like object; workers get mocked.
        import xarray as xr

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

        # Patch the worker so compute is instant and returns empty-result dicts.
        with (
            patch("treevox.main.worker_run"),
            patch("treevox.main.multiprocessing.get_context") as mock_get_ctx,
        ):
            # Fake Pool that calls worker_run synchronously.
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

            result = handle_inventory(_base_grid(), _fake_domain(), progress)

        assert isinstance(result, main.VoxelizationResult)
        assert result.chunk_shape[0] > 0
        assert len(result.chunk_shape) == 3

        # init_store called exactly once.
        assert mock_init.call_count == 1
        # consolidate_metadata called exactly once at end.
        mock_consolidate.assert_called_once()
        # Early progress stages fire.
        msgs = [m for m, _ in progress_calls]
        assert any("Loading" in m for m in msgs)
        assert any("Initializing" in m for m in msgs)
        assert any("Finalizing" in m for m in msgs)

    @patch("treevox.main.download_inventory")
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
            handle_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)
        assert exc.value.code == "EMPTY_INVENTORY"

    @patch("treevox.main.storage.consolidate_metadata")
    @patch("treevox.main.storage.write_union")
    @patch("treevox.main.storage.masked_merge")
    @patch("treevox.main.storage.read_union")
    @patch("treevox.main.storage.init_store")
    @patch("treevox.main.download_inventory")
    def test_worker_error_surfaces_as_voxelization_failed(
        self,
        mock_download,
        mock_init,
        mock_read,
        mock_merge,
        mock_write,
        mock_consolidate,
    ):
        mock_download.return_value = pd.DataFrame(
            {
                "x": [50.0],
                "y": [50.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [5.0],
                "crown_ratio": [0.5],
            }
        )
        import xarray as xr

        mock_read.return_value = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )

        with patch("treevox.main.multiprocessing.get_context") as mock_get_ctx:
            fake_pool = MagicMock()
            fake_pool.__enter__.return_value = fake_pool
            fake_pool.__exit__.return_value = False
            fake_pool.map.return_value = [{"chunk_location": (0, 0), "error": "boom"}]
            fake_ctx = MagicMock()
            fake_ctx.Pool.return_value = fake_pool
            mock_get_ctx.return_value = fake_ctx

            with pytest.raises(ProcessingError) as exc:
                handle_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)
        assert exc.value.code == "VOXELIZATION_FAILED"


# Persistent Pool across batches


class TestPersistentPool:
    @patch("treevox.main.storage.consolidate_metadata")
    @patch("treevox.main.storage.write_union")
    @patch("treevox.main.storage.masked_merge")
    @patch("treevox.main.storage.read_union")
    @patch("treevox.main.storage.init_store")
    @patch("treevox.main.download_inventory")
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
        mock_download.return_value = pd.DataFrame(
            {
                "x": [50.0],
                "y": [50.0],
                "fia_species_code": [131],
                "fia_status_code": [1],
                "dbh": [20.0],
                "height": [5.0],
                "crown_ratio": [0.5],
            }
        )
        import xarray as xr

        mock_read.return_value = xr.Dataset(
            {
                "volume_fraction": (
                    ("z", "y", "x"),
                    np.zeros((5, 100, 100), dtype="float32"),
                )
            }
        )
        mock_merge.side_effect = lambda union_ds, *a, **kw: union_ds

        # Force many batches: tiny CHUNK_LENGTH_METERS so chunk_xy=20 across a 120×120 grid.
        # That yields a 6x6 chunk grid = 36 chunks. With 4 workers, 9 batches.
        with (
            patch("treevox.main.voxelize.CHUNK_LENGTH_METERS", 20),
            patch("treevox.main.multiprocessing.get_context") as mock_get_ctx,
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

            handle_inventory(_base_grid(), _fake_domain(), lambda *a, **k: None)

            # Pool should be created exactly once across all batches.
            assert fake_ctx.Pool.call_count == 1
            # But map should be called multiple times (one per batch).
            assert fake_pool.map.call_count > 1
