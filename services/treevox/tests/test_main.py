"""Unit tests for treevox.main — HTTP entry + retry/error arms only.

Source routing, the voxelization job, inventory I/O, and Firestore helpers are
covered by their own test modules (test_dispatch.py, handlers/test_voxelize.py,
and lib's test_inventory_io.py). The Cloud Function layer is exercised here by
mocking `dispatch_handler` and the Firestore helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from treevox.errors import CancelledException, ProcessingError
from treevox.handlers.voxelize import VoxelizationResult
from treevox.main import MockRequest, process_grid_request


class TestProcessGridRequest:
    def test_missing_grid_id_returns_400(self):
        response, status = process_grid_request(MockRequest(data={}))
        assert status == 400
        assert "id" in response.lower()

    def test_empty_body_returns_400(self):
        response, status = process_grid_request(MockRequest(data=None))
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
        _, status = process_grid_request(MockRequest(data={"id": "missing"}))
        assert status == 200

    @patch("treevox.main.update_metadata")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main.load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_happy_path(
        self,
        mock_load_grid,
        mock_status,
        mock_load_domain,
        mock_dispatch,
        mock_update_metadata,
    ):
        mock_load_grid.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_load_domain.return_value = MagicMock()
        mock_dispatch.return_value = VoxelizationResult(
            gcs_path="gs://bucket/g1",
            georeference={"shape": [5, 10, 10]},
            chunk_shape=[5, 10, 10],
        )

        _, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 200

        calls = [c.args for c in mock_status.call_args_list]
        assert calls[0] == ("g1", "running")
        assert calls[1][1] == "completed"

        mock_update_metadata.assert_any_call(
            "g1",
            {
                "chunks": {
                    "shape": [5, 10, 10],
                    "count": 1,
                    "count_by_axis": {"z": 1, "y": 1, "x": 1},
                }
            },
        )

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main.load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_processing_error_returns_200_and_deletes_zarr(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_delete
    ):
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
        last = mock_status.call_args_list[-1]
        assert last.args[1] == "failed"

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main.load_domain")
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
    @patch("treevox.main.load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_cancelled_during_processing_deletes_zarr(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_delete
    ):
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

    @patch("treevox.main.storage.delete_zarr")
    @patch("treevox.main.update_metadata")
    @patch("treevox.main.dispatch_handler")
    @patch("treevox.main.load_domain")
    @patch("treevox.main.update_status")
    @patch("treevox.main.load_grid")
    def test_own_doc_deleted_midrun_cancels_gracefully(
        self, mock_load, mock_status, mock_domain, mock_dispatch, mock_meta, mock_delete
    ):
        """Deleting the grid mid-run makes the chunks write-back surface as
        CancelledException — the job ends at 200 and cleans up its zarr, with
        no unhandled 500 / Cloud Tasks retry (#593)."""
        mock_load.return_value = {
            "id": "g1",
            "domain_id": "d1",
            "source": {"name": "inventory"},
            "bands": [{"key": "volume_fraction"}],
        }
        mock_domain.return_value = MagicMock()
        mock_dispatch.return_value = VoxelizationResult(
            gcs_path="gs://bucket/g1",
            georeference={"shape": [5, 10, 10]},
            chunk_shape=[5, 10, 10],
        )
        mock_meta.side_effect = CancelledException("cancelled")

        _, status = process_grid_request(MockRequest(data={"id": "g1"}))
        assert status == 200
        mock_delete.assert_called_once()
        assert not any(c.args[1] == "completed" for c in mock_status.call_args_list)


class TestUpdateMetadata:
    """The guarded own-doc write-back helper in firestore_io (#593)."""

    @patch("treevox.firestore_io.update_document")
    def test_success_writes_to_grids_collection(self, mock_update):
        from treevox.firestore_io import update_metadata

        from lib.config import GRIDS_COLLECTION

        update_metadata("g1", {"chunks": {"count": 1}})
        mock_update.assert_called_once_with(
            GRIDS_COLLECTION, "g1", {"chunks": {"count": 1}}
        )

    @patch("treevox.firestore_io.update_document")
    def test_deleted_doc_raises_cancelled(self, mock_update):
        from treevox.firestore_io import update_metadata

        from lib.firestore import DocumentNotFoundError

        mock_update.side_effect = DocumentNotFoundError("grids", "g1")
        with pytest.raises(CancelledException):
            update_metadata("g1", {"chunks": {}})


class TestErrorsModuleSurface:
    """Smoke tests ensuring the error types stay importable from `treevox.errors`."""

    def test_processing_error_to_dict_includes_code_and_message(self):
        err = ProcessingError(code="X", message="m")
        assert err.to_dict() == {"code": "X", "message": "m"}

    def test_processing_error_to_dict_includes_optional_fields(self):
        err = ProcessingError(code="X", message="m", suggestion="s", traceback="tb")
        d = err.to_dict()
        assert d["suggestion"] == "s"
        assert d["traceback"] == "tb"

    def test_cancelled_exception_is_exception(self):
        assert issubclass(CancelledException, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
