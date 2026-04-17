"""Unit tests for treevox.inventory_io — tabular parquet I/O only.

These tests don't hit GCS — they substitute `download_file` with a local copy.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from treevox import inventory_io
from treevox.errors import ProcessingError
from treevox.inventory_io import (
    assign_tree_ids,
    download_inventory,
    filter_live,
)


class TestDownloadInventory:
    def test_success_roundtrip(self, tmp_path, monkeypatch):
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
            shutil.copy(src, local_path)

        monkeypatch.setattr(inventory_io, "download_file", fake_download)

        result = download_inventory("inv123", str(tmp_path))
        pd.testing.assert_frame_equal(result, df_in)

    def test_missing_inventory_raises_processing_error(self, monkeypatch, tmp_path):
        def raising_download(gcs_path, local_path):
            raise FileNotFoundError(gcs_path)

        monkeypatch.setattr(inventory_io, "download_file", raising_download)

        with pytest.raises(ProcessingError) as exc:
            download_inventory("missing", str(tmp_path))
        assert exc.value.code == "INVENTORY_NOT_FOUND"

    def test_unexpected_io_error_also_maps_to_not_found(self, monkeypatch, tmp_path):
        """gcsfs sometimes raises permission/timeout errors; all map to NOT_FOUND."""

        def raising_download(gcs_path, local_path):
            raise PermissionError("denied")

        monkeypatch.setattr(inventory_io, "download_file", raising_download)

        with pytest.raises(ProcessingError) as exc:
            download_inventory("x", str(tmp_path))
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
        out = filter_live(self._df())
        assert len(out) == 2
        assert (out["fia_status_code"] == 1).all()

    def test_drops_nulls_on_required_columns(self):
        df = self._df()
        df.loc[0, "dbh"] = None
        out = filter_live(df)
        assert len(out) == 1

    def test_biomass_column_non_null_required_when_specified(self):
        df = self._df()
        df["fuel_load"] = [10.0, 20.0, None]
        out = filter_live(df, biomass_column="fuel_load")
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
