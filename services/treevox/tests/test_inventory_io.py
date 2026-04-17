"""Unit tests for treevox.inventory_io — tabular parquet I/O only.

These tests don't hit GCS — they substitute `pd.read_parquet` (on the
`inventory_io` module) with an in-memory / local stand-in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from treevox import inventory_io
from treevox.errors import ProcessingError
from treevox.inventory_io import (
    REQUIRED_COLUMNS,
    assign_tree_ids,
    drop_null_rows,
    read_inventory,
)


class TestReadInventory:
    def test_success_roundtrip(self, monkeypatch):
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

        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["path"] = path
            captured["columns"] = columns
            captured["filters"] = filters
            return df_in[columns] if columns else df_in

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        result = read_inventory("inv123")
        pd.testing.assert_frame_equal(result, df_in[REQUIRED_COLUMNS])
        assert captured["path"].startswith("gs://")
        assert captured["path"].endswith("inv123")
        # Column projection and status pushdown both make it to parquet.
        assert captured["columns"] == REQUIRED_COLUMNS
        assert captured["filters"] == [("fia_status_code", "=", 1)]

    def test_biomass_column_appended_to_projection(self, monkeypatch):
        df_in = pd.DataFrame(
            {col: [1.0] for col in REQUIRED_COLUMNS} | {"my_load": [42.0]}
        )
        df_in["fia_species_code"] = [131]
        df_in["fia_status_code"] = [1]

        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return df_in[columns]

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", biomass_column="my_load")
        assert "my_load" in captured["columns"]

    def test_biomass_column_already_required_not_duplicated(self, monkeypatch):
        """If the biomass column name happens to collide with REQUIRED_COLUMNS,
        it must not appear twice (pyarrow would reject a duplicated projection)."""
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", biomass_column="dbh")
        assert captured["columns"].count("dbh") == 1

    def test_missing_inventory_raises_processing_error(self, monkeypatch):
        def raising(path, **kwargs):
            raise FileNotFoundError(path)

        monkeypatch.setattr(inventory_io.pd, "read_parquet", raising)

        with pytest.raises(ProcessingError) as exc:
            read_inventory("missing")
        assert exc.value.code == "INVENTORY_NOT_FOUND"

    def test_unexpected_io_error_also_maps_to_not_found(self, monkeypatch):
        """gcsfs / pyarrow may surface permission or transport errors; map all to NOT_FOUND."""

        def raising(path, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(inventory_io.pd, "read_parquet", raising)

        with pytest.raises(ProcessingError) as exc:
            read_inventory("x")
        assert exc.value.code == "INVENTORY_NOT_FOUND"


class TestDropNullRows:
    """`drop_null_rows` sees post-pushdown input — all rows are already live —
    so fixtures use `fia_status_code == 1` throughout."""

    def _df(self, **overrides):
        data = {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "fia_species_code": [131, 131, 131],
            "fia_status_code": [1, 1, 1],
            "dbh": [20.0, 20.0, 20.0],
            "height": [15.0, 15.0, 15.0],
            "crown_ratio": [0.4, 0.4, 0.4],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_drops_rows_with_null_required_columns(self):
        df = self._df()
        df.loc[0, "dbh"] = None
        out = drop_null_rows(df)
        assert len(out) == 2

    def test_biomass_column_non_null_required_when_specified(self):
        df = self._df()
        df["fuel_load"] = [10.0, 20.0, None]
        out = drop_null_rows(df, biomass_column="fuel_load")
        assert len(out) == 2
        assert list(out["fuel_load"]) == [10.0, 20.0]

    def test_resets_index(self):
        df = self._df()
        df.loc[0, "dbh"] = None  # drop the first row
        out = drop_null_rows(df)
        assert list(out.index) == [0, 1]


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
