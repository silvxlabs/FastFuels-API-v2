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
    assign_tree_ids,
    drop_null_rows,
    read_inventory,
)

# Every column a full tree inventory carries (e.g. an upload or PIM expansion).
ALL_COLUMNS = [
    "x",
    "y",
    "fia_species_code",
    "fia_status_code",
    "dbh",
    "height",
    "crown_ratio",
]


def _stub_schema(monkeypatch, columns):
    """Make `read_inventory`'s schema probe return ``columns`` without GCS."""
    monkeypatch.setattr(inventory_io, "_available_columns", lambda inv: set(columns))


def _stub_open_raising(monkeypatch, exc):
    """Make the schema probe's `gcsfs_client.open` raise ``exc``."""

    class _FakeFS:
        def open(self, *args, **kwargs):
            raise exc

    monkeypatch.setattr(inventory_io, "gcsfs_client", _FakeFS())


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

        _stub_schema(monkeypatch, ALL_COLUMNS)
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        result = read_inventory("inv123")
        assert set(result.columns) == set(ALL_COLUMNS)
        assert list(result["dbh"]) == [20.0]
        assert captured["path"].startswith("gs://")
        assert captured["path"].endswith("inv123")
        # Full column projection and the live-tree pushdown both reach parquet.
        assert set(captured["columns"]) == set(ALL_COLUMNS)
        assert captured["filters"] == [("fia_status_code", "=", 1)]

    def test_missing_required_column_raises_missing_columns(self, monkeypatch):
        """A height-only (CHM/ITD) inventory can't be voxelized — surface that as
        MISSING_COLUMNS, not the old misleading INVENTORY_NOT_FOUND."""
        _stub_schema(monkeypatch, ["x", "y", "height"])
        monkeypatch.setattr(
            inventory_io.pd,
            "read_parquet",
            lambda *a, **k: pytest.fail("should not read data when columns missing"),
        )

        with pytest.raises(ProcessingError) as exc:
            read_inventory("chm")
        assert exc.value.code == "MISSING_COLUMNS"
        assert "dbh" in exc.value.message

    def test_biomass_column_appended_to_projection(self, monkeypatch):
        df_in = pd.DataFrame({col: [1.0] for col in ALL_COLUMNS} | {"my_load": [42.0]})

        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return df_in[columns]

        _stub_schema(monkeypatch, ALL_COLUMNS + ["my_load"])
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", biomass_column="my_load")
        assert "my_load" in captured["columns"]

    def test_missing_biomass_column_raises_missing_columns(self, monkeypatch):
        """A requested biomass/crown column that isn't in the inventory is a
        missing required column too."""
        _stub_schema(monkeypatch, ALL_COLUMNS)
        monkeypatch.setattr(
            inventory_io.pd,
            "read_parquet",
            lambda *a, **k: pytest.fail("should not read data when columns missing"),
        )

        with pytest.raises(ProcessingError) as exc:
            read_inventory("inv1", biomass_column="my_load")
        assert exc.value.code == "MISSING_COLUMNS"
        assert "my_load" in exc.value.message

    def test_biomass_column_already_required_not_duplicated(self, monkeypatch):
        """If the biomass column name collides with a required column, it must
        not appear twice (pyarrow would reject a duplicated projection)."""
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        _stub_schema(monkeypatch, ALL_COLUMNS)
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", biomass_column="dbh")
        assert captured["columns"].count("dbh") == 1

    def test_crown_radius_column_appended_to_projection(self, monkeypatch):
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        _stub_schema(monkeypatch, ALL_COLUMNS + ["lidar_max_radius"])
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", crown_radius_column="lidar_max_radius")
        assert "lidar_max_radius" in captured["columns"]

    def test_biomass_and_crown_radius_columns_both_projected(self, monkeypatch):
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        _stub_schema(monkeypatch, ALL_COLUMNS + ["my_load", "lidar_max_radius"])
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory(
            "inv1",
            biomass_column="my_load",
            crown_radius_column="lidar_max_radius",
        )
        assert "my_load" in captured["columns"]
        assert "lidar_max_radius" in captured["columns"]

    def test_crown_radius_column_already_required_not_duplicated(self, monkeypatch):
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        _stub_schema(monkeypatch, ALL_COLUMNS)
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", crown_radius_column="dbh")
        assert captured["columns"].count("dbh") == 1

    def test_same_biomass_and_crown_radius_column_not_duplicated(self, monkeypatch):
        """If both the biomass and max-crown-radius roles map to the same custom
        column, it must appear once — pyarrow rejects a duplicated projection."""
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        _stub_schema(monkeypatch, ALL_COLUMNS + ["my_col"])
        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", biomass_column="my_col", crown_radius_column="my_col")
        assert captured["columns"].count("my_col") == 1

    def test_missing_inventory_raises_processing_error(self, monkeypatch):
        _stub_open_raising(monkeypatch, FileNotFoundError("no _metadata"))

        with pytest.raises(ProcessingError) as exc:
            read_inventory("missing")
        assert exc.value.code == "INVENTORY_NOT_FOUND"

    def test_unexpected_io_error_also_maps_to_not_found(self, monkeypatch):
        """gcsfs / pyarrow may surface permission or transport errors; map all to NOT_FOUND."""
        _stub_open_raising(monkeypatch, PermissionError("denied"))

        with pytest.raises(ProcessingError) as exc:
            read_inventory("x")
        assert exc.value.code == "INVENTORY_NOT_FOUND"

    def test_data_read_failure_maps_to_not_found(self, monkeypatch):
        """The footer probe can succeed (all columns present) yet the data read
        still fail — e.g. a corrupt row group or transport error mid-scan. That
        branch must also map to NOT_FOUND, distinct from the footer-probe path."""
        _stub_schema(monkeypatch, ALL_COLUMNS)

        def raising(*a, **k):
            raise OSError("row group read failed")

        monkeypatch.setattr(inventory_io.pd, "read_parquet", raising)

        with pytest.raises(ProcessingError) as exc:
            read_inventory("inv1")
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

    def test_crown_radius_column_non_null_required_when_specified(self):
        df = self._df()
        df["lidar_max_radius"] = [2.5, None, 4.0]
        out = drop_null_rows(df, crown_radius_column="lidar_max_radius")
        assert len(out) == 2
        assert list(out["lidar_max_radius"]) == [2.5, 4.0]

    def test_biomass_and_crown_radius_columns_drop_independently(self):
        df = self._df()
        df["fuel_load"] = [10.0, 20.0, 30.0]
        df["lidar_max_radius"] = [2.5, None, 4.0]
        out = drop_null_rows(
            df,
            biomass_column="fuel_load",
            crown_radius_column="lidar_max_radius",
        )
        assert len(out) == 2
        assert list(out["fuel_load"]) == [10.0, 30.0]
        assert list(out["lidar_max_radius"]) == [2.5, 4.0]

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
