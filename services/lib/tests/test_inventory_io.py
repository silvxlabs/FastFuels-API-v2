"""Unit tests for lib.inventory_io — tabular parquet I/O only.

These tests don't hit GCS — they substitute `pd.read_parquet` (on the
`inventory_io` module) with an in-memory / local stand-in.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.compute as pc
import pytest

from lib import inventory_io
from lib.errors import ProcessingError
from lib.inventory_io import (
    REQUIRED_COLUMNS,
    canopy_required_columns,
    read_inventory,
    require_complete_rows,
)

_STATUS = pc.field("fia_status_code")
LIVE_FILTER = (_STATUS == 1) | _STATUS.is_null()


class TestReadInventory:
    @pytest.fixture(autouse=True)
    def _status_column_present(self, monkeypatch):
        """Default: the inventory carries every required column, so
        `read_inventory` keeps its full projection and live-tree pushdown. This
        also keeps these tests off GCS — the real schema probe reads a parquet
        footer. Individual tests override the probe to exercise other paths."""
        monkeypatch.setattr(
            inventory_io,
            "_inventory_column_names",
            lambda inventory_id: set(REQUIRED_COLUMNS),
        )

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
        assert captured["filters"].equals(LIVE_FILTER)

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

    def test_crown_radius_column_appended_to_projection(self, monkeypatch):
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", crown_radius_column="lidar_max_radius")
        assert "lidar_max_radius" in captured["columns"]

    def test_biomass_and_crown_radius_columns_both_projected(self, monkeypatch):
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [] for c in columns})

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

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv1", crown_radius_column="dbh")
        assert captured["columns"].count("dbh") == 1

    def test_missing_status_column_defaults_all_trees_live(self, monkeypatch):
        """A CHM/GDAM inventory has no `fia_status_code` column: it must not be
        projected or pushed as a filter, and every tree defaults to live (1)."""
        present = [c for c in REQUIRED_COLUMNS if c != "fia_status_code"]
        monkeypatch.setattr(
            inventory_io, "_inventory_column_names", lambda _id: set(present)
        )
        df_in = pd.DataFrame({c: [1.0, 2.0] for c in present})

        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            captured["filters"] = filters
            return df_in[columns].copy()

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        result = read_inventory("chm_inv")
        assert "fia_status_code" not in captured["columns"]
        assert captured["filters"] is None
        assert list(result["fia_status_code"]) == [1, 1]

    def test_schema_probe_failure_keeps_pushdown(self, monkeypatch):
        """If the schema can't be read (probe returns None), fall back to the
        historical projection + live-tree pushdown rather than dropping it."""
        monkeypatch.setattr(inventory_io, "_inventory_column_names", lambda _id: None)

        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            captured["filters"] = filters
            return pd.DataFrame({c: [1.0] for c in columns})

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        read_inventory("inv")
        assert captured["columns"] == REQUIRED_COLUMNS
        assert captured["filters"].equals(LIVE_FILTER)

    def test_missing_morphology_columns_raises_actionable_error(self, monkeypatch):
        """A CHM-only inventory (position + height, no morphology) raises a clear
        error pointing at the allometry endpoint before any read is attempted."""
        monkeypatch.setattr(
            inventory_io,
            "_inventory_column_names",
            lambda _id: {"x", "y", "height"},
        )

        def fail(*a, **k):
            raise AssertionError("read_parquet must not be called")

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fail)

        with pytest.raises(ProcessingError) as exc:
            read_inventory("chm_only")
        assert exc.value.code == "INVENTORY_MISSING_MORPHOLOGY"
        assert "dbh" in exc.value.message
        assert "allometry" in exc.value.suggestion
        # The reader is shared by the canopy handler, not just voxelization, so
        # its guidance must not be voxelization-specific.
        assert "voxeliz" not in exc.value.message.lower()
        assert "voxeliz" not in exc.value.suggestion.lower()

    def test_required_columns_trims_projection_and_morphology_check(self, monkeypatch):
        """A canopy request that reads neither dbh nor species passes a reduced
        required set: those columns are neither required-in-schema nor projected,
        while the live-status column is still handled."""
        present = {"x", "y", "height", "crown_ratio", "acf_kg", "fia_status_code"}
        monkeypatch.setattr(
            inventory_io, "_inventory_column_names", lambda _id: present
        )
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kwargs):
            captured["columns"] = columns
            return pd.DataFrame({c: [1.0] for c in columns})

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)

        # dbh / fia_species_code are absent from the schema, but the reduced
        # required set does not name them, so the read must not raise and must
        # not project them.
        read_inventory(
            "inv",
            biomass_column="acf_kg",
            required_columns=["x", "y", "height", "crown_ratio"],
        )
        assert "dbh" not in captured["columns"]
        assert "fia_species_code" not in captured["columns"]
        assert "acf_kg" in captured["columns"]
        assert "fia_status_code" in captured["columns"]

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

    def test_null_status_is_live(self, monkeypatch, tmp_path):
        """The pushdown keeps live and null-status trees and drops dead ones;
        null status is returned as 1."""
        path = tmp_path / "inv.parquet"
        pd.DataFrame(
            {col: [1.0, 2.0, 3.0] for col in REQUIRED_COLUMNS}
            | {"fia_status_code": pd.array([1, 2, None], dtype="Int64")}
        ).to_parquet(path)
        real_read_parquet = pd.read_parquet
        monkeypatch.setattr(
            inventory_io.pd,
            "read_parquet",
            lambda _path, **kwargs: real_read_parquet(path, **kwargs),
        )
        result = read_inventory("inv")
        assert list(result["x"]) == [1.0, 3.0]
        assert list(result["fia_status_code"]) == [1, 1]
        assert result["fia_status_code"].dtype == "int64"


class TestRequireCompleteRows:
    """`require_complete_rows` sees post-pushdown input, so every row is live."""

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

    def test_complete_rows_pass(self):
        require_complete_rows(self._df())

    def test_null_required_column_fails_with_counts(self):
        df = self._df(dbh=[None, None, 20.0], height=[15.0, None, 15.0])
        with pytest.raises(ProcessingError) as exc:
            require_complete_rows(df)
        assert exc.value.code == "INCOMPLETE_INVENTORY"
        assert "dbh: 2" in exc.value.message
        assert "height: 1" in exc.value.message

    def test_null_biomass_column_fails(self):
        df = self._df()
        df["fuel_load"] = [10.0, None, 30.0]
        with pytest.raises(ProcessingError) as exc:
            require_complete_rows(df, biomass_column="fuel_load")
        assert "fuel_load: 1" in exc.value.message

    def test_null_crown_radius_is_not_checked(self):
        """A null radius means "not measured" and falls back to allometry."""
        df = self._df()
        df["crown_radius"] = [2.5, None, 4.0]
        require_complete_rows(df)

    def test_required_columns_restricts_check(self):
        """A null in a column the request does not read is not an error."""
        df = self._df(dbh=[None, 20.0, 20.0], fia_species_code=[131, None, 131])
        require_complete_rows(df, required_columns=["x", "y", "height", "crown_ratio"])


class TestCanopyRequiredColumns:
    """The morphology columns a canopy source's selected methods actually read —
    the single authority the API router and the griddle handler share."""

    def _source(self, **overrides):
        source = {
            "biomass_source": {"type": "allometry", "equations": "brown_1978"},
            "vertical_distribution": "reinhardt_2006",
            "species_inclusion": "fuelcalc_default",
            "crown_class_adjustment": {"method": "fuelcalc_table"},
            "max_crown_radius_source": {
                "type": "allometry",
                "equations": "crookston_stage",
            },
        }
        source.update(overrides)
        return source

    def test_allometry_defaults_need_dbh_and_species(self):
        cols = canopy_required_columns(self._source())
        assert {"x", "y", "height", "crown_ratio", "dbh", "fia_species_code"} <= cols

    def test_column_radius_needs_dbh_and_species_for_fallback(self):
        """A null column radius falls back to Purves, which reads dbh and species."""
        cols = canopy_required_columns(
            self._source(
                biomass_source={
                    "type": "inventory_column",
                    "column": "acf_kg",
                    "unit": "kg",
                },
                vertical_distribution="uniform",
                species_inclusion="all_species",
                crown_class_adjustment={"method": "none"},
                max_crown_radius_source={
                    "type": "inventory_column",
                    "column": "crad_m",
                    "unit": "m",
                },
            )
        )
        assert cols == {"x", "y", "height", "crown_ratio", "dbh", "fia_species_code"}

    def test_reinhardt_distribution_needs_species(self):
        cols = canopy_required_columns(
            self._source(
                biomass_source={
                    "type": "inventory_column",
                    "column": "acf_kg",
                    "unit": "kg",
                },
                vertical_distribution="reinhardt_2006",
                species_inclusion="all_species",
                crown_class_adjustment={"method": "none"},
                max_crown_radius_source={
                    "type": "inventory_column",
                    "column": "crad_m",
                    "unit": "m",
                },
            )
        )
        assert "fia_species_code" in cols

    def test_fuelcalc_species_inclusion_needs_species(self):
        cols = canopy_required_columns(
            self._source(
                biomass_source={
                    "type": "inventory_column",
                    "column": "acf_kg",
                    "unit": "kg",
                },
                vertical_distribution="uniform",
                species_inclusion="fuelcalc_default",
                crown_class_adjustment={"method": "none"},
                max_crown_radius_source={
                    "type": "inventory_column",
                    "column": "crad_m",
                    "unit": "m",
                },
            )
        )
        assert "fia_species_code" in cols

    def test_fuelcalc_table_crown_class_needs_species(self):
        cols = canopy_required_columns(
            self._source(
                biomass_source={
                    "type": "inventory_column",
                    "column": "acf_kg",
                    "unit": "kg",
                },
                vertical_distribution="uniform",
                species_inclusion="all_species",
                crown_class_adjustment={"method": "fuelcalc_table"},
                max_crown_radius_source={
                    "type": "inventory_column",
                    "column": "crad_m",
                    "unit": "m",
                },
            )
        )
        assert "fia_species_code" in cols


class TestReadInventoryTreeId:
    def _read(self, monkeypatch, available, **kwargs):
        monkeypatch.setattr(
            inventory_io, "_inventory_column_names", lambda _id: available
        )
        captured: dict = {}

        def fake_read_parquet(path, columns=None, filters=None, **kw):
            captured["columns"] = columns
            return pd.DataFrame({c: [1] for c in columns})

        monkeypatch.setattr(inventory_io.pd, "read_parquet", fake_read_parquet)
        read_inventory("inv", **kwargs)
        return captured["columns"]

    def test_projected_when_requested(self, monkeypatch):
        columns = self._read(
            monkeypatch, {*REQUIRED_COLUMNS, "tree_id"}, include_tree_id=True
        )
        assert "tree_id" in columns

    def test_not_projected_by_default(self, monkeypatch):
        columns = self._read(monkeypatch, {*REQUIRED_COLUMNS, "tree_id"})
        assert "tree_id" not in columns
