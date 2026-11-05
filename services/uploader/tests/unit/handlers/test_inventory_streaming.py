"""
The streaming inventory upload against the eager, whole-file reference (#764).

``_eager`` is the pre-#764 handler's data path: the whole file read with
pandas / geopandas and validated with pandera at once. The streaming handler
must produce the same rows, tree IDs, bounds, provided columns, Parquet dtypes
and errors, except for the four accepted differences tested at the end.
Batches are kept small so every case spans several of them.
"""

import ast
import importlib.util
import math
import sys
import types
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pytest
import shapely
from pandera.errors import SchemaErrors
from shapely.geometry import box
from uploader.handlers import inventory, inventory_reader

from lib.errors import ProcessingError

DOMAIN_CRS = "EPSG:32610"
XMIN, YMIN, XMAX, YMAX = 500_000.0, 4_200_000.0, 510_000.0, 4_210_000.0
BOUNDS = (XMIN, YMIN, XMAX, YMAX)
BATCH_SIZE = 1000

# File column names for the mapped-CSV cases (v2 name → file name).
CSV_COLUMNS = {"height": "HT", "dbh": "DIA", "fia_species_code": "SPCD"}
SPECIES = np.array([122, 202, 108, 15, 19, 93, 81, 116, 746, 65])

_CACHE_PY = (
    Path(__file__).parents[4] / "api/api/resources/inventories/cache.py"
).resolve()


@pytest.fixture(autouse=True)
def small_batches(monkeypatch):
    monkeypatch.setattr(
        inventory,
        "csv_batches",
        partial(inventory_reader.csv_batches, batch_size=BATCH_SIZE),
    )
    monkeypatch.setattr(
        inventory,
        "geo_batches",
        partial(inventory_reader.geo_batches, batch_size=BATCH_SIZE),
    )


def _trees(n: int, seed: int, with_id: bool = True) -> pd.DataFrame:
    """Synthetic trees in the domain, about 5% in a band just outside it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(XMIN, XMAX, n)
    y = rng.uniform(YMIN, YMAX, n)
    out = rng.random(n) < 0.05
    side = rng.integers(0, 4, out.sum())
    d = rng.uniform(1, 500, out.sum())
    x[out] = np.where(side == 0, XMIN - d, np.where(side == 1, XMAX + d, x[out]))
    y[out] = np.where(side == 2, YMIN - d, np.where(side == 3, YMAX + d, y[out]))
    height = np.clip(rng.lognormal(2.6, 0.5, n), 1.5, 80).round(1)
    dbh = np.clip(height * rng.normal(1.6, 0.3, n), 2.5, 250).round(1)
    cr = np.clip(rng.beta(4, 4, n), 0.05, 0.95).round(3)

    def masked(values, fraction):
        return np.where(rng.random(n) < fraction, np.nan, values)

    df = pd.DataFrame(
        {
            "tree_id": rng.permutation(np.arange(1, n + 1)) * 3,
            "x": x.round(2),
            "y": y.round(2),
            "height": height,
            "dbh": masked(dbh, 0.08),
            "fia_species_code": masked(SPECIES[rng.integers(0, 10, n)], 0.03),
            "fia_status_code": np.where(rng.random(n) < 0.9, 1, 2),
            "fia_crown_class_code": masked(rng.integers(1, 6, n), 0.2),
            "crown_ratio": cr,
            "crown_base_height": masked(np.floor(height * (1 - cr) * 100) / 100, 0.05),
            "crown_radius": np.clip(0.08 * dbh + rng.normal(0, 0.3, n), 0.3, 15).round(
                2
            ),
            "plot_id": rng.integers(1, 5000, n),
        }
    )
    return df if with_id else df.drop(columns="tree_id")


def _geo(df: pd.DataFrame, crs: str, multipoint_every: int = 0) -> gpd.GeoDataFrame:
    """Trees as point features in ``crs``; every k-th feature folds in the next
    tree as a MultiPoint."""
    tx, ty = pyproj.Transformer.from_crs(DOMAIN_CRS, crs, always_xy=True).transform(
        df["x"].to_numpy(), df["y"].to_numpy()
    )
    points = shapely.points(tx, ty)
    props = df.drop(columns=["x", "y"]).reset_index(drop=True)
    if multipoint_every:
        geoms, keep, i = [], [], 0
        while i < len(points):
            if i % multipoint_every == 0 and i + 1 < len(points):
                geoms.append(shapely.multipoints(points[i : i + 2]))
                keep.append(i)
                i += 2
            else:
                geoms.append(points[i])
                keep.append(i)
                i += 1
        props = props.iloc[keep].reset_index(drop=True)
        points = np.array(geoms, dtype=object)
    return gpd.GeoDataFrame(props, geometry=points, crs=crs)


def _write_geo(gdf: gpd.GeoDataFrame, path: Path, rfc7946: bool = True) -> Path:
    if path.suffix == ".gpkg":
        gdf.to_file(path, driver="GPKG", engine="pyogrio")
    else:
        options = {"RFC7946": "YES" if rfc7946 else "NO"}
        gdf.to_file(path, driver="GeoJSON", engine="pyogrio", layer_options=options)
    return path


def _eager(fmt: str, path: Path, col_map: dict) -> dict:
    """The pre-#764 handler on the whole file: parse, pad, validate, filter,
    number. Returns its rows (as Parquet would store them), bounds, provided
    columns and pandas column summaries."""
    rename = {user_col: v2_name for v2_name, user_col in col_map.items()}
    if fmt == "csv":
        df = pd.read_csv(path).rename(columns=rename)
        df = df[[c for c in df.columns if c in inventory._V2_COLUMNS]]
    else:
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ProcessingError(
                code="EMPTY_FILE", message="The uploaded file contains no features."
            )
        geom_types = set(gdf.geometry.geom_type.unique())
        unsupported = geom_types - {"Point", "MultiPoint"}
        if unsupported:
            raise ProcessingError(
                code="INVALID_GEOMETRY_TYPE",
                message=f"Expected Point or MultiPoint geometries, found: {unsupported}.",
            )
        if "MultiPoint" in geom_types:
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        if gdf.crs is None and fmt == "geojson":
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs is not None:
            gdf = gdf.to_crs(DOMAIN_CRS)
        df = pd.DataFrame({"x": gdf.geometry.x.values, "y": gdf.geometry.y.values})
        for col in gdf.columns:
            v2_col = rename.get(col, col)
            if v2_col in inventory._V2_COLUMNS and v2_col not in ("x", "y"):
                df[v2_col] = gdf[col].values

    provided = [c for c in inventory._COLUMN_METADATA if c in df.columns]
    for col in inventory._OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    try:
        df = inventory._InventorySchema.validate(df, lazy=True)
    except SchemaErrors as e:
        raise ProcessingError(
            code="SCHEMA_VALIDATION_ERROR",
            message=f"Schema validation failed with {len(e.failure_cases)} error(s).",
            suggestion=str(e.failure_cases.head(100).to_dict("records")),
        )
    df = df[(df.x >= XMIN) & (df.x <= XMAX) & (df.y >= YMIN) & (df.y <= YMAX)]
    if df.empty:
        raise ProcessingError(
            code="EMPTY_AFTER_FILTER",
            message="No trees remain after filtering to domain bounds.",
        )
    if "tree_id" in df.columns:
        df["tree_id"] = df["tree_id"].astype("int32")
    else:
        df = df.reset_index(drop=True)
        df.insert(0, "tree_id", np.arange(1, len(df) + 1, dtype="int32"))
        provided.append("tree_id")
    df = df.reset_index(drop=True)
    stored = pa.Table.from_pandas(df, preserve_index=False).to_pandas()
    summaries = {}
    for key, (col_type, _) in inventory._COLUMN_METADATA.items():
        if key in provided:
            s = df[key]
            summaries[key] = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
            }
            if col_type == "continuous":
                summaries[key] |= {
                    "min": s.min(),
                    "max": s.max(),
                    "mean": s.mean(),
                    "std": s.std(),
                }
            elif key != "tree_id":
                summaries[key]["unique_count"] = int(s.nunique())
    return {
        "rows": stored,
        "bounds": [df.x.min(), df.y.min(), df.x.max(), df.y.max()],
        "provided": [c for c in inventory._COLUMN_METADATA if c in provided],
        "summaries": summaries,
    }


def _stream(tmp_path: Path, monkeypatch, fmt: str, path: Path, col_map: dict) -> dict:
    """Run handle_inventory on ``path`` with Firestore and GCS stubbed."""
    out = tmp_path / "out"
    out.mkdir()
    updates = {}

    class _Snap:
        def to_dict(self):
            return {"crs": DOMAIN_CRS}

    read_table = inventory._read_table
    write_parquet = inventory._write_parquet
    monkeypatch.setattr(
        inventory,
        "_read_table",
        lambda fmt, src, *args: read_table(fmt, str(path), *args),
    )
    monkeypatch.setattr(inventory, "get_document", lambda c, i: (None, _Snap()))
    monkeypatch.setattr(
        inventory,
        "parse_domain_gdf",
        lambda d: gpd.GeoDataFrame(geometry=[box(*BOUNDS)], crs=DOMAIN_CRS),
    )
    monkeypatch.setattr(
        inventory,
        "_write_parquet",
        lambda t, b, p, *a: write_parquet(t, b, str(out), *a),
    )
    monkeypatch.setattr(inventory, "storage_size", lambda p: 0)
    monkeypatch.setattr(inventory, "delete_file", lambda p: None)
    monkeypatch.setattr(inventory, "update_resource", lambda c, i, u: updates.update(u))
    doc = {"domain_id": "d", "source": {"format": fmt, "columns": col_map}}
    inventory.handle_inventory("inv", "bucket", "inventories/inv/upload", doc)
    return {"out": out, **updates}


def _outcome(run) -> ProcessingError | dict:
    try:
        return run()
    except ProcessingError as e:
        return e


def _cases(error: ProcessingError) -> list[dict]:
    return ast.literal_eval(error.suggestion.replace("nan", "None"))


def _assert_same_result(eager: dict, streamed: dict) -> None:
    rows = pd.read_parquet(streamed["out"])
    # Accepted difference: the Parquet column order is canonical.
    assert list(rows.columns) == list(inventory._COLUMN_METADATA)
    rows = rows.sort_values("tree_id").reset_index(drop=True)
    expected = eager["rows"].sort_values("tree_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[expected.columns], expected, check_exact=True)

    assert streamed["georeference"] == {"crs": DOMAIN_CRS, "bounds": eager["bounds"]}
    assert [c["key"] for c in streamed["columns"]] == eager["provided"]
    # Accepted difference: Arrow float statistics match pandas to rounding.
    for column in streamed["columns"]:
        for stat, expected_value in eager["summaries"][column["key"]].items():
            value = column["summary"][stat]
            if isinstance(expected_value, float) and not math.isfinite(expected_value):
                assert value is None
            else:
                assert value == pytest.approx(expected_value, rel=1e-9)


def _assert_same_error(eager, streamed) -> None:
    assert isinstance(eager, ProcessingError), eager
    assert isinstance(streamed, ProcessingError), streamed
    assert (streamed.code, streamed.message) == (eager.code, eager.message)
    if eager.code == "SCHEMA_VALIDATION_ERROR":
        assert streamed.suggestion == eager.suggestion


def _compare(tmp_path, monkeypatch, fmt, path, col_map=None):
    col_map = col_map or {}
    eager = _outcome(lambda: _eager(fmt, path, col_map))
    streamed = _outcome(lambda: _stream(tmp_path, monkeypatch, fmt, path, col_map))
    if isinstance(eager, ProcessingError):
        _assert_same_error(eager, streamed)
    else:
        assert not isinstance(streamed, ProcessingError), streamed
        _assert_same_result(eager, streamed)
    return streamed


class TestMatchesEager:
    def test_csv_user_tree_id_and_column_mapping(self, tmp_path, monkeypatch):
        path = tmp_path / "trees.csv"
        col_map = {**CSV_COLUMNS, "tree_id": "TreeID"}
        _trees(6000, 1).rename(columns=col_map).to_csv(path, index=False)
        _compare(tmp_path, monkeypatch, "csv", path, col_map)

    def test_csv_generated_ids_null_and_absent_columns(self, tmp_path, monkeypatch):
        """No tree_id, an all-empty dbh, crown_radius absent (padded, not
        reported)."""
        df = _trees(6000, 2, with_id=False).drop(columns="crown_radius")
        df["dbh"] = np.nan
        path = tmp_path / "trees.csv"
        df.to_csv(path, index=False)
        _compare(tmp_path, monkeypatch, "csv", path)

    def test_geojson_without_crs_and_multipoints(self, tmp_path, monkeypatch):
        gdf = _geo(_trees(3000, 3, with_id=False), "EPSG:4326", multipoint_every=7)
        path = _write_geo(gdf, tmp_path / "trees.geojson")
        _compare(tmp_path, monkeypatch, "geojson", path)

    def test_geojson_in_domain_crs_with_mapped_tree_id(self, tmp_path, monkeypatch):
        df = _trees(3000, 4).rename(columns={"tree_id": "TID"})
        path = _write_geo(_geo(df, DOMAIN_CRS), tmp_path / "trees.geojson", False)
        _compare(tmp_path, monkeypatch, "geojson", path, {"tree_id": "TID"})

    def test_geopackage_in_other_crs_with_multipoints(self, tmp_path, monkeypatch):
        gdf = _geo(_trees(3000, 5, with_id=False), "EPSG:5070", multipoint_every=11)
        path = _write_geo(gdf, tmp_path / "trees.gpkg")
        _compare(tmp_path, monkeypatch, "geopackage", path)

    @pytest.mark.parametrize("suffix", [".geojson", ".gpkg"])
    def test_empty_geo_file(self, tmp_path, monkeypatch, suffix):
        gdf = _geo(_trees(10, 6), "EPSG:4326").iloc[:0]
        path = _write_geo(gdf, tmp_path / f"empty{suffix}")
        fmt = "geojson" if suffix == ".geojson" else "geopackage"
        streamed = _compare(tmp_path, monkeypatch, fmt, path)
        assert streamed.code == "EMPTY_FILE"

    @pytest.mark.parametrize("text", ["x,y,height\n", "x,y,height", "x,y,height\r"])
    def test_header_only_csv(self, tmp_path, monkeypatch, text):
        path = tmp_path / "trees.csv"
        path.write_bytes(text.encode())
        streamed = _compare(tmp_path, monkeypatch, "csv", path)
        assert streamed.code == "EMPTY_AFTER_FILTER"

    @pytest.mark.parametrize(
        "text",
        [
            "﻿x,y,Höhe\n500100,4200100,3\n",
            "x,y,Höhe\r\n500100,4200100,3\r\n",
            "x,y,Höhe\r500100,4200100,3\r500200,4200200,4",
        ],
        ids=["bom_unicode_header", "crlf", "cr"],
    )
    def test_csv_encodings_and_line_endings(self, tmp_path, monkeypatch, text):
        path = tmp_path / "trees.csv"
        path.write_bytes(text.encode())
        _compare(tmp_path, monkeypatch, "csv", path, {"height": "Höhe"})

    def test_all_trees_outside_domain(self, tmp_path, monkeypatch):
        df = _trees(3000, 13)
        df["x"] += 50_000
        path = tmp_path / "trees.csv"
        df.to_csv(path, index=False)
        streamed = _compare(tmp_path, monkeypatch, "csv", path)
        assert streamed.code == "EMPTY_AFTER_FILTER"

    def test_missing_required_column(self, tmp_path, monkeypatch):
        path = tmp_path / "trees.csv"
        _trees(3000, 12).drop(columns="x").to_csv(path, index=False)
        streamed = _compare(tmp_path, monkeypatch, "csv", path)
        assert streamed.code == "SCHEMA_VALIDATION_ERROR"

    def test_failures_spread_across_batches(self, tmp_path, monkeypatch):
        """Range, isin and crown-base-height failures in several batches, and
        duplicate tree_ids both across and within batches."""
        df = _trees(6000, 8)
        rows = [5, 2500, 5900]
        df.loc[rows, "crown_base_height"] = df.loc[rows, "height"] + 1
        df.loc[[10, 4000], "height"] = 130.0
        df.loc[11, "crown_ratio"] = 1.5
        df.loc[[12, 5000], "fia_status_code"] = 7
        df.loc[4500, "tree_id"] = df.loc[3, "tree_id"]
        df.loc[4501, "tree_id"] = df.loc[4502, "tree_id"]
        path = tmp_path / "trees.csv"
        df.rename(columns=CSV_COLUMNS).to_csv(path, index=False)
        streamed = _compare(tmp_path, monkeypatch, "csv", path, CSV_COLUMNS)
        assert "field_uniqueness" in streamed.suggestion

    @pytest.mark.parametrize("fmt", ["csv", "geojson"])
    def test_padded_columns_in_crown_base_height_failures(
        self, tmp_path, monkeypatch, fmt
    ):
        """Frames that fail the crown-base-height check, with the other
        optional columns missing and padded with typed nulls (#778)."""
        df = _trees(6000, 16, with_id=False)[["x", "y", "height", "crown_base_height"]]
        rows = [5, 2500, 5900]
        df.loc[rows, "crown_base_height"] = df.loc[rows, "height"] + 1
        if fmt == "csv":
            path = tmp_path / "trees.csv"
            df.to_csv(path, index=False)
        else:
            path = _write_geo(_geo(df, "EPSG:4326"), tmp_path / "trees.geojson")
        streamed = _compare(tmp_path, monkeypatch, fmt, path)
        assert streamed.code == "SCHEMA_VALIDATION_ERROR"
        assert {c["check"] for c in _cases(streamed)} == {
            "crown_base_height must not exceed height"
        }
        assert {c["index"] for c in _cases(streamed)} == set(rows)

    def test_more_than_100_failures(self, tmp_path, monkeypatch):
        """Several failing checks, so the order of the first 100 cases matters."""
        df = _trees(6000, 9, with_id=False)
        df.loc[np.arange(100, 6000, 41), "height"] = -1.0
        bad_cbh = np.arange(50, 6000, 53)
        df.loc[bad_cbh, "crown_base_height"] = df.loc[bad_cbh, "height"] + 3
        df.loc[np.arange(7, 6000, 501), "crown_radius"] = 25.0
        path = tmp_path / "trees.csv"
        df.to_csv(path, index=False)
        streamed = _compare(tmp_path, monkeypatch, "csv", path)
        assert len(_cases(streamed)) == 100

    def test_geojson_failures_index_exploded_points(self, tmp_path, monkeypatch):
        gdf = _geo(_trees(3000, 10, with_id=False), "EPSG:4326", multipoint_every=5)
        gdf.loc[gdf.index[::31], "height"] = 200.0
        gdf.loc[gdf.index[5::77], "crown_ratio"] = -0.2
        path = _write_geo(gdf, tmp_path / "trees.geojson")
        _compare(tmp_path, monkeypatch, "geojson", path)

    def test_non_numeric_cell_late_in_file(self, tmp_path, monkeypatch):
        """Forces the CSV re-read as strings."""
        df = _trees(6000, 11, with_id=False).astype({"height": object})
        df.loc[5500, "height"] = "abc"
        df.loc[[20, 4000], "crown_ratio"] = 2.0
        path = tmp_path / "trees.csv"
        df.to_csv(path, index=False)
        streamed = _compare(tmp_path, monkeypatch, "csv", path)
        assert "'failure_case': 'abc'" in streamed.suggestion

    def test_polygon_after_schema_failures(self, tmp_path, monkeypatch):
        """A bad geometry wins over schema failures in earlier batches."""
        gdf = _geo(_trees(3000, 7), "EPSG:4326")
        gdf.loc[3, "height"] = -5.0
        gdf.loc[2000, "geometry"] = shapely.box(-122.9, 37.9, -122.8, 38.0)
        path = _write_geo(gdf, tmp_path / "trees.geojson")
        streamed = _compare(tmp_path, monkeypatch, "geojson", path)
        assert streamed.code == "INVALID_GEOMETRY_TYPE"


class TestAcceptedDifferences:
    def test_range_failures_reported_beside_a_non_numeric_cell(
        self, tmp_path, monkeypatch
    ):
        """1. With a non-numeric cell, the eager handler reported one generic
        type error for every range check on that column. Batches without the
        cell now report their real range failures."""
        df = _trees(6000, 11, with_id=False).astype({"height": object})
        df.loc[5500, "height"] = "abc"
        df.loc[50, "height"] = -1.0
        path = tmp_path / "trees.csv"
        df.to_csv(path, index=False)

        eager = _outcome(lambda: _eager("csv", path, {}))
        streamed = _outcome(lambda: _stream(tmp_path, monkeypatch, "csv", path, {}))
        assert streamed.code == eager.code == "SCHEMA_VALIDATION_ERROR"
        assert not any(c["index"] == 50 for c in _cases(eager))
        [range_case] = [
            c
            for c in _cases(streamed)
            if c["index"] == 50 and c["check"] == "greater_than_or_equal_to(0)"
        ]
        assert range_case["failure_case"] == -1.0

    def test_geometry_types_from_first_failing_batch(self, tmp_path, monkeypatch):
        """2. INVALID_GEOMETRY_TYPE lists the bad types of the first failing
        batch, not the whole file."""
        gdf = _geo(_trees(3000, 7), "EPSG:4326")
        gdf.loc[1500, "geometry"] = shapely.box(-122.9, 37.9, -122.8, 38.0)
        gdf.loc[2500, "geometry"] = shapely.linestrings([[-122.9, 37.9], [-122.8, 38]])
        path = _write_geo(gdf, tmp_path / "trees.geojson")

        eager = _outcome(lambda: _eager("geojson", path, {}))
        streamed = _outcome(lambda: _stream(tmp_path, monkeypatch, "geojson", path, {}))
        assert streamed.code == eager.code == "INVALID_GEOMETRY_TYPE"
        assert "'LineString'" in eager.message
        assert streamed.message == (
            "Expected Point or MultiPoint geometries, found: {'Polygon'}."
        )

    # 3. Float summaries within 1e-9 and 4. canonical Parquet column order are
    # asserted for every success case by _assert_same_result.


class TestReadErrors:
    def test_missing_upload_is_file_not_found(self, tmp_path):
        """main.py records SOURCE_NOT_FOUND for this instead of retrying."""
        with pytest.raises(FileNotFoundError):
            inventory._read_table(
                "geojson", str(tmp_path / "gone.geojson"), {}, DOMAIN_CRS, BOUNDS
            )

    @pytest.mark.parametrize(
        "bad_row, detail",
        [("500300,4200300", "got 2"), ("500300,4200300,12,4", "got 4")],
        ids=["short_row", "long_row"],
    )
    def test_malformed_row_is_terminal(self, tmp_path, bad_row, detail):
        """A row with too few or too many fields fails the upload instead of
        being retried."""
        rows = [f"{500_000 + i},{4_200_000 + i},10" for i in range(3000)]
        rows[2500] = bad_row
        path = tmp_path / "trees.csv"
        path.write_text("x,y,height\n" + "\n".join(rows) + "\n")
        with pytest.raises(ProcessingError) as exc_info:
            inventory._read_table("csv", str(path), {}, DOMAIN_CRS, BOUNDS)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"
        assert "Expected 3 columns" in exc_info.value.message
        assert detail in exc_info.value.message
        assert bad_row in exc_info.value.message

    def test_empty_csv_is_retried(self, tmp_path):
        """A zero-byte CSV raises a non-terminal error, as before #764."""
        path = tmp_path / "trees.csv"
        path.write_bytes(b"")
        with pytest.raises(pa.ArrowInvalid):
            inventory._read_table("csv", str(path), {}, DOMAIN_CRS, BOUNDS)


def test_read_table_has_one_chunk_per_column(tmp_path):
    """take on a many-chunk table is superlinear, so batches are combined."""
    path = tmp_path / "trees.csv"
    _trees(6000, 15).to_csv(path, index=False)
    table = inventory._read_table("csv", str(path), {}, DOMAIN_CRS, BOUNDS)
    assert {table.column(c).num_chunks for c in table.column_names} == {1}


def _rows_table(x, y) -> pa.Table:
    n = len(x)
    return pa.table(
        {
            "tree_id": pa.array(np.arange(1, n + 1), pa.int32()),
            "x": x,
            "y": y,
            "height": np.full(n, 10.0),
        }
    )


class TestRowGroupLayout:
    @pytest.mark.parametrize(
        "n, sizes",
        [
            (1, [1]),
            (inventory.ROW_GROUP_SIZE, [inventory.ROW_GROUP_SIZE]),
            (inventory.ROW_GROUP_SIZE + 1, [inventory.ROW_GROUP_SIZE, 1]),
        ],
    )
    def test_row_groups_capped(self, tmp_path, n, sizes):
        rng = np.random.default_rng(0)
        table = _rows_table(rng.uniform(XMIN, XMAX, n), rng.uniform(YMIN, YMAX, n))
        inventory._write_parquet(table, BOUNDS, str(tmp_path))
        metadata = pq.read_metadata(tmp_path / "_metadata")
        assert [
            metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
        ] == sizes

    def test_row_groups_spatially_compact(self, tmp_path):
        """1M trees in random order over a 10 km domain: each 50k-row group
        covers a few neighbouring 1 km tiles, not the whole domain."""
        n = 20 * inventory.ROW_GROUP_SIZE
        rng = np.random.default_rng(1)
        table = _rows_table(rng.uniform(XMIN, XMAX, n), rng.uniform(YMIN, YMAX, n))
        inventory._write_parquet(table, BOUNDS, str(tmp_path))

        pf = pq.ParquetFile(tmp_path / "part.0.parquet")
        assert pf.num_row_groups == 20
        extents = []
        for i in range(pf.num_row_groups):
            rg = pf.read_row_group(i, columns=["x", "y"]).to_pandas()
            extents.append(max(np.ptp(rg["x"]), np.ptp(rg["y"])))
            assert len(set(zip(rg["x"] // 1000, rg["y"] // 1000))) <= 8
        assert np.median(extents) <= 3_500
        assert max(extents) <= 5_500

    def test_sort_is_by_tile_then_y_then_x(self, tmp_path):
        x = np.array([500_900.0, 500_100.0, 500_500.0, 501_500.0])
        y = np.array([4_200_100.0, 4_200_100.0, 4_200_050.0, 4_200_000.0])
        inventory._write_parquet(_rows_table(x, y), BOUNDS, str(tmp_path))
        stored = pd.read_parquet(tmp_path)
        assert stored["tree_id"].tolist() == [3, 2, 1, 4]


class TestApiReadsOutput:
    """The API's own ``_metadata`` parser and row-group reader page the output."""

    @pytest.fixture
    def api_cache(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "ring", types.SimpleNamespace(lru=lambda **k: lambda f: f)
        )
        spec = importlib.util.spec_from_file_location("api_inventory_cache", _CACHE_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_partitions_page_every_row(self, tmp_path, monkeypatch, api_cache):
        path = tmp_path / "trees.csv"
        _trees(3000, 14).to_csv(path, index=False)
        monkeypatch.setattr(inventory, "ROW_GROUP_SIZE", 1000)
        streamed = _stream(tmp_path, monkeypatch, "csv", path, {})

        out = streamed["out"]
        meta = api_cache._parse_metadata(pq.read_metadata(out / "_metadata"))
        assert meta.columns == list(inventory._COLUMN_METADATA)
        assert meta.num_partitions == 3
        assert all(p.num_rows <= 1000 for p in meta.partitions)
        pages = [
            api_cache._read_row_group_sync(str(out / p.path), p.row_group, None)
            for p in meta.partitions
        ]
        stored = pd.concat(pages, ignore_index=True)
        assert len(stored) == meta.total_rows == len(pd.read_parquet(out))
        assert stored["tree_id"].is_unique
        assert stored["fia_species_code"].dtype == "Int64"
