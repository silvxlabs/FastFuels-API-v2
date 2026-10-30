"""Tests for the standgen crown base height allometry handler."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from standgen import crown_base
from standgen.crown_base import crown_base_height, fill_crown_base_height
from standgen.handlers import crown_base_height as handler
from standgen.handlers import gdam
from standgen.region import in_pacific_northwest

from lib.errors import ProcessingError

_INVENTORY = {
    "id": "newinv",
    "columns": [
        {"key": "x", "type": "continuous", "unit": "m"},
        {"key": "y", "type": "continuous", "unit": "m"},
        {"key": "height", "type": "continuous", "unit": "m"},
        {"key": "dbh", "type": "continuous", "unit": "cm"},
        {"key": "crown_ratio", "type": "continuous"},
        {"key": "fia_species_code", "type": "categorical"},
        {"key": "crown_base_height", "type": "continuous", "unit": "m"},
    ],
    "type": "tree",
}


_UTM10N = "EPSG:32610"


def _noop_progress(*args, **kwargs):
    pass


def _no_core_model(*args, **kwargs):
    raise AssertionError("the gdam model must not run a core crown base model")


def _frame(crown_base_heights=None, species=(202, 122, 202)):
    frame = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "height": [20.0, 15.0, 25.0],
            "dbh": [30.0, 25.0, 40.0],
            "crown_ratio": [0.4, 0.5, 0.0],
            "fia_species_code": pd.array(species, dtype="Int64"),
        }
    )
    if crown_base_heights is not None:
        frame["crown_base_height"] = crown_base_heights
    return frame


def _expected(frame, domain_gdf, model="fia_national_logit_2026"):
    return crown_base_height(
        frame["fia_species_code"],
        frame["crown_ratio"],
        frame["dbh"],
        frame["height"],
        pacific_northwest=in_pacific_northwest(domain_gdf),
        model=model,
    )


@pytest.fixture
def oregon_domain_gdf():
    """A domain near Eugene, Oregon (UTM zone 10N)."""
    return gpd.GeoDataFrame(
        geometry=[box(490000, 4870000, 490100, 4870100)], crs="EPSG:32610"
    )


@contextmanager
def _patched(source_frame, npartitions=1):
    """Patch the handler's parquet I/O; ``save`` computes the lazy result."""
    ddf = dd.from_pandas(source_frame, npartitions=npartitions)
    saved = {}

    def fake_save(inventory_id, result_ddf, columns, inventory_type, domain_gdf):
        saved["id"] = inventory_id
        saved["df"] = result_ddf.compute(scheduler="synchronous")
        saved["columns"] = columns
        return f"gs://test/{inventory_id}", {}, None

    with (
        patch.object(handler, "load_inventory_parquet", return_value=ddf),
        patch.object(handler, "save_parquet_with_summary", side_effect=fake_save),
    ):
        yield saved


def _run(frame, domain_gdf, **source):
    source = {"source_tree_inventory_id": "src", **source}
    with _patched(frame) as saved:
        out = handler.handle_crown_base_height(
            dict(_INVENTORY), source, domain_gdf, _noop_progress
        )
    return saved, out


class TestFillVsOverwrite:
    def test_absent_column_is_added(self, mock_domain_gdf):
        frame = _frame()
        saved, out = _run(frame, mock_domain_gdf)
        result = saved["df"]
        assert result["crown_base_height"].to_numpy() == pytest.approx(
            _expected(frame, mock_domain_gdf)
        )
        # No live crown -> crown base at the top.
        assert result.loc[2, "crown_base_height"] == 25.0
        assert saved["id"] == "newinv"
        assert out["georeference"]["crs"].upper().endswith("32610")

    def test_fill_keeps_existing_values(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, np.nan, np.nan])
        saved, _ = _run(frame, mock_domain_gdf)
        result = saved["df"]
        expected = _expected(frame, mock_domain_gdf)
        assert result.loc[0, "crown_base_height"] == 3.0
        assert result.loc[1, "crown_base_height"] == pytest.approx(expected[1])
        assert result.loc[2, "crown_base_height"] == pytest.approx(expected[2])

    def test_overwrite_recomputes_every_tree(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, 4.0, np.nan])
        saved, _ = _run(frame, mock_domain_gdf, overwrite=True)
        assert saved["df"]["crown_base_height"].to_numpy() == pytest.approx(
            _expected(frame, mock_domain_gdf)
        )

    def test_other_columns_untouched(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[np.nan, np.nan, np.nan])
        saved, _ = _run(frame, mock_domain_gdf)
        result = saved["df"]
        for col in ("x", "y", "height", "dbh", "crown_ratio", "fia_species_code"):
            pd.testing.assert_series_equal(result[col], frame[col])


class TestModel:
    def test_model_name_passed_to_core(self, mock_domain_gdf, monkeypatch):
        seen = {}

        def fake_cbh(*args, pacific_northwest, model):
            seen["model"] = model
            return np.full(len(args[0]), 2.0)

        monkeypatch.setattr(crown_base, "crown_base_height", fake_cbh)
        saved, _ = _run(_frame(), mock_domain_gdf, model={"name": "randolph_2010"})
        assert seen["model"] == "randolph_2010"
        assert list(saved["df"]["crown_base_height"]) == [2.0, 2.0, 2.0]

    def test_default_is_refit(self, mock_domain_gdf, monkeypatch):
        seen = {}

        def fake_cbh(*args, pacific_northwest, model):
            seen["model"] = model
            return np.full(len(args[0]), 2.0)

        monkeypatch.setattr(crown_base, "crown_base_height", fake_cbh)
        _run(_frame(), mock_domain_gdf)
        assert seen["model"] == "fia_national_logit_2026"

    def test_uncovered_species_gets_nan(self, mock_domain_gdf):
        # Loblolly pine (131) is outside Toney and Reeves (2009)'s species.
        frame = _frame(species=(131, 131, 131))
        frame["crown_ratio"] = [0.4, 0.5, 0.3]
        saved, _ = _run(frame, mock_domain_gdf, model={"name": "toney_reeves_2009"})
        assert saved["df"]["crown_base_height"].isna().all()

    def test_pacific_northwest_applies_to_refit_only(
        self, mock_domain_gdf, oregon_domain_gdf
    ):
        assert in_pacific_northwest(oregon_domain_gdf)
        assert not in_pacific_northwest(mock_domain_gdf)
        frame = _frame()

        def cbh(domain_gdf, model):
            saved, _ = _run(frame, domain_gdf, model={"name": model})
            return saved["df"]["crown_base_height"].to_numpy()

        refit = "fia_national_logit_2026"
        assert cbh(oregon_domain_gdf, refit)[0] != pytest.approx(
            cbh(mock_domain_gdf, refit)[0]
        )
        published = "monleon_2004_eq1"
        np.testing.assert_allclose(
            cbh(oregon_domain_gdf, published), cbh(mock_domain_gdf, published)
        )


class TestErrors:
    def test_nothing_to_fill_is_terminal(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, 4.0, 5.0])
        with _patched(frame) as saved:
            with pytest.raises(ProcessingError) as exc:
                handler.handle_crown_base_height(
                    dict(_INVENTORY),
                    {"source_tree_inventory_id": "src"},
                    mock_domain_gdf,
                    _noop_progress,
                )
        assert exc.value.code == "NOTHING_TO_FILL"
        assert "overwrite: true" in exc.value.suggestion
        assert "df" not in saved

    def test_complete_column_with_overwrite_runs(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, 4.0, 5.0])
        saved, _ = _run(frame, mock_domain_gdf, overwrite=True)
        assert saved["df"]["crown_base_height"].to_numpy() == pytest.approx(
            _expected(frame, mock_domain_gdf)
        )

    def test_absent_column_is_not_nothing_to_fill(self, mock_domain_gdf):
        saved, _ = _run(_frame(), mock_domain_gdf)
        assert saved["df"]["crown_base_height"].notna().all()

    def test_missing_model_input_raises(self, mock_domain_gdf):
        frame = _frame().drop(columns=["dbh"])
        with _patched(frame):
            with pytest.raises(ProcessingError) as exc:
                handler.handle_crown_base_height(
                    dict(_INVENTORY),
                    {"source_tree_inventory_id": "src"},
                    mock_domain_gdf,
                    _noop_progress,
                )
        assert exc.value.code == "MISSING_CROWN_BASE_INPUTS"
        assert "dbh" in exc.value.message

    def test_source_not_found_raises(self, mock_domain_gdf):
        with patch.object(
            handler, "load_inventory_parquet", side_effect=FileNotFoundError("gone")
        ):
            with pytest.raises(ProcessingError) as exc:
                handler.handle_crown_base_height(
                    dict(_INVENTORY),
                    {"source_tree_inventory_id": "src"},
                    mock_domain_gdf,
                    _noop_progress,
                )
        assert exc.value.code == "SOURCE_INVENTORY_NOT_FOUND"


class TestMeta:
    @pytest.mark.parametrize("overwrite", [False, True])
    @pytest.mark.parametrize(
        "columns",
        [
            ["x", "y", "height", "dbh", "crown_ratio", "fia_species_code"],
            [
                "crown_base_height",
                "x",
                "y",
                "height",
                "dbh",
                "crown_ratio",
                "fia_species_code",
                "tree_id",
            ],
            ["tree_id", "dbh", "crown_ratio", "crown_base_height", "fia_species_code"],
        ],
    )
    def test_partition_output_matches_meta(self, columns, overwrite):
        values = {
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "height": [20.0, 15.0],
            "dbh": [30.0, 25.0],
            "crown_ratio": [0.4, 0.5],
            "fia_species_code": pd.array([202, 122], dtype="Int64"),
            "crown_base_height": np.array([np.nan, 3.0], dtype="float32"),
            "tree_id": np.array([1, 2], dtype="int32"),
        }
        if "height" not in columns:
            columns = [*columns, "height"]
        frame = pd.DataFrame({c: values[c] for c in columns})
        meta = handler._result_meta(dd.from_pandas(frame, npartitions=1))
        for part in (frame, frame.iloc[:0]):
            out = fill_crown_base_height(part, False, overwrite=overwrite)
            assert list(out.columns) == list(meta.columns)
            assert out.dtypes.to_dict() == meta.dtypes.to_dict()


def _gdam_post(uncrcd=55.0):
    """A GDAM /predict/batch mock: 0-based index, fixed predictions, and DIA,
    CR and SPCD that differ from every source value."""

    def post(url, json=None, timeout=None):
        n = len(json["trees"]["index"])
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "predictions": {
                "columns": ["DIA", "CR", "SPCD", "UNCRCD"],
                "index": list(range(n)),
                "data": [[50.0, 10.0, 999.0, uncrcd] for _ in range(n)],
            }
        }
        return resp

    return post


def _run_gdam(frame, domain_gdf, post=None, **source):
    source = {"source_tree_inventory_id": "src", "model": {"name": "gdam"}, **source}
    with (
        _patched(frame) as saved,
        patch.object(gdam.httpx, "post", side_effect=post or _gdam_post()) as mock,
    ):
        handler.handle_crown_base_height(
            dict(_INVENTORY), source, domain_gdf, _noop_progress
        )
    return saved, mock


def _sent(mock_post):
    trees = mock_post.call_args.kwargs["json"]["trees"]
    return [dict(zip(trees["columns"], row)) for row in trees["data"]]


class TestGdamModel:
    """model: gdam takes the crown base from GDAM's UNCRCD and writes only it."""

    def test_needs_only_position_and_height(self, mock_domain_gdf, monkeypatch):
        monkeypatch.setattr(crown_base, "crown_base_height", _no_core_model)
        frame = pd.DataFrame(
            {"x": [10.0, 20.0], "y": [10.0, 20.0], "height": [10.0, 20.0]}
        )
        saved, mock_post = _run_gdam(frame, mock_domain_gdf)
        result = saved["df"]
        assert list(result.columns) == ["x", "y", "height", "crown_base_height"]
        # UNCRCD 55 percent -> crown base at 45% of the height.
        assert list(result["crown_base_height"]) == pytest.approx([4.5, 9.0])
        assert mock_post.call_args.kwargs["json"]["model"]["version"] == "v3_conus.1"

    def test_writes_only_crown_base_height(self, mock_domain_gdf):
        frame = _frame()
        frame.loc[1, ["dbh", "crown_ratio"]] = np.nan
        frame["fia_species_code"] = pd.array([202, pd.NA, 202], dtype="Int64")
        saved, mock_post = _run_gdam(frame, mock_domain_gdf)
        result = saved["df"]
        # GDAM drew dbh/CR/species for row 1, but none of them are written.
        pd.testing.assert_frame_equal(
            result.drop(columns=["crown_base_height"]), frame, check_like=False
        )
        assert result["crown_base_height"].to_numpy() == pytest.approx(
            frame["height"].to_numpy() * 0.45
        )
        # Conditioned on whatever is present; a missing value is sent as null.
        sent = _sent(mock_post)
        assert sent[0]["DIA"] == pytest.approx(30.0 / 2.54)
        assert sent[0]["CR"] == pytest.approx(40.0)
        assert sent[0]["SPCD"] == 202
        assert sent[1]["DIA"] is None
        assert sent[1]["SPCD"] is None
        assert "UNCRCD" not in sent[0]

    def test_fill_sends_and_keeps_existing_crown_bases(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, np.nan, np.nan])
        saved, mock_post = _run_gdam(frame, mock_domain_gdf)
        sent = _sent(mock_post)
        assert sent[0]["UNCRCD"] == pytest.approx(100.0 * (1 - 3.0 / 20.0))
        assert sent[1]["UNCRCD"] is None
        result = saved["df"]
        assert result.loc[0, "crown_base_height"] == 3.0
        assert list(result["crown_base_height"][1:]) == pytest.approx(
            [15.0 * 0.45, 25.0 * 0.45]
        )

    def test_overwrite_does_not_send_existing_crown_bases(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, 4.0, 5.0])
        saved, mock_post = _run_gdam(frame, mock_domain_gdf, overwrite=True)
        assert "UNCRCD" not in _sent(mock_post)[0]
        result = saved["df"]
        assert result["crown_base_height"].to_numpy() == pytest.approx(
            frame["height"].to_numpy() * 0.45
        )
        assert list(result["crown_ratio"]) == list(frame["crown_ratio"])

    def test_nothing_to_fill_is_terminal(self, mock_domain_gdf):
        frame = _frame(crown_base_heights=[3.0, 4.0, 5.0])
        with pytest.raises(ProcessingError) as exc:
            _run_gdam(frame, mock_domain_gdf)
        assert exc.value.code == "NOTHING_TO_FILL"

    def test_missing_height_column_raises(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [10.0], "y": [10.0], "dbh": [20.0]})
        with pytest.raises(ProcessingError) as exc:
            _run_gdam(frame, mock_domain_gdf)
        assert exc.value.code == "MISSING_REQUIRED_COLUMNS"

    def test_pacific_northwest_has_no_refit_fallback(
        self, oregon_domain_gdf, monkeypatch
    ):
        monkeypatch.setattr(crown_base, "crown_base_height", _no_core_model)
        saved, _ = _run_gdam(_frame(), oregon_domain_gdf)
        assert saved["df"]["crown_base_height"].to_numpy() == pytest.approx(
            _frame()["height"].to_numpy() * 0.45
        )

    def test_outside_coverage_is_terminal(self, mock_domain_gdf):
        def outside(url, json=None, timeout=None):
            return httpx.Response(
                400,
                json={"detail": "tree(s) outside GDAM coverage"},
                request=httpx.Request("POST", url),
            )

        with pytest.raises(ProcessingError) as exc:
            _run_gdam(_frame(), mock_domain_gdf, post=outside)
        assert exc.value.code == "GDAM_REQUEST_REJECTED"
        assert "outside GDAM coverage" in exc.value.message

    @pytest.mark.parametrize("overwrite", [False, True])
    @pytest.mark.parametrize(
        "columns",
        [
            ["x", "y", "height"],
            ["crown_base_height", "x", "y", "height", "crown_ratio", "tree_id"],
        ],
    )
    def test_partition_output_matches_meta(self, columns, overwrite):
        values = {
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "height": [20.0, 15.0],
            "crown_ratio": [0.4, 0.5],
            "crown_base_height": np.array([np.nan, 3.0], dtype="float32"),
            "tree_id": np.array([1, 2], dtype="int32"),
        }
        frame = pd.DataFrame({c: values[c] for c in columns})
        args = (_UTM10N, False, ["crown_base_height"], "inv", overwrite, "gdam")
        meta = gdam._result_meta(
            dd.from_pandas(frame, npartitions=1), ["crown_base_height"], overwrite
        )
        with patch.object(gdam.httpx, "post", side_effect=_gdam_post()):
            full = gdam._process_partition(frame, *args, partition_info={"number": 0})
        empty = gdam._process_partition(
            frame.iloc[:0], *args, partition_info={"number": 1}
        )
        for out in (full, empty):
            assert list(out.columns) == list(meta.columns)
            assert out.dtypes.to_dict() == meta.dtypes.to_dict()
