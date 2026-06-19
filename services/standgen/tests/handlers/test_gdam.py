"""Tests for standgen GDAM allometry handler marshalling."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import httpx
import numpy as np
import pandas as pd
import pytest
from standgen.handlers import gdam
from standgen.handlers.gdam import (
    build_batch_payload,
    fill_missing,
    parse_gdam_response,
)

from lib.errors import ProcessingError

DATA_DIR = Path(__file__).parent / "data"

# Domain CRS used in the handler fixtures (UTM zone 10N, meters).
_UTM10N = "EPSG:32610"


@pytest.fixture
def recorded_gdam_response():
    """A real GDAM /predict/batch response captured by the planning probe."""
    return json.loads((DATA_DIR / "gdam_batch_response.json").read_text())


class TestParseGdamResponse:
    """parse_gdam_response: GDAM predictions -> v2-unit imputed columns."""

    def test_parses_recorded_fixture(self, recorded_gdam_response):
        out = parse_gdam_response(recorded_gdam_response)

        # Indexed by the response index; one row per prediction.
        assert list(out.index) == [0, 1, 2]
        assert list(out.columns) == ["dbh", "crown_ratio", "fia_species_code"]

        # Row 0 of the fixture: DIA=10.78449535, CR=43.199234, SPCD=748.0
        assert out.loc[0, "dbh"] == pytest.approx(10.78449535 * 2.54)
        assert out.loc[0, "crown_ratio"] == pytest.approx(0.4319923400, rel=1e-6)
        assert out.loc[0, "fia_species_code"] == 748
        assert list(out["fia_species_code"]) == [748, 757, 748]

    def test_crown_ratio_is_a_fraction(self, recorded_gdam_response):
        """GDAM CR comes back as a percent; we store a 0-1 fraction."""
        out = parse_gdam_response(recorded_gdam_response)
        assert (out["crown_ratio"] >= 0).all()
        assert (out["crown_ratio"] <= 1).all()

    def test_selects_by_name_not_position(self):
        """Response columns may be reordered; selection must be by name."""
        response = {
            "predictions": {
                # Deliberately reordered vs the request and vs the fixture.
                "columns": ["SPCD", "inference_time_ms", "CR", "HT", "DIA"],
                "index": [5, 9],
                "data": [
                    [122.0, 1.0, 50.0, 60.0, 10.0],
                    [202.0, 1.0, 25.0, 80.0, 20.0],
                ],
            }
        }
        out = parse_gdam_response(response)
        assert list(out.index) == [5, 9]
        assert out.loc[5, "dbh"] == pytest.approx(10.0 * 2.54)
        assert out.loc[5, "crown_ratio"] == pytest.approx(0.50)
        assert out.loc[5, "fia_species_code"] == 122
        assert out.loc[9, "fia_species_code"] == 202


class TestBuildBatchPayload:
    """build_batch_payload: v2 inventory chunk -> GDAM request body."""

    def test_required_fields_and_height_conversion(self):
        df = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        payload = build_batch_payload(df, _UTM10N)
        trees = payload["trees"]

        # Only the required fields when no conditioning columns are present.
        assert trees["columns"] == ["Lat", "Lon", "HT"]
        assert trees["index"] == [0]

        lat, lon, ht = trees["data"][0]
        # 500000E on the UTM-10N central meridian -> ~ -123 lon, ~40.6 lat.
        assert lat == pytest.approx(40.6, abs=0.5)
        assert lon == pytest.approx(-123.0, abs=0.2)
        # 10 m -> ~32.808 ft.
        assert ht == pytest.approx(10.0 * 3.280839895)

    def test_conditioning_columns_converted(self):
        df = pd.DataFrame(
            {
                "x": [500000.0],
                "y": [4500000.0],
                "height": [10.0],
                "dbh": [25.4],  # cm -> 10 in
                "crown_ratio": [0.45],  # fraction -> 45 percent
                # Int64 is the dtype the uploader's pandera schema produces.
                "fia_species_code": pd.array([122], dtype="Int64"),
            }
        )
        payload = build_batch_payload(df, _UTM10N)
        trees = payload["trees"]
        assert trees["columns"] == ["Lat", "Lon", "HT", "DIA", "CR", "SPCD"]

        row = dict(zip(trees["columns"], trees["data"][0]))
        assert row["DIA"] == pytest.approx(10.0)
        assert row["CR"] == pytest.approx(45.0)
        # SPCD stays an int (no cast) because the Int64 dtype is preserved.
        assert row["SPCD"] == 122
        assert isinstance(row["SPCD"], int)

    def test_missing_values_become_null(self):
        df = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "dbh": [25.4, np.nan],  # second tree missing dbh
                # Int64 with a missing value -> int for the present one, null for NA.
                "fia_species_code": pd.array([122, pd.NA], dtype="Int64"),
            }
        )
        payload = build_batch_payload(df, _UTM10N)
        rows = [
            dict(zip(payload["trees"]["columns"], r)) for r in payload["trees"]["data"]
        ]
        assert rows[0]["DIA"] == pytest.approx(10.0)
        assert rows[0]["SPCD"] == 122
        assert isinstance(rows[0]["SPCD"], int)
        assert rows[1]["DIA"] is None
        assert rows[1]["SPCD"] is None

    def test_environmental_fields_never_sent(self):
        df = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        cols = build_batch_payload(df, _UTM10N)["trees"]["columns"]
        assert not ({"elevation", "slope", "aspect"} & set(cols))


class TestFillMissing:
    """fill_missing: fill only originally-null morphology cells."""

    def test_only_fills_nulls(self):
        source = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "height": [10.0, 12.0],
                "dbh": [30.0, np.nan],  # first present, second missing
            }
        )
        predicted = pd.DataFrame(
            {
                "dbh": [99.0, 42.0],
                "crown_ratio": [0.5, 0.6],
                "fia_species_code": pd.array([122, 202], dtype="Int64"),
            }
        )
        result = fill_missing(source, predicted)
        # Existing dbh preserved; missing one filled from prediction.
        assert list(result["dbh"]) == [30.0, 42.0]

    def test_creates_absent_column(self):
        source = pd.DataFrame(
            {"x": [1.0], "y": [1.0], "height": [10.0]}  # no morphology columns
        )
        predicted = pd.DataFrame(
            {
                "dbh": [42.0],
                "crown_ratio": [0.6],
                "fia_species_code": pd.array([202], dtype="Int64"),
            }
        )
        result = fill_missing(source, predicted)
        assert result.loc[0, "dbh"] == 42.0
        assert result.loc[0, "crown_ratio"] == 0.6
        assert result.loc[0, "fia_species_code"] == 202

    def test_preserves_position_and_height(self):
        source = pd.DataFrame(
            {"x": [1.5], "y": [2.5], "height": [11.0], "dbh": [np.nan]}
        )
        predicted = pd.DataFrame(
            {
                "dbh": [42.0],
                "crown_ratio": [0.6],
                "fia_species_code": pd.array([202], dtype="Int64"),
            }
        )
        result = fill_missing(source, predicted)
        assert result.loc[0, "x"] == 1.5
        assert result.loc[0, "y"] == 2.5
        assert result.loc[0, "height"] == 11.0

    def test_aligns_by_index_not_position(self):
        source = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "height": [10.0, 12.0],
                "dbh": [np.nan, np.nan],
            },
            index=[7, 3],
        )
        # Predicted rows intentionally in a different index order.
        predicted = pd.DataFrame(
            {
                "dbh": [11.0, 77.0],
                "crown_ratio": [0.1, 0.2],
                "fia_species_code": pd.array([1, 2], dtype="Int64"),
            },
            index=[3, 7],
        )
        result = fill_missing(source, predicted)
        assert result.loc[7, "dbh"] == 77.0
        assert result.loc[3, "dbh"] == 11.0

    def test_only_fills_selected_columns(self):
        source = pd.DataFrame({"x": [1.0], "y": [1.0], "height": [10.0]})
        predicted = pd.DataFrame(
            {
                "dbh": [42.0],
                "crown_ratio": [0.6],
                "fia_species_code": pd.array([202], dtype="Int64"),
            }
        )
        result = fill_missing(source, predicted, ["fia_species_code"])
        # Only the requested column is added; the rest are left out entirely.
        assert result.loc[0, "fia_species_code"] == 202
        assert "dbh" not in result.columns
        assert "crown_ratio" not in result.columns


# --- handle_gdam orchestration (GDAM HTTP + storage mocked) ---

_INVENTORY = {"id": "newinv"}
_SOURCE = {"source_tree_inventory_id": "src"}


def _noop_progress(*args, **kwargs):
    pass


def _predictions(index, dia=11.811, cr=40.0, spcd=122.0):
    """A GDAM predictions block with a faithful 0-based index (columns reordered).

    The live GDAM API always returns a fresh 0-based index regardless of the
    request's index (verified by the planning probe), so this mock does too:
    `index` is consumed only for its row *count*, never echoed back. Modelling
    the echo here is exactly what let the partition index-mismatch bug slip
    past the unit suite.

    DIA 11.811 in -> ~30 cm, CR 40 percent -> 0.40 fraction, SPCD 122.
    """
    n = len(list(index))
    return {
        "predictions": {
            "columns": [
                "HT",
                "elevation",
                "slope",
                "aspect",
                "DIA",
                "CR",
                "SPCD",
                "inference_time_ms",
            ],
            "index": list(range(n)),
            "data": [[10.0, 0, 0, 0, dia, cr, spcd, 1.0] for _ in range(n)],
        }
    }


def _ok_post(url, json=None, timeout=None):
    """A successful httpx response that predicts every sent tree."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = _predictions(json["trees"]["index"])
    return resp


@contextmanager
def _patched(source_frame, post_side_effect):
    """Patch the handler's parquet I/O and GDAM client for one call.

    ``save_parquet`` computes the lazy ddf (mirroring the real ``to_parquet``)
    using the synchronous scheduler, so the ``map_partitions`` GDAM calls actually
    run and any error surfaces inside ``handle_gdam`` — exactly as in production.
    Yields ``(saved, mock_post)`` where ``saved`` holds the computed result.
    """
    ddf = dd.from_pandas(source_frame, npartitions=1)
    saved = {}

    def fake_save(inventory_id, result_ddf):
        saved["id"] = inventory_id
        saved["df"] = result_ddf.compute(scheduler="synchronous")

    with (
        patch.object(gdam, "load_inventory_parquet", return_value=ddf),
        patch.object(gdam, "save_parquet", side_effect=fake_save),
        patch.object(gdam.httpx, "post", side_effect=post_side_effect) as mock_post,
    ):
        yield saved, mock_post


class TestHandleGdam:
    """End-to-end handler behavior with GDAM and storage mocked."""

    def test_happy_path_fills_missing_and_saves(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "dbh": [99.0, np.nan],  # first present, second missing
            }
        )
        with _patched(frame, _ok_post) as (saved, _):
            out = gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )

        result = saved["df"]
        assert saved["id"] == "newinv"
        # Existing dbh preserved; missing dbh filled from GDAM.
        assert result.loc[0, "dbh"] == 99.0
        assert result.loc[1, "dbh"] == pytest.approx(30.0, abs=1e-2)
        # crown_ratio / species columns created from predictions.
        assert result.loc[0, "crown_ratio"] == pytest.approx(0.40)
        assert int(result.loc[0, "fia_species_code"]) == 122
        # Position and height are untouched.
        assert list(result["height"]) == [10.0, 12.0]
        assert out["georeference"]["crs"].upper().endswith("32610")

    def test_imputes_only_selected_columns(self, mock_domain_gdf):
        # Position+height source, request only species -> dbh/crown_ratio absent.
        frame = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        source = {
            "source_tree_inventory_id": "src",
            "impute_columns": ["fia_species_code"],
        }
        with _patched(frame, _ok_post) as (saved, _):
            gdam.handle_gdam(dict(_INVENTORY), source, mock_domain_gdf, _noop_progress)
        result = saved["df"]
        assert result["fia_species_code"].notna().all()
        assert "dbh" not in result.columns
        assert "crown_ratio" not in result.columns

    def test_missing_height_raises(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0], "y": [1.0], "height": [np.nan]})
        with _patched(frame, _ok_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "MISSING_REQUIRED_HEIGHT"

    def test_missing_required_column_raises(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0], "y": [1.0]})  # no height column
        with _patched(frame, _ok_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "MISSING_REQUIRED_COLUMNS"

    def test_source_not_found_raises(self, mock_domain_gdf):
        with (
            patch.object(gdam, "load_inventory_parquet", side_effect=FileNotFoundError),
            patch.object(gdam, "save_parquet"),
            patch.object(gdam.httpx, "post", side_effect=_ok_post),
        ):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "SOURCE_INVENTORY_NOT_FOUND"

    def test_one_gdam_call_per_partition(self, mock_domain_gdf, monkeypatch):
        # Batch size of 2 over 5 trees -> 3 partitions -> 3 GDAM requests.
        monkeypatch.setattr(gdam.config, "GDAM_BATCH_SIZE", 2)
        frame = pd.DataFrame(
            {
                "x": [500000.0 + i for i in range(5)],
                "y": [4500000.0] * 5,
                "height": [10.0] * 5,
                "dbh": [np.nan] * 5,
            }
        )
        with _patched(frame, _ok_post) as (saved, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 3
        result = saved["df"]
        assert len(result) == 5
        assert result["dbh"].notna().all()
        assert sorted(result.index) == [0, 1, 2, 3, 4]

    def test_multi_partition_aligns_predictions_to_source_rows(self, mock_domain_gdf):
        """Each tree gets ITS prediction even though GDAM returns a 0-based index.

        Regression for the partition index-mismatch bug. With batch size 2, the
        second partition carries a non-zero source index ([2, 3]) while GDAM's
        response is always 0-based ([0, 1]). The handler must restore the source
        index positionally so predictions land on the right rows — and must not
        raise PARTIAL_PREDICTION. The mock encodes each tree's height into DIA, so
        any misalignment surfaces as a dbh that doesn't match the row's height.

        Pre-fix, this fails: the second partition's {2, 3} index is absent from
        the 0-based response and trips PARTIAL_PREDICTION.
        """
        heights = [10.0, 20.0, 30.0, 40.0]
        frame = pd.DataFrame(
            {
                "x": [500000.0 + i for i in range(4)],
                "y": [4500000.0] * 4,
                "height": heights,
                "dbh": [np.nan] * 4,
            }
        )

        def distinct_post(url, json=None, timeout=None):
            """Faithful 0-based GDAM mock; DIA(in) == each row's HT(ft)."""
            trees = json["trees"]
            ht_pos = trees["columns"].index("HT")
            rows = trees["data"]
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "predictions": {
                    "columns": [
                        "HT",
                        "elevation",
                        "slope",
                        "aspect",
                        "DIA",
                        "CR",
                        "SPCD",
                        "inference_time_ms",
                    ],
                    "index": list(range(len(rows))),
                    "data": [
                        [r[ht_pos], 0, 0, 0, r[ht_pos], 40.0, 122.0, 1.0] for r in rows
                    ],
                }
            }
            return resp

        with patch.object(gdam.config, "GDAM_BATCH_SIZE", 2):
            with _patched(frame, distinct_post) as (saved, mock_post):
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )

        result = saved["df"]
        assert mock_post.call_count == 2  # 4 trees / batch 2 -> 2 partitions
        assert len(result) == 4
        assert result["dbh"].notna().all()
        # dbh(cm) = DIA(in) * 2.54 = HT(ft) * 2.54 = height(m) / 0.3048 * 2.54.
        # Asserting per row proves each prediction reached its own source tree.
        for _, row in result.iterrows():
            expected_dbh = row["height"] / 0.3048 * 2.54
            assert row["dbh"] == pytest.approx(expected_dbh, rel=1e-6)

    def test_reordered_response_aligns_by_returned_index(self, mock_domain_gdf):
        """GDAM may return rows out of order; predictions follow the returned index.

        Each tree gets a distinct DIA, and the response is returned reversed (rows
        AND index together, the way the returned index labels each row's sent
        position). Each prediction must still land on the correct source row — a
        positional assignment would swap them. This is the test that distinguishes
        the index-mapping restore (``pdf.index[predicted.index]``) from a plain
        positional one (``predicted.index = pdf.index``).
        """
        frame = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "dbh": [np.nan, np.nan],
            }
        )

        def reordered(url, json=None, timeout=None):
            # 0-based, ascending by sent position: pos 0 -> DIA 10in/SPCD 122,
            # pos 1 -> DIA 20in/SPCD 202.
            idx = list(range(len(json["trees"]["index"])))
            rows = [[10.0, 40.0, 122.0, 1.0], [20.0, 40.0, 202.0, 1.0]]
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "predictions": {
                    "columns": ["DIA", "CR", "SPCD", "inference_time_ms"],
                    "index": idx[::-1],
                    "data": rows[::-1],
                }
            }
            return resp

        with _patched(frame, reordered) as (saved, _):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        by_x = saved["df"].set_index("x")
        assert by_x.loc[500000.0, "dbh"] == pytest.approx(10.0 * 2.54)
        assert by_x.loc[500030.0, "dbh"] == pytest.approx(20.0 * 2.54)
        assert int(by_x.loc[500000.0, "fia_species_code"]) == 122
        assert int(by_x.loc[500030.0, "fia_species_code"]) == 202

    def test_transport_error_raises_gdam_request_failed(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0], "y": [1.0], "height": [10.0]})

        def boom(url, json=None, timeout=None):
            raise httpx.ConnectError("boom")

        with _patched(frame, boom):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"

    def test_non_2xx_raises_gdam_request_failed(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0], "y": [1.0], "height": [10.0]})

        def server_error(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            return resp

        with _patched(frame, server_error):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"

    def test_partial_response_retries_then_raises(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "height": [10.0, 12.0]})

        def partial(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            # Drop one tree from the response.
            resp.json.return_value = _predictions(json["trees"]["index"][:1])
            return resp

        with _patched(frame, partial) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "PARTIAL_PREDICTION"
        assert mock_post.call_count == 2  # initial attempt + one retry

    def test_partial_then_complete_succeeds(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "height": [10.0, 12.0],
                "dbh": [np.nan, np.nan],
            }
        )
        calls = {"n": 0}

        def flaky(url, json=None, timeout=None):
            calls["n"] += 1
            idx = json["trees"]["index"]
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            # Partial on the first attempt, complete on the retry.
            resp.json.return_value = _predictions(idx[:1] if calls["n"] == 1 else idx)
            return resp

        with _patched(frame, flaky) as (saved, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 2
        assert saved["df"]["dbh"].notna().all()
