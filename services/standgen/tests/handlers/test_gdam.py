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
from standgen.crown_base import crown_base_height
from standgen.handlers import gdam
from standgen.handlers.gdam import (
    batch_seed,
    build_batch_payload,
    fill_missing,
    parse_gdam_response,
)
from standgen.region import in_pacific_northwest

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
        payload = build_batch_payload(df, _UTM10N, seed=1)
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
        payload = build_batch_payload(df, _UTM10N, seed=1)
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
        payload = build_batch_payload(df, _UTM10N, seed=1)
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
        cols = build_batch_payload(df, _UTM10N, seed=1)["trees"]["columns"]
        assert not ({"elevation", "slope", "aspect"} & set(cols))

    def test_names_pinned_flow_model_with_seed(self):
        df = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        payload = build_batch_payload(df, _UTM10N, seed=42)
        assert payload["model"] == {
            "name": "flow_matching",
            "version": "v2_west.2",
            "tree_observation": "overhead",
            "seed": 42,
        }


class TestBatchSeed:
    """batch_seed: stable per (inventory, partition), distinct across them."""

    def test_stable(self):
        assert batch_seed("inv-a", 3) == batch_seed("inv-a", 3)

    def test_distinct_across_partitions_and_inventories(self):
        seeds = {batch_seed(inv, p) for inv in ("inv-a", "inv-b") for p in range(50)}
        assert len(seeds) == 100

    def test_in_gdam_seed_range(self):
        for p in range(200):
            assert 0 <= batch_seed("inv-a", p) <= 2**32 - 1


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

_INVENTORY = {
    "id": "newinv",
    "columns": [
        {"key": "x", "type": "continuous", "unit": "m"},
        {"key": "y", "type": "continuous", "unit": "m"},
        {"key": "height", "type": "continuous", "unit": "m"},
        {"key": "dbh", "type": "continuous", "unit": "cm"},
        {"key": "crown_ratio", "type": "continuous"},
        {"key": "fia_species_code", "type": "categorical"},
    ],
    "type": "tree",
}
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


def _flaky_post(exc, fail_times):
    """A post side_effect that raises ``exc`` the first ``fail_times`` calls, then
    delegates to ``_ok_post`` — drives the transient-retry path in ``_post_batch``."""
    calls = {"n": 0}

    def side_effect(url, json=None, timeout=None):
        if calls["n"] < fail_times:
            calls["n"] += 1
            raise exc
        return _ok_post(url, json=json, timeout=timeout)

    return side_effect


def _status_then_ok(status, fail_times, retry_after=None):
    """A post side_effect whose first ``fail_times`` responses carry ``status``
    (with the given Retry-After headers, one per failure), then ``_ok_post``."""
    calls = {"n": 0}

    def side_effect(url, json=None, timeout=None):
        if calls["n"] < fail_times:
            headers = {}
            if retry_after:
                headers["Retry-After"] = retry_after[calls["n"]]
            calls["n"] += 1
            return httpx.Response(
                status, headers=headers, request=httpx.Request("POST", url)
            )
        return _ok_post(url, json=json, timeout=timeout)

    return side_effect


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

    def fake_save(inventory_id, result_ddf, columns, inventory_type, domain_gdf):
        saved["id"] = inventory_id
        saved["df"] = result_ddf.compute(scheduler="synchronous")
        saved["columns"] = columns
        return f"gs://test/{inventory_id}", {}, None

    with (
        patch.object(gdam, "load_inventory_parquet", return_value=ddf),
        patch.object(gdam, "save_parquet_with_summary", side_effect=fake_save),
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
            patch.object(
                gdam,
                "save_parquet_with_summary",
                return_value=("gs://test/newinv", {}, None),
            ),
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

    def test_each_partition_sends_its_own_stable_seed(
        self, mock_domain_gdf, monkeypatch
    ):
        # 5 trees / batch 2 -> 3 partitions, each seeded batch_seed(id, partition).
        monkeypatch.setattr(gdam.config, "GDAM_BATCH_SIZE", 2)
        frame = pd.DataFrame(
            {
                "x": [500000.0 + i for i in range(5)],
                "y": [4500000.0] * 5,
                "height": [10.0] * 5,
            }
        )
        runs = []
        for _ in range(2):
            with _patched(frame, _ok_post) as (_, mock_post):
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
            runs.append(
                sorted(
                    c.kwargs["json"]["model"]["seed"] for c in mock_post.call_args_list
                )
            )
        expected = sorted(batch_seed(_INVENTORY["id"], p) for p in range(3))
        assert runs[0] == expected
        assert runs[1] == expected  # a rerun reproduces the seeds

    def test_non_positive_height_raises(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "height": [10.0, 0.0]})
        with _patched(frame, _ok_post) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "NON_POSITIVE_HEIGHT"
        assert mock_post.call_count == 0

    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    def test_retries_transient_status_then_succeeds(
        self, mock_domain_gdf, monkeypatch, status
    ):
        sleeps = []
        monkeypatch.setattr(gdam.time, "sleep", sleeps.append)
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [np.nan]}
        )
        side_effect = _status_then_ok(status, fail_times=2)
        with _patched(frame, side_effect) as (saved, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 3
        assert sleeps == [0.5, 1.0]  # exponential backoff, no Retry-After
        assert saved["df"].loc[0, "dbh"] == pytest.approx(30.0, abs=1e-2)

    def test_honours_retry_after_capped(self, mock_domain_gdf, monkeypatch):
        sleeps = []
        monkeypatch.setattr(gdam.time, "sleep", sleeps.append)
        frame = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        side_effect = _status_then_ok(429, fail_times=2, retry_after=["7", "600"])
        with _patched(frame, side_effect) as (_, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 3
        assert sleeps == [7.0, gdam._MAX_RETRY_AFTER_S]

    def test_terminal_after_exhausting_429_retries(self, mock_domain_gdf, monkeypatch):
        monkeypatch.setattr(gdam.time, "sleep", lambda *_: None)
        frame = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        side_effect = _status_then_ok(429, fail_times=99)
        with _patched(frame, side_effect) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"
        assert mock_post.call_count == gdam.config.GDAM_MAX_ATTEMPTS

    @pytest.mark.parametrize("status", [400, 422, 500])
    def test_other_status_is_terminal_without_retry(self, mock_domain_gdf, status):
        frame = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        with _patched(frame, _status_then_ok(status, fail_times=99)) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"
        assert mock_post.call_count == 1

    def test_retries_transient_transport_error_then_succeeds(
        self, mock_domain_gdf, monkeypatch
    ):
        # Two dropped connections, then success — the inventory should complete.
        monkeypatch.setattr(gdam.time, "sleep", lambda *_: None)  # skip real backoff
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [np.nan]}
        )
        side_effect = _flaky_post(httpx.ConnectError("connection reset"), fail_times=2)
        with _patched(frame, side_effect) as (saved, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 3  # 2 failures + 1 success
        assert saved["df"].loc[0, "dbh"] == pytest.approx(30.0, abs=1e-2)

    def test_terminal_after_exhausting_transport_retries(
        self, mock_domain_gdf, monkeypatch
    ):
        monkeypatch.setattr(gdam.time, "sleep", lambda *_: None)
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [np.nan]}
        )
        side_effect = _flaky_post(
            httpx.RemoteProtocolError("server disconnected"), fail_times=99
        )
        with _patched(frame, side_effect) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"
        assert mock_post.call_count == gdam.config.GDAM_MAX_ATTEMPTS

    def test_read_timeout_is_terminal_without_retry(self, mock_domain_gdf, monkeypatch):
        # A slow timeout already burned the per-request budget — never retried.
        monkeypatch.setattr(gdam.time, "sleep", lambda *_: None)
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [np.nan]}
        )
        side_effect = _flaky_post(httpx.ReadTimeout("timed out"), fail_times=99)
        with _patched(frame, side_effect) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"
        assert mock_post.call_count == 1

    def test_http_status_error_is_terminal_without_retry(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [np.nan]}
        )

        def bad_status(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            return resp

        with _patched(frame, bad_status) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "GDAM_REQUEST_FAILED"
        assert mock_post.call_count == 1

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

    def test_preserves_source_tree_ids(self, mock_domain_gdf):
        """Every tree keeps its source tree_id, gaps included, across
        partitions (#611)."""
        frame = pd.DataFrame(
            {
                "tree_id": np.array([4, 900, 12, 7], dtype="int32"),
                "x": [500000.0 + i for i in range(4)],
                "y": [4500000.0] * 4,
                "height": [10.0, 20.0, 30.0, 40.0],
            }
        )
        with patch.object(gdam.config, "GDAM_BATCH_SIZE", 2):
            with _patched(frame, _ok_post) as (saved, _):
                gdam.handle_gdam(
                    dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
                )

        result = saved["df"].reset_index(drop=True)
        assert result["tree_id"].dtype == np.int32
        assert list(result["tree_id"]) == [4, 900, 12, 7]
        assert list(result["height"]) == [10.0, 20.0, 30.0, 40.0]

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


class TestCrownBaseHeight:
    """crown_base_height comes from the crown ratio model, never from GDAM."""

    def test_default_fills_from_gdam_filled_columns(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "crown_ratio": [0.7, np.nan],
            }
        )
        with _patched(frame, _ok_post) as (saved, _):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        result = saved["df"]
        expected = crown_base_height(
            result["fia_species_code"],
            result["crown_ratio"],
            result["dbh"],
            result["height"],
            pacific_northwest=in_pacific_northwest(mock_domain_gdf),
        )
        assert list(result["crown_ratio"]) == pytest.approx([0.7, 0.40])
        assert result["crown_base_height"].to_numpy() == pytest.approx(expected)
        assert result["crown_base_height"].notna().all()

    def test_keeps_source_values_and_fills_nulls(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "crown_base_height": [2.5, np.nan],
            }
        )
        with _patched(frame, _ok_post) as (saved, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        result = saved["df"]
        # Not sent to GDAM as conditioning.
        sent = mock_post.call_args.kwargs["json"]["trees"]["columns"]
        assert sent == ["Lat", "Lon", "HT"]
        expected = crown_base_height(
            result["fia_species_code"],
            result["crown_ratio"],
            result["dbh"],
            result["height"],
            pacific_northwest=in_pacific_northwest(mock_domain_gdf),
        )
        assert result.loc[0, "crown_base_height"] == 2.5
        assert result.loc[1, "crown_base_height"] == pytest.approx(expected[1])

    def test_only_crown_base_height_makes_no_gdam_request(self, mock_domain_gdf):
        # A zero and a missing height would fail a GDAM run; here they only
        # leave those trees without a crown base.
        frame = pd.DataFrame(
            {
                "x": [500000.0 + i for i in range(4)],
                "y": [4500000.0] * 4,
                "height": [10.0, 20.0, 0.0, np.nan],
                "dbh": [20.0, 40.0, 10.0, 10.0],
                "crown_ratio": [0.5, 0.0, 0.5, 0.5],
                "fia_species_code": pd.array([122, 202, 122, 122], dtype="Int64"),
            }
        )
        source = {
            "source_tree_inventory_id": "src",
            "impute_columns": ["crown_base_height"],
        }
        with _patched(frame, _ok_post) as (saved, mock_post):
            gdam.handle_gdam(dict(_INVENTORY), source, mock_domain_gdf, _noop_progress)
        assert mock_post.call_count == 0
        result = saved["df"]
        expected = crown_base_height(
            frame["fia_species_code"],
            frame["crown_ratio"],
            frame["dbh"],
            frame["height"],
            pacific_northwest=in_pacific_northwest(mock_domain_gdf),
        )
        cbh = result["crown_base_height"].to_numpy()
        assert cbh == pytest.approx(expected, nan_ok=True)
        assert cbh[1] == 20.0  # no live crown -> crown base at the top
        assert np.isnan(cbh[2:]).all()
        assert list(result["dbh"]) == [20.0, 40.0, 10.0, 10.0]

    def test_dbh_only_adds_no_crown_base_height(self, mock_domain_gdf):
        frame = pd.DataFrame({"x": [500000.0], "y": [4500000.0], "height": [10.0]})
        source = {"source_tree_inventory_id": "src", "impute_columns": ["dbh"]}
        with _patched(frame, _ok_post) as (saved, _):
            gdam.handle_gdam(dict(_INVENTORY), source, mock_domain_gdf, _noop_progress)
        assert "crown_base_height" not in saved["df"].columns

    def test_missing_model_input_raises(self, mock_domain_gdf):
        frame = pd.DataFrame(
            {"x": [500000.0], "y": [4500000.0], "height": [10.0], "dbh": [20.0]}
        )
        source = {
            "source_tree_inventory_id": "src",
            "impute_columns": ["crown_base_height", "fia_species_code"],
        }
        with _patched(frame, _ok_post) as (_, mock_post):
            with pytest.raises(ProcessingError) as exc:
                gdam.handle_gdam(
                    dict(_INVENTORY), source, mock_domain_gdf, _noop_progress
                )
        assert exc.value.code == "MISSING_CROWN_BASE_INPUTS"
        assert mock_post.call_count == 0

    @pytest.mark.parametrize(
        "source_columns",
        [
            ["x", "y", "height"],
            ["x", "y", "height", "crown_base_height", "tree_id"],
            ["x", "crown_base_height", "y", "height", "crown_ratio"],
        ],
    )
    def test_partition_output_matches_meta(self, mock_domain_gdf, source_columns):
        values = {
            "x": [500000.0, 500001.0],
            "y": [4500000.0] * 2,
            "height": [10.0, 12.0],
            "crown_base_height": [np.nan, 3.0],
            "crown_ratio": [0.5, np.nan],
            "tree_id": np.array([1, 2], dtype="int32"),
        }
        frame = pd.DataFrame({c: values[c] for c in source_columns})
        ddf = dd.from_pandas(frame, npartitions=1)
        columns = list(gdam.IMPUTED_COLUMNS)
        meta = gdam._result_meta(ddf, columns)

        with patch.object(gdam.httpx, "post", side_effect=_ok_post):
            full = gdam._process_partition(
                frame, _UTM10N, True, columns, "inv", partition_info={"number": 0}
            )
        empty = gdam._process_partition(
            frame.iloc[:0], _UTM10N, True, columns, "inv", partition_info={"number": 1}
        )
        for out in (full, empty):
            assert list(out.columns) == list(meta.columns)
            assert out.dtypes.to_dict() == meta.dtypes.to_dict()
