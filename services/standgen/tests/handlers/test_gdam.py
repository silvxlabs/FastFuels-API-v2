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
                "fia_species_code": [122],
            }
        )
        payload = build_batch_payload(df, _UTM10N)
        trees = payload["trees"]
        assert trees["columns"] == ["Lat", "Lon", "HT", "DIA", "CR", "SPCD"]

        row = dict(zip(trees["columns"], trees["data"][0]))
        assert row["DIA"] == pytest.approx(10.0)
        assert row["CR"] == pytest.approx(45.0)
        assert row["SPCD"] == 122
        assert isinstance(row["SPCD"], int)

    def test_missing_values_become_null(self):
        df = pd.DataFrame(
            {
                "x": [500000.0, 500030.0],
                "y": [4500000.0, 4500030.0],
                "height": [10.0, 12.0],
                "dbh": [25.4, np.nan],  # second tree missing dbh
                "fia_species_code": [122, np.nan],  # and species
            }
        )
        payload = build_batch_payload(df, _UTM10N)
        rows = [
            dict(zip(payload["trees"]["columns"], r)) for r in payload["trees"]["data"]
        ]
        assert rows[0]["DIA"] == pytest.approx(10.0)
        assert rows[0]["SPCD"] == 122
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


# --- handle_gdam orchestration (GDAM HTTP + storage mocked) ---

_INVENTORY = {"id": "newinv"}
_SOURCE = {"source_tree_inventory_id": "src"}


def _noop_progress(*args, **kwargs):
    pass


def _predictions(index, dia=11.811, cr=40.0, spcd=122.0):
    """A GDAM predictions block echoing `index` (columns reordered vs request).

    DIA 11.811 in -> ~30 cm, CR 40 percent -> 0.40 fraction, SPCD 122.
    """
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
            "index": list(index),
            "data": [[10.0, 0, 0, 0, dia, cr, spcd, 1.0] for _ in index],
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
    """Patch the handler's parquet I/O and GDAM client for one call."""
    ddf = dd.from_pandas(source_frame, npartitions=1)
    with (
        patch.object(gdam, "load_inventory_parquet", return_value=ddf),
        patch.object(gdam, "save_parquet") as mock_save,
        patch.object(gdam.httpx, "post", side_effect=post_side_effect) as mock_post,
    ):
        yield mock_save, mock_post


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
        with _patched(frame, _ok_post) as (mock_save, _):
            out = gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )

        mock_save.assert_called_once()
        inv_id, result_ddf = mock_save.call_args[0]
        result = result_ddf.compute()
        assert inv_id == "newinv"
        # Existing dbh preserved; missing dbh filled from GDAM.
        assert result.loc[0, "dbh"] == 99.0
        assert result.loc[1, "dbh"] == pytest.approx(30.0, abs=1e-2)
        # crown_ratio / species columns created from predictions.
        assert result.loc[0, "crown_ratio"] == pytest.approx(0.40)
        assert int(result.loc[0, "fia_species_code"]) == 122
        # Position and height are untouched.
        assert list(result["height"]) == [10.0, 12.0]
        assert out["georeference"]["crs"].upper().endswith("32610")

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

    def test_chunks_large_inventory_and_reassembles(self, mock_domain_gdf, monkeypatch):
        monkeypatch.setattr(gdam.config, "GDAM_BATCH_SIZE", 2)
        frame = pd.DataFrame(
            {
                "x": [500000.0 + i for i in range(5)],
                "y": [4500000.0] * 5,
                "height": [10.0] * 5,
                "dbh": [np.nan] * 5,
            }
        )
        with _patched(frame, _ok_post) as (mock_save, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        # 5 rows / batch 2 -> 3 chunk requests.
        assert mock_post.call_count == 3
        result = mock_save.call_args[0][1].compute()
        assert len(result) == 5
        assert result["dbh"].notna().all()
        assert list(result.index) == [0, 1, 2, 3, 4]

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

        with _patched(frame, flaky) as (mock_save, mock_post):
            gdam.handle_gdam(
                dict(_INVENTORY), dict(_SOURCE), mock_domain_gdf, _noop_progress
            )
        assert mock_post.call_count == 2
        result = mock_save.call_args[0][1].compute()
        assert result["dbh"].notna().all()
