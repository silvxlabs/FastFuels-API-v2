"""
Live contract test for the GDAM /predict/batch API.

The unit tests in ``tests/handlers/test_gdam.py`` mock GDAM, so they only prove
the handler is correct *given our assumptions about GDAM's contract*. They cannot
detect GDAM changing its response shape, units, or row/index semantics — the mock
would keep encoding the old contract and stay green.

This test calls the **live** GDAM API directly (not through the deployed standgen
container, so it is independent of ``DEPLOYMENT_ENV`` and needs no GCS/Firestore).
It pins every property ``standgen/handlers/gdam.py`` relies on. If an assertion
here fails, the live contract has drifted and the handler must be updated to
match — each assertion message names the handler line that depends on it.

Contract pinned:
  * one prediction row per sent tree                  -> _predict count check
  * a fresh 0-based positional index, NOT an echo of
    the request index                                 -> _process_partition's
                                                         ``pdf.index[predicted.index]``
                                                         restore (needs 0-based
                                                         positions, not an echo)
  * rows returned in request order (via echoed HT)    -> canary only: the restore
                                                         maps by returned index, so
                                                         correctness survives a
                                                         reorder; this pins that
                                                         GDAM doesn't reorder today
  * DIA / CR / SPCD present *by name* (cols reorder)  -> parse_gdam_response
  * CR is a 0-100 percent, not a 0-1 fraction         -> parse_gdam_response's /100
  * SPCD is a positive FIA code                       -> fia_species_code mapping
"""

import httpx
import pandas as pd
import pytest
from standgen import config as standgen_config
from standgen.handlers.gdam import build_batch_payload

# Three points on land near the Blue Mountain domain (Missoula, MT), already in
# lon/lat so build_batch_payload's reprojection is an identity (CRS EPSG:4326).
# Distinct heights let us verify the response preserves request row order.
_LONS = [-114.110, -114.098, -114.090]
_LATS = [46.826, 46.832, 46.840]
_HEIGHTS_M = [12.0, 25.0, 18.0]
# Deliberately non-zero, shuffled index: if GDAM ever *echoed* the request index
# the response would come back [7, 3, 5]; the live contract returns [0, 1, 2].
_REQUEST_INDEX = [7, 3, 5]
_M_PER_FT = 0.3048


@pytest.fixture(scope="module")
def gdam_contract_response():
    """POST one small, controlled batch to the live GDAM API for the module.

    Returns ``(request_payload, response_json)``. A transport/HTTP failure here
    (the ``raise_for_status``) is itself a contract signal — GDAM is unreachable
    or rejected a request the handler builds the same way.
    """
    df = pd.DataFrame(
        {"x": _LONS, "y": _LATS, "height": _HEIGHTS_M}, index=_REQUEST_INDEX
    )
    payload = build_batch_payload(df, "EPSG:4326")

    response = httpx.post(
        f"{standgen_config.GDAM_API_URL}/predict/batch",
        json=payload,
        timeout=standgen_config.GDAM_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return payload, response.json()


def _frame(predictions: dict) -> pd.DataFrame:
    """Build the response frame exactly as parse_gdam_response does."""
    return pd.DataFrame(
        predictions["data"],
        columns=predictions["columns"],
        index=predictions["index"],
    )


def test_one_prediction_per_sent_tree(gdam_contract_response):
    _, body = gdam_contract_response
    n = len(body["predictions"]["data"])
    assert n == len(_HEIGHTS_M), (
        f"GDAM returned {n} predictions for {len(_HEIGHTS_M)} trees — "
        "_predict assumes one row per sent tree."
    )


def test_index_is_zero_based_not_echoed(gdam_contract_response):
    """The load-bearing property: GDAM returns 0-based, not the request index."""
    payload, body = gdam_contract_response
    idx = list(body["predictions"]["index"])
    assert idx == list(range(len(_HEIGHTS_M))), (
        f"GDAM index is no longer 0-based positional (got {idx}). "
        "_process_partition maps it back via pdf.index[predicted.index], which "
        "assumes the returned index is 0-based positions — revisit it."
    )
    assert idx != payload["trees"]["index"], (
        "GDAM now echoes the request index. _process_partition uses the returned "
        "index as positions into pdf.index, so an echoed (non-0-based) index "
        "would index out of bounds."
    )


def test_rows_returned_in_request_order(gdam_contract_response):
    """Canary: GDAM returns rows in request order (echoed HT matches sent HT).

    The handler no longer depends on this — it aligns by the returned index via
    pdf.index[predicted.index], so a reorder is handled correctly. This pins the
    simpler invariant GDAM provides today; the 'reordered' unit test is what
    actually exercises the out-of-order path.
    """
    _, body = gdam_contract_response
    frame = _frame(body["predictions"])
    assert "HT" in frame.columns, "GDAM no longer echoes HT; can't verify row order."
    sent_ht_ft = [h / _M_PER_FT for h in _HEIGHTS_M]
    returned_ht_ft = frame["HT"].to_numpy(dtype=float)
    assert returned_ht_ft == pytest.approx(sent_ht_ft, rel=1e-3), (
        "GDAM no longer returns rows in request order. Alignment is still correct "
        "via pdf.index[predicted.index], but the contract has shifted."
    )


def test_imputed_fields_present_by_name(gdam_contract_response):
    _, body = gdam_contract_response
    cols = set(body["predictions"]["columns"])
    missing = {"DIA", "CR", "SPCD"} - cols
    assert not missing, (
        f"GDAM response is missing expected field(s) {missing}; "
        "parse_gdam_response selects DIA/CR/SPCD by name."
    )


def test_crown_ratio_is_percent_not_fraction(gdam_contract_response):
    """CR must come back as a 0-100 percent (the documented GDAM gotcha)."""
    _, body = gdam_contract_response
    cr = _frame(body["predictions"])["CR"].to_numpy(dtype=float)
    assert ((cr >= 0) & (cr <= 100)).all(), f"CR outside 0-100: {cr}"
    assert cr.max() > 1.5, (
        "GDAM CR now looks like a 0-1 fraction, not a 0-100 percent. "
        "parse_gdam_response divides CR by 100 — that conversion must change."
    )


def test_species_is_positive_fia_code(gdam_contract_response):
    _, body = gdam_contract_response
    spcd = _frame(body["predictions"])["SPCD"].to_numpy(dtype=float)
    assert (spcd > 0).all(), f"GDAM returned a non-positive SPCD: {spcd}"
