"""Unit tests for the compose router's `_validate_alignment` helper.

Pure function over plain grid dicts — no Firestore, no server — so the CRS,
shape, and transform checks can be exercised with hand-built fixtures.
"""

import pytest
from api.resources.grids.compose.router import _validate_alignment
from api.resources.grids.compose.schema import ComposeInput
from fastapi import HTTPException

_TRANSFORM = [2.0, 0.0, 720226.0, 0.0, -2.0, 5190646.0]


def _grid(crs: str) -> dict:
    return {
        "georeference": {"shape": [50, 100], "transform": list(_TRANSFORM), "crs": crs}
    }


def _inputs(aliases: list[str]) -> dict[str, ComposeInput]:
    return {a: ComposeInput(grid_id=f"grid-{a}", alias=a) for a in aliases}


def test_equivalent_crs_spellings_pass():
    # Same CRS spelled two ways (EPSG vs OGC URN) — must not be rejected.
    source_grids = {"a": _grid("EPSG:32611"), "b": _grid("urn:ogc:def:crs:EPSG::32611")}
    _validate_alignment(source_grids, _inputs(["a", "b"]))


def test_matching_crs_passes():
    source_grids = {"a": _grid("EPSG:32611"), "b": _grid("EPSG:32611")}
    _validate_alignment(source_grids, _inputs(["a", "b"]))


def test_different_crs_rejected():
    source_grids = {"a": _grid("EPSG:32611"), "b": _grid("EPSG:4326")}
    with pytest.raises(HTTPException) as exc:
        _validate_alignment(source_grids, _inputs(["a", "b"]))
    assert exc.value.status_code == 422
    assert "CRS" in exc.value.detail
