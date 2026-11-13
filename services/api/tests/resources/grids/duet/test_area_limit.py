"""
Unit tests for the DUET source-grid area limit in api/resources/grids/duet/router.py.

Pure function tests with no external dependencies.
"""

import pytest
from api.resources.grids.duet.router import validate_source_area
from fastapi import HTTPException


def _grid(height: int, width: int, resolution: float = 2.0) -> dict:
    return {
        "georeference": {
            "crs": "EPSG:32611",
            "transform": [resolution, 0.0, 500000.0, 0.0, -resolution, 4000000.0],
            "shape": [37, height, width],
            "z_resolution": 1.0,
            "z_origin": 0.0,
        }
    }


class TestValidateSourceArea:
    def test_exactly_one_square_kilometre_is_allowed(self):
        validate_source_area(_grid(500, 500), "g1")  # 1000 m x 1000 m

    def test_a_larger_grid_is_rejected_with_its_area(self):
        # Blue Mountain, the docs and SDK test stand: 442 x 654 at 2 m.
        with pytest.raises(HTTPException) as exc:
            validate_source_area(_grid(442, 654), "g1")
        assert exc.value.status_code == 422
        assert "1.16 km²" in exc.value.detail
        assert "limited to 1 km²" in exc.value.detail

    def test_area_follows_the_cell_size(self):
        # The same cell count at 1 m covers a quarter of the area.
        validate_source_area(_grid(884, 1000, resolution=1.0), "g1")
        with pytest.raises(HTTPException):
            validate_source_area(_grid(884, 1000, resolution=2.0), "g1")
