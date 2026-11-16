"""
Unit tests for the DUET source-grid size limit in api/resources/grids/duet/router.py.

Pure function tests with no external dependencies.
"""

import pytest
from api.resources.grids.duet.router import validate_source_size
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


class TestValidateSourceSize:
    def test_exactly_the_cell_limit_is_allowed(self):
        validate_source_size(_grid(600, 1000), "g1")

    def test_one_cell_over_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            validate_source_size(_grid(601, 1000), "g1")
        assert exc.value.status_code == 422

    def test_the_limit_is_cells_not_area(self):
        # The same 1.16 km² Blue Mountain stand: fine at 2 m, too many cells at 1 m.
        validate_source_size(_grid(442, 654, resolution=2.0), "g1")
        with pytest.raises(HTTPException):
            validate_source_size(_grid(884, 1308, resolution=1.0), "g1")

    def test_error_names_the_cells_and_the_area_allowed_at_this_cell_size(self):
        # The SDK's shared test grid, from #755.
        with pytest.raises(HTTPException) as exc:
            validate_source_size(_grid(703, 863, resolution=2.0), "g1")
        detail = exc.value.detail
        assert "606,689 horizontal cells (703 × 863, 2.43 km²)" in detail
        assert "limited to 600,000 cells: 2.40 km² at this grid's cell size" in detail
