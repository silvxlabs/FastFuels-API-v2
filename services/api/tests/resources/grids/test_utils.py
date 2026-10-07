"""
Unit tests for api/v2/resources/grids/utils.py validators.

These are pure-logic tests: they call the validators directly with fabricated
grid_data dicts and assert on the raised HTTPException. No server or Firestore
is required.
"""

from unittest.mock import patch

import pytest
from api.resources.exports.schema import GridExportFormat
from api.resources.grids.utils import (
    validate_format_supports_grid,
    validate_grids_share_horizontal_lattice,
    validate_lfps_coverage,
)
from fastapi import HTTPException

from lib.landfire import CoverageStatus
from tests.fixtures import make_domain_data


def _grid(shape):
    """Minimal grid_data dict carrying a georeference of the given shape."""
    return {"georeference": {"shape": shape}}


_TRANSFORM = (2.0, 0.0, 500000.0, 0.0, -2.0, 5201000.0)


def _lattice_grid(
    grid_id="grid",
    shape=(40, 40),
    crs="EPSG:32611",
    transform=_TRANSFORM,
):
    return {
        "id": grid_id,
        "georeference": {
            "shape": shape,
            "crs": crs,
            "transform": transform,
        },
    }


class TestValidateGridsShareHorizontalLattice:
    def test_matching_3d_and_2d_grids_pass(self):
        validate_grids_share_horizontal_lattice(
            _lattice_grid("lad-grid", shape=(6, 40, 40)),
            _lattice_grid("terrain-grid", shape=(40, 40)),
        )

    def test_equivalent_crs_spellings_pass(self):
        validate_grids_share_horizontal_lattice(
            _lattice_grid("lad-grid", crs="EPSG:32611"),
            _lattice_grid("terrain-grid", crs="urn:ogc:def:crs:EPSG::32611"),
        )

    def test_different_crs_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            validate_grids_share_horizontal_lattice(
                _lattice_grid("lad-grid", crs="EPSG:32611"),
                _lattice_grid("terrain-grid", crs="EPSG:4326"),
            )
        assert exc.value.status_code == 422
        assert "CRS" in exc.value.detail

    def test_different_horizontal_shape_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            validate_grids_share_horizontal_lattice(
                _lattice_grid("lad-grid", shape=(6, 40, 40)),
                _lattice_grid("terrain-grid", shape=(20, 20)),
            )
        assert exc.value.status_code == 422
        assert "shape" in exc.value.detail

    def test_different_transform_raises_422(self):
        shifted = (*_TRANSFORM[:2], 500001.0, *_TRANSFORM[3:])
        with pytest.raises(HTTPException) as exc:
            validate_grids_share_horizontal_lattice(
                _lattice_grid("lad-grid"),
                _lattice_grid("terrain-grid", transform=shifted),
            )
        assert exc.value.status_code == 422
        assert "transform" in exc.value.detail


class TestValidateFormatSupportsGrid:
    """validate_format_supports_grid rejects 3D grids for 2D-only formats."""

    def test_geotiff_3d_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            validate_format_supports_grid(
                _grid((5, 34, 34)), "grid-1", GridExportFormat.geotiff
            )
        assert exc.value.status_code == 422
        detail = exc.value.detail
        assert "geotiff" in detail
        assert "netcdf" in detail and "zarr" in detail

    def test_geotiff_2d_passes(self):
        # A 2D grid is fine for geotiff — no exception.
        validate_format_supports_grid(
            _grid((34, 34)), "grid-1", GridExportFormat.geotiff
        )

    def test_zarr_3d_passes(self):
        # zarr supports 3D natively.
        validate_format_supports_grid(
            _grid((5, 34, 34)), "grid-1", GridExportFormat.zarr
        )

    def test_netcdf_3d_passes(self):
        # netcdf supports 3D natively.
        validate_format_supports_grid(
            _grid((5, 34, 34)), "grid-1", GridExportFormat.netcdf
        )

    def test_missing_georeference_is_noop(self):
        # Without a georeference the dimensionality is unknown; the check is a
        # no-op (the router's completed-status fetch guarantees one in practice).
        validate_format_supports_grid({}, "grid-1", GridExportFormat.geotiff)

    class TestValidateLfpsCoverage:
        """validate_lfps_coverage checks LFPS coverage before grid creation.

        covers_annual/covers_seasonal are mocked so no real HTTP call to LFPS
        happens.
        """

        def _domain(self):
            return make_domain_data()

        @pytest.mark.parametrize(
            "coverage", [CoverageStatus.FULL, CoverageStatus.PARTIAL]
        )
        def test_seasonal_full_or_partial_passes(self, coverage):
            with patch(
                "api.resources.grids.utils.covers_seasonal", return_value=coverage
            ):
                validate_lfps_coverage("fbfm40", "2025", self._domain(), season="SP")

        @pytest.mark.parametrize(
            "coverage", [CoverageStatus.FULL, CoverageStatus.PARTIAL]
        )
        def test_annual_full_or_partial_passes(self, coverage):
            with patch(
                "api.resources.grids.utils.covers_annual", return_value=coverage
            ):
                validate_lfps_coverage("fbfm40", "2024", self._domain())

        @pytest.mark.parametrize(
            "coverage", [CoverageStatus.NONE, CoverageStatus.NO_SUCH_PRODUCT]
        )
        def test_seasonal_none_or_no_such_product_raises_422(self, coverage):
            with patch(
                "api.resources.grids.utils.covers_seasonal", return_value=coverage
            ):
                with pytest.raises(HTTPException) as exc:
                    validate_lfps_coverage(
                        "fbfm40", "2025", self._domain(), season="SP"
                    )
            assert exc.value.status_code == 422

        @pytest.mark.parametrize(
            "coverage", [CoverageStatus.NONE, CoverageStatus.NO_SUCH_PRODUCT]
        )
        def test_annual_none_or_no_such_product_raises_422(self, coverage):
            with patch(
                "api.resources.grids.utils.covers_annual", return_value=coverage
            ):
                with pytest.raises(HTTPException) as exc:
                    validate_lfps_coverage("fbfm40", "2024", self._domain())
            assert exc.value.status_code == 422

        def test_season_none_calls_covers_annual_not_seasonal(self):
            with patch(
                "api.resources.grids.utils.covers_annual",
                return_value=CoverageStatus.FULL,
            ) as mock_annual:
                with patch(
                    "api.resources.grids.utils.covers_seasonal"
                ) as mock_seasonal:
                    validate_lfps_coverage("fbfm40", "2024", self._domain())
            mock_annual.assert_called_once()
            mock_seasonal.assert_not_called()

        def test_season_set_calls_covers_seasonal_not_annual(self):
            with patch(
                "api.resources.grids.utils.covers_seasonal",
                return_value=CoverageStatus.FULL,
            ) as mock_seasonal:
                with patch("api.resources.grids.utils.covers_annual") as mock_annual:
                    validate_lfps_coverage(
                        "fbfm40", "2025", self._domain(), season="SP"
                    )
            mock_seasonal.assert_called_once()
            mock_annual.assert_not_called()

        def test_seasonal_call_args(self):
            with patch(
                "api.resources.grids.utils.covers_seasonal",
                return_value=CoverageStatus.FULL,
            ) as mock_seasonal:
                validate_lfps_coverage("fbfm40", "2025", self._domain(), season="FA")
            args, _ = mock_seasonal.call_args
            assert args[0] == "fbfm40"
            assert args[1] == "2025"
            assert args[2] == "FA"

        def test_no_such_product_message_mentions_publication(self):
            with patch(
                "api.resources.grids.utils.covers_seasonal",
                return_value=CoverageStatus.NO_SUCH_PRODUCT,
            ):
                with pytest.raises(HTTPException) as exc:
                    validate_lfps_coverage(
                        "fbfm40", "2025", self._domain(), season="FA"
                    )
            assert "isn't currently published" in exc.value.detail
            assert "landfire.gov/fuel/seasonal_fuels" in exc.value.detail

        def test_none_message_mentions_domain_location(self):
            with patch(
                "api.resources.grids.utils.covers_annual",
                return_value=CoverageStatus.NONE,
            ):
                with pytest.raises(HTTPException) as exc:
                    validate_lfps_coverage("fbfm40", "2024", self._domain())
            assert "doesn't currently cover this domain's location" in exc.value.detail
            assert "landfire.gov/data" in exc.value.detail
