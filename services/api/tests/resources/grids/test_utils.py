"""
Unit tests for api/v2/resources/grids/utils.py validators.

These are pure-logic tests: they call the validators directly with fabricated
grid_data dicts and assert on the raised HTTPException. No server or Firestore
is required.
"""

from unittest.mock import patch

import pytest
from api.resources.exports.schema import GridExportFormat
from api.resources.features.crown.schema import CreateChmCrownFeatureRequest
from api.resources.grids.alignment import GridLatticeGridTarget
from api.resources.grids.compose.schema import CreateComposeRequest
from api.resources.grids.lookup.fbfm40.schema import CreateFbfm40LookupRequest
from api.resources.grids.providers.landfire import (
    LANDFIRE_DISTURBANCE_FDIST_VERSION,
    LANDFIRE_DISTURBANCE_LDIST_VERSION,
    LANDFIRE_DISTURBANCE_RELEASES,
    LandfireDisturbanceRequest,
)
from api.resources.grids.utils import (
    incomplete_inventory_columns,
    validate_band_of,
    resolve_disturbance_plan,
    validate_format_supports_grid,
    validate_grid_input,
    validate_grids_share_horizontal_lattice,
    validate_lfps_coverage,
    validate_target_grid_alignment,
)
from api.resources.grids.voxelize.inventory.tree.schema import (
    CreateTreeInventoryRequest,
)
from api.resources.input_requirements import X_INPUT_REQUIREMENTS, requirements_for
from api.resources.inventories.tree.pim.schema import CreatePimInventoryRequest
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


class TestValidateTargetGridAlignmentCrs:
    @pytest.mark.parametrize(
        "domain_crs,grid_crs,ok",
        [
            ("EPSG:32611", "urn:ogc:def:crs:EPSG::32611", True),
            ("EPSG:32611", "EPSG:5070", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_target_grid_crs_must_match_domain_crs(
        self, domain_crs, grid_crs, ok
    ):
        snapshot = type(
            "S", (), {"to_dict": lambda self: _lattice_grid("g", crs=grid_crs)}
        )()
        alignment = GridLatticeGridTarget(target="grid", grid_id="g")
        with patch(
            "api.resources.grids.utils.get_document_async",
            return_value=(None, snapshot),
        ):
            call = validate_target_grid_alignment(
                alignment, "owner", "dom", domain_crs=domain_crs
            )
            if ok:
                await call
                return
            with pytest.raises(HTTPException) as exc:
                await call
        assert exc.value.status_code == 422
        assert grid_crs in exc.value.detail
        assert domain_crs in exc.value.detail


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


class TestResolveDisturbancePlan:
    """resolve_disturbance_plan checks the baseline and newest-release coverage.

    covers_annual is mocked so no real HTTP call to LFPS happens.
    """

    BASELINES = [
        (product, releases["baseline"])
        for product, releases in LANDFIRE_DISTURBANCE_RELEASES.items()
    ]

    def _domain(self):
        return make_domain_data()

    def _request(self):
        return LandfireDisturbanceRequest(source="landfire")

    @pytest.mark.parametrize("product,baseline", BASELINES)
    @pytest.mark.parametrize(
        "coverage",
        [CoverageStatus.PARTIAL, CoverageStatus.NONE, CoverageStatus.NO_SUCH_PRODUCT],
    )
    def test_newest_not_fully_covering_returns_plan(self, product, baseline, coverage):
        with patch("api.resources.grids.utils.covers_annual", return_value=coverage):
            plan = resolve_disturbance_plan(
                product, baseline, self._domain(), self._request()
            )
        assert plan.source == "landfire"
        assert plan.mode == "ldist_and_last_year_fdist"
        assert plan.ldist_version == LANDFIRE_DISTURBANCE_LDIST_VERSION
        assert plan.fdist_version == LANDFIRE_DISTURBANCE_FDIST_VERSION

    @pytest.mark.parametrize("product,baseline", BASELINES)
    def test_newest_fully_covering_raises_422(self, product, baseline):
        with patch(
            "api.resources.grids.utils.covers_annual",
            return_value=CoverageStatus.FULL,
        ):
            with pytest.raises(HTTPException) as exc:
                resolve_disturbance_plan(
                    product, baseline, self._domain(), self._request()
                )
        assert exc.value.status_code == 422
        assert "already fully covers" in exc.value.detail

    @pytest.mark.parametrize("product,baseline", BASELINES)
    def test_checks_the_newest_release(self, product, baseline):
        with patch(
            "api.resources.grids.utils.covers_annual",
            return_value=CoverageStatus.NONE,
        ) as mock_annual:
            resolve_disturbance_plan(product, baseline, self._domain(), self._request())
        args, _ = mock_annual.call_args
        assert args[0] == product
        assert args[1] == LANDFIRE_DISTURBANCE_RELEASES[product]["newest"]

    def test_non_baseline_version_raises_422_without_calling_lfps(self):
        with patch("api.resources.grids.utils.covers_annual") as mock_annual:
            with pytest.raises(HTTPException) as exc:
                resolve_disturbance_plan(
                    "fbfm13", "2023", self._domain(), self._request()
                )
        assert exc.value.status_code == 422
        baseline = LANDFIRE_DISTURBANCE_RELEASES["fbfm13"]["baseline"]
        assert f"'{baseline}'" in exc.value.detail
        mock_annual.assert_not_called()

    @pytest.mark.parametrize("disturbance", [None, object()])
    def test_non_landfire_disturbance_is_skipped(self, disturbance):
        with patch("api.resources.grids.utils.covers_annual") as mock_annual:
            plan = resolve_disturbance_plan(
                "fbfm13", "2023", self._domain(), disturbance
            )
        assert plan is None
        mock_annual.assert_not_called()


class TestIncompleteInventoryColumns:
    def _inventory(self, null_counts: dict) -> dict:
        return {
            "columns": [
                {"key": key, "summary": {"null_count": n}}
                for key, n in null_counts.items()
            ]
        }

    def test_complete_columns_pass(self):
        inv = self._inventory({"x": 0, "dbh": 0})
        assert incomplete_inventory_columns(inv, {"x", "dbh"}) == set()

    def test_missing_and_null_columns_are_incomplete(self):
        inv = self._inventory({"x": 0, "dbh": 3})
        assert incomplete_inventory_columns(inv, {"x", "dbh", "height"}) == {
            "dbh",
            "height",
        }

    def test_nullable_column_only_needs_to_be_listed(self):
        inv = self._inventory({"crown_radius": 5})
        assert (
            incomplete_inventory_columns(inv, {"crown_radius"}, {"crown_radius"})
            == set()
        )
        assert incomplete_inventory_columns({}, {"crown_radius"}, {"crown_radius"}) == {
            "crown_radius"
        }

    def test_columns_without_summaries_count_as_complete(self):
        """Older inventories have no summaries; the worker backstop covers them."""
        inv = {"columns": [{"key": "dbh"}, "x"]}
        assert incomplete_inventory_columns(inv, {"dbh", "x"}) == set()


def _input_grid(bands, source="pim", shape=(4, 4)):
    return {
        "source": {"name": source},
        "bands": [{"key": key} for key in bands],
        "georeference": {"shape": list(shape)},
    }


class TestValidateGridInput:
    body = CreatePimInventoryRequest(source_pim_grid_id="g1")

    def test_declared_requirements_met(self):
        validate_grid_input(self.body, "source_pim_grid_id", _input_grid(["tm_id"]))

    def test_wrong_source_rejected(self):
        with pytest.raises(HTTPException, match="'pim' grid"):
            validate_grid_input(
                self.body,
                "source_pim_grid_id",
                _input_grid(["tm_id"], source="compose"),
            )

    def test_missing_band_rejected(self):
        with pytest.raises(HTTPException, match="missing required bands"):
            validate_grid_input(
                self.body, "source_pim_grid_id", _input_grid(["plt_cn"])
            )


class TestValidateBandOf:
    def _body(self, band):
        return CreateFbfm40LookupRequest(
            source_grid_id="g1", source_band=band, bands=["fuel_load.1hr"]
        )

    def test_band_of_declared_quantity_accepted(self):
        grid = _input_grid(["chm", "fbfm"])
        validate_band_of(self._body("fbfm"), "source_band", grid)

    def test_band_of_other_quantity_rejected(self):
        grid = _input_grid(["chm", "fbfm"])
        with pytest.raises(HTTPException, match=r"must name a 'fbfm' band.*\['fbfm'\]"):
            validate_band_of(self._body("chm"), "source_band", grid)

    def test_missing_band_rejected(self):
        with pytest.raises(HTTPException, match="missing required bands"):
            validate_band_of(self._body("fbfm"), "source_band", _input_grid(["chm"]))


class TestRequirementsFor:
    def test_when_applies_only_to_matching_requests(self):
        parametric = CreateTreeInventoryRequest(source_inventory_id="i1")
        measured = CreateTreeInventoryRequest(
            source_inventory_id="i1",
            crown_geometry_source={"type": "crown_feature", "feature_id": "f1"},
        )

        def columns(body):
            return [
                c
                for r in requirements_for(body, "source_inventory_id")
                for c in r["columns"]
            ]

        assert "tree_id" not in columns(parametric)
        assert "tree_id" in columns(measured)

    def test_nested_schema_entries_apply_to_their_own_fields(self):
        target = GridLatticeGridTarget(target="grid", grid_id="g1")
        assert requirements_for(target, "grid_id") == [
            {"field": "grid_id", "resource": "grid"}
        ]

    def test_crown_inventory_derived_from_chm_grid(self):
        body = CreateChmCrownFeatureRequest(
            source_inventory_id="i1", source_chm_grid_id="g1"
        )
        (req,) = requirements_for(body, "source_inventory_id")
        assert req["derived_from"] == "source_chm_grid_id"

    def test_compose_inputs_share_one_lattice(self):
        extra = CreateComposeRequest.model_config["json_schema_extra"]
        assert extra[X_INPUT_REQUIREMENTS] == [
            {
                "field": "inputs.*.grid_id",
                "resource": "grid",
                "dims": 2,
                "same_lattice_as": "inputs.*.grid_id",
            }
        ]
