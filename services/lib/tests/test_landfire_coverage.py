"""
Unit tests for lib.landfire_coverage.

All tests run against synthetic boundary polygons (patched onto the module)
and mocked `list_products()` responses -- no live LFPS, no GDAL/Parquet I/O.

Run with: uv run --extra landfire_coverage --extra lfps pytest tests/test_landfire_coverage.py -v
"""

from unittest.mock import patch

from shapely.geometry import box

from lib.landfire_coverage import CoverageStatus, covers_annual, covers_seasonal
from lib.landfire_lfps import LfpsProduct

# Two adjacent, non-overlapping GeoAreas -- used to test geometry spanning a
# covered/uncovered boundary.
SW = box(0, 0, 10, 10)
NW = box(10, 0, 20, 10)
GEOAREAS = {"SW": SW, "NW": NW}

# Two adjacent Seasonal Fuels regions, keyed the same way SEASONAL_BOUNDARIES
# is: raw two-letter season codes.
ES_REGION = box(0, 0, 10, 10)
SP_REGION = box(10, 0, 20, 10)
SEASONAL = {"ES": ES_REGION, "SP": SP_REGION}


def _make_product(
    acronym: str,
    version: str,
    geo_areas: str = "",
    theme: str = "Fuels",
    season: str | None = None,
) -> LfpsProduct:
    """Build an `LfpsProduct` directly, for pre-seeding `list_products()` in tests."""
    return LfpsProduct(
        layer_name=f"{version}_{acronym}",
        product_name=f"{acronym} product",
        theme=theme,
        acronym=acronym,
        version=version,
        conus=True,
        geo_areas=geo_areas,
        season=season,
    )


class TestCoversAnnual:
    """Tests for annual product/version coverage."""

    def test_no_matching_product_returns_no_such_product(self):
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=[]):
                status = covers_annual("fbfm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.NO_SUCH_PRODUCT

    def test_geo_areas_all_returns_full_regardless_of_geometry(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="All")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                # Geometry entirely outside every known GeoArea.
                status = covers_annual("fbfm40", "2024", box(100, 100, 101, 101))

        assert status == CoverageStatus.FULL

    def test_covered_geoarea_returns_full(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_uncovered_geoarea_returns_none(self):
        """Geometry sits entirely inside a real, known GeoArea (NW) that this
        product just doesn't serve -- not FULL, not PARTIAL, just NONE."""
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(11, 1, 12, 2))

        assert status == CoverageStatus.NONE

    def test_geometry_spanning_covered_and_uncovered_geoarea_is_partial(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                spanning = box(5, 1, 15, 2)  # crosses the SW/NW boundary at x=10
                status = covers_annual("fbfm40", "2024", spanning)

        assert status == CoverageStatus.PARTIAL

    def test_geometry_outside_all_known_geoareas_returns_none(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(1000, 1000, 1001, 1001))

        assert status == CoverageStatus.NONE

    def test_product_matched_case_insensitively(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_annual("FbFm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_bare_year_version_matches_lf_prefixed_catalog_entry(self):
        products = [_make_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                miss = covers_annual("fbfm40", "2025", box(1, 1, 2, 2))
                hit = covers_annual("fbfm40", "2024", box(1, 1, 2, 2))

        assert miss == CoverageStatus.NO_SUCH_PRODUCT
        assert hit == CoverageStatus.FULL

    def test_seasonal_fuels_entries_sharing_acronym_version_are_ignored(self):
        """Real LFPS data has Seasonal Fuels entries sharing acronym/version
        with the annual product (e.g. FBFM40/LF2025 exists as both a "Fuels"
        entry and several "Seasonal Fuels" entries). Only the "Fuels" entry
        should be matched -- picking up a Seasonal Fuels entry here would use
        its geo_areas/conus values instead of the real annual product's."""
        products = [
            _make_product(
                "FBFM40", "LF2025", geo_areas="", theme="Seasonal Fuels", season="SP"
            ),
            _make_product("FBFM40", "LF2025", geo_areas="SW", theme="Fuels"),
        ]
        with patch("lib.landfire_coverage.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_annual("fbfm40", "2025", box(11, 1, 12, 2))

        assert status == CoverageStatus.NONE


class TestCoversSeasonal:
    """Tests for Seasonal Fuels product/version/season coverage."""

    def test_season_not_live_returns_no_such_product(self):
        products = [
            _make_product("FBFM40", "LF2025", theme="Seasonal Fuels", season="SP")
        ]
        with patch("lib.landfire_coverage.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                # FA isn't in the live catalog -- e.g. fall not published yet.
                status = covers_seasonal("fbfm40", "2025", "FA", box(1, 1, 2, 2))

        assert status == CoverageStatus.NO_SUCH_PRODUCT

    def test_live_season_fully_covers_geometry(self):
        products = [
            _make_product("FBFM40", "LF2025", theme="Seasonal Fuels", season="ES")
        ]
        with patch("lib.landfire_coverage.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_seasonal("fbfm40", "2025", "ES", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_live_season_geometry_outside_boundary_returns_none(self):
        products = [
            _make_product("FBFM40", "LF2025", theme="Seasonal Fuels", season="ES")
        ]
        with patch("lib.landfire_coverage.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                status = covers_seasonal(
                    "fbfm40", "2025", "ES", box(1000, 1000, 1001, 1001)
                )

        assert status == CoverageStatus.NONE

    def test_live_season_geometry_spanning_boundary_is_partial(self):
        products = [
            _make_product("FBFM40", "LF2025", theme="Seasonal Fuels", season="ES")
        ]
        with patch("lib.landfire_coverage.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                spanning = box(5, 1, 15, 2)  # crosses the ES/SP boundary at x=10
                status = covers_seasonal("fbfm40", "2025", "ES", spanning)

        assert status == CoverageStatus.PARTIAL

    def test_product_version_mismatch_returns_no_such_product(self):
        """A live "SP" season exists, but not for this product/version."""
        products = [
            _make_product("FBFM40", "LF2025", theme="Seasonal Fuels", season="SP")
        ]
        with patch("lib.landfire_coverage.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire_coverage.list_products", return_value=products):
                wrong_version = covers_seasonal("fbfm40", "2026", "SP", box(1, 1, 2, 2))
                wrong_product = covers_seasonal("fccs", "2025", "SP", box(1, 1, 2, 2))

        assert wrong_version == CoverageStatus.NO_SUCH_PRODUCT
        assert wrong_product == CoverageStatus.NO_SUCH_PRODUCT
