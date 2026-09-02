"""
Unit tests for lib.landfire.lfps: LFPS API client and its coverage check.

Covers job submission/polling/download against mocked LFPS responses, and
GeoArea/Seasonal boundary matching for `covers_annual`/`covers_seasonal`.
No network access -- the product cache is patched directly so
`list_products` never reaches out.

Run with: uv run --extra landfire pytest tests/test_landfire_lfps.py -v
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import box

from lib.landfire.lfps import (
    _SEASON_PATTERN,
    LFPS_PRODUCTS_TTL_SECONDS,
    CoverageStatus,
    LandfireRelease,
    LfpsJob,
    LfpsJobFailedError,
    LfpsProduct,
    covers_annual,
    covers_seasonal,
    download,
    list_products,
    list_releases,
    poll_status,
    resolve_seasonal_product,
    submit_job,
)


def make_products_response(names: list[str]) -> dict:
    """Build a synthetic /api/products response body."""
    return {
        "products": [
            {
                "productName": f"{name} product",
                "theme": "Fuels",
                "layerName": name,
                "acronym": name.split("_")[-1],
                "version": name.split("_")[0].removeprefix("LF"),
                "conus": True,
                "ak": True,
                "hi": False,
                "prvi": False,
                "geoAreas": "SW, NW",
            }
            for name in names
        ]
    }


def _make_product(
    layer_name: str, season: str | None = None, season_year: int | None = None
) -> LfpsProduct:
    """Build an `LfpsProduct` directly, for pre-seeding the cache in tests."""
    # Strip any trailing season suffix so the acronym is the real product
    # token for seasonal names too ("LF2025_FBFM40_SP26" -> "FBFM40"), the
    # way LFPS reports `acronym` independently of the layer name.
    core = _SEASON_PATTERN.sub("", layer_name)
    return LfpsProduct(
        layer_name=layer_name,
        product_name=f"{layer_name} product",
        theme="Fuels",
        acronym=core.split("_")[-1],
        version=core.split("_")[0],
        conus=True,
        geo_areas="SW, NW",
        season=season,
        season_year=season_year,
    )


def mock_response(json_body: dict, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_body
    response.content = content
    response.raise_for_status.return_value = None
    return response


class TestListProducts:
    """Tests for product-catalog loading and its TTL cache."""

    def test_fetches_and_parses_products(self):
        body = make_products_response(["LF2024_FBFM40"])
        with patch("lib.landfire.lfps._products", None):
            with patch(
                "lib.landfire.lfps.requests.get",
                return_value=mock_response(body),
            ) as mock_get:
                products = list_products()

        assert len(products) == 1
        assert products[0].layer_name == "LF2024_FBFM40"
        assert products[0].conus is True
        mock_get.assert_called_once()

    def test_cached_products_are_reused_within_the_ttl(self):
        cached = [_make_product("LF2024_FBFM40")]
        with patch("lib.landfire.lfps._products", cached):
            with patch("lib.landfire.lfps._products_fetched_on", datetime.now(UTC)):
                with patch("lib.landfire.lfps.requests.get") as mock_get:
                    result = list_products()

        assert result == cached
        mock_get.assert_not_called()

    def test_refresh_forces_a_re_fetch(self):
        cached = [_make_product("OLD")]
        fresh_body = make_products_response(["NEW"])
        with patch("lib.landfire.lfps._products", cached):
            with patch("lib.landfire.lfps._products_fetched_on", datetime.now(UTC)):
                with patch(
                    "lib.landfire.lfps.requests.get",
                    return_value=mock_response(fresh_body),
                ) as mock_get:
                    result = list_products(refresh=True)

        assert [p.layer_name for p in result] == ["NEW"]
        mock_get.assert_called_once()

    def test_expired_cache_is_reloaded(self):
        cached = [_make_product("OLD")]
        fresh_body = make_products_response(["OLD", "NEW"])
        expired = datetime.now(UTC) - timedelta(seconds=LFPS_PRODUCTS_TTL_SECONDS + 1)
        with patch("lib.landfire.lfps._products", cached):
            with patch("lib.landfire.lfps._products_fetched_on", expired):
                with patch(
                    "lib.landfire.lfps.requests.get",
                    return_value=mock_response(fresh_body),
                ):
                    result = list_products()

        assert [p.layer_name for p in result] == ["OLD", "NEW"]

    def test_seasonal_layer_parses_season_code(self):
        body = make_products_response(["LF2025_FBFM40_SP26"])
        with patch("lib.landfire.lfps._products", None):
            with patch(
                "lib.landfire.lfps.requests.get",
                return_value=mock_response(body),
            ):
                products = list_products()

        assert products[0].season == "SP"

    def test_seasonal_layer_parses_season_year_from_the_suffix(self):
        # The "_SP26" suffix on an LF2025 layer is spring 2026 -- read from the
        # layer name, not assumed to be version + 1.
        body = make_products_response(["LF2025_FBFM40_SP26"])
        with patch("lib.landfire.lfps._products", None):
            with patch(
                "lib.landfire.lfps.requests.get",
                return_value=mock_response(body),
            ):
                products = list_products()

        assert products[0].season_year == 2026

    def test_annual_layer_has_no_season(self):
        body = make_products_response(["LF2025_FBFM40"])
        with patch("lib.landfire.lfps._products", None):
            with patch(
                "lib.landfire.lfps.requests.get",
                return_value=mock_response(body),
            ):
                products = list_products()

        assert products[0].season is None
        assert products[0].season_year is None


class TestSubmitJob:
    """Tests for job submission and its query params."""

    def test_builds_expected_query_params(self):
        response_body = {
            "jobId": "job-123",
            "status": "Pending",
            "messages": [],
        }
        with patch("lib.landfire.lfps.LANDFIRE_USER_EMAIL", "test@example.com"):
            with patch(
                "lib.landfire.lfps.requests.get",
                return_value=mock_response(response_body),
            ) as mock_get:
                submit_job(
                    ["LF2024_FBFM40", "LF2024_FBFM13"], "-114.2 46.8 -114.1 46.9"
                )

        args, kwargs = mock_get.call_args
        assert args[0].endswith("/job/submit")
        assert kwargs["params"] == {
            "Email": "test@example.com",
            "Layer_List": "LF2024_FBFM40;LF2024_FBFM13",
            "Area_of_Interest": "-114.2 46.8 -114.1 46.9",
        }

    def test_parses_the_submitted_job(self):
        response_body = {
            "jobId": "job-123",
            "status": "Pending",
            "messages": [],
        }
        with patch(
            "lib.landfire.lfps.requests.get",
            return_value=mock_response(response_body),
        ):
            job = submit_job(["LF2024_FBFM40"], "-114.2 46.8 -114.1 46.9")

        assert job.job_id == "job-123"
        assert job.status == "Pending"
        assert job.output_file is None


class TestPollStatus:
    """Tests for single-shot status checks and Failed-status error surfacing."""

    def test_pending_status(self):
        body = {"jobId": "job-123", "status": "Pending", "messages": []}
        with patch("lib.landfire.lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Pending"
        assert job.output_file is None

    def test_executing_status(self):
        body = {"jobId": "job-123", "status": "Executing", "messages": []}
        with patch("lib.landfire.lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Executing"

    def test_succeeded_status_carries_output_file_and_geo_area(self):
        body = {
            "jobId": "job-123",
            "status": "Succeeded",
            "messages": [],
            "outputFile": "https://lfps.usgs.gov/.../job-123.zip",
            "geoArea": "SW",
        }
        with patch("lib.landfire.lfps.requests.get", return_value=mock_response(body)):
            job = poll_status("job-123")

        assert job.status == "Succeeded"
        assert job.output_file == "https://lfps.usgs.gov/.../job-123.zip"
        assert job.geo_area == "SW"

    def test_failed_status_raises_with_the_error_line(self):
        body = {
            "jobId": "job-123",
            "status": "Failed",
            "messages": [
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": "Start Time: ...",
                },
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": (
                        "ERROR:  Output geotif with no raster statistics. "
                        "Possible Reason: Area of Interest falls outside the "
                        "LF layer list input data."
                    ),
                },
                {"type": "esriJobMessageTypeInformative", "description": "Failed."},
            ],
        }
        with patch("lib.landfire.lfps.requests.get", return_value=mock_response(body)):
            with pytest.raises(LfpsJobFailedError) as exc_info:
                poll_status("job-123")

        message = str(exc_info.value)
        assert "no raster statistics" in message
        assert "Start Time" not in message
        assert message.strip() != "Failed."

    def test_failed_status_falls_back_to_full_trail_without_an_error_line(self):
        body = {
            "jobId": "job-123",
            "status": "Failed",
            "messages": [
                {
                    "type": "esriJobMessageTypeInformative",
                    "description": "Start Time: ...",
                },
                {"type": "esriJobMessageTypeInformative", "description": "Failed."},
            ],
        }
        with patch("lib.landfire.lfps.requests.get", return_value=mock_response(body)):
            with pytest.raises(LfpsJobFailedError) as exc_info:
                poll_status("job-123")

        message = str(exc_info.value)
        assert "Start Time" in message
        assert "Failed." in message


class TestDownload:
    """Tests for fetching a succeeded job's output."""

    def test_fetches_from_output_file(self):
        job = LfpsJob(
            job_id="job-123",
            status="Succeeded",
            output_file="https://lfps.usgs.gov/.../job-123.zip",
        )
        with patch(
            "lib.landfire.lfps.requests.get",
            return_value=mock_response({}, content=b"PK\x03\x04zip-bytes"),
        ) as mock_get:
            content = download(job)

        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "https://lfps.usgs.gov/.../job-123.zip"
        assert content == b"PK\x03\x04zip-bytes"


# --- Coverage check tests --------------------------------------------

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


def _make_coverage_product(
    acronym: str,
    version: str,
    geo_areas: str = "",
    theme: str = "Fuels",
    season: str | None = None,
    season_year: int | None = None,
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
        season_year=season_year,
    )


class TestCoversAnnual:
    """Tests for annual product/version coverage."""

    def test_no_matching_product_returns_no_such_product(self):
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=[]):
                status = covers_annual("fbfm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.NO_SUCH_PRODUCT

    def test_geo_areas_all_returns_full_regardless_of_geometry(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="All")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                # Geometry entirely outside every known GeoArea.
                status = covers_annual("fbfm40", "2024", box(100, 100, 101, 101))

        assert status == CoverageStatus.FULL

    def test_covered_geoarea_returns_full(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_uncovered_geoarea_returns_none(self):
        """Geometry sits entirely inside a real, known GeoArea (NW) that this
        product just doesn't serve -- not FULL, not PARTIAL, just NONE."""
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(11, 1, 12, 2))

        assert status == CoverageStatus.NONE

    def test_geometry_spanning_covered_and_uncovered_geoarea_is_partial(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                spanning = box(5, 1, 15, 2)  # crosses the SW/NW boundary at x=10
                status = covers_annual("fbfm40", "2024", spanning)

        assert status == CoverageStatus.PARTIAL

    def test_geometry_outside_all_known_geoareas_returns_none(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_annual("fbfm40", "2024", box(1000, 1000, 1001, 1001))

        assert status == CoverageStatus.NONE

    def test_product_matched_case_insensitively(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_annual("FbFm40", "2024", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_bare_year_version_matches_lf_prefixed_catalog_entry(self):
        products = [_make_coverage_product("FBFM40", "LF2024", geo_areas="SW")]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
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
            _make_coverage_product(
                "FBFM40", "LF2025", geo_areas="", theme="Seasonal Fuels", season="SP"
            ),
            _make_coverage_product("FBFM40", "LF2025", geo_areas="SW", theme="Fuels"),
        ]
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_annual("fbfm40", "2025", box(11, 1, 12, 2))

        assert status == CoverageStatus.NONE


class TestCoversSeasonal:
    """Tests for Seasonal Fuels product/version/season coverage."""

    def test_season_not_live_returns_no_such_product(self):
        products = [
            _make_coverage_product(
                "FBFM40", "LF2025", theme="Seasonal Fuels", season="SP"
            )
        ]
        with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                # FA isn't in the live catalog -- e.g. fall not published yet.
                status = covers_seasonal("fbfm40", "2025", "FA", box(1, 1, 2, 2))

        assert status == CoverageStatus.NO_SUCH_PRODUCT

    def test_live_season_fully_covers_geometry(self):
        products = [
            _make_coverage_product(
                "FBFM40", "LF2025", theme="Seasonal Fuels", season="ES"
            )
        ]
        with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_seasonal("fbfm40", "2025", "ES", box(1, 1, 2, 2))

        assert status == CoverageStatus.FULL

    def test_live_season_geometry_outside_boundary_returns_none(self):
        products = [
            _make_coverage_product(
                "FBFM40", "LF2025", theme="Seasonal Fuels", season="ES"
            )
        ]
        with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                status = covers_seasonal(
                    "fbfm40", "2025", "ES", box(1000, 1000, 1001, 1001)
                )

        assert status == CoverageStatus.NONE

    def test_live_season_geometry_spanning_boundary_is_partial(self):
        products = [
            _make_coverage_product(
                "FBFM40", "LF2025", theme="Seasonal Fuels", season="ES"
            )
        ]
        with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                spanning = box(5, 1, 15, 2)  # crosses the ES/SP boundary at x=10
                status = covers_seasonal("fbfm40", "2025", "ES", spanning)

        assert status == CoverageStatus.PARTIAL

    def test_product_version_mismatch_returns_no_such_product(self):
        """A live "SP" season exists, but not for this product/version."""
        products = [
            _make_coverage_product(
                "FBFM40", "LF2025", theme="Seasonal Fuels", season="SP"
            )
        ]
        with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", SEASONAL):
            with patch("lib.landfire.lfps.list_products", return_value=products):
                wrong_version = covers_seasonal("fbfm40", "2026", "SP", box(1, 1, 2, 2))
                wrong_product = covers_seasonal("fccs", "2025", "SP", box(1, 1, 2, 2))

        assert wrong_version == CoverageStatus.NO_SUCH_PRODUCT
        assert wrong_product == CoverageStatus.NO_SUCH_PRODUCT


class TestResolveSeasonalProduct:
    """Tests for resolving the live Seasonal Fuels catalog entry."""

    def test_returns_matching_product_with_its_real_season_year(self):
        products = [
            _make_product("LF2025_FBFM40_SP26", season="SP", season_year=2026),
            _make_product("LF2024_FBFM40", season=None),
        ]
        with patch("lib.landfire.lfps.list_products", return_value=products):
            match = resolve_seasonal_product("fbfm40", "2025", "SP")

        assert match is not None
        assert match.layer_name == "LF2025_FBFM40_SP26"
        assert match.season_year == 2026

    def test_product_matched_case_insensitively(self):
        products = [_make_product("LF2025_FBFM40_SP26", season="SP", season_year=2026)]
        with patch("lib.landfire.lfps.list_products", return_value=products):
            assert resolve_seasonal_product("FBFM40", "2025", "SP") is not None

    def test_returns_none_when_season_not_live(self):
        products = [_make_product("LF2025_FBFM40_SP26", season="SP", season_year=2026)]
        with patch("lib.landfire.lfps.list_products", return_value=products):
            assert resolve_seasonal_product("fbfm40", "2025", "FA") is None

    def test_returns_none_on_version_mismatch(self):
        products = [_make_product("LF2025_FBFM40_SP26", season="SP", season_year=2026)]
        with patch("lib.landfire.lfps.list_products", return_value=products):
            assert resolve_seasonal_product("fbfm40", "2024", "SP") is None


# A registry with one on-demand vintage and two staged ones, patched in for
# list_releases so the tests don't track the real LANDFIRE_VERSIONS.
RELEASE_REGISTRY = {
    "fbfm40": {
        "available": ["2023", "2024"],
        "lfps_available": ["2025"],
        "default": "2024",
    },
    "fbfm13": {
        "available": ["2024"],
        "lfps_available": ["2025"],
        "default": "2024",
    },
}
ALL_SEASONS = {"ES": SW, "SP": SW, "SU": SW, "FA": SW}


def _patched_releases(products, product="fbfm40", geometry=None, seasons=()):
    """Run list_releases with the registry, catalog, and boundaries patched."""
    geometry = geometry if geometry is not None else box(1, 1, 2, 2)
    with patch("lib.landfire.lfps.LANDFIRE_VERSIONS", RELEASE_REGISTRY):
        with patch("lib.landfire.lfps.GEOAREA_BOUNDARIES", GEOAREAS):
            with patch("lib.landfire.lfps.SEASONAL_BOUNDARIES", ALL_SEASONS):
                with patch("lib.landfire.lfps.list_products", return_value=products):
                    return list_releases(product, geometry, seasons)


class TestListReleases:
    """Tests for enumerating every release with its coverage, newest first."""

    def test_ordered_newest_first_with_seasons_ahead_of_their_vintage(self):
        products = [
            _make_coverage_product("FBFM40", "LF2025", geo_areas="SW"),
            _make_coverage_product(
                "FBFM40",
                "LF2025",
                theme="Seasonal Fuels",
                season="ES",
                season_year=2026,
            ),
            _make_coverage_product(
                "FBFM40",
                "LF2025",
                theme="Seasonal Fuels",
                season="SP",
                season_year=2026,
            ),
        ]
        releases = _patched_releases(products, seasons=("ES", "SP", "SU", "FA"))

        assert [(r.version, r.season) for r in releases] == [
            ("2025", "FA"),
            ("2025", "SU"),
            ("2025", "SP"),
            ("2025", "ES"),
            ("2025", None),
            ("2024", None),
            ("2023", None),
        ]

    def test_unpublished_seasons_have_no_year_and_published_read_catalog_year(self):
        """The season year comes from the catalog, not `version + 1`."""
        products = [
            _make_coverage_product("FBFM40", "LF2025", geo_areas="SW"),
            _make_coverage_product(
                "FBFM40",
                "LF2025",
                theme="Seasonal Fuels",
                season="SP",
                season_year=2027,
            ),
        ]
        releases = _patched_releases(products, seasons=("SP", "FA"))
        by_season = {r.season: r for r in releases if r.season is not None}

        assert by_season["SP"].year == 2027
        assert by_season["SP"].coverage == CoverageStatus.FULL
        assert by_season["FA"].year is None
        assert by_season["FA"].coverage == CoverageStatus.NO_SUCH_PRODUCT

    def test_staged_versions_are_full_wherever_the_geometry_is(self):
        """Staged vintages are national rasters, so no GeoArea check applies."""
        outside_every_geoarea = box(1000, 1000, 1001, 1001)
        releases = _patched_releases([], geometry=outside_every_geoarea)

        assert {r.version: r.coverage for r in releases} == {
            "2025": CoverageStatus.NO_SUCH_PRODUCT,
            "2024": CoverageStatus.FULL,
            "2023": CoverageStatus.FULL,
        }

    def test_on_demand_annual_uses_live_geoarea_coverage(self):
        products = [_make_coverage_product("FBFM40", "LF2025", geo_areas="SW")]
        in_nw = box(11, 1, 12, 2)
        releases = _patched_releases(products, geometry=in_nw)

        assert releases[0] == LandfireRelease("2025", None, 2025, CoverageStatus.NONE)

    def test_no_seasons_yields_annual_entries_only(self):
        products = [
            _make_coverage_product("FBFM13", "LF2025", geo_areas="SW"),
            _make_coverage_product(
                "FBFM13",
                "LF2025",
                theme="Seasonal Fuels",
                season="SP",
                season_year=2026,
            ),
        ]
        releases = _patched_releases(products, product="fbfm13")

        assert [(r.version, r.season, r.year) for r in releases] == [
            ("2025", None, 2025),
            ("2024", None, 2024),
        ]
