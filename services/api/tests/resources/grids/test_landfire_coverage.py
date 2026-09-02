"""
Tests for the shared LANDFIRE coverage response builder.

Pure schema tests: no server, no LFPS. Router-level coverage tests live in
each product's test_router.py.
"""

from api.resources.grids.providers.landfire import (
    LandfireCoverage,
    build_landfire_coverage_response,
)

from lib.landfire import CoverageStatus, LandfireRelease

HREF = "https://api.example/domains/abc/grids/fbfm40/landfire"


def _release(version, season, year, coverage):
    return LandfireRelease(version, season, year, coverage)


class TestBuildLandfireCoverageResponse:
    def test_create_link_present_only_for_creatable_releases(self):
        releases = [
            _release("2025", "FA", None, CoverageStatus.NO_SUCH_PRODUCT),
            _release("2025", "SU", 2026, CoverageStatus.FULL),
            _release("2025", "ES", 2026, CoverageStatus.NONE),
            _release("2025", None, 2025, CoverageStatus.PARTIAL),
        ]
        response = build_landfire_coverage_response("fbfm40", releases, HREF)
        links = [r.links.create for r in response.releases]

        assert links[0] is None
        assert links[2] is None
        assert links[1].href == HREF
        assert links[1].method == "POST"
        assert links[1].body == {"version": "2025", "season": "SU"}
        assert links[3].body == {"version": "2025"}

    def test_coverage_status_maps_to_user_facing_enum(self):
        releases = [
            _release("2025", "FA", None, CoverageStatus.NO_SUCH_PRODUCT),
            _release("2025", None, 2025, CoverageStatus.NONE),
        ]
        response = build_landfire_coverage_response("fbfm40", releases, HREF)

        assert [r.coverage for r in response.releases] == [
            LandfireCoverage.unpublished,
            LandfireCoverage.none,
        ]

    def test_latest_is_first_fully_covered_release(self):
        releases = [
            _release("2025", "FA", None, CoverageStatus.NO_SUCH_PRODUCT),
            _release("2025", "SU", 2026, CoverageStatus.PARTIAL),
            _release("2025", "SP", 2026, CoverageStatus.FULL),
            _release("2024", None, 2024, CoverageStatus.FULL),
        ]
        response = build_landfire_coverage_response("fbfm40", releases, HREF)

        assert response.latest is not None
        assert (response.latest.version, response.latest.season) == ("2025", "SP")
        assert response.latest.links.create.body == {"version": "2025", "season": "SP"}

    def test_latest_is_null_when_nothing_fully_covers(self):
        releases = [
            _release("2025", None, 2025, CoverageStatus.PARTIAL),
            _release("2024", None, 2024, CoverageStatus.NONE),
        ]
        response = build_landfire_coverage_response("fbfm40", releases, HREF)

        assert response.latest is None
        assert response.product == "fbfm40"
