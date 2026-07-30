"""
Unit tests for lib.entwine 3DEP EPT discovery.

All tests run against a synthetic in-memory catalog — no network access. The
catalog cache is patched directly so ``load_catalog`` never reaches out.

Run with: uv run --extra entwine pytest tests/test_entwine.py -v
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from lib.entwine import (
    CATALOG_TTL,
    MIN_CONTRIBUTION_FRACTION,
    DatasetNotFoundError,
    DatasetOutsideDomainError,
    EptCatalogError,
    ept_url,
    load_catalog,
    search_3dep_ept,
    select_datasets,
)

# A metric CRS so geometry coordinates are metres and areas are trivially
# checkable. The ROI below is a 1000 m square = 1e6 m**2.
CRS = "EPSG:32612"


def make_catalog(entries) -> gpd.GeoDataFrame:
    """Build a synthetic catalog. Entries are (name, geometry, count)."""
    return gpd.GeoDataFrame(
        {
            "name": [e[0] for e in entries],
            "id": list(range(len(entries))),
            "count": [e[2] for e in entries],
            "url": [ept_url(e[0]) for e in entries],
            "geometry": [e[1] for e in entries],
        },
        crs=CRS,
    )


@pytest.fixture
def roi() -> gpd.GeoDataFrame:
    """A 1000 m x 1000 m square domain, area 1e6 m**2."""
    return gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)


@pytest.fixture
def patched_catalog():
    """Install a synthetic catalog in place of the real one.

    ``search_3dep_ept`` reprojects the catalog to the ROI CRS, so the synthetic
    catalog is authored in the ROI CRS and the reprojection is a no-op.
    """

    def install(catalog: gpd.GeoDataFrame):
        return patch("lib.entwine._catalog", catalog)

    return install


class TestLoadCatalog:
    """Tests for catalog loading and its fallback path."""

    def test_falls_back_to_source_when_mirror_missing(self):
        """A missing GCS mirror should fall through to the live catalog."""
        sentinel = make_catalog([("A", box(0, 0, 1, 1), 10)])
        with patch("lib.entwine._catalog", None):
            with patch("geopandas.read_parquet", side_effect=FileNotFoundError("no")):
                with patch("lib.entwine.requests.get") as mock_get:
                    mock_get.return_value.text = "{}"
                    with patch("geopandas.read_file", return_value=sentinel):
                        assert len(load_catalog(refresh=True)) == 1
                    mock_get.assert_called_once()

    def test_cached_catalog_is_reused_within_the_ttl(self):
        sentinel = make_catalog([("A", box(0, 0, 1, 1), 10)])
        with patch("lib.entwine._catalog", sentinel):
            with patch("lib.entwine._catalog_fetched_on", datetime.now(UTC)):
                with patch("geopandas.read_parquet") as mock_read:
                    load_catalog()
                    mock_read.assert_not_called()

    def test_expired_catalog_is_reloaded(self):
        """Caching forever lets two warm instances disagree indefinitely.

        USGS publishes new surveys, so a process that never re-reads keeps
        answering coverage questions from a snapshot that drifts further out of
        date the longer it stays up.
        """
        stale = make_catalog([("OLD", box(0, 0, 1, 1), 10)])
        fresh = make_catalog(
            [("OLD", box(0, 0, 1, 1), 10), ("NEW", box(2, 2, 3, 3), 20)]
        )
        expired = datetime.now(UTC) - timedelta(seconds=CATALOG_TTL + 1)

        with patch("lib.entwine._catalog", stale):
            with patch("lib.entwine._catalog_fetched_on", expired):
                with patch("geopandas.read_parquet", return_value=fresh):
                    assert list(load_catalog()["name"]) == ["OLD", "NEW"]

    def test_raises_when_both_sources_fail(self):
        """Losing the mirror and the source is an error, not an empty catalog."""
        with patch("lib.entwine._catalog", None):
            with patch("geopandas.read_parquet", side_effect=FileNotFoundError("no")):
                with patch(
                    "lib.entwine.requests.get", side_effect=ConnectionError("down")
                ):
                    with pytest.raises(EptCatalogError):
                        load_catalog(refresh=True)


class TestSearch:
    """Tests for finding candidate acquisitions."""

    def test_returns_only_overlapping_datasets(self, roi, patched_catalog):
        catalog = make_catalog(
            [
                ("OVERLAPS", box(500, 500, 1500, 1500), 1_000_000),
                ("MISSES", box(5000, 5000, 6000, 6000), 1_000_000),
            ]
        )
        with patched_catalog(catalog):
            hits = search_3dep_ept(roi)
        assert list(hits["name"]) == ["OVERLAPS"]

    def test_empty_when_nothing_overlaps(self, roi, patched_catalog):
        catalog = make_catalog([("FAR", box(9000, 9000, 9500, 9500), 100)])
        with patched_catalog(catalog):
            assert search_3dep_ept(roi).empty

    def test_density_is_points_per_square_metre(self, roi, patched_catalog):
        """Density comes from the whole footprint, not the overlapping part."""
        # 1000 m square footprint = 1e6 m**2 holding 5e6 points -> 5 pts/m**2.
        catalog = make_catalog([("A", box(0, 0, 1000, 1000), 5_000_000)])
        with patched_catalog(catalog):
            hits = search_3dep_ept(roi)
        # Equal-area reprojection distorts slightly; 10% is plenty tight.
        assert hits["density"].iloc[0] == pytest.approx(5.0, rel=0.1)


class TestSelectionCoverage:
    """Tests for coverage accounting."""

    def test_coverage_is_union_not_sum(self, roi, patched_catalog):
        """Two fully-overlapping acquisitions cover 100%, not 200%.

        Regression test: summing per-acquisition overlap percentages is the
        obvious implementation and it is wrong. Real catalog entries overlap
        freely — a live domain was measured summing to 144%.
        """
        catalog = make_catalog(
            [
                ("FULL_A", box(-100, -100, 1100, 1100), 2_000_000),
                ("FULL_B", box(-100, -100, 1100, 1100), 1_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert selection.coverage_fraction == pytest.approx(1.0, abs=1e-6)

    def test_zero_coverage_when_no_candidates(self, roi, patched_catalog):
        with patched_catalog(make_catalog([("FAR", box(9000, 9000, 9500, 9500), 10)])):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert selection.datasets == []
        assert selection.coverage_fraction == 0.0
        assert selection.estimated_point_count == 0

    def test_partial_coverage_is_reported(self, roi, patched_catalog):
        """Half-covering the domain reports 0.5, and still selects."""
        catalog = make_catalog([("HALF", box(0, 0, 500, 1000), 1_000_000)])
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert len(selection.datasets) == 1
        assert selection.coverage_fraction == pytest.approx(0.5, abs=1e-3)


class TestSelectionStrategy:
    """Tests for which acquisitions get chosen, and in what order."""

    def test_single_complete_dataset_wins_on_density(self, roi, patched_catalog):
        """Two acquisitions each covering the domain: take the denser one alone."""
        catalog = make_catalog(
            [
                ("SPARSE", box(-10, -10, 1010, 1010), 1_000_000),
                ("DENSE", box(-10, -10, 1010, 1010), 9_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in selection.datasets] == ["DENSE"]

    def test_prefers_one_near_complete_over_a_mosaic(self, roi, patched_catalog):
        """A 99.5% acquisition beats mosaicking two halves — fewer seams."""
        catalog = make_catalog(
            [
                ("ALMOST_ALL", box(0, 0, 1000, 995), 1_000_000),
                ("LEFT", box(0, 0, 500, 1000), 9_000_000),
                ("RIGHT", box(500, 0, 1000, 1000), 9_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in selection.datasets] == ["ALMOST_ALL"]

    def test_marginal_greedy_fills_the_domain(self, roi, patched_catalog):
        """With no near-complete option, fill by marginal coverage."""
        catalog = make_catalog(
            [
                ("BIG", box(0, 0, 800, 1000), 8_000_000),
                ("SMALL", box(700, 0, 1000, 1000), 3_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in selection.datasets] == ["BIG", "SMALL"]
        assert selection.coverage_fraction == pytest.approx(1.0, abs=1e-3)
        # SMALL overlaps BIG by 100 m but is only credited the 200 m it adds.
        assert selection.datasets[1].contribution_fraction == pytest.approx(
            0.2, abs=1e-3
        )

    def test_greedy_picks_by_marginal_not_absolute_coverage(self, roi, patched_catalog):
        """The second pick maximizes what is still uncovered, not total area.

        REDUNDANT covers more of the domain than FILLER, but all of it is
        already covered by BIG, so FILLER must be chosen second.
        """
        catalog = make_catalog(
            [
                ("BIG", box(0, 0, 900, 1000), 9_000_000),
                ("REDUNDANT", box(0, 0, 850, 1000), 8_000_000),
                ("FILLER", box(880, 0, 1000, 1000), 1_200_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in selection.datasets] == ["BIG", "FILLER"]

    def test_contributions_are_disjoint(self, roi, patched_catalog):
        """Overlapping acquisitions must never both claim the same ground."""
        catalog = make_catalog(
            [
                ("A", box(0, 0, 700, 1000), 7_000_000),
                ("B", box(300, 0, 1000, 1000), 7_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))

        contributions = selection.contributions
        assert len(contributions) == 2
        overlap = contributions[0].intersection(contributions[1]).area
        assert overlap == pytest.approx(0.0, abs=1e-6)
        total = sum(c.area for c in contributions)
        assert total == pytest.approx(1e6, rel=1e-3)

    def test_sliver_contributions_are_dropped(self, roi, patched_catalog):
        """A hairline filler is not worth an octree traversal."""
        catalog = make_catalog(
            [
                ("MAIN", box(0, 0, 999.99, 1000), 9_000_000),
                ("SLIVER", box(999.99, 0, 1000, 1000), 100),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in selection.datasets] == ["MAIN"]
        assert all(
            d.contribution_fraction >= MIN_CONTRIBUTION_FRACTION
            for d in selection.datasets
        )

    def test_selection_is_deterministic(self, roi, patched_catalog):
        """Identical candidates must resolve the same way every time."""
        catalog = make_catalog(
            [
                ("B_TIED", box(0, 0, 600, 1000), 6_000_000),
                ("A_TIED", box(0, 0, 600, 1000), 6_000_000),
                ("FILL", box(600, 0, 1000, 1000), 4_000_000),
            ]
        )
        with patched_catalog(catalog):
            first = select_datasets(roi, search_3dep_ept(roi))
            second = select_datasets(roi, search_3dep_ept(roi))
        assert [d.name for d in first.datasets] == [d.name for d in second.datasets]

    def test_rotated_domain_clips_to_polygon_not_envelope(self, patched_catalog):
        """A non-axis-aligned domain must be covered by its own shape.

        Legacy domains are rotated quadrilaterals (blue_mtn is one), so the
        contribution must follow the polygon, not its bounding box.
        """
        diamond = Polygon([(500, 0), (1000, 500), (500, 1000), (0, 500)])
        roi = gpd.GeoDataFrame(geometry=[diamond], crs=CRS)
        catalog = make_catalog([("COVER", box(-10, -10, 1010, 1010), 1_000_000)])
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))

        contribution = selection.contributions[0]
        assert contribution.area == pytest.approx(diamond.area, rel=1e-3)
        assert contribution.area < diamond.envelope.area


class TestPinnedDatasets:
    """Tests for an explicit user-supplied acquisition list."""

    def test_pin_is_honored_verbatim_and_in_order(self, roi, patched_catalog):
        """Pinning bypasses selection; list order decides overlap priority."""
        catalog = make_catalog(
            [
                ("DENSE", box(0, 0, 700, 1000), 9_000_000),
                ("SPARSE", box(300, 0, 1000, 1000), 1_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(
                roi, search_3dep_ept(roi), pinned=["SPARSE", "DENSE"]
            )

        assert [d.name for d in selection.datasets] == ["SPARSE", "DENSE"]
        # SPARSE was listed first, so it keeps the 300-1000 overlap band.
        assert selection.datasets[0].contribution_fraction == pytest.approx(
            0.7, abs=1e-3
        )
        assert selection.datasets[1].contribution_fraction == pytest.approx(
            0.3, abs=1e-3
        )

    def test_unknown_pinned_name_raises(self, roi, patched_catalog):
        catalog = make_catalog([("REAL", box(0, 0, 1000, 1000), 1_000_000)])
        with patched_catalog(catalog):
            with pytest.raises(DatasetNotFoundError, match="NOT_A_DATASET"):
                select_datasets(roi, search_3dep_ept(roi), pinned=["NOT_A_DATASET"])

    def test_non_intersecting_pinned_name_raises(self, roi, patched_catalog):
        catalog = make_catalog(
            [
                ("NEAR", box(0, 0, 1000, 1000), 1_000_000),
                ("FAR", box(9000, 9000, 9500, 9500), 1_000_000),
            ]
        )
        with patched_catalog(catalog):
            with pytest.raises(DatasetOutsideDomainError, match="FAR"):
                select_datasets(roi, search_3dep_ept(roi), pinned=["FAR"])


class TestEstimates:
    """Tests for the point-count estimate that gates a fetch."""

    def test_estimate_scales_with_contribution_area(self, roi, patched_catalog):
        """Half the domain at 4 pts/m**2 is ~2M points, not the full 4M."""
        catalog = make_catalog([("HALF", box(0, 0, 500, 1000), 2_000_000)])
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        # 5e5 m**2 footprint holding 2e6 points = 4 pts/m**2 over 5e5 m**2.
        assert selection.estimated_point_count == pytest.approx(2_000_000, rel=0.15)

    def test_estimate_sums_across_datasets(self, roi, patched_catalog):
        catalog = make_catalog(
            [
                ("LEFT", box(0, 0, 500, 1000), 2_000_000),
                ("RIGHT", box(500, 0, 1000, 1000), 4_000_000),
            ]
        )
        with patched_catalog(catalog):
            selection = select_datasets(roi, search_3dep_ept(roi))
        total = sum(d.estimated_points for d in selection.datasets)
        assert selection.estimated_point_count == total
        assert total == pytest.approx(6_000_000, rel=0.15)
