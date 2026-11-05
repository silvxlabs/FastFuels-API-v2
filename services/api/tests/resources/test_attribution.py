"""Tests for the attribution inheritance helpers in api/resources/attribution.py."""

from api.resources.attribution import inherit_attributions, merge_attributions

from tests.fixtures import make_osm_attribution


def _entry(dataset: str, license_name: str, accessed_on: str) -> dict:
    return {
        "dataset": dataset,
        "license_name": license_name,
        "license_url": f"https://example.com/{license_name}",
        "citation": f"© {dataset}",
        "access_url": f"https://example.com/{dataset}",
        "accessed_on": accessed_on,
    }


class TestMergeAttributions:
    def test_no_inputs(self):
        assert merge_attributions() == []

    def test_empty_and_none_inputs(self):
        assert merge_attributions([], None, []) == []

    def test_single_list_passes_through(self):
        osm = make_osm_attribution("2026-11-05")
        assert merge_attributions([osm]) == [osm]

    def test_dedupes_by_dataset_and_license(self):
        osm = make_osm_attribution("2026-11-05")
        assert merge_attributions([osm], [dict(osm)], None) == [osm]

    def test_earliest_accessed_on_wins(self):
        later = make_osm_attribution("2026-11-05")
        earlier = make_osm_attribution("2026-10-01")
        assert merge_attributions([later], [earlier]) == [earlier]
        assert merge_attributions([earlier], [later]) == [earlier]

    def test_same_dataset_different_license_kept_apart(self):
        a = _entry("Overture Maps Foundation", "ODbL-1.0", "2026-11-05")
        b = _entry("Overture Maps Foundation", "CDLA-Permissive-2.0", "2026-11-05")
        assert merge_attributions([a, b]) == [b, a]

    def test_deterministic_order(self):
        osm = make_osm_attribution("2026-11-05")
        meta = _entry("Meta CHM", "CC-BY-4.0", "2026-11-05")
        overture = _entry("Overture Maps Foundation", "ODbL-1.0", "2026-11-05")
        expected = [meta, osm, overture]
        assert merge_attributions([osm, overture], [meta]) == expected
        assert merge_attributions([meta], [overture], [osm]) == expected

    def test_does_not_mutate_or_alias_inputs(self):
        osm = make_osm_attribution("2026-11-05")
        merged = merge_attributions([osm])
        merged[0]["accessed_on"] = "1999-01-01"
        assert osm["accessed_on"] == "2026-11-05"


class TestInheritAttributions:
    def test_no_parents(self):
        assert inherit_attributions([]) == []

    def test_missing_and_none_attributions(self):
        assert inherit_attributions([{}, {"attributions": None}]) == []

    def test_accepts_any_iterable(self):
        osm = make_osm_attribution("2026-11-05")
        parents = ({"attributions": [osm]} for _ in range(2))
        assert inherit_attributions(parents) == [osm]

    def test_merges_parent_lists(self):
        osm = make_osm_attribution("2026-11-05")
        meta = _entry("Meta CHM", "CC-BY-4.0", "2026-11-05")
        parents = [{"attributions": [osm]}, {"attributions": [meta, dict(osm)]}]
        assert inherit_attributions(parents) == [meta, osm]
