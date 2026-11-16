"""Tests for the Overture building feature handler."""

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from etcher.handlers.building import (
    building_columns,
    handle_overture,
    upstream_attributions,
)
from etcher.osm_source import OSM_ATTRIBUTION
from etcher.overture_source import OVERTURE_ATTRIBUTION
from shapely.geometry import box

RELEASE = "2026-10-21.0"
CRS = "EPSG:32610"

OSM = {"property": "", "dataset": "OpenStreetMap", "license": "ODbL-1.0"}
MS = {"property": "", "dataset": "Microsoft ML Buildings", "license": "ODbL-1.0"}
MS_HEIGHT = {
    "property": "/properties/height",
    "dataset": "Microsoft ML Buildings",
    "license": "ODbL-1.0",
}


@pytest.fixture
def domain_gdf():
    """A 1 km x 1 km domain in UTM."""
    return gpd.GeoDataFrame(geometry=[box(500000, 4500000, 501000, 4501000)], crs=CRS)


@pytest.fixture
def feature():
    return {"id": "feat-123", "domain_id": "dom-456", "type": "building"}


def overture_read(rows):
    """A fake ``read_overture_buildings`` result from (utm_geom, height, sources)."""
    geoms, heights, sources = zip(*rows) if rows else ([], [], [])
    gdf = gpd.GeoDataFrame(
        {
            "id": [f"b{i}" for i in range(len(rows))],
            "height": list(heights),
            # pyarrow hands list<struct> columns to pandas as numpy arrays.
            "sources": [np.array(s, dtype=object) for s in sources],
        },
        geometry=list(geoms),
        crs=CRS,
    )
    return gdf.to_crs(epsg=4326)


def run(feature, domain_gdf, rows, source=None):
    """Run the handler with a fake Overture read; returns (result, saved gdf)."""
    with (
        patch("etcher.handlers.building.latest_release", return_value=RELEASE),
        patch(
            "etcher.handlers.building.read_overture_buildings",
            return_value=overture_read(rows),
        ) as mock_read,
        patch("etcher.handlers.building.save_features") as mock_save,
    ):
        result = handle_overture(feature, source or {}, domain_gdf, MagicMock())
    mock_save.assert_called_once()
    assert mock_save.call_args.args[:2] == ("dom-456", "feat-123")
    assert mock_read.call_args.args[1] == RELEASE
    return result, mock_save.call_args.args[2]


def inside(i):
    """A 10 m footprint well inside the domain."""
    x = 500100 + 20 * i
    return box(x, 4500100, x + 10, 4500110)


class TestBuildingColumns:
    def test_columns_and_crs(self, feature, domain_gdf):
        _, saved = run(feature, domain_gdf, [(inside(0), 6.0, [OSM])])

        assert list(saved.columns) == [
            "geometry",
            "height",
            "height_source",
            "source_dataset",
        ]
        assert saved.crs == CRS
        assert saved["height"].dtype == "float64"

    def test_height_source_from_height_property(self, feature, domain_gdf):
        """An OSM footprint with a Microsoft height credits Microsoft for it."""
        _, saved = run(feature, domain_gdf, [(inside(0), 7.5, [OSM, MS_HEIGHT])])

        row = saved.iloc[0]
        assert row["height"] == 7.5
        assert row["height_source"] == "Microsoft ML Buildings"
        assert row["source_dataset"] == "OpenStreetMap"

    def test_height_source_falls_back_to_whole_building_dataset(
        self, feature, domain_gdf
    ):
        _, saved = run(feature, domain_gdf, [(inside(0), 4.0, [MS])])

        row = saved.iloc[0]
        assert row["height_source"] == "Microsoft ML Buildings"
        assert row["source_dataset"] == "Microsoft ML Buildings"

    def test_null_height_has_null_height_source(self, feature, domain_gdf):
        _, saved = run(feature, domain_gdf, [(inside(0), None, [OSM])])

        row = saved.iloc[0]
        assert np.isnan(row["height"])
        assert row["height_source"] is None
        assert row["source_dataset"] == "OpenStreetMap"

    def test_null_sources(self):
        buildings = gpd.GeoDataFrame(
            {"height": [3.0], "sources": [None]}, geometry=[inside(0)], crs=CRS
        )
        out = building_columns(buildings, CRS)
        assert out.iloc[0]["source_dataset"] is None
        assert out.iloc[0]["height_source"] is None


class TestClipping:
    def test_boundary_crossing_footprint_is_clipped(self, feature, domain_gdf):
        straddling = box(500990, 4500100, 501010, 4500110)  # 20 m wide, half out
        outside = box(502000, 4500100, 502010, 4500110)
        _, saved = run(
            feature,
            domain_gdf,
            [(straddling, 5.0, [OSM]), (outside, 5.0, [MS])],
        )

        assert len(saved) == 1
        assert saved.geometry.iloc[0].area == pytest.approx(100.0)
        assert saved.total_bounds[2] <= 501000

    def test_extent_buffer_keeps_overhang(self, feature, domain_gdf):
        straddling = box(500990, 4500100, 501010, 4500110)
        result, saved = run(
            feature,
            domain_gdf,
            [(straddling, 5.0, [OSM])],
            source={"extent_buffer_m": 50},
        )

        assert saved.geometry.iloc[0].area == pytest.approx(200.0, rel=1e-3)
        assert result["georeference"]["bounds"][2] == pytest.approx(501050)


class TestEmpty:
    def test_no_buildings(self, feature, domain_gdf):
        result, saved = run(feature, domain_gdf, [])

        assert saved.empty
        assert saved.crs == CRS
        assert list(saved.columns) == [
            "geometry",
            "height",
            "height_source",
            "source_dataset",
        ]
        assert result["release"] == RELEASE
        assert [a["dataset"] for a in result["attributions"]] == [
            "Overture Maps Foundation"
        ]

    def test_all_buildings_outside_domain(self, feature, domain_gdf):
        outside = box(502000, 4500100, 502010, 4500110)
        result, saved = run(feature, domain_gdf, [(outside, 5.0, [MS])])

        assert saved.empty
        assert [a["dataset"] for a in result["attributions"]] == [
            "Overture Maps Foundation"
        ]


class TestAttributions:
    today = date.today().isoformat()

    def test_osm_only(self, feature, domain_gdf):
        result, _ = run(feature, domain_gdf, [(inside(0), 5.0, [OSM])])

        assert result["attributions"] == [
            {**OVERTURE_ATTRIBUTION, "accessed_on": self.today},
            {**OSM_ATTRIBUTION, "accessed_on": self.today},
        ]

    def test_microsoft_and_osm(self, feature, domain_gdf):
        result, _ = run(
            feature,
            domain_gdf,
            [(inside(0), 5.0, [OSM, MS_HEIGHT]), (inside(1), 3.0, [MS])],
        )

        datasets = [a["dataset"] for a in result["attributions"]]
        assert datasets == [
            "Overture Maps Foundation",
            "Microsoft ML Buildings",
            "OpenStreetMap",
        ]
        microsoft = result["attributions"][1]
        assert microsoft == {
            "dataset": "Microsoft ML Buildings",
            "license_name": "ODbL-1.0",
            "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "citation": "Microsoft Global ML Building Footprints",
            "access_url": "https://github.com/microsoft/GlobalMLBuildingFootprints",
            "accessed_on": self.today,
        }

    def test_only_datasets_in_the_clipped_result(self, feature, domain_gdf):
        outside = box(502000, 4500100, 502010, 4500110)
        result, _ = run(
            feature,
            domain_gdf,
            [(inside(0), 5.0, [OSM]), (outside, 5.0, [MS])],
        )

        assert [a["dataset"] for a in result["attributions"]] == [
            "Overture Maps Foundation",
            "OpenStreetMap",
        ]

    def test_unknown_dataset_warns_and_completes(self, feature, domain_gdf, caplog):
        esri = {
            "property": "",
            "dataset": "Esri Community Maps",
            "license": "CC-BY-4.0",
        }
        with caplog.at_level(logging.WARNING, logger="etcher.handlers.building"):
            result, saved = run(feature, domain_gdf, [(inside(0), 5.0, [esri])])

        assert len(saved) == 1
        assert result["attributions"][1] == {
            "dataset": "Esri Community Maps",
            "license_name": "CC-BY-4.0",
            "license_url": "https://docs.overturemaps.org/attribution/",
            "citation": "Esri Community Maps",
            "access_url": "https://docs.overturemaps.org/attribution/",
            "accessed_on": self.today,
        }
        assert any(
            r.levelno == logging.WARNING and "Esri Community Maps" in r.getMessage()
            for r in caplog.records
        )

    def test_upstream_attributions_empty(self):
        empty = gpd.GeoDataFrame({"sources": []}, geometry=[], crs=CRS)
        assert upstream_attributions(empty) == []


def test_returns_release_and_georeference(feature, domain_gdf):
    result, _ = run(feature, domain_gdf, [(inside(0), 5.0, [OSM])])

    assert result["release"] == RELEASE
    assert result["georeference"] == {
        "crs": CRS,
        "bounds": [500000.0, 4500000.0, 501000.0, 4501000.0],
    }


def test_reader_error_propagates(feature, domain_gdf):
    with (
        patch("etcher.handlers.building.latest_release", return_value=RELEASE),
        patch(
            "etcher.handlers.building.read_overture_buildings",
            side_effect=RuntimeError("s3 read failed"),
        ),
        patch("etcher.handlers.building.save_features") as mock_save,
    ):
        with pytest.raises(RuntimeError):
            handle_overture(feature, {}, domain_gdf, MagicMock())
    mock_save.assert_not_called()
