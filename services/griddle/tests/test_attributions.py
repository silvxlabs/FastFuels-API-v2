"""
Tests for the attributions griddle adds to source grids (#790).
"""

from datetime import date

import pytest
from griddle.attributions import (
    US_PUBLIC_DOMAIN_LICENSE,
    add_attributions,
    source_attributions,
    threedep_attributions,
)

ACCESSED = date(2026, 11, 13)
ELEVATION = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"

OSM = {
    "dataset": "OpenStreetMap",
    "license_name": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "citation": "© OpenStreetMap contributors",
    "access_url": "https://www.openstreetmap.org/copyright",
    "accessed_on": "2026-11-05",
}


def datasets(entries: list[dict]) -> list[str]:
    return [e["dataset"] for e in entries]


class TestThreedepAttributions:
    def test_arc_second_products(self):
        entries = threedep_attributions(
            [
                f"{ELEVATION}/13/TIFF/current/n47w114/USGS_13_n47w114.tif",
                f"{ELEVATION}/13/TIFF/current/n48w114/USGS_13_n48w114.tif",
                f"{ELEVATION}/1/TIFF/current/n47w114/USGS_1_n47w114.tif",
            ],
            ACCESSED,
        )
        assert datasets(entries) == [
            "USGS 3DEP 1 arc-second DEM",
            "USGS 3DEP 1/3 arc-second DEM",
        ]

    def test_s1m(self):
        entries = threedep_attributions(
            [f"{ELEVATION}/S1M/n52w14/n5260w1400/S1M_n5260w1400_20230515.tif"],
            ACCESSED,
        )
        assert datasets(entries) == ["USGS 3DEP Seamless 1 m DEM (S1M)"]

    def test_one_entry_per_lidar_project(self):
        project = "MT_NRCS_MT_2018_D18"
        other = "MT_Statewide_2020_B20"
        entries = threedep_attributions(
            [
                f"{ELEVATION}/1m/Projects/{project}/TIFF/USGS_1M_12_x70y532_{project}.tif",
                f"{ELEVATION}/1m/Projects/{project}/TIFF/USGS_1M_12_x71y532_{project}.tif",
                f"{ELEVATION}/1m/Projects/{other}/TIFF/USGS_1M_12_x70y533_{other}.tif",
            ],
            ACCESSED,
        )
        assert datasets(entries) == [
            f"USGS 3DEP 1 m lidar DEM ({project})",
            f"USGS 3DEP 1 m lidar DEM ({other})",
        ]
        assert entries[0]["citation"] == (
            "U.S. Geological Survey, 3D Elevation Program 1-Meter Resolution "
            f"Digital Elevation Model ({project}), accessed November 13, 2026 at "
            "https://www.usgs.gov/3d-elevation-program."
        )

    def test_unrecognized_url_still_credits_3dep(self):
        entries = threedep_attributions([f"{ELEVATION}/2m/new.tif"], ACCESSED)
        assert datasets(entries) == ["USGS 3D Elevation Program (3DEP)"]

    def test_entry_fields(self):
        (entry,) = threedep_attributions(
            [f"{ELEVATION}/13/TIFF/current/n47w114/USGS_13_n47w114.tif"], ACCESSED
        )
        assert entry["license_name"] == US_PUBLIC_DOMAIN_LICENSE
        assert entry["license_url"].startswith("https://www.usgs.gov/")
        assert entry["access_url"] == "https://www.usgs.gov/3d-elevation-program"
        assert entry["accessed_on"] == "2026-11-13"


class TestSourceAttributions:
    def test_3dep_reads_tiles_from_tile_metadata(self):
        source = {
            "name": "3dep",
            "product": "topography",
            "tile_metadata": {
                "tiles": [f"{ELEVATION}/13/TIFF/current/n47w114/USGS_13_n47w114.tif"]
            },
        }
        assert datasets(source_attributions(source, ACCESSED)) == [
            "USGS 3DEP 1/3 arc-second DEM"
        ]

    def test_landfire_fuel_model(self):
        source = {"name": "landfire", "product": "fbfm40", "version": "2024"}
        (entry,) = source_attributions(source, ACCESSED)
        assert entry["dataset"] == (
            "LANDFIRE 40 Scott and Burgan Fire Behavior Fuel Models (LF 2024)"
        )
        assert entry["citation"] == (
            "LANDFIRE, 40 Scott and Burgan Fire Behavior Fuel Models, LANDFIRE "
            "2024, U.S. Department of the Interior, Geological Survey, and U.S. "
            "Department of Agriculture. Accessed 13 November 2026 at "
            "https://landfire.gov."
        )
        assert entry["license_name"] == US_PUBLIC_DOMAIN_LICENSE

    def test_landfire_default_version(self):
        source = {"name": "landfire", "product": "fccs"}
        assert datasets(source_attributions(source, ACCESSED)) == [
            "LANDFIRE Fuel Characteristic Classification System Fuelbeds (LF 2023)"
        ]

    def test_landfire_season(self):
        source = {
            "name": "landfire",
            "product": "fbfm40",
            "version": "2025",
            "season": "SP",
        }
        assert datasets(source_attributions(source, ACCESSED)) == [
            "LANDFIRE 40 Scott and Burgan Fire Behavior Fuel Models, SP season "
            "(LF 2025)"
        ]

    def test_landfire_topography_one_entry_per_band(self):
        source = {
            "name": "landfire",
            "product": "topography",
            "version": "2020",
            "bands": ["elevation", "slope"],
        }
        assert datasets(source_attributions(source, ACCESSED)) == [
            "LANDFIRE Elevation (LF 2020)",
            "LANDFIRE Slope Degrees (LF 2020)",
        ]

    def test_landfire_disturbance_credits_every_layer_read(self):
        source = {
            "name": "landfire",
            "product": "fbfm40",
            "version": "2024",
            "disturbance": {
                "source": "landfire",
                "mode": "ldist_and_last_year_fdist",
                "ldist_version": "2025",
                "fdist_version": "2024",
            },
        }
        names = datasets(source_attributions(source, ACCESSED))
        assert names[0] == (
            "LANDFIRE 40 Scott and Burgan Fire Behavior Fuel Models (LF 2024)"
        )
        assert "LANDFIRE Fuel Disturbance (LF 2024)" in names
        assert "LANDFIRE Annual Disturbance (LF 2025)" in names
        for title in (
            "Fuel Vegetation Type",
            "Fuel Vegetation Cover",
            "Fuel Vegetation Height",
            "Biophysical Settings",
        ):
            assert any(n.startswith(f"LANDFIRE {title} (LF ") for n in names)

    def test_landfire_fdist_only_disturbance_skips_annual(self):
        source = {
            "name": "landfire",
            "product": "fbfm13",
            "version": "2024",
            "disturbance": {
                "source": "landfire",
                "mode": "fdist",
                "fdist_version": "2026",
            },
        }
        names = datasets(source_attributions(source, ACCESSED))
        assert "LANDFIRE Fuel Disturbance (LF 2026)" in names
        assert not any("Annual Disturbance" in n for n in names)

    def test_landfire_annual_disturbance(self):
        source = {
            "name": "landfire",
            "product": "annual_disturbance",
            "version": "2025",
        }
        assert datasets(source_attributions(source, ACCESSED)) == [
            "LANDFIRE Annual Disturbance (LF 2025)"
        ]

    def test_landfire_canopy_one_entry_per_band(self):
        source = {
            "name": "canopy",
            "product": "landfire",
            "version": "2024",
            "bands": ["cbd", "cbh"],
        }
        assert datasets(source_attributions(source, ACCESSED)) == [
            "LANDFIRE Forest Canopy Bulk Density (LF 2024)",
            "LANDFIRE Forest Canopy Base Height (LF 2024)",
        ]

    def test_treemap(self):
        source = {"name": "pim", "product": "treemap", "version": "2022"}
        (entry,) = source_attributions(source, ACCESSED)
        assert entry["dataset"] == "TreeMap 2022"
        assert entry["citation"].startswith("Houtman, Rachel M.;")
        assert entry["citation"].endswith("https://doi.org/10.2737/RDS-2025-0032")

    @pytest.mark.parametrize("version", ["2014", "2016", "2020", "2022"])
    def test_every_treemap_version_is_cited(self, version):
        source = {"name": "pim", "product": "treemap", "version": version}
        assert datasets(source_attributions(source, ACCESSED)) == [f"TreeMap {version}"]

    def test_meta_chm_copies_source_attribution(self):
        source_attribution = {
            "license_name": "DINOv3",
            "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
            "citation": "Brandt et al. CHMv2 ...",
            "access_url": "https://registry.opendata.aws/dataforgood-fb-forests",
            "accessed_on": "2026-11-13",
        }
        source = {
            "name": "canopy",
            "product": "meta",
            "version": "2",
            "attribution": dict(source_attribution),
        }
        (entry,) = source_attributions(source, ACCESSED)
        assert entry == {
            **source_attribution,
            "dataset": "Meta CHMv2 Global Canopy Height Maps",
            "license_name": "LicenseRef-DINOv3",
        }
        # source.attribution itself is left as it was.
        assert source["attribution"] == source_attribution

    @pytest.mark.parametrize(
        "source",
        [
            {"name": "uniform"},
            {"name": "compose"},
            {"name": "resample"},
            {"name": "lookup", "table": "fbfm40"},
            {"name": "fosberg"},
            {"name": "layerset"},
            {"name": "canopy", "product": "naip"},
            {"name": "canopy", "product": "point_cloud"},
            {"name": "canopy", "product": "inventory"},
        ],
    )
    def test_sources_without_third_party_entries(self, source):
        assert source_attributions(source, ACCESSED) == []


class TestAddAttributions:
    def test_keeps_existing_entries(self):
        landfire = source_attributions(
            {"name": "landfire", "product": "fbfm40", "version": "2024"}, ACCESSED
        )
        merged = add_attributions([OSM], landfire)
        assert datasets(merged) == [landfire[0]["dataset"], "OpenStreetMap"]

    def test_none_existing(self):
        assert add_attributions(None, [OSM]) == [OSM]

    def test_duplicate_keeps_earliest_access(self):
        later = {**OSM, "accessed_on": "2026-11-13"}
        assert add_attributions([OSM], [later]) == [OSM]
        assert add_attributions([later], [OSM]) == [OSM]
