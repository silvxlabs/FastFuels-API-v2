"""
Attributions for the third-party data a source grid is built from.

Griddle adds these to the grid's top-level ``attributions`` when a source
handler finishes. They are a resource-level record only: never write them into
the grid's data or stored files.
"""

import re
from datetime import date

from griddle.handlers.landfire import disturbance_layer_versions
from lib.landfire import LANDFIRE_VERSIONS

US_PUBLIC_DOMAIN_LICENSE = "LicenseRef-US-Government-Public-Domain"
USGS_PUBLIC_DOMAIN_URL = (
    "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
)

THREEDEP_ACCESS_URL = "https://www.usgs.gov/3d-elevation-program"
_THREEDEP_PROJECT = re.compile(r"/Elevation/1m/Projects/([^/]+)/")
# (dataset, USGS dataset title) by the product directory in the tile URL.
_THREEDEP_PRODUCTS = {
    "/Elevation/S1M/": (
        "USGS 3DEP Seamless 1 m DEM (S1M)",
        "3D Elevation Program Seamless 1-Meter Digital Elevation Model (S1M)",
    ),
    "/Elevation/13/": (
        "USGS 3DEP 1/3 arc-second DEM",
        "3D Elevation Program 1/3 Arc-Second Digital Elevation Model",
    ),
    "/Elevation/1/": (
        "USGS 3DEP 1 arc-second DEM",
        "3D Elevation Program 1 Arc-Second Digital Elevation Model",
    ),
}
_THREEDEP_OTHER = (
    "USGS 3D Elevation Program (3DEP)",
    "3D Elevation Program Digital Elevation Model",
)

LANDFIRE_ACCESS_URL = "https://landfire.gov"
LANDFIRE_PUBLISHER = (
    "U.S. Department of the Interior, Geological Survey, and "
    "U.S. Department of Agriculture"
)
# LANDFIRE product titles by the registry / band key griddle reads.
LANDFIRE_TITLES = {
    "fbfm13": "13 Anderson Fire Behavior Fuel Models",
    "fbfm40": "40 Scott and Burgan Fire Behavior Fuel Models",
    "fccs": "Fuel Characteristic Classification System Fuelbeds",
    "annual_disturbance": "Annual Disturbance",
    "fdist": "Fuel Disturbance",
    "fvt": "Fuel Vegetation Type",
    "fvc": "Fuel Vegetation Cover",
    "fvh": "Fuel Vegetation Height",
    "bps": "Biophysical Settings",
    "chm": "Forest Canopy Height",
    "cbd": "Forest Canopy Bulk Density",
    "cbh": "Forest Canopy Base Height",
    "cc": "Forest Canopy Cover",
    "elevation": "Elevation",
    "slope": "Slope Degrees",
    "aspect": "Aspect",
}

# Citations as the USDA Forest Service Research Data Archive gives them, by
# TreeMap version, with the archive's catalog page.
TREEMAP_CITATIONS = {
    "2014": (
        "Riley, Karin L.; Grenfell, Isaac C.; Finney, Mark A.; Wiener, Jason M.; "
        "Houtman, Rachel M. 2019. Fire Lab tree list: A tree-level model of the "
        "conterminous United States landscape circa 2014. Fort Collins, CO: Forest "
        "Service Research Data Archive. https://doi.org/10.2737/RDS-2019-0026",
        "https://www.fs.usda.gov/rds/archive/catalog/RDS-2019-0026",
    ),
    "2016": (
        "Riley, Karin L.; Grenfell, Isaac C.; Finney, Mark A.; Shaw, John D. 2021. "
        "TreeMap 2016: A tree-level model of the forests of the conterminous United "
        "States circa 2016. Fort Collins, CO: Forest Service Research Data Archive. "
        "https://doi.org/10.2737/RDS-2021-0074",
        "https://www.fs.usda.gov/rds/archive/catalog/RDS-2021-0074",
    ),
    "2020": (
        "Zimmer, Scott N.; Houtman, Rachel M.; Leatherman, Lila S. T.; Housman, "
        "Ian W.; Shrestha, Abhinav; Shaw, John D.; Riley, Karin L. 2025. TreeMap "
        "2020 CONUS: A tree-level model of the forests of the conterminous United "
        "States circa 2020. Fort Collins, CO: Forest Service Research Data Archive. "
        "https://doi.org/10.2737/RDS-2025-0031",
        "https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0031",
    ),
    "2022": (
        "Houtman, Rachel M.; Leatherman, Lila S. T.; Zimmer, Scott N.; Housman, "
        "Ian W.; Shrestha, Abhinav; Shaw, John D.; Riley, Karin L. 2025. TreeMap "
        "2022 CONUS: A tree-level model of the forests of the conterminous United "
        "States circa 2022. Fort Collins, CO: Forest Service Research Data Archive. "
        "https://doi.org/10.2737/RDS-2025-0032",
        "https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0032",
    ),
}

# Dataset name and SPDX license id of the Meta CHM entry griddle writes to
# source.attribution, by version.
META_CHM_DATASETS = {
    "1": ("High Resolution Canopy Height Maps by WRI and Meta", "CC-BY-4.0"),
    "2": ("Meta CHMv2 Global Canopy Height Maps", "LicenseRef-DINOv3"),
}


def add_attributions(existing: list[dict] | None, entries: list[dict]) -> list[dict]:
    """Add ``entries`` to a grid's ``attributions``, keeping what is there.

    The same rule as the API's ``merge_attributions``: entries are
    de-duplicated by ``(dataset, license_name)``, the earliest ``accessed_on``
    wins, and the result is sorted by dataset, then license name.
    """
    merged: dict[tuple[str, str], dict] = {}
    for entry in [*(existing or []), *entries]:
        key = (entry["dataset"], entry["license_name"])
        kept = merged.get(key)
        if kept is None or entry["accessed_on"] < kept["accessed_on"]:
            merged[key] = dict(entry)
    return [merged[key] for key in sorted(merged)]


def source_attributions(source: dict, accessed_on: date) -> list[dict]:
    """Attributions for the third-party data a source handler read.

    Call after the handler has run: 3DEP reads its tiles from, and Meta CHM
    its entry from, what the handler wrote to ``source``. Sources built from
    no third-party data return an empty list.
    """
    match source["name"], source.get("product"):
        case "3dep", "topography":
            return threedep_attributions(source["tile_metadata"]["tiles"], accessed_on)
        case "landfire", product:
            return _landfire_source_attributions(source, product, accessed_on)
        case "canopy", "landfire":
            version = source.get("version", "2024")
            return [
                landfire_attribution(band, version, accessed_on)
                for band in source["bands"]
            ]
        case "canopy", "meta":
            return [meta_chm_attribution(source)]
        case "pim", "treemap":
            return [treemap_attribution(source.get("version", "2022"), accessed_on)]
        case _:
            return []


def threedep_attributions(tile_urls: list[str], accessed_on: date) -> list[dict]:
    """One entry per 3DEP product, and per lidar project, among ``tile_urls``."""
    products = set()
    for url in tile_urls:
        if project := _THREEDEP_PROJECT.search(url):
            name = project.group(1)
            products.add(
                (
                    f"USGS 3DEP 1 m lidar DEM ({name})",
                    "3D Elevation Program 1-Meter Resolution Digital Elevation "
                    f"Model ({name})",
                )
            )
            continue
        products.add(
            next(
                (p for part, p in _THREEDEP_PRODUCTS.items() if part in url),
                _THREEDEP_OTHER,
            )
        )
    return [
        {
            "dataset": dataset,
            "license_name": US_PUBLIC_DOMAIN_LICENSE,
            "license_url": USGS_PUBLIC_DOMAIN_URL,
            "citation": (
                f"U.S. Geological Survey, {title}, accessed "
                f"{_month_day_year(accessed_on)} at {THREEDEP_ACCESS_URL}."
            ),
            "access_url": THREEDEP_ACCESS_URL,
            "accessed_on": accessed_on.isoformat(),
        }
        for dataset, title in sorted(products)
    ]


def landfire_attribution(
    product: str, version: str, accessed_on: date, season: str | None = None
) -> dict:
    """The entry for one LANDFIRE product at ``version``."""
    title = LANDFIRE_TITLES[product]
    if season is not None:
        title = f"{title}, {season} season"
    return {
        "dataset": f"LANDFIRE {title} (LF {version})",
        "license_name": US_PUBLIC_DOMAIN_LICENSE,
        "license_url": USGS_PUBLIC_DOMAIN_URL,
        "citation": (
            f"LANDFIRE, {title}, LANDFIRE {version}, {LANDFIRE_PUBLISHER}. "
            f"Accessed {accessed_on:%d %B %Y} at {LANDFIRE_ACCESS_URL}."
        ),
        "access_url": LANDFIRE_ACCESS_URL,
        "accessed_on": accessed_on.isoformat(),
    }


def treemap_attribution(version: str, accessed_on: date) -> dict:
    """The entry for TreeMap at ``version``."""
    citation, catalog_url = TREEMAP_CITATIONS[version]
    return {
        "dataset": f"TreeMap {version}",
        "license_name": US_PUBLIC_DOMAIN_LICENSE,
        "license_url": catalog_url,
        "citation": citation,
        "access_url": catalog_url,
        "accessed_on": accessed_on.isoformat(),
    }


def meta_chm_attribution(source: dict) -> dict:
    """The resource-level entry for the Meta CHM ``source.attribution``."""
    dataset, license_name = META_CHM_DATASETS[source.get("version", "2")]
    return {
        **source["attribution"],
        "dataset": dataset,
        "license_name": license_name,
    }


def _landfire_source_attributions(
    source: dict, product: str, accessed_on: date
) -> list[dict]:
    if product == "topography":
        version = source.get("version", "2020")
        return [
            landfire_attribution(band, version, accessed_on) for band in source["bands"]
        ]
    version = source.get("version", LANDFIRE_VERSIONS[product]["default"])
    entries = [
        landfire_attribution(product, version, accessed_on, source.get("season"))
    ]
    disturbance = source.get("disturbance")
    if disturbance and product in ("fbfm13", "fbfm40", "fccs"):
        entries += [
            landfire_attribution(layer, layer_version, accessed_on)
            for layer, layer_version in disturbance_layer_versions(disturbance).items()
        ]
    return entries


def _month_day_year(day: date) -> str:
    """``November 13, 2026``, the date form in USGS's citation examples."""
    return f"{day:%B} {day.day}, {day.year}"
