"""
Building feature handlers for Overture Maps data.

Reads building footprints from the latest Overture Maps release, clips them to
the domain, and saves footprint, height and the upstream dataset of each
building as GeoParquet.
"""

import logging
from datetime import date

import geopandas as gpd
import pandas as pd

from etcher.handlers.road import compute_georeference
from etcher.osm_source import OSM_ATTRIBUTION
from etcher.overture_source import (
    latest_release,
    overture_attribution,
    read_overture_buildings,
)
from etcher.storage import save_features
from lib.domain_utils import buffer_domain

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["geometry", "height", "height_source", "source_dataset"]

# Overture `sources[].property` values: the dataset that supplied the whole
# building, and the dataset that supplied its height.
WHOLE_BUILDING_PROPERTY = ""
HEIGHT_PROPERTY = "/properties/height"

ODBL_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
OVERTURE_ATTRIBUTION_URL = "https://docs.overturemaps.org/attribution/"

# Citation and links for the upstream datasets named in Overture's sources[],
# keyed by their `dataset` value. The license comes from Overture's own label.
UPSTREAM_ATTRIBUTIONS = {
    "OpenStreetMap": {
        "license_url": OSM_ATTRIBUTION["license_url"],
        "citation": OSM_ATTRIBUTION["citation"],
        "access_url": OSM_ATTRIBUTION["access_url"],
    },
    "Microsoft ML Buildings": {
        "license_url": ODBL_URL,
        "citation": "Microsoft Global ML Building Footprints",
        "access_url": "https://github.com/microsoft/GlobalMLBuildingFootprints",
    },
}


def handle_overture(
    feature: dict, source: dict, domain_gdf: gpd.GeoDataFrame, progress
) -> dict:
    """Process an Overture Maps building feature request.

    Buildings that cross the (buffered) domain boundary are clipped to it.

    Args:
        feature: Full feature document from Firestore
        source: Source dict with Overture-specific fields
        domain_gdf: Domain geometry as GeoDataFrame
        progress: Callback for progress reporting

    Returns:
        Dict with 'georeference', 'attributions' and 'release' keys
    """
    feature_id = feature["id"]
    domain_id = feature["domain_id"]

    progress("Preparing domain boundary...", 10)

    native_crs = domain_gdf.crs
    domain_gdf = buffer_domain(domain_gdf, source.get("extent_buffer_m", 0))

    # Overture data is in EPSG:4326.
    bbox = tuple(domain_gdf.to_crs(epsg=4326).total_bounds)

    progress("Finding the latest Overture Maps release...", 20)
    release = latest_release()

    progress("Reading Overture Maps buildings...", 30)
    buildings = read_overture_buildings(bbox, release)

    progress("Clipping to domain boundary...", 70)
    if buildings.empty:
        logger.info(
            f"No building features found for domain {domain_id}",
            extra={"feature_id": feature_id},
        )
        clipped = buildings
    else:
        clipped = buildings.to_crs(native_crs).clip(domain_gdf, keep_geom_type=True)

    attributions = [overture_attribution(), *upstream_attributions(clipped)]
    final_gdf = building_columns(clipped, native_crs)

    progress("Saving features to storage...", 90)
    save_features(domain_id, feature_id, final_gdf)

    progress("Computing georeference...", 95)
    georeference = compute_georeference(domain_gdf)

    progress("Complete", 100)

    return {
        "georeference": georeference,
        "attributions": attributions,
        "release": release,
    }


def sources_list(sources) -> list[dict]:
    """A building's ``sources`` entries; empty when Overture left it null."""
    return [] if sources is None else list(sources)


def dataset_for(sources, prop: str) -> str | None:
    """Dataset of the first ``sources`` entry with the given ``property``."""
    for entry in sources_list(sources):
        if entry.get("property") == prop:
            return entry.get("dataset")
    return None


def building_columns(buildings: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """Reduce Overture buildings to geometry, height and their source datasets.

    ``height_source`` is the dataset that supplied the height, falling back to
    the dataset that supplied the whole building; null when height is null.
    """
    if buildings.empty:
        return gpd.GeoDataFrame(
            {
                "height": pd.Series(dtype="float64"),
                "height_source": pd.Series(dtype="object"),
                "source_dataset": pd.Series(dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )[OUTPUT_COLUMNS]

    height = pd.to_numeric(buildings["height"], errors="coerce").astype("float64")
    source_dataset = buildings["sources"].map(
        lambda s: dataset_for(s, WHOLE_BUILDING_PROPERTY)
    )
    height_source = buildings["sources"].map(lambda s: dataset_for(s, HEIGHT_PROPERTY))
    height_source = height_source.where(height_source.notna(), source_dataset)
    height_source = height_source.where(height.notna(), None)

    return gpd.GeoDataFrame(
        {
            "geometry": buildings.geometry.values,
            "height": height.values,
            "height_source": height_source.values,
            "source_dataset": source_dataset.values,
        },
        crs=crs,
    ).reset_index(drop=True)[OUTPUT_COLUMNS]


def upstream_attributions(buildings: gpd.GeoDataFrame) -> list[dict]:
    """One attribution per upstream dataset named in the buildings' ``sources``.

    Unknown datasets get their name and Overture's license label with links to
    Overture's attribution page, and a logged warning.
    """
    licenses: dict[str, str | None] = {}
    if not buildings.empty:
        for sources in buildings["sources"]:
            for entry in sources_list(sources):
                dataset = entry.get("dataset")
                if dataset:
                    licenses.setdefault(dataset, entry.get("license"))

    accessed_on = date.today().isoformat()
    attributions = []
    for dataset in sorted(licenses):
        known = UPSTREAM_ATTRIBUTIONS.get(dataset)
        if known is None:
            logger.warning(
                f"Unknown Overture source dataset '{dataset}'; "
                "attributing it with Overture's license label"
            )
            known = {
                "license_url": OVERTURE_ATTRIBUTION_URL,
                "citation": dataset,
                "access_url": OVERTURE_ATTRIBUTION_URL,
            }
        attributions.append(
            {
                "dataset": dataset,
                "license_name": licenses[dataset] or "ODbL-1.0",
                **known,
                "accessed_on": accessed_on,
            }
        )
    return attributions
