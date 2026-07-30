"""
Refresh the mirrored USGS 3DEP EPT catalog.

The published catalog is a ~9 MB GeoJSON file in a GitHub repository with no
availability guarantee, and the point cloud coverage endpoint depends on it, so
we serve a GeoParquet mirror from GCS and fall back to the source. Run this to
create or update the mirror.

USGS publishes new acquisitions periodically; re-run when coverage for a known
area looks stale.

    uv run python scripts/refresh_ept_catalog.py
"""

import sys

import geopandas as gpd

from lib.entwine import CATALOG_MIRROR, CATALOG_URL


def main() -> int:
    print(f"Reading catalog from {CATALOG_URL}")
    catalog = gpd.read_file(CATALOG_URL)
    print(f"  {len(catalog)} acquisitions, CRS {catalog.crs}")

    missing = {"name", "url", "count"} - set(catalog.columns)
    if missing:
        print(f"  catalog is missing expected columns: {sorted(missing)}")
        return 1

    print(f"Writing mirror to {CATALOG_MIRROR}")
    # write_covering_bbox stores a bbox column so readers can prune row groups
    # instead of loading every footprint.
    catalog.to_parquet(CATALOG_MIRROR, write_covering_bbox=True, index=False)

    verify = gpd.read_parquet(CATALOG_MIRROR)
    print(f"  verified {len(verify)} acquisitions readable from the mirror")
    return 0


if __name__ == "__main__":
    sys.exit(main())
