"""
api/v2/resources/grids/exports/landscape/examples.py

OpenAPI example payloads for the landscape export endpoint.

These are surfaced in Swagger UI and exercised by integration tests so the
docs stay in sync with the schema.
"""

EXAMPLE_LANDSCAPE_MINIMAL = {
    "fire_behavior_fuel_model": "fbfm40",
    "elevation": {"grid_id": "topo_abc", "band": "elevation"},
    "slope": {"grid_id": "topo_abc", "band": "slope"},
    "aspect": {"grid_id": "topo_abc", "band": "aspect"},
    "fuel_model": {"grid_id": "fbfm_def", "band": "fbfm"},
    "canopy_cover": {"grid_id": "canopy_xyz", "band": "cc"},
    "canopy_height": {"grid_id": "canopy_xyz", "band": "chm"},
    "canopy_base_height": {"grid_id": "canopy_xyz", "band": "cbh"},
    "canopy_bulk_density": {"grid_id": "canopy_xyz", "band": "cbd"},
    "name": "Landscape",
}

EXAMPLE_LANDSCAPE_CUSTOM_RESOLUTION = {
    **EXAMPLE_LANDSCAPE_MINIMAL,
    "alignment": {"target": "domain", "resolution": 10.0},
    "name": "Landscape (10 m)",
}

EXAMPLE_LANDSCAPE_GRID_ALIGNED = {
    **EXAMPLE_LANDSCAPE_MINIMAL,
    "alignment": {"target": "grid", "grid_id": "master_native_fbfm40"},
    "name": "Landscape (aligned to master grid)",
}

CREATE_LANDSCAPE_EXPORT_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_LANDSCAPE_MINIMAL,
        "summary": "Minimal: 8-band landscape (default 30 m domain lattice)",
        "description": (
            "The smallest valid landscape export request: elevation, slope, "
            "and aspect from a topography grid; fuel model codes from an "
            "FBFM40 grid; canopy cover, height, base height, and bulk "
            "density from a canopy grid. Alignment defaults to "
            "Domain-anchored at 30 m (LANDFIRE-native), so grids built from "
            "LANDFIRE sources with default alignment need no alignment "
            "configuration."
        ),
    },
    "custom_resolution": {
        "value": EXAMPLE_LANDSCAPE_CUSTOM_RESOLUTION,
        "summary": "Custom landscape resolution",
        "description": (
            "Tile the Domain bbox at 10 m instead of the default 30 m. Every "
            "role grid must already be built at 10 m on the domain lattice — "
            "the exporter never resamples. Use "
            "`POST .../grids/{grid_id}/resample` to bring grids onto the "
            "target lattice first."
        ),
    },
    "grid_aligned": {
        "value": EXAMPLE_LANDSCAPE_GRID_ALIGNED,
        "summary": "Aligned to an existing grid's lattice",
        "description": (
            "Use `alignment.target='grid'` to anchor the landscape to an "
            "existing grid's CRS/transform/shape instead of the Domain bbox. "
            "Useful when role grids share a non-Domain-anchored lattice "
            "(e.g. all chained off a `target='native'` master grid)."
        ),
    },
}
