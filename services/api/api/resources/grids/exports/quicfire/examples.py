"""
api/v2/resources/grids/exports/quicfire/examples.py

OpenAPI example payloads for the QUIC-Fire combined export endpoint.

These are surfaced in Swagger UI and exercised by integration tests so the
docs stay in sync with the schema.
"""

EXAMPLE_QUICFIRE_MINIMAL = {
    "canopy_bulk_density": {"grid_id": "tree_xyz", "band": "bulk_density.foliage.live"},
    "canopy_moisture": {"grid_id": "tree_xyz", "band": "fuel_moisture.live"},
    "surface_fuel_load": {"grid_id": "lookup_abc", "band": "fuel_load.1hr"},
    "surface_fuel_depth": {"grid_id": "lookup_abc", "band": "fuel_depth"},
    "surface_moisture": {"grid_id": "uniform_def", "band": "fuel_moisture.1hr"},
    "name": "QUIC-Fire inputs",
}

EXAMPLE_QUICFIRE_WITH_TOPOGRAPHY = {
    **EXAMPLE_QUICFIRE_MINIMAL,
    "topography": {"grid_id": "topo_xyz", "band": "elevation"},
    "name": "QUIC-Fire inputs (with terrain)",
}

EXAMPLE_QUICFIRE_WITH_SAVR = {
    **EXAMPLE_QUICFIRE_WITH_TOPOGRAPHY,
    "canopy_savr": {"grid_id": "tree_xyz", "band": "savr.foliage"},
    "surface_savr": {"grid_id": "lookup_abc", "band": "savr.1hr"},
    "name": "QUIC-Fire inputs (with terrain and SAVR)",
}

CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_QUICFIRE_MINIMAL,
        "summary": "Minimal: surface + canopy fuel only",
        "description": (
            "The smallest valid QUIC-Fire export request. Produces "
            "treesrhof.dat, treesmoist.dat, treesfueldepth.dat, "
            "metadata.json, and domain.geojson."
        ),
    },
    "with_topography": {
        "value": EXAMPLE_QUICFIRE_WITH_TOPOGRAPHY,
        "summary": "With terrain",
        "description": "Adds topo.dat to the output zip.",
    },
    "with_savr": {
        "value": EXAMPLE_QUICFIRE_WITH_SAVR,
        "summary": "With terrain and SAVR",
        "description": (
            "Adds treesss.dat to the output zip. Requires both canopy_savr "
            "and surface_savr to be provided together."
        ),
    },
}
