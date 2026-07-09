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

EXAMPLE_QUICFIRE_EXPLICIT_DOMAIN = {
    **EXAMPLE_QUICFIRE_MINIMAL,
    "alignment": {"target": "domain", "dx": 1.0, "dy": 1.0, "dz": 1},
    "name": "QUIC-Fire inputs (1m horizontal, 1m vertical)",
}

EXAMPLE_QUICFIRE_SHORT_DOMAIN = {
    **EXAMPLE_QUICFIRE_MINIMAL,
    "alignment": {"dx": 1.0, "dy": 1.0},
    "name": "QUIC-Fire inputs (1m, target omitted)",
}

EXAMPLE_QUICFIRE_EXPLICIT_MERGE = {
    **EXAMPLE_QUICFIRE_WITH_SAVR,
    "rhof_merge": "sum",
    "moist_merge": "weighted_avg",
    "savr_merge": "weighted_avg",
    "name": "QUIC-Fire inputs (explicit merge fields)",
}

EXAMPLE_QUICFIRE_GRID_ALIGNED = {
    **EXAMPLE_QUICFIRE_MINIMAL,
    "alignment": {"target": "grid", "grid_id": "master_native_fbfm40"},
    "name": "QUIC-Fire inputs (aligned to master grid)",
}

CREATE_QUICFIRE_EXPORT_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_QUICFIRE_MINIMAL,
        "summary": "Minimal: surface + canopy fuel only (default 2 m / 1 m fire grid)",
        "description": (
            "The smallest valid QUIC-Fire export request. Alignment defaults "
            "to Domain-anchored at QUIC-Fire's recommended values "
            "(dx=dy=2 m, dz=1 m). Produces treesrhof.dat, treesmoist.dat, "
            "treesfueldepth.dat, metadata.json, and domain.geojson."
        ),
    },
    "explicit_domain": {
        "value": EXAMPLE_QUICFIRE_EXPLICIT_DOMAIN,
        "summary": "Explicit Domain alignment with custom resolution",
        "description": (
            "Override the default 2 m / 1 m fire grid. Useful for coarser "
            "simulations (e.g. dx=4 m) or finer vertical resolution (e.g. "
            "dz=0.5 m). All role grids must still be lattice-aligned and "
            "cover the resulting fire-grid extent."
        ),
    },
    "short_domain": {
        "value": EXAMPLE_QUICFIRE_SHORT_DOMAIN,
        "summary": "1 m fire grid, target omitted (shorthand)",
        "description": (
            "`alignment.target` defaults to `'domain'`, so you can pass just "
            '`{"dx": 1, "dy": 1}` without repeating the target. Every role '
            "grid must be built at 1 m."
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
    "explicit_merge": {
        "value": EXAMPLE_QUICFIRE_EXPLICIT_MERGE,
        "summary": "With explicit merge fields",
        "description": (
            "Same as 'with_savr' but with the surface/canopy stitching fields "
            "set explicitly. `moist_merge='weighted_avg'` opts into mass-"
            "weighted moisture at k=0 (default is `'max'`, which matches v1)."
        ),
    },
    "grid_aligned": {
        "value": EXAMPLE_QUICFIRE_GRID_ALIGNED,
        "summary": "Aligned to an existing grid's lattice",
        "description": (
            "Use `alignment.target='grid'` to anchor the fire grid to an "
            "existing grid's CRS/transform/shape instead of the Domain bbox. "
            "Useful when role grids share a non-Domain-anchored lattice "
            "(e.g. all chained off a `target='native'` master grid)."
        ),
    },
}
