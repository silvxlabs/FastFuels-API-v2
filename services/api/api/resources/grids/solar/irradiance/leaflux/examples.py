CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES = {
    "minimal": {
        "summary": "Minimal - default surface band, no terrain",
        "value": {
            "source_lad_grid_id": "LAD_GRID_ID",
            "date_time": "2025-07-01T19:00:00Z",
        },
    },
    "full": {
        "summary": "Full - canopy + surface with terrain",
        "value": {
            "name": "Midday irradiance",
            "description": "Relative canopy and surface irradiance.",
            "tags": ["solar", "irradiance"],
            "source_lad_grid_id": "LAD_GRID_ID",
            "source_terrain_grid_id": "TERRAIN_GRID_ID",
            "bands": [
                "irradiance.canopy.relative",
                "irradiance.surface.relative",
            ],
            "date_time": "2025-07-01T19:00:00Z",
            "extinction_coefficient": 0.5,
        },
    },
    "canopy_only": {
        "summary": "Canopy only - no terrain needed",
        "value": {
            "source_lad_grid_id": "LAD_GRID_ID",
            "bands": ["irradiance.canopy.relative"],
            "date_time": "2025-07-01T19:00:00Z",
        },
    },
    "surface_flat": {
        "summary": "Surface only - flat plane, no terrain supplied",
        "value": {
            "source_lad_grid_id": "LAD_GRID_ID",
            "bands": ["irradiance.surface.relative"],
            "date_time": "2025-07-01T19:00:00Z",
        },
    },
    "surface_terrain": {
        "summary": "Surface only - draped on a supplied terrain grid",
        "value": {
            "source_lad_grid_id": "LAD_GRID_ID",
            "source_terrain_grid_id": "TERRAIN_GRID_ID",
            "bands": ["irradiance.surface.relative"],
            "date_time": "2025-07-01T19:00:00Z",
        },
    },
}

ALL_LEAFLUX_IRRADIANCE_EXAMPLE_VALUES = [
    (name, example["value"])
    for name, example in CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES.items()
]
