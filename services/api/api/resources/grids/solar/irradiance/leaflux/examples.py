CREATE_LEAFLUX_IRRADIANCE_GRID_EXAMPLES = {
    "minimal": {
        "summary": "Minimal - canopy irradiance at an instant",
        "value": {
            "source_grid_id": "GRID_ID",
            "date_time": "2025-07-01T19:00:00Z",
        },
    },
    "full": {
        "summary": "Full - canopy + surface with terrain and explicit location",
        "value": {
            "name": "Midday irradiance",
            "description": "Relative canopy and surface irradiance.",
            "tags": ["solar", "irradiance"],
            "source_grid_id": "GRID_ID",
            "source_terrain_grid_id": "TERRAIN_GRID_ID",
            "bands": [
                "irradiance.canopy.relative",
                "irradiance.surface.relative",
            ],
            "date_time": "2025-07-01T19:00:00Z",
            "latitude": 39.0,
            "longitude": -120.0,
            "extinction_coefficient": 0.5,
        },
    },
}
