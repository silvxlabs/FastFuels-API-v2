"""
api/v2/resources/features/layerset/examples.py

Example request bodies for Layerset endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# The core hierarchical GeoJSON payload with the required 8 strata
_HIERARCHICAL_GEOJSON = {
    "type": "FeatureCollection",
    "metadata": {"created": "2026-05-04T19:34:00.000Z", "total_features": 8},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "strata": "Shrub1",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {
                            "percent_cover": 50,
                            "height": 2,
                            "species": ["kinnikinnick", "common juniper"],
                        },
                        "polygons": {
                            "type": "MultiPolygon",
                            "coordinates": [
                                [
                                    [
                                        [
                                            [-114.09545796676623, 46.8324794598619],
                                            [-114.11217537297199, 46.8324794598619],
                                            [-114.11217537297199, 46.82496749915157],
                                            [-114.09545796676623, 46.82496749915157],
                                            [-114.09545796676623, 46.8324794598619],
                                        ]
                                    ]
                                ]
                            ],
                        },
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "Shrub2",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"percent_cover": 0, "height": 0},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "Herb1",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"percent_cover": 50, "height": 2},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "Herb2",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"percent_cover": 0, "height": 0},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "DownWoodFine",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"load_1hr": 0, "load_10hr": 0, "load_100hr": 0},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "DownWoodCoarse",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"load_1000hr_sound": 0, "load_1000hr_rotten": 0},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "LitterLichenMoss",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"depth": 0, "arrangement": "Freshly Fallen"},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata": "GroundFuels",
                "fuelbeds": [
                    {
                        "number": 22,
                        "settings": {"upper_depth": 0, "lower_depth": 0},
                        "polygons": {"type": "MultiPolygon", "coordinates": []},
                    }
                ],
            },
        },
    ],
}

EXAMPLE_LAYERSET_MINIMAL = {
    "type": "layerset",
    "geojson": _HIERARCHICAL_GEOJSON,
    "description": "TO BE PAIRED WITH BLACKFOOT EXAMPLE DOMAIN",
}

EXAMPLE_LAYERSET_WITH_METADATA = {
    "type": "layerset",
    "name": "Custom Surface Fuels",
    "description": "TO BE PAIRED WITH BLACKFOOT EXAMPLE DOMAIN",
    "tags": ["surface-fuels", "custom", "shrub"],
    "geojson": _HIERARCHICAL_GEOJSON,
}

CREATE_LAYERSET_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_LAYERSET_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Uploads a hierarchical GeoJSON layerset with default metadata. "
            "The system will automatically extract bounding box coordinates."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_LAYERSET_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Uploads a named layerset with custom tags for organization. "
            "Useful when maintaining multiple custom fuel scenarios within a single domain."
        ),
    },
}

LAYERSET_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_LAYERSET_MINIMAL),
    ("with_metadata", EXAMPLE_LAYERSET_WITH_METADATA),
]
