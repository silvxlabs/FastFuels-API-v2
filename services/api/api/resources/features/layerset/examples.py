"""
api/v2/resources/features/layerset/examples.py

Example request bodies for Layerset endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

# The core example layerset GeoJSON. Polygon shapes derived from a Lubrecht
# site layerset, translated into the bounds of the BLACKFOOT example domain
# so the documented `/features/layerset/geojson` example produces meaningful
# downstream rasterization results when paired with that domain.
_HIERARCHICAL_GEOJSON = {
    "type": "FeatureCollection",
    "name": "blackfoot_example_layerset",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32612"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "Shrub1_52",
                "fuel_type": "shrub",
                "fuel_loading": 1,
                "fuel_height": 2.0,
                "percent_cover": 15,
                "distribution": "uniform_random",
                "patch_size": None,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294083.48046647, 5199205.57570166],
                            [294284.18598006, 5199133.99758107],
                            [294072.44232996, 5198853.44471689],
                            [294083.48046647, 5199205.57570166],
                        ]
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "Shrub2_53",
                "fuel_type": "shrub",
                "fuel_loading": 2,
                "fuel_height": 2.0,
                "percent_cover": 22,
                "distribution": "random_clusters",
                "patch_size": 8.0,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294072.26437098, 5199682.19987124],
                            [294573.5386922, 5199594.76306425],
                            [294186.10735175, 5199345.84167541],
                            [294072.26437098, 5199682.19987124],
                        ]
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "Herb1_52",
                "fuel_type": "herb",
                "fuel_loading": 1,
                "fuel_height": 1.0,
                "percent_cover": 22,
                "distribution": "uniform_random",
                "patch_size": None,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294130.69988096, 5199249.69856456],
                            [294419.42243816, 5199273.35748891],
                            [294128.26474852, 5198962.69921029],
                            [294130.69988096, 5199249.69856456],
                        ]
                    ],
                    [
                        [
                            [294532.26226275, 5199530.76412423],
                            [294849.82095037, 5199638.29520852],
                            [294741.71628427, 5199322.00825025],
                            [294532.26226275, 5199530.76412423],
                        ]
                    ],
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "Herb1_53",
                "fuel_type": "herb",
                "fuel_loading": 1,
                "fuel_height": 3.0,
                "percent_cover": 50,
                "distribution": "uniform_random",
                "patch_size": None,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294439.59770565, 5199759.67483683],
                            [294688.91756758, 5199675.13200658],
                            [294475.54357416, 5199443.66854859],
                            [294439.59770565, 5199759.67483683],
                        ]
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "Herb2_22",
                "fuel_type": "herb",
                "fuel_loading": 3,
                "fuel_height": 1.0,
                "percent_cover": 20,
                "distribution": "random_clusters",
                "patch_size": 5.0,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294195.40890056, 5199148.56814316],
                            [294253.21398324, 5199138.63796577],
                            [294219.57968408, 5199082.63227565],
                            [294175.69358256, 5199093.79503399],
                            [294195.40890056, 5199148.56814316],
                        ]
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "LitterLichenMoss_52",
                "fuel_type": "litter",
                "fuel_loading": 1,
                "fuel_height": 0.08,
                "percent_cover": 60,
                "distribution": "homogeneous",
                "patch_size": None,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294092.8211293, 5199608.14575947],
                            [294769.78988294, 5199485.85688618],
                            [294174.92507172, 5198988.93255592],
                            [294092.8211293, 5199608.14575947],
                        ]
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "strata_fb": "LitterLichenMoss_53",
                "fuel_type": "litter",
                "fuel_loading": 2,
                "fuel_height": 0.05,
                "percent_cover": 10,
                "distribution": "homogeneous",
                "patch_size": None,
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [294029.28510358, 5199877.74955579],
                            [294658.80237681, 5199704.65830969],
                            [294143.54426584, 5199345.61822959],
                            [294029.28510358, 5199877.74955579],
                        ]
                    ]
                ],
            },
        },
    ],
}


EXAMPLE_LAYERSET_MINIMAL = {
    "type": "layerset",
    "geojson": _HIERARCHICAL_GEOJSON,
    # "description": "TO BE PAIRED WITH BLACKFOOT EXAMPLE DOMAIN",
}

EXAMPLE_LAYERSET_WITH_METADATA = {
    "type": "layerset",
    "name": "Custom Surface Fuels",
    # "description": "TO BE PAIRED WITH BLACKFOOT EXAMPLE DOMAIN",
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
