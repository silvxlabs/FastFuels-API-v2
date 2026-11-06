"""
api/v2/resources/features/building/examples.py

Example request bodies for Building feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_BUILDING_MINIMAL = {"type": "building"}

EXAMPLE_BUILDING_WITH_METADATA = {
    "type": "building",
    "name": "Overture Buildings",
    "description": "Building footprints with heights from Overture Maps.",
    "tags": ["infrastructure", "overture", "buildings"],
}

EXAMPLE_BUILDING_WITH_BUFFER = {
    "type": "building",
    "name": "Overture Buildings (with overhang)",
    "description": "Buildings clipped to the domain plus 25 m.",
    "tags": ["infrastructure", "overture", "buildings"],
    "extent_buffer_m": 25,
}

CREATE_BUILDING_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_BUILDING_MINIMAL,
        "summary": "Minimal request",
        "description": "Building footprints clipped to the domain boundary.",
    },
    "with_metadata": {
        "value": EXAMPLE_BUILDING_WITH_METADATA,
        "summary": "With name and tags",
        "description": "A named building feature with tags for organization.",
    },
    "with_buffer": {
        "value": EXAMPLE_BUILDING_WITH_BUFFER,
        "summary": "With clip-extent buffer",
        "description": (
            "Expands the clip extent by 25 m, so buildings that cross the "
            "domain boundary are not cut at the edge."
        ),
    },
}

BUILDING_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_BUILDING_MINIMAL),
    ("with_metadata", EXAMPLE_BUILDING_WITH_METADATA),
    ("with_buffer", EXAMPLE_BUILDING_WITH_BUFFER),
]

# Two hand-drawn buildings in the Blackfoot example domain, in WGS84 lon/lat.
EXAMPLE_BUILDING_GEOJSON = {
    "type": "FeatureCollection",
    "name": "Surveyed buildings",
    "description": "Footprints and heights from a site survey.",
    "tags": ["buildings", "survey"],
    "features": [
        {
            "type": "Feature",
            "properties": {"height": 6.5},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.700314, 46.9157445],
                        [-113.700373, 46.9158446],
                        [-113.7005677, 46.9157908],
                        [-113.7005088, 46.9156907],
                        [-113.700314, 46.9157445],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"height": 4.0},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.6979716, 46.9149877],
                        [-113.6979533, 46.9150767],
                        [-113.6981352, 46.9150943],
                        [-113.6981536, 46.9150053],
                        [-113.6979716, 46.9149877],
                    ]
                ],
            },
        },
    ],
}

CREATE_BUILDING_GEOJSON_OPENAPI_EXAMPLES = {
    "surveyed": {
        "value": EXAMPLE_BUILDING_GEOJSON,
        "summary": "Surveyed buildings",
        "description": (
            "Uploads two building footprints with heights in WGS84 (no `crs` "
            "block). They are reprojected to the domain CRS and clipped to the "
            "domain."
        ),
    },
}

BUILDING_GEOJSON_EXAMPLE_VALUES = [("surveyed", EXAMPLE_BUILDING_GEOJSON)]

EXAMPLE_BUILDING_CONCATENATE = {
    "source_feature_ids": [
        "4f6c1b2e9a7d4c3b8e5f0a1d2c3b4a59",
        "b81e7d6c5a4f4e3d9c2b1a0f9e8d7c6b",
    ],
    "name": "All buildings",
    "description": "Overture buildings plus surveyed buildings Overture is missing.",
    "tags": ["buildings"],
}

CONCATENATE_BUILDING_OPENAPI_EXAMPLES = {
    "overture_and_survey": {
        "value": EXAMPLE_BUILDING_CONCATENATE,
        "summary": "Overture plus uploaded buildings",
        "description": (
            "Appends the rows of an Overture building feature and an uploaded "
            "building feature into one building feature."
        ),
    },
}
