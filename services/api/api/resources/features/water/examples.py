"""
api/v2/resources/features/water/examples.py

Example request bodies for Water feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_WATER_MINIMAL = {"type": "water"}

EXAMPLE_WATER_WITH_METADATA = {
    "type": "water",
    "name": "OSM Water Bodies and Streams",
    "description": "OpenStreetMap lakes, ponds, rivers, and streams. Linear waterways are automatically buffered into polygons.",
    "tags": ["hydrology", "osm", "water"],
}

EXAMPLE_WATER_WITH_BUFFER = {
    "type": "water",
    "name": "OSM Water Features (with overhang)",
    "description": "Water features clipped to the domain plus 50 m, so streams and rivers crossing the boundary continue past it.",
    "tags": ["hydrology", "osm", "water"],
    "extent_buffer_m": 50,
}

CREATE_WATER_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_WATER_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Kicks off a background task to extract, buffer, and clip water "
            "features for this domain using default OpenStreetMap data."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_WATER_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named water feature with custom tags for organization. "
            "Useful for distinguishing between different hydrology layers or scenarios."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_WATER_WITH_BUFFER,
        "summary": "With clip-extent buffer",
        "description": (
            "Expands the clip extent by 50 m before clipping, so streams and "
            "rivers that cross the domain boundary continue past it instead "
            "of being cut at the edge."
        ),
    },
}

WATER_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_WATER_MINIMAL),
    ("with_metadata", EXAMPLE_WATER_WITH_METADATA),
    ("with_buffer", EXAMPLE_WATER_WITH_BUFFER),
]

# A hand-drawn stock pond in the Blackfoot example domain, in WGS84 lon/lat.
EXAMPLE_WATER_GEOJSON = {
    "type": "FeatureCollection",
    "name": "Stock pond",
    "description": "A pond that OpenStreetMap is missing.",
    "tags": ["hydrology"],
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Stock pond"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.6971539, 46.9177891],
                        [-113.6971814, 46.9177074],
                        [-113.6972649, 46.9176462],
                        [-113.6973819, 46.9176217],
                        [-113.697501, 46.9176406],
                        [-113.6975904, 46.9176978],
                        [-113.6976261, 46.9177779],
                        [-113.6975986, 46.9178596],
                        [-113.6975151, 46.9179209],
                        [-113.6973981, 46.9179453],
                        [-113.697279, 46.9179264],
                        [-113.6971896, 46.9178692],
                        [-113.6971539, 46.9177891],
                    ]
                ],
            },
        },
    ],
}

CREATE_WATER_GEOJSON_OPENAPI_EXAMPLES = {
    "pond": {
        "value": EXAMPLE_WATER_GEOJSON,
        "summary": "Hand-drawn pond",
        "description": (
            "Uploads one pond polygon in WGS84 (no `crs` block). It is "
            "reprojected to the domain CRS and clipped to the domain."
        ),
    },
}

WATER_GEOJSON_EXAMPLE_VALUES = [("pond", EXAMPLE_WATER_GEOJSON)]

EXAMPLE_WATER_CONCATENATE = {
    "source_feature_ids": [
        "4f6c1b2e9a7d4c3b8e5f0a1d2c3b4a59",
        "b81e7d6c5a4f4e3d9c2b1a0f9e8d7c6b",
    ],
    "name": "Water and ponds",
    "description": "OpenStreetMap water plus hand-drawn ponds.",
    "tags": ["hydrology"],
}

CONCATENATE_WATER_OPENAPI_EXAMPLES = {
    "water_and_ponds": {
        "value": EXAMPLE_WATER_CONCATENATE,
        "summary": "OSM water plus uploaded ponds",
        "description": (
            "Appends the rows of an OpenStreetMap water feature and an uploaded "
            "pond feature into one water feature."
        ),
    },
}
