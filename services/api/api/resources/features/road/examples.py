"""
api/v2/resources/features/road/examples.py

Example request bodies for Road feature endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
"""

EXAMPLE_ROAD_MINIMAL = {"type": "road"}

EXAMPLE_ROAD_WITH_METADATA = {
    "type": "road",
    "name": "OSM Road Network",
    "description": "OpenStreetMap roads automatically buffered to realistic widths based on highway classification.",
    "tags": ["infrastructure", "osm", "roads"],
}

EXAMPLE_ROAD_WITH_BUFFER = {
    "type": "road",
    "name": "OSM Road Network (with overhang)",
    "description": "Roads clipped to the domain plus 50 m, so segments exiting the boundary are not truncated at the edge.",
    "tags": ["infrastructure", "osm", "roads"],
    "extent_buffer_m": 50,
}

CREATE_ROAD_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_ROAD_MINIMAL,
        "summary": "Minimal request",
        "description": (
            "Kicks off a background task to extract, buffer, and clip the road "
            "network for this domain using default OpenStreetMap data."
        ),
    },
    "with_metadata": {
        "value": EXAMPLE_ROAD_WITH_METADATA,
        "summary": "With name and tags",
        "description": (
            "Creates a named road feature with custom tags for organization. "
            "Useful when managing multiple infrastructure layers."
        ),
    },
    "with_buffer": {
        "value": EXAMPLE_ROAD_WITH_BUFFER,
        "summary": "With clip-extent buffer",
        "description": (
            "Expands the clip extent by 50 m before clipping, so roads that "
            "cross the domain boundary continue past it instead of being cut "
            "at the edge."
        ),
    },
}

ROAD_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_ROAD_MINIMAL),
    ("with_metadata", EXAMPLE_ROAD_WITH_METADATA),
    ("with_buffer", EXAMPLE_ROAD_WITH_BUFFER),
]

# Two hand-drawn driveways in the Blackfoot example domain, in WGS84 lon/lat.
# OpenStreetMap maps neither.
EXAMPLE_ROAD_GEOJSON = {
    "type": "FeatureCollection",
    "name": "Driveways",
    "description": "Hand-drawn driveways that OpenStreetMap is missing.",
    "tags": ["roads", "driveways"],
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "driveway"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.7011095, 46.9162721],
                        [-113.7004257, 46.9158347],
                        [-113.7004615, 46.9158084],
                        [-113.7011347, 46.916239],
                        [-113.7021861, 46.9163929],
                        [-113.7021751, 46.9164281],
                        [-113.7011095, 46.9162721],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"type": "driveway"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.6980575, 46.9149857],
                        [-113.6980168, 46.9149711],
                        [-113.6985099, 46.9143295],
                        [-113.6985506, 46.9143441],
                        [-113.6980575, 46.9149857],
                    ]
                ],
            },
        },
    ],
}

CREATE_ROAD_GEOJSON_OPENAPI_EXAMPLES = {
    "driveways": {
        "value": EXAMPLE_ROAD_GEOJSON,
        "summary": "Hand-drawn driveways",
        "description": (
            "Uploads two driveway polygons in WGS84 (no `crs` block). They are "
            "reprojected to the domain CRS and clipped to the domain."
        ),
    },
}

ROAD_GEOJSON_EXAMPLE_VALUES = [("driveways", EXAMPLE_ROAD_GEOJSON)]

EXAMPLE_ROAD_CONCATENATE = {
    "source_feature_ids": [
        "4f6c1b2e9a7d4c3b8e5f0a1d2c3b4a59",
        "b81e7d6c5a4f4e3d9c2b1a0f9e8d7c6b",
    ],
    "name": "Roads and driveways",
    "description": "OpenStreetMap roads plus hand-drawn driveways.",
    "tags": ["roads"],
}

CONCATENATE_ROAD_OPENAPI_EXAMPLES = {
    "roads_and_driveways": {
        "value": EXAMPLE_ROAD_CONCATENATE,
        "summary": "OSM roads plus uploaded driveways",
        "description": (
            "Appends the rows of an OpenStreetMap road feature and an uploaded "
            "driveway feature into one road feature."
        ),
    },
}
