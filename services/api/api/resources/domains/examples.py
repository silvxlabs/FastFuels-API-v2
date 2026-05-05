"""
Example request bodies for the Domain resource endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

All examples represent valid domain creation requests for areas within CONUS
(Continental United States). Examples cover different CRS formats.

IMPORTANT: The v2 API only accepts FeatureCollection input, not individual Features.
If you have a single Feature, wrap it in a FeatureCollection with a single-element
features array.
"""

# =============================================================================
# Example: WGS84 FeatureCollection (Default CRS)
# =============================================================================
# Blue Mountain Recreation Area near Missoula, Montana
# Demonstrates the simplest case: WGS84 coordinates without explicit CRS

EXAMPLE_WGS84_DEFAULT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-114.09545796676623, 46.8324794598619],
                        [-114.11217537297199, 46.8324794598619],
                        [-114.11217537297199, 46.82496749915157],
                        [-114.09545796676623, 46.82496749915157],
                        [-114.09545796676623, 46.8324794598619],
                    ]
                ],
            },
        }
    ],
    "name": "Blue Mountain Recreation Area",
    "description": "Approximately 1 square kilometer in the Blue Mountain Recreation Area near Missoula, Montana.",
}


# =============================================================================
# Example: WGS84 FeatureCollection (Explicit CRS)
# =============================================================================
# Same location but with explicit EPSG:4326 CRS specified

EXAMPLE_WGS84_EXPLICIT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-114.19991252294375, 46.7747518267752],
                        [-114.19991252294375, 46.77023218168779],
                        [-114.19133479742844, 46.77023218168779],
                        [-114.19133479742844, 46.7747518267752],
                        [-114.19991252294375, 46.7747518267752],
                    ]
                ],
            },
        }
    ],
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "name": "3DEP Example Area",
    "description": "Small domain behind Blue Mountain, suitable for 3DEP point cloud testing.",
}


# =============================================================================
# Example: EPSG:5070 (CONUS Albers Equal Area)
# =============================================================================
# Demonstrates using a projected CRS that covers all of CONUS

EXAMPLE_EPSG5070 = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-1377797.8087433458, 2780720.35945401],
                        [-1379055.2389321132, 2780962.840146559],
                        [-1379212.7978295316, 2780146.1745799617],
                        [-1377955.2239776908, 2779903.6661836714],
                        [-1377797.8087433458, 2780720.35945401],
                    ]
                ],
            },
        }
    ],
    "crs": {"type": "name", "properties": {"name": "EPSG:5070"}},
    "name": "EPSG:5070 Example",
    "description": "Domain using CONUS Albers Equal Area projection coordinates.",
}


# =============================================================================
# Example: UTM Zone 11N (EPSG:32611)
# =============================================================================
# Demonstrates using UTM coordinates with URN-format CRS specification

EXAMPLE_UTM = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [721502.7544906491, 5190645.048516054],
                        [720227.9398802927, 5190598.00908098],
                        [720258.6480286171, 5189763.323999467],
                        [721533.6406826023, 5189810.364218195],
                        [721502.7544906491, 5190645.048516054],
                    ]
                ],
            },
        }
    ],
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32611"}},
    "name": "UTM Zone 11N Example",
    "description": "Domain using UTM Zone 11N coordinates (Montana/Idaho region).",
}


# =============================================================================
# Example: Blackfoot River Area
# =============================================================================
# Rich example area with multiple feature types (3DEP, roads, water)

EXAMPLE_BLACKFOOT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-113.70402605601585, 46.91933213469031],
                        [-113.70402605601585, 46.91262776084295],
                        [-113.69531269250463, 46.91262776084295],
                        [-113.69531269250463, 46.91933213469031],
                        [-113.70402605601585, 46.91933213469031],
                    ]
                ],
            },
        }
    ],
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "name": "Blackfoot River Area",
    "description": "Domain near the Blackfoot River containing 3DEP, road, and water features.",
}


# =============================================================================
# Example: Bondurant, Wyoming (S1M 1-Meter Coverage)
# =============================================================================
# Small domain in western Wyoming with known 3DEP S1M (Seamless 1-Meter)
# coverage. Useful for testing 1m resolution topography requests.

EXAMPLE_BONDURANT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-110.72191134418632, 42.635711976574456],
                        [-110.71581298102096, 42.63569701132406],
                        [-110.71579249057953, 42.64019978690923],
                        [-110.72189129343488, 42.64021475450427],
                        [-110.72191134418632, 42.635711976574456],
                    ]
                ],
            },
        }
    ],
    "name": "Bondurant, Wyoming",
    "description": "Small domain in western Wyoming with known 3DEP S1M 1-meter coverage.",
}


# =============================================================================
# Example: Padded Domain (pad_to_resolution)
# =============================================================================
# Demonstrates pad_to_resolution=30 for cross-resolution grid alignment.
# When set, the domain bounding box is snapped to multiples of 30 meters.
# Grids at 30m, 10m, 5m, 2m, 1m (any divisor of 30) will share an aligned
# extent on this domain.

EXAMPLE_PADDED = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-114.09545796676623, 46.8324794598619],
                        [-114.11217537297199, 46.8324794598619],
                        [-114.11217537297199, 46.82496749915157],
                        [-114.09545796676623, 46.82496749915157],
                        [-114.09545796676623, 46.8324794598619],
                    ]
                ],
            },
        }
    ],
    "name": "Padded Domain Example",
    "description": "Domain with pad_to_resolution=30 for cross-resolution grid alignment.",
    "pad_to_resolution": 30,
}


# =============================================================================
# OpenAPI Examples Dictionary
# =============================================================================

CREATE_DOMAIN_OPENAPI_EXAMPLES = {
    "wgs84_default": {
        "value": EXAMPLE_WGS84_DEFAULT,
        "summary": "WGS84 FeatureCollection (Default CRS)",
        "description": (
            "Creates a domain from a FeatureCollection in WGS84 coordinates (EPSG:4326). "
            "When no CRS is specified, the API assumes WGS84. The geometry is automatically "
            "reprojected to an appropriate UTM zone for accurate area calculations."
        ),
    },
    "wgs84_explicit": {
        "value": EXAMPLE_WGS84_EXPLICIT,
        "summary": "WGS84 FeatureCollection (Explicit CRS)",
        "description": (
            "Creates a domain from a FeatureCollection with explicitly specified WGS84 CRS. "
            "Functionally equivalent to omitting the CRS, but demonstrates the explicit format."
        ),
    },
    "epsg5070": {
        "value": EXAMPLE_EPSG5070,
        "summary": "EPSG:5070 FeatureCollection (Albers)",
        "description": (
            "Creates a domain from a FeatureCollection in EPSG:5070 (CONUS Albers Equal Area). "
            "Since this is already a projected coordinate system, the API uses it directly "
            "without reprojection to UTM."
        ),
    },
    "utm": {
        "value": EXAMPLE_UTM,
        "summary": "UTM FeatureCollection (EPSG:32611)",
        "description": (
            "Creates a domain from a FeatureCollection in UTM Zone 11N (EPSG:32611). "
            "Demonstrates using URN format for CRS specification. UTM coordinates "
            "are already projected, so no reprojection is performed."
        ),
    },
    "blackfoot": {
        "value": EXAMPLE_BLACKFOOT,
        "summary": "Blackfoot River FeatureCollection",
        "description": (
            "Creates a domain near the Blackfoot River in Montana. This area contains "
            "diverse feature types including 3DEP point cloud data, roads, and water bodies."
        ),
    },
    "bondurant": {
        "value": EXAMPLE_BONDURANT,
        "summary": "Bondurant, Wyoming (S1M 1m Coverage)",
        "description": (
            "Creates a domain in western Wyoming with known 3DEP Seamless 1-Meter (S1M) "
            "coverage. Use this area to test 1m resolution topography grid requests."
        ),
    },
    "padded": {
        "value": EXAMPLE_PADDED,
        "summary": "Padded Domain (pad_to_resolution=30)",
        "description": (
            "Creates a domain with pad_to_resolution=30. The domain bounding box is "
            "snapped to multiples of 30 meters, so grids at 30m, 10m, 5m, 2m, 1m (any "
            "divisor of 30) on this domain will share an aligned extent. Use this "
            "pattern when you need to compare or compose grids at multiple resolutions."
        ),
    },
}


# =============================================================================
# Test Data Export
# =============================================================================

# All example values for integration testing
# Each tuple contains (name, example_value) for parameterized tests
ALL_EXAMPLE_VALUES = [
    ("wgs84_default", EXAMPLE_WGS84_DEFAULT),
    ("wgs84_explicit", EXAMPLE_WGS84_EXPLICIT),
    ("epsg5070", EXAMPLE_EPSG5070),
    ("utm", EXAMPLE_UTM),
    ("blackfoot", EXAMPLE_BLACKFOOT),
    ("bondurant", EXAMPLE_BONDURANT),
    ("padded", EXAMPLE_PADDED),
]
