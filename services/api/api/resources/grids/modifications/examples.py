"""
Example request bodies for the in-place grid modifications endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation
2. Integration tests

They mirror the worked examples on ``GridModification.model_config``
(json_schema_extra), wrapped in the endpoint's request envelope.
"""

# Wipe all surface fuel from roads and water bodies. Conditions within a rule
# are ANDed, so a union of two features needs two rules: one for the road
# (linestring + target=cell catches every cell the road crosses) and one for
# the water body (polygon — the default centroid test covers an area feature).
EXAMPLE_ZERO_FUEL_ON_ROADS_AND_WATER = {
    "modifications": [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_road_abc",
                    "target": "cell",
                }
            ],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "replace", "value": 0},
                {"band": "fuel_load.10hr", "modifier": "replace", "value": 0},
                {"band": "fuel_load.100hr", "modifier": "replace", "value": 0},
            ],
        },
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_water_xyz",
                }
            ],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "replace", "value": 0},
                {"band": "fuel_load.10hr", "modifier": "replace", "value": 0},
                {"band": "fuel_load.100hr", "modifier": "replace", "value": 0},
            ],
        },
    ],
}

# Remove 90% of surface fuel along a 4 m road buffer (multiply 0.1)
EXAMPLE_REDUCE_FUEL_ALONG_ROAD_BUFFER = {
    "modifications": [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 4,
                }
            ],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.1},
                {"band": "fuel_load.10hr", "modifier": "multiply", "value": 0.1},
                {"band": "fuel_load.100hr", "modifier": "multiply", "value": 0.1},
            ],
        }
    ],
}

# Inline geometry variant with buffer
EXAMPLE_ZERO_FUEL_IN_POLYGON = {
    "modifications": [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-120.0, 38.0],
                                [-119.5, 38.0],
                                [-119.5, 38.5],
                                [-120.0, 38.5],
                                [-120.0, 38.0],
                            ]
                        ],
                    },
                    "buffer_m": 5,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0}],
        }
    ],
}

# Reclassify a fuel model: every GR1 (101) cell becomes GR2 (102). The fbfm
# band stores the numeric Scott-Burgan codes, so the rule compares and
# replaces code values.
EXAMPLE_REPLACE_GR1_WITH_GR2 = {
    "modifications": [
        {
            "conditions": [{"band": "fbfm", "operator": "eq", "value": 101}],
            "actions": [{"band": "fbfm", "modifier": "replace", "value": 102}],
        }
    ],
}

# Reclassify a fuel model only inside a polygon: attribute and spatial
# conditions in one rule are ANDed, so only GR1 cells whose centroid falls
# inside the polygon become GR2.
EXAMPLE_REPLACE_GR1_WITH_GR2_IN_POLYGON = {
    "modifications": [
        {
            "conditions": [
                {"band": "fbfm", "operator": "eq", "value": 101},
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-120.0, 38.0],
                                [-119.5, 38.0],
                                [-119.5, 38.5],
                                [-120.0, 38.5],
                                [-120.0, 38.0],
                            ]
                        ],
                    },
                },
            ],
            "actions": [{"band": "fbfm", "modifier": "replace", "value": 102}],
        }
    ],
}

# Attribute condition: halve the 1-hour load wherever it is non-zero
EXAMPLE_HALVE_NONZERO_FUEL = {
    "modifications": [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "gt", "value": 0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "multiply", "value": 0.5}
            ],
        }
    ],
}

APPLY_GRID_MODIFICATIONS_OPENAPI_EXAMPLES = {
    "replace_gr1_with_gr2": {
        "summary": "Replace all GR1 fuel models with GR2",
        "description": (
            "Reclassify every GR1 cell to GR2 on an FBFM40 grid. The `fbfm` "
            "band stores numeric Scott-Burgan codes (GR1 = 101, GR2 = 102), "
            "so the condition matches code 101 and the action replaces it "
            "with 102."
        ),
        "value": EXAMPLE_REPLACE_GR1_WITH_GR2,
    },
    "replace_gr1_with_gr2_in_polygon": {
        "summary": "Replace GR1 with GR2 only inside a polygon",
        "description": (
            "Conditions in one rule are ANDed: the attribute condition "
            "matches GR1 cells (code 101) and the spatial condition limits "
            "the rule to cells whose centroid falls inside the supplied "
            "polygon, so only GR1 cells inside the polygon become GR2 (102)."
        ),
        "value": EXAMPLE_REPLACE_GR1_WITH_GR2_IN_POLYGON,
    },
    "zero_fuel_on_roads_and_water": {
        "summary": "Zero surface fuel on roads and water bodies",
        "description": (
            "Wipe the 1/10/100-hour fuel loads inside both a road Feature and "
            "a water Feature. Because conditions in a single rule are ANDed, "
            "the union is expressed as two rules — one per Feature. The road "
            "rule uses `target=cell` so every cell its linestring crosses is "
            "caught; the water rule uses the default centroid test for its "
            "polygon."
        ),
        "value": EXAMPLE_ZERO_FUEL_ON_ROADS_AND_WATER,
    },
    "reduce_fuel_along_road_buffer": {
        "summary": "Reduce fuel 90% along a buffered road",
        "description": (
            "Buffer the road Feature by 4 m in the domain CRS, then multiply "
            "the surface fuel loads by 0.1 inside the buffer."
        ),
        "value": EXAMPLE_REDUCE_FUEL_ALONG_ROAD_BUFFER,
    },
    "zero_fuel_in_polygon": {
        "summary": "Zero fuel inside an inline polygon",
        "description": (
            "Supply GeoJSON directly instead of referencing a Feature; the "
            "polygon is buffered 5 m before testing cell centroids."
        ),
        "value": EXAMPLE_ZERO_FUEL_IN_POLYGON,
    },
    "halve_nonzero_fuel": {
        "summary": "Halve all non-zero 1-hour fuel load",
        "description": (
            "Attribute condition with no spatial component: every cell whose "
            "1-hour load is positive is multiplied by 0.5."
        ),
        "value": EXAMPLE_HALVE_NONZERO_FUEL,
    },
}
