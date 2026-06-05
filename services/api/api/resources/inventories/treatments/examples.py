"""
Example request bodies for the standalone inventory treatments endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation
2. Integration tests
"""

# Thin from below to a diameter limit (remove trees smaller than 10 cm dbh)
EXAMPLE_DIAMETER_FROM_BELOW = {
    "treatments": [
        {
            "metric": "diameter",
            "method": "from_below",
            "value": 10.0,
        }
    ],
}

# Thin from below to a residual basal area target
EXAMPLE_BASAL_AREA_FROM_BELOW = {
    "treatments": [
        {
            "metric": "basal_area",
            "method": "from_below",
            "value": 25.0,
        }
    ],
}

# Proportional basal-area thin (remove across all diameter classes)
EXAMPLE_BASAL_AREA_PROPORTIONAL = {
    "treatments": [
        {
            "metric": "basal_area",
            "method": "proportional",
            "value": 18.0,
        }
    ],
}

# Spatially scoped treatment: thin only within a buffered road Feature
EXAMPLE_TREATMENT_IN_ROAD_BUFFER = {
    "treatments": [
        {
            "metric": "basal_area",
            "method": "from_below",
            "value": 20.0,
            "conditions": [
                {
                    "source": "feature",
                    "operator": "within",
                    "feature_id": "feat_road_abc",
                    "buffer_m": 30,
                }
            ],
        }
    ],
}

# Diameter limit supplied in inches (unit conversion to native cm)
EXAMPLE_DIAMETER_INCHES = {
    "treatments": [
        {
            "metric": "diameter",
            "method": "from_above",
            "value": 16.0,
            "unit": "in",
        }
    ],
}

APPLY_TREATMENTS_OPENAPI_EXAMPLES = {
    "diameter_from_below": {
        "value": EXAMPLE_DIAMETER_FROM_BELOW,
        "summary": "Thin from below to a diameter limit",
        "description": (
            "Remove trees smaller than 10 cm dbh (a low thinning that clears "
            "suppressed understory stems)."
        ),
    },
    "basal_area_from_below": {
        "value": EXAMPLE_BASAL_AREA_FROM_BELOW,
        "summary": "Thin from below to a residual basal area",
        "description": (
            "Remove the smallest trees first until the stand reaches a residual "
            "basal area of 25 m**2/ha."
        ),
    },
    "basal_area_proportional": {
        "value": EXAMPLE_BASAL_AREA_PROPORTIONAL,
        "summary": "Proportional basal-area thin",
        "description": (
            "Reduce the stand to 18 m**2/ha by removing trees across all "
            "diameter classes proportionally, preserving the diameter "
            "distribution."
        ),
    },
    "treatment_in_road_buffer": {
        "value": EXAMPLE_TREATMENT_IN_ROAD_BUFFER,
        "summary": "Thin only within a buffered road Feature",
        "description": (
            "Scope the treatment to a region with spatial conditions. Here the "
            "thin applies only to trees within 30 m of the referenced road "
            "Feature, which must belong to the same domain as the inventory."
        ),
    },
    "diameter_inches": {
        "value": EXAMPLE_DIAMETER_INCHES,
        "summary": "Diameter limit in inches (unit conversion)",
        "description": (
            "Remove trees larger than 16 inches dbh. The `unit` field converts "
            "the value to the native cm before the treatment is applied."
        ),
    },
}

ALL_TREATMENTS_EXAMPLE_VALUES = [
    ("diameter_from_below", EXAMPLE_DIAMETER_FROM_BELOW),
    ("basal_area_from_below", EXAMPLE_BASAL_AREA_FROM_BELOW),
    ("basal_area_proportional", EXAMPLE_BASAL_AREA_PROPORTIONAL),
    ("treatment_in_road_buffer", EXAMPLE_TREATMENT_IN_ROAD_BUFFER),
    ("diameter_inches", EXAMPLE_DIAMETER_INCHES),
]
