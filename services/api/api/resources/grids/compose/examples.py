"""Example request bodies for the compose endpoint."""

EXAMPLE_COMPOSE_BASIC_COMPUTE = {
    "name": "Combined fuel loads",
    "inputs": [
        {"grid_id": "grid_fbfm40", "alias": "a"},
        {"grid_id": "grid_fccs", "alias": "b"},
    ],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "add",
            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
        }
    ],
}

EXAMPLE_COMPOSE_SELECT_AND_COMPUTE = {
    "name": "Composed surface fuels",
    "inputs": [
        {"grid_id": "grid_fbfm40", "alias": "a"},
        {"grid_id": "grid_fccs", "alias": "b"},
    ],
    "bands": [
        {"key": "fbfm", "type": "categorical", "unit": None},
        {"key": "fuel_depth", "type": "continuous", "unit": "m"},
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
        {"key": "fuel_load.10hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {"output": "fbfm", "from": "a.fbfm"},
        {"output": "fuel_depth", "from": "a.fuel_depth"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "add",
            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
        },
        {
            "output": "fuel_load.10hr",
            "operator": "max",
            "operands": ["a.fuel_load.10hr", "b.fuel_load.10hr"],
        },
    ],
}

EXAMPLE_COMPOSE_SINGLE_GRID_MATH = {
    "name": "FCCS with combined 1hr fuels",
    "inputs": [{"grid_id": "grid_fccs", "alias": "a"}],
    "bands": [
        {"key": "fuel_depth", "type": "continuous", "unit": "m"},
        {"key": "moisture_of_extinction", "type": "continuous", "unit": "%"},
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {"output": "fuel_depth", "from": "a.fuel_depth"},
        {"output": "moisture_of_extinction", "from": "a.moisture_of_extinction"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "add",
            "operands": ["a.dead_grass", "a.litter"],
        }
    ],
}

EXAMPLE_COMPOSE_WITH_LITERAL = {
    "name": "Adjusted fuel loads",
    "inputs": [{"grid_id": "grid_surface", "alias": "a"}],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "multiply",
            "operands": ["a.fuel_load.1hr", 0.5],
        }
    ],
}

EXAMPLE_COMPOSE_CONDITIONAL_COMPUTE = {
    "name": "Averaged fuel loads where both valid",
    "inputs": [
        {"grid_id": "grid_landfire", "alias": "a"},
        {"grid_id": "grid_rap", "alias": "b"},
    ],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "average",
            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
            "conditions": [
                {"band": "a.fuel_load.1hr", "operator": "gt", "value": 0},
                {"band": "b.fuel_load.1hr", "operator": "gt", "value": 0},
            ],
            "else": "a.fuel_load.1hr",
        }
    ],
}

EXAMPLE_COMPOSE_CONDITIONAL_FALLBACK = {
    "name": "RAP with LANDFIRE fallback",
    "inputs": [
        {"grid_id": "grid_landfire", "alias": "a"},
        {"grid_id": "grid_rap", "alias": "b"},
    ],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {
            "output": "fuel_load.1hr",
            "from": "a.fuel_load.1hr",
            "conditions": [{"band": "b.fuel_load.1hr", "operator": "eq", "value": 0}],
            "else": "b.fuel_load.1hr",
        }
    ],
}

EXAMPLE_COMPOSE_SET_MEMBERSHIP = {
    "name": "RAP for grass fuel models",
    "inputs": [
        {"grid_id": "grid_landfire", "alias": "a"},
        {"grid_id": "grid_rap", "alias": "b"},
    ],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {
            "output": "fuel_load.1hr",
            "from": "b.fuel_load.1hr",
            "conditions": [
                {
                    "band": "a.fbfm",
                    "operator": "in",
                    "value": [
                        101,
                        102,
                        103,
                        104,
                        105,
                        106,
                        107,
                        108,
                        109,
                    ],
                }
            ],
            "else": "a.fuel_load.1hr",
        }
    ],
}

EXAMPLE_COMPOSE_INLINE_COMPUTE_FALLBACK = {
    "name": "LANDFIRE infill with averaging",
    "inputs": [
        {"grid_id": "grid_landfire", "alias": "a"},
        {"grid_id": "grid_rap", "alias": "b"},
    ],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {
            "output": "fuel_load.1hr",
            "from": "a.fuel_load.1hr",
            "conditions": [{"band": "b.fuel_load.1hr", "operator": "eq", "value": 0}],
            "else": {
                "operator": "average",
                "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
            },
        }
    ],
}

EXAMPLE_COMPOSE_MIXED_CONDITIONS = {
    "name": "Conditional average inside treatment area",
    "inputs": [
        {"grid_id": "grid_landfire", "alias": "a"},
        {"grid_id": "grid_rap", "alias": "b"},
    ],
    "bands": [
        {"key": "fbfm", "type": "categorical", "unit": None},
        {"key": "fuel_depth", "type": "continuous", "unit": "m"},
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {"output": "fbfm", "from": "a.fbfm"},
        {"output": "fuel_depth", "from": "a.fuel_depth"},
    ],
    "compute": [
        {
            "output": "fuel_load.1hr",
            "operator": "average",
            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
            "conditions": [
                {"band": "a.fuel_load.1hr", "operator": "gt", "value": 0},
                {"band": "b.fuel_load.1hr", "operator": "gt", "value": 0},
                {
                    "source": "geometry",
                    "operator": "within",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [500000.0, 5200000.0],
                                [500500.0, 5200000.0],
                                [500500.0, 5200500.0],
                                [500000.0, 5200500.0],
                                [500000.0, 5200000.0],
                            ]
                        ],
                    },
                },
            ],
            "else": "a.fuel_load.1hr",
        }
    ],
}

EXAMPLE_COMPOSE_TYPED_LITERAL_FALLBACK = {
    "name": "Zero non-burnable fuels",
    "inputs": [{"grid_id": "grid_surface", "alias": "a"}],
    "bands": [
        {"key": "fuel_load.1hr", "type": "continuous", "unit": "kg/m**2"},
    ],
    "select": [
        {
            "output": "fuel_load.1hr",
            "from": "a.fuel_load.1hr",
            "conditions": [
                {"band": "a.fbfm", "operator": "ne", "value": [91, 92, 93, 98, 99]}
            ],
            "else": {"type": "literal", "value": 0, "unit": "kg/m**2"},
        }
    ],
}

CREATE_COMPOSE_OPENAPI_EXAMPLES = {
    "basic_compute": {
        "value": EXAMPLE_COMPOSE_BASIC_COMPUTE,
        "summary": "Compute one output from two grids",
        "description": (
            "Adds the same continuous band from two aligned source grids and "
            "writes the result as a single output band. This is the smallest "
            "multi-grid compose request: `inputs` gives each source grid an "
            "alias, `bands` declares the output metadata, and `compute` names "
            "the output plus the arithmetic operation. Use this pattern when "
            "two products carry compatible units and should contribute to the "
            "same fire-model quantity."
        ),
    },
    "select_and_compute": {
        "value": EXAMPLE_COMPOSE_SELECT_AND_COMPUTE,
        "summary": "Select bands and compute multiple outputs",
        "description": (
            "Builds a richer output grid by copying some bands directly from "
            "one source and computing others from two sources. `select` "
            "preserves categorical or already-valid bands such as `fbfm` and "
            "`fuel_depth`; `compute` derives new continuous outputs. The "
            "`bands` list is intentionally explicit and ordered: it defines "
            "the output band metadata and final band order, and its keys must "
            "exactly match the `select` and `compute` outputs."
        ),
    },
    "single_grid_math": {
        "value": EXAMPLE_COMPOSE_SINGLE_GRID_MATH,
        "summary": "Combine bands within one grid",
        "description": (
            "Uses compose with a single input grid. This is useful when one "
            "source already contains the component bands needed to create a "
            "new aggregate band. Here `fuel_depth` and "
            "`moisture_of_extinction` are copied through unchanged while "
            "`dead_grass + litter` becomes the composed `fuel_load.1hr` band. "
            "A single-grid compose still creates a new async Grid with normal "
            "lineage and output band metadata."
        ),
    },
    "literal_operand": {
        "value": EXAMPLE_COMPOSE_WITH_LITERAL,
        "summary": "Compute using a scalar literal",
        "description": (
            "Scales a raster band by a bare numeric literal. Bare numbers are "
            "allowed in `compute.operands`; for `multiply` and `divide` they "
            "are treated as dimensionless, so multiplying `kg/m**2` by `0.5` "
            "keeps the output unit `kg/m**2`. Use a tagged literal instead "
            "when the literal itself has a unit."
        ),
    },
    "conditional_compute": {
        "value": EXAMPLE_COMPOSE_CONDITIONAL_COMPUTE,
        "summary": "Compute only where both source bands are valid",
        "description": (
            "Computes an average only where both source bands pass the "
            "conditions. Conditions are ANDed together, and attribute "
            "conditions evaluate false on nodata. Because `conditions` is "
            "present, `else` is required; here the fallback is the LANDFIRE "
            "band. This pattern is useful for combining a preferred source "
            "with a conservative fallback when one input has zeros or gaps."
        ),
    },
    "conditional_fallback": {
        "value": EXAMPLE_COMPOSE_CONDITIONAL_FALLBACK,
        "summary": "Conditional fallback to another input band",
        "description": (
            "Selects from one source when a condition on another source is "
            "true, otherwise falls back to that other source. In this example, "
            "LANDFIRE fills cells where RAP is zero; everywhere else the RAP "
            "band is used. This is the direct conditional-infill pattern for "
            "choosing between two aligned rasters without doing arithmetic."
        ),
    },
    "set_membership": {
        "value": EXAMPLE_COMPOSE_SET_MEMBERSHIP,
        "summary": "Use set membership to choose the source grid",
        "description": (
            "Uses a categorical source band to decide which raster should "
            "supply the output value. The `in` operator checks whether "
            "`a.fbfm` is one of the numeric Scott-Burgan grass fuel model "
            "codes (GR1-GR9 are stored as 101-109); matching cells come from "
            "RAP and all other cells come from LANDFIRE. For categorical "
            "conditions, compose supports `eq`, `ne`, and `in`."
        ),
    },
    "inline_compute_fallback": {
        "value": EXAMPLE_COMPOSE_INLINE_COMPUTE_FALLBACK,
        "summary": "Fallback to an inline computation",
        "description": (
            "Uses a normal `select` for the primary branch and an inline "
            "`compute` object for the fallback branch. Cells where RAP is zero "
            "use LANDFIRE directly; all other cells use the average of "
            "LANDFIRE and RAP. Inline computes are allowed only as fallback "
            "values and do not have their own nested conditions."
        ),
    },
    "mixed_conditions": {
        "value": EXAMPLE_COMPOSE_MIXED_CONDITIONS,
        "summary": "Combine attribute and spatial conditions",
        "description": (
            "Combines copied context bands with a computed band whose "
            "conditions mix raster attributes and spatial geometry. All "
            "conditions are ANDed: both source fuel-load bands must be "
            "positive and the cell must be inside the treatment polygon. "
            "Cells outside that intersection fall back to LANDFIRE. Use this "
            "pattern for treatment-area or analysis-area overrides that "
            "should apply only where source data also passes quality checks."
        ),
    },
    "typed_literal_fallback": {
        "value": EXAMPLE_COMPOSE_TYPED_LITERAL_FALLBACK,
        "summary": "Conditional fallback to a typed literal",
        "description": (
            "Uses a tagged literal as the fallback value. Tagged literals make "
            "the value type explicit and can carry a unit, which lets the API "
            "validate that the fallback is compatible with the output band "
            "before enqueueing the job. This example writes zero fuel load "
            "with unit `kg/m**2` for non-burnable FBFM cells. LANDFIRE FBFM "
            "bands store numeric Scott-Burgan codes, so NB1, NB2, NB3, NB8, "
            "and NB9 are referenced as 91, 92, 93, 98, and 99."
        ),
    },
}

ALL_COMPOSE_EXAMPLE_VALUES = [
    ("basic_compute", EXAMPLE_COMPOSE_BASIC_COMPUTE),
    ("select_and_compute", EXAMPLE_COMPOSE_SELECT_AND_COMPUTE),
    ("single_grid_math", EXAMPLE_COMPOSE_SINGLE_GRID_MATH),
    ("literal_operand", EXAMPLE_COMPOSE_WITH_LITERAL),
    ("conditional_compute", EXAMPLE_COMPOSE_CONDITIONAL_COMPUTE),
    ("conditional_fallback", EXAMPLE_COMPOSE_CONDITIONAL_FALLBACK),
    ("set_membership", EXAMPLE_COMPOSE_SET_MEMBERSHIP),
    ("inline_compute_fallback", EXAMPLE_COMPOSE_INLINE_COMPUTE_FALLBACK),
    ("mixed_conditions", EXAMPLE_COMPOSE_MIXED_CONDITIONS),
    ("typed_literal_fallback", EXAMPLE_COMPOSE_TYPED_LITERAL_FALLBACK),
]
