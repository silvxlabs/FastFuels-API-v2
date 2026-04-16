"""
Example request bodies for the tree/inventory endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation — Users see these as example payloads
2. Integration tests — Each example is tested to ensure documentation stays
   accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Standard QUIC-Fire inputs (all defaults).
EXAMPLE_STANDARD_QF = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": [
        "bulk_density.foliage",
        "fuel_moisture.live",
        "savr.foliage",
        "spcd",
    ],
}

# Minimal — just foliage density.
EXAMPLE_MINIMAL = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage"],
}

# FDS high-resolution with volume fraction.
EXAMPLE_FDS_HIGH_RES = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [1.0, 1.0, 0.5],
    "bands": [
        "bulk_density.foliage",
        "savr.foliage",
        "fuel_moisture.live",
        "volume_fraction",
    ],
    "crown_profile_model": "beta",
    "biomass_model": "jenkins",
    "moisture_model": {"method": "uniform", "live": 97.0},
}

# With tree linkage for downstream analysis.
EXAMPLE_WITH_TREE_LINKAGE = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": [
        "bulk_density.foliage",
        "spcd",
        "tree_id",
        "volume_fraction",
    ],
}

CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES = {
    "standard_qf": {
        "value": EXAMPLE_STANDARD_QF,
        "summary": "Standard QUIC-Fire inputs (all defaults)",
        "description": (
            "Voxelizes a tree inventory with default models (purves crown "
            "profile, nsvb biomass) into the bands QUIC-Fire needs."
        ),
    },
    "minimal": {
        "value": EXAMPLE_MINIMAL,
        "summary": "Minimal — just foliage bulk density",
        "description": "Voxelizes a tree inventory to a single foliage bulk density band.",
    },
    "fds_high_res": {
        "value": EXAMPLE_FDS_HIGH_RES,
        "summary": "FDS high-resolution with volume fraction",
        "description": (
            "High-resolution voxelization with beta crown profile and Jenkins "
            "biomass. Includes volume_fraction for diagnosing crown overlap."
        ),
    },
    "with_tree_linkage": {
        "value": EXAMPLE_WITH_TREE_LINKAGE,
        "summary": "With tree linkage for analysis",
        "description": (
            "Includes spcd and tree_id so voxels can be traced back to the "
            "source inventory records."
        ),
    },
}

ALL_TREE_INVENTORY_EXAMPLE_VALUES = [
    ("standard_qf", EXAMPLE_STANDARD_QF),
    ("minimal", EXAMPLE_MINIMAL),
    ("fds_high_res", EXAMPLE_FDS_HIGH_RES),
    ("with_tree_linkage", EXAMPLE_WITH_TREE_LINKAGE),
]
