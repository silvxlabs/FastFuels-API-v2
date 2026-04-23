"""
Example request bodies for the tree/inventory endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation — Users see these as example payloads
2. Integration tests — Each example is tested to ensure documentation stays
   accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimum request — only required fields.
EXAMPLE_MINIMAL = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage"],
}

# Explicit bands list.
EXAMPLE_WITH_BANDS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": [
        "bulk_density.foliage",
        "fuel_moisture.live",
        "savr.foliage",
        "spcd",
        "tree_id",
        "volume_fraction",
    ],
}

# Moisture model configured explicitly.
EXAMPLE_WITH_MOISTURE_MODEL = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage", "fuel_moisture.live"],
    "moisture_model": {"method": "uniform", "live": 75.0},
}

# Non-default crown profile and biomass models.
EXAMPLE_WITH_ALTERNATE_MODELS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage"],
    "crown_profile_model": "beta",
    "biomass_model": "jenkins",
}

# Pinned seed for reproducible voxelization.
EXAMPLE_WITH_SEED = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage"],
    "seed": 42,
}

CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_MINIMAL,
        "summary": "Minimum request",
        "description": (
            "Voxelizes a tree inventory with all defaults. Produces a single "
            "`bulk_density.foliage` band (kg/m³) at 2 m × 2 m × 1 m voxel "
            "resolution using the Purves crown profile and NSVB biomass "
            "models. Use this when you only need foliage mass per voxel."
        ),
    },
    "with_bands": {
        "value": EXAMPLE_WITH_BANDS,
        "summary": "Request every available band",
        "description": (
            "Produces all six tree-grid bands: `bulk_density.foliage`, "
            "`fuel_moisture.live`, `savr.foliage`, `spcd`, `tree_id`, and "
            "`volume_fraction`. `fuel_moisture.live` defaults to a uniform "
            "100% because no `moisture_model` is provided. `spcd` and "
            "`tree_id` record which species and inventory record occupy "
            "each voxel (tallest tree wins when crowns overlap). "
            "`volume_fraction` sums per-tree crown occupancy and exceeds "
            "1.0 where crowns overlap — useful for diagnosing dense canopy."
        ),
    },
    "with_moisture_model": {
        "value": EXAMPLE_WITH_MOISTURE_MODEL,
        "summary": "Configure live fuel moisture",
        "description": (
            "Sets live fuel moisture to a uniform 75% across the grid. "
            "Supply `moisture_model` whenever you request "
            "`fuel_moisture.live` and the default of 100% is not "
            "appropriate for your scenario (e.g., late-season or "
            "fire-weather conditions). Only the `uniform` method is "
            "available today; additional methods will appear as new "
            "`method` values."
        ),
    },
    "alternate_models": {
        "value": EXAMPLE_WITH_ALTERNATE_MODELS,
        "summary": "Override crown-profile and biomass models",
        "description": (
            "Switches from the default Purves crown profile to the Beta "
            "profile (Ferrarese et al. 2015, 10 Jenkins species groups) and "
            "from NSVB biomass to Jenkins biomass (Jenkins et al. 2003). "
            "Use these alternates when your species composition is not "
            "well-represented by Purves/NSVB or when you need continuity "
            "with prior FastFuels outputs. Set `biomass_model` to "
            '`"inventory"` and supply `biomass_column` to read biomass '
            "directly from an inventory column instead of modeling it."
        ),
    },
    "with_seed": {
        "value": EXAMPLE_WITH_SEED,
        "summary": "Pin the random seed for reproducibility",
        "description": (
            "Supplies an explicit `seed` so repeated voxelizations of this "
            "inventory at this resolution produce bit-identical output. "
            "The seed is persisted on the grid document; omit it to have "
            "the API generate and persist one for you (re-runs are still "
            "deterministic against the stored seed)."
        ),
    },
}

ALL_TREE_INVENTORY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_MINIMAL),
    ("with_bands", EXAMPLE_WITH_BANDS),
    ("with_moisture_model", EXAMPLE_WITH_MOISTURE_MODEL),
    ("alternate_models", EXAMPLE_WITH_ALTERNATE_MODELS),
    ("with_seed", EXAMPLE_WITH_SEED),
]
