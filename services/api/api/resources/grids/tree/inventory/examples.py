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

# Explicit foliage-compatible bands list.
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
    "biomass_source": {
        "type": "allometry",
        "equations": "jenkins",
        "components": ["foliage"],
    },
}

# Foliage biomass supplied directly by the inventory.
EXAMPLE_WITH_INVENTORY_BIOMASS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.foliage"],
    "biomass_source": {
        "type": "inventory_columns",
        "columns": {
            "foliage": {
                "column": "foliage_biomass",
                "unit": "kg",
            },
        },
        "components": ["foliage"],
    },
}

# Branchwood biomass request. Accepted by the API; Treevox returns a
# not-implemented processing error until branchwood compute lands.
EXAMPLE_WITH_BRANCHWOOD_BIOMASS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.branchwood"],
    "biomass_source": {
        "type": "allometry",
        "equations": "nsvb",
        "components": ["branchwood"],
    },
}

# Fine biomass derived from foliage plus a branchwood fraction. Accepted by
# the API; Treevox returns a not-implemented processing error until fine
# component compute lands.
EXAMPLE_WITH_DERIVED_FINE_BIOMASS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": [2.0, 2.0, 1.0],
    "bands": ["bulk_density.fine"],
    "biomass_source": {
        "type": "allometry",
        "equations": "nsvb",
        "components": ["fine"],
        "fine": {
            "recipe": "foliage_plus_branchwood_fraction",
            "branchwood_fraction": 0.1,
        },
    },
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
        "summary": "Request every foliage-compatible band",
        "description": (
            "Produces all six foliage-compatible tree-grid bands: "
            "`bulk_density.foliage`, "
            "`fuel_moisture.live`, `savr.foliage`, `spcd`, `tree_id`, and "
            "`volume_fraction`. `fuel_moisture.live` defaults to a uniform "
            "100% because no `moisture_model` is provided. `spcd` and "
            "`tree_id` record which species and inventory record occupy "
            "each voxel (tallest tree wins when crowns overlap). "
            "`volume_fraction` sums per-tree crown occupancy and exceeds "
            "1.0 where crowns overlap — useful for diagnosing dense canopy. "
            "Branchwood and fine bands are accepted by the API but currently "
            "fail asynchronously in Treevox with a not-implemented error."
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
            "from NSVB allometry to Jenkins allometry (Jenkins et al. 2003). "
            "Use these alternates when your species composition is not "
            "well-represented by Purves/NSVB or when you need continuity "
            "with prior FastFuels outputs. Use a `biomass_source.type` of "
            '`"inventory_columns"` to read per-tree kg biomass values '
            "directly from inventory columns instead of modeling them."
        ),
    },
    "with_inventory_biomass": {
        "value": EXAMPLE_WITH_INVENTORY_BIOMASS,
        "summary": "Use inventory foliage biomass",
        "description": (
            "Reads per-tree foliage biomass from the `foliage_biomass` "
            "inventory column instead of estimating foliage biomass with "
            "allometric equations. The column values must be per-tree kg; "
            "area-normalized fuel loads such as kg/m² are not accepted."
        ),
    },
    "with_branchwood_biomass": {
        "value": EXAMPLE_WITH_BRANCHWOOD_BIOMASS,
        "summary": "Request branchwood biomass",
        "description": (
            "Requests a `bulk_density.branchwood` output band using NSVB "
            "allometry. The API accepts this request shape, but Treevox "
            "currently marks the asynchronous job failed with a "
            "not-implemented error until branchwood component compute is "
            "available."
        ),
    },
    "with_derived_fine_biomass": {
        "value": EXAMPLE_WITH_DERIVED_FINE_BIOMASS,
        "summary": "Request derived fine biomass",
        "description": (
            "Requests `bulk_density.fine` as foliage plus 10% of branchwood. "
            "The API accepts this request shape, but Treevox currently marks "
            "the asynchronous job failed with a not-implemented error until "
            "fine component compute is available."
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
    ("with_inventory_biomass", EXAMPLE_WITH_INVENTORY_BIOMASS),
    ("with_branchwood_biomass", EXAMPLE_WITH_BRANCHWOOD_BIOMASS),
    ("with_derived_fine_biomass", EXAMPLE_WITH_DERIVED_FINE_BIOMASS),
    ("with_seed", EXAMPLE_WITH_SEED),
]
