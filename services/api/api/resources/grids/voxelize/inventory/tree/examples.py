"""
Example request bodies for the voxelize/inventory/tree endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation — Users see these as example payloads
2. Integration tests — Each example is tested to ensure documentation stays
   accurate

domain_id comes from the URL path parameter, not the request body.
"""

# Minimum request — source inventory only; other fields resolve to defaults.
EXAMPLE_MINIMAL = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
}

# Basic request with default grid options shown explicitly.
EXAMPLE_BASIC = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live"],
}

# Explicit foliage-compatible bands list.
EXAMPLE_WITH_BANDS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": [
        "bulk_density.foliage.live",
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
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live", "fuel_moisture.live"],
    "moisture_model": {"live": {"method": "uniform", "value": 75.0}},
}

# Split one component into live/dead biomass states and assign state moisture.
EXAMPLE_WITH_LIVE_DEAD_PARTITION = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": [
        "bulk_density.foliage.live",
        "bulk_density.foliage.dead",
        "fuel_moisture.live",
        "fuel_moisture.dead",
    ],
    "biomass_source": {
        "type": "allometry",
        "equations": "nsvb",
        "components": ["foliage"],
        "component_states": {
            "foliage": {
                "live": 0.85,
                "dead": 0.15,
            },
        },
    },
    "moisture_model": {
        "live": {"method": "uniform", "value": 95.0},
        "dead": {"method": "uniform", "value": 8.0},
    },
}

# Non-default crown profile and biomass models.
EXAMPLE_WITH_ALTERNATE_MODELS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live"],
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
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live"],
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
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.branchwood.live"],
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
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.fine.live"],
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

# Per-tree max crown radius supplied by the inventory (e.g. from LiDAR).
EXAMPLE_WITH_INVENTORY_CROWN_RADIUS = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live"],
    "max_crown_radius_source": {
        "type": "inventory_column",
        "column": "max_crown_radius",
        "unit": "m",
    },
}

# Pinned seed for reproducible voxelization.
EXAMPLE_WITH_SEED = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": ["bulk_density.foliage.live"],
    "seed": 42,
}

EXAMPLE_WITH_LAD = {
    "source_inventory_id": "PLACEHOLDER_INVENTORY_ID",
    "resolution": {"horizontal": 2.0, "vertical": 1.0},
    "bands": [
        "bulk_density.foliage.live",
        "leaf_area_density",
    ],
}

CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_MINIMAL,
        "summary": "Minimum request",
        "description": (
            "Voxelizes a tree inventory with all defaults. The request only "
            "needs the source inventory ID. Produces a single "
            "`bulk_density.foliage.live` band (kg/m**3) at 2 m × 2 m × 1 m voxel"
            "resolution using the Purves crown profile and NSVB biomass "
            "models. Use this when you only need foliage mass per voxel."
        ),
    },
    "basic": {
        "value": EXAMPLE_BASIC,
        "summary": "Basic request with explicit defaults",
        "description": (
            "Equivalent to the minimum request, but shows the default "
            "`resolution` and `bands` values explicitly."
        ),
    },
    "with_bands": {
        "value": EXAMPLE_WITH_BANDS,
        "summary": "Request every foliage-compatible band",
        "description": (
            "Produces all six foliage-compatible tree-grid bands: "
            "`bulk_density.foliage.live`, "
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
    "with_live_dead_partition": {
        "value": EXAMPLE_WITH_LIVE_DEAD_PARTITION,
        "summary": "Split foliage into live and dead states",
        "description": (
            "Splits allometry-estimated foliage biomass into 85% live and "
            "15% dead density bands, then assigns separate uniform live and "
            "dead fuel moisture values. Omit `component_states` to use the "
            "default 100% live, 0% dead partition for each requested "
            "component."
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
            "area-normalized fuel loads such as kg/m**2 are not accepted."
        ),
    },
    "with_branchwood_biomass": {
        "value": EXAMPLE_WITH_BRANCHWOOD_BIOMASS,
        "summary": "Request branchwood biomass",
        "description": (
            "Requests a `bulk_density.branchwood.live` output band using NSVB "
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
            "Requests `bulk_density.fine.live` as foliage plus 10% of branchwood. "
            "The API accepts this request shape, but Treevox currently marks "
            "the asynchronous job failed with a not-implemented error until "
            "fine component compute is available."
        ),
    },
    "with_inventory_crown_radius": {
        "value": EXAMPLE_WITH_INVENTORY_CROWN_RADIUS,
        "summary": "Use per-tree max crown radius from inventory",
        "description": (
            "Reads each tree's maximum crown radius (m) from the "
            "`max_crown_radius` inventory column instead of estimating it "
            "from the crown profile model. The crown profile model still "
            "drives the crown shape — the supplied radius rescales the "
            "profile so its peak matches the per-tree value. Useful when "
            "max crown radius has been measured externally (e.g. from "
            "LiDAR) and is more reliable than the allometric estimate."
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
    "with_leaf_area_density": {
        "value": EXAMPLE_WITH_LAD,
        "summary": "Request leaf area density band",
        "description": (
            "Produces a leaf area density (LAD) (m^2/m^3) band using canopy "
            "bulk density (kg/m^3) and a species-derived specific leaf area "
            "(SLA) (m^2/kg). SLA per tree is calculated in fastfuels-core "
            "(https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/trees.py) "
            "using a database of per species SLA assembled from TRY Plant "
            "Trait Database (https://www.try-db.org/TryWeb/Home.php) data, "
            "with fallbacks for genus and Jenkin's group. Database values are "
            "per species means with outliers removed, see "
            "https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/data/REF_TRY_DB_LEAF.csv "
            "for full database. SLA is petiole excluded. "
            "LAD can be used as an input into the LeafLux "
            "light dynamics model (https://github.com/silvxlabs/leaflux-core). "
        ),
    },
}

ALL_TREE_INVENTORY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_MINIMAL),
    ("basic", EXAMPLE_BASIC),
    ("with_bands", EXAMPLE_WITH_BANDS),
    ("with_moisture_model", EXAMPLE_WITH_MOISTURE_MODEL),
    ("with_live_dead_partition", EXAMPLE_WITH_LIVE_DEAD_PARTITION),
    ("alternate_models", EXAMPLE_WITH_ALTERNATE_MODELS),
    ("with_inventory_biomass", EXAMPLE_WITH_INVENTORY_BIOMASS),
    ("with_branchwood_biomass", EXAMPLE_WITH_BRANCHWOOD_BIOMASS),
    ("with_derived_fine_biomass", EXAMPLE_WITH_DERIVED_FINE_BIOMASS),
    ("with_inventory_crown_radius", EXAMPLE_WITH_INVENTORY_CROWN_RADIUS),
    ("with_seed", EXAMPLE_WITH_SEED),
    ("with_leaf_area_density", EXAMPLE_WITH_LAD),
]
