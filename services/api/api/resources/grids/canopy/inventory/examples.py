"""
Example request bodies for the inventory canopy endpoint.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

domain_id comes from the URL path parameter, not the request body.
Replace the inventory id with a completed tree inventory in your domain.
"""

EXAMPLE_INVENTORY_CANOPY_MINIMAL = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
}

EXAMPLE_INVENTORY_CANOPY_EXPLICIT_DEFAULTS = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "alignment": {"target": "domain", "resolution": 30.0},
    "bands": ["cbd", "cbh", "chm", "cc"],
    "biomass_source": {"type": "allometry", "equations": "nsvb"},
    "available_fuel": {
        "foliage_fraction": 1.0,
        "branchwood": {"size_partition": "none", "fraction": 0.075},
    },
    "species_inclusion": "all_species",
    "crown_class_adjustment": {"method": "none"},
    "min_tree_height": 0.0,
    "vertical_distribution": "reinhardt_2006",
    "layer_depth": 0.3048,
    "horizontal_distribution": "crown_projected",
    "max_crown_radius_source": {"type": "allometry"},
    "cbd": {"method": "maximum_running_mean", "window": 3.0},
    "cbh": {
        "method": "bulk_density_threshold",
        "threshold": 0.012,
        "relative_threshold_fraction": 0.1,
        "smoothing_window": None,
    },
    "chm": {
        "method": "bulk_density_threshold",
        "threshold": 0.012,
        "relative_threshold_fraction": 0.1,
        "smoothing_window": None,
    },
    "cc": {"method": "crown_union"},
}

EXAMPLE_INVENTORY_CANOPY_FUELCALC_COMPARISON = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "name": "FuelCalc-comparable canopy fuels",
    "tags": ["fuelcalc", "landfire-comparison"],
    "biomass_source": {"type": "allometry", "equations": "brown_1978"},
    "species_inclusion": "fuelcalc_default",
    "crown_class_adjustment": {
        "method": "fuelcalc_table",
        "missing_crown_class": "other_none",
    },
    "horizontal_distribution": "stem",
    "max_crown_radius_source": {"type": "allometry", "equations": "crookston_stage"},
    "cbd": {
        "method": "maximum_running_mean",
        "window": 1.524,
        "edge": "ground_clamped",
    },
    "cbh": {
        "method": "bulk_density_threshold",
        "smoothing_window": 1.524,
        "smoothing_edge": "ground_clamped",
    },
    "chm": {
        "method": "bulk_density_threshold",
        "smoothing_window": 1.524,
        "smoothing_edge": "ground_clamped",
    },
    "cc": {"method": "crown_overlap"},
}

EXAMPLE_INVENTORY_CANOPY_FFE_COMPARISON = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "name": "FFE-FVS-comparable canopy fuels",
    "tags": ["ffe-fvs"],
    "biomass_source": {"type": "allometry", "equations": "brown_1978"},
    "vertical_distribution": "uniform",
    "min_tree_height": 1.83,
    "cbd": {"method": "maximum_running_mean", "window": 3.9624},
    "cbh": {
        "method": "bulk_density_threshold",
        "threshold": 0.011,
        "relative_threshold_fraction": None,
        "smoothing_window": 0.9144,
    },
    "chm": {
        "method": "bulk_density_threshold",
        "threshold": 0.011,
        "relative_threshold_fraction": None,
        "smoothing_window": 0.9144,
    },
}

EXAMPLE_INVENTORY_CANOPY_WITH_CFL = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "bands": ["cbd", "cbh", "chm", "cc", "cfl"],
}

EXAMPLE_INVENTORY_CANOPY_10M = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "alignment": {"target": "domain", "resolution": 10.0},
    "name": "High-resolution canopy fuels",
}

EXAMPLE_INVENTORY_CANOPY_ALIGNED_TO_LANDFIRE = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "alignment": {"target": "grid", "grid_id": "REPLACE_WITH_GRID_ID"},
}

EXAMPLE_INVENTORY_CANOPY_FLAT_THRESHOLD = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "bands": ["cbd", "cbh", "chm"],
    "cbh": {
        "method": "bulk_density_threshold",
        "threshold": 0.037,
        "relative_threshold_fraction": None,
    },
    "chm": {
        "method": "bulk_density_threshold",
        "threshold": 0.037,
        "relative_threshold_fraction": None,
    },
}

EXAMPLE_INVENTORY_CANOPY_VAN_WAGNER_CBD = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "bands": ["cbd", "cbh", "chm"],
    "cbd": {"method": "load_over_depth", "depth": "canopy_depth"},
}

EXAMPLE_INVENTORY_CANOPY_TOTAL_BRANCHWOOD = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "available_fuel": {
        "foliage_fraction": 1.0,
        "branchwood": {"size_partition": "none", "fraction": 0.2},
    },
}

EXAMPLE_INVENTORY_CANOPY_COLUMN_FUEL = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "biomass_source": {
        "type": "inventory_column",
        "column": "available_canopy_fuel_kg",
        "unit": "kg",
    },
}

EXAMPLE_INVENTORY_CANOPY_LIDAR_CROWNS = {
    "source_inventory_id": "9c1f2ab4708d4290a8ab6ecf35f21ab4",
    "max_crown_radius_source": {
        "type": "inventory_column",
        "column": "crown_radius_m",
        "unit": "m",
    },
}

CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES: dict = {
    "minimal": {
        "value": EXAMPLE_INVENTORY_CANOPY_MINIMAL,
        "summary": "Canopy fuels from your inventory (defaults)",
        "description": (
            "The only required field is the tree inventory to summarize. "
            "Produces the four canopy bands a landscape file needs — `cbd`, "
            "`cbh`, `chm`, `cc` — at 30 m on the domain lattice, computed "
            "per cell from the inventory's actual stem positions and crowns."
            "\n\n"
            "Defaults follow a FuelCalc-style profile method with NSVB "
            "allometry: "
            "available fuel is all foliage plus a national fraction (0.075) "
            "of NSVB branchwood — the `none` partition, which prices every "
            "species where the FuelCalc fine-share crosswalk cannot — "
            "distributed over each crown by the Reinhardt et al. (2006) "
            "species curves, summed into 1 ft layers per cell; CBD is the "
            "maximum 3 m running mean of that profile, CBH and CHM are the "
            "lowest and highest threshold crossings, and cover is the "
            "geometric union of projected crowns.\n\n"
            "The inventory must be a completed tree inventory in this "
            "domain with species, dbh, height, and crown ratio columns."
        ),
    },
    "explicit_defaults": {
        "value": EXAMPLE_INVENTORY_CANOPY_EXPLICIT_DEFAULTS,
        "summary": "Every default written out",
        "description": (
            "Identical to the minimal request, with every modeling choice "
            "stated explicitly. Useful as a template: change one field at a "
            "time to see its effect. The stored grid records these resolved "
            "choices either way, so a grid is always reproducible from its "
            "own `source`."
        ),
    },
    "fuelcalc_comparison": {
        "value": EXAMPLE_INVENTORY_CANOPY_FUELCALC_COMPARISON,
        "summary": "FuelCalc / LANDFIRE comparison settings",
        "description": (
            "Switches the pieces where the FastFuels defaults deliberately "
            "depart from FuelCalc 1.7: Brown (1978) crown allometry, "
            "FuelCalc's hardwood exclusion, its crown-class factors (every "
            "tree takes the Other/none factor because inventories carry no "
            "crown class), plot-style stem attribution, its 1.524 m (5 ft) "
            "CBD window, and Crookston–Stage overlap cover. No "
            "`min_tree_height` is set because FuelCalc 1.7 includes small "
            "trees; add 1.83 m to match FFE-FVS or the original FuelCalc "
            "method (RMRS-P-41) instead.\n\n"
            "The 5 ft window is the FuelCalc 1.7 User Guide's stated "
            "value; the guide's own Appendix D reads unsmoothed layers "
            "(`window: null`) and the original method used 15 ft "
            "(`window: 4.572`), so pin the window against a reference "
            "FuelCalc run when exact agreement matters.\n\n"
            "Use this for method-comparison studies against FuelCalc runs "
            "or LANDFIRE canopy layers. Expect characterized differences, "
            "not identical rasters: published LANDFIRE canopy products "
            "include mapping, imputation, and product-level adjustments "
            "beyond the plot calculation."
        ),
    },
    "ffe_fvs_comparison": {
        "value": EXAMPLE_INVENTORY_CANOPY_FFE_COMPARISON,
        "summary": "FFE-FVS comparison settings",
        "description": (
            "The FFE-FVS canopy-fuel algorithm: uniform vertical "
            "distribution over each crown, trees under 1.83 m (6 ft) "
            "treated as surface fuel, CBD as the maximum 3.9624 m (13 ft) "
            "running mean, and CBH/CHM crossing a 0.9144 m (3 ft) running "
            "mean at a flat 0.011 kg/m**3 (30 lb/acre/ft) — FFE applies no "
            "relative threshold reduction. One residual difference: FFE's "
            "crown biomass comes from Brown & Johnston (1976), which is "
            "not offered; Brown (1978) is the closest available family."
        ),
    },
    "with_cfl": {
        "value": EXAMPLE_INVENTORY_CANOPY_WITH_CFL,
        "summary": "Include canopy fuel load",
        "description": (
            "Adds the `cfl` band (kg/m**2): the total available canopy fuel "
            "per unit ground area. CFL cannot be reconstructed from "
            "`cbd * (chm - cbh)` unless the profile is vertically uniform, "
            "which it rarely is — request it here if you need it. It is not "
            "a landscape-file band; the other four remain the export set."
        ),
    },
    "high_resolution_10m": {
        "value": EXAMPLE_INVENTORY_CANOPY_10M,
        "summary": "10 m cells",
        "description": (
            "A 10 m landscape resolves within-stand structure that 30 m "
            "products smooth over, and operational tools accept it (IFTDSS "
            "takes 1–270 m cells). Note the scale trade-off: a 10 m cell is "
            "smaller than a large crown, so cell values describe crown-scale "
            "rather than stand-scale fuel. The default `crown_projected` "
            "attribution matters most here — it spreads each crown's mass "
            "over the cells it actually covers."
        ),
    },
    "aligned_to_landfire_grid": {
        "value": EXAMPLE_INVENTORY_CANOPY_ALIGNED_TO_LANDFIRE,
        "summary": "Match an existing grid's cells",
        "description": (
            "Produces canopy bands on exactly the named grid's lattice — "
            "same origin, cell size, and shape. Align to a LANDFIRE canopy "
            "grid to compare the two cell-for-cell, or to any grid you plan "
            "to export alongside."
        ),
    },
    "flat_threshold": {
        "value": EXAMPLE_INVENTORY_CANOPY_FLAT_THRESHOLD,
        "summary": "Flat CBH/CHM threshold (Sando & Wick)",
        "description": (
            "Setting `relative_threshold_fraction` to null applies the "
            "bulk-density threshold flat, with no relative reduction in "
            "sparse cells — here Sando & Wick's 0.037 kg/m**3. In open or "
            "treated stands a flat threshold reports higher CBH (or no "
            "canopy at all) compared to the FuelCalc rule, which drops the "
            "threshold to 10% of the cell's maximum profile density."
        ),
    },
    "van_wagner_cbd": {
        "value": EXAMPLE_INVENTORY_CANOPY_VAN_WAGNER_CBD,
        "summary": "Load-over-depth CBD",
        "description": (
            "Computes CBD as canopy fuel load divided by canopy depth "
            "(chm - cbh) — the van Wagner-consistent average density — "
            "instead of the running-mean maximum. The two conventions "
            "differ systematically: the running-mean maximum reads higher, "
            "predicting active crowning at lower windspeeds. The stored "
            "source records which convention produced the grid. `cbh` and "
            "`chm` must be requested with the threshold method, since they "
            "define the depth.\n\n"
            "For the Cruz et al. (2003) variant, use `depth: "
            '"mean_crown_length"` with foliage-only availability '
            '(`available_fuel: {"branchwood": {"size_partition": "none", '
            '"fraction": 0.0}}`).'
        ),
    },
    "total_branchwood_fraction": {
        "value": EXAMPLE_INVENTORY_CANOPY_TOTAL_BRANCHWOOD,
        "summary": "Fraction of total branchwood",
        "description": (
            '`size_partition: "none"` skips the fine-class partition, so '
            "`fraction` applies to the equation family's total branchwood. "
            "The single number then combines two things the default keeps "
            "separate — the fine-branch share of the crown and the share of "
            "fine branchwood that burns. Use it when you have a local "
            "estimate of the available share of all branchwood, or when "
            "Brown's Interior-West fine-branch proportions are not "
            "appropriate for your forest type."
        ),
    },
    "precomputed_fuel_column": {
        "value": EXAMPLE_INVENTORY_CANOPY_COLUMN_FUEL,
        "summary": "Per-tree fuel from an inventory column",
        "description": (
            "Reads available canopy fuel per tree from an inventory column "
            "instead of estimating it, bypassing allometry, the "
            "available-fuel fractions, species inclusion, and crown-class "
            "adjustment. Use this when you have your own crown fuel "
            "estimates — from destructive sampling, a local allometry, or "
            "species-specific consumable fractions. Vertical and horizontal "
            "distribution still apply."
        ),
    },
    "lidar_crown_radii": {
        "value": EXAMPLE_INVENTORY_CANOPY_LIDAR_CROWNS,
        "summary": "Measured crown radii",
        "description": (
            "Reads each tree's maximum crown radius from an inventory "
            "column — e.g. segment areas from LiDAR or NAIP individual-tree "
            "detection — instead of the allometric value. The radius drives "
            "`crown_projected` attribution and the geometric cover methods, "
            "so measured crowns directly improve `cc` and the spatial "
            "pattern of all four bands."
        ),
    },
}

INVENTORY_CANOPY_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_INVENTORY_CANOPY_MINIMAL),
    ("explicit_defaults", EXAMPLE_INVENTORY_CANOPY_EXPLICIT_DEFAULTS),
    ("fuelcalc_comparison", EXAMPLE_INVENTORY_CANOPY_FUELCALC_COMPARISON),
    ("ffe_fvs_comparison", EXAMPLE_INVENTORY_CANOPY_FFE_COMPARISON),
    ("with_cfl", EXAMPLE_INVENTORY_CANOPY_WITH_CFL),
    ("high_resolution_10m", EXAMPLE_INVENTORY_CANOPY_10M),
    ("flat_threshold", EXAMPLE_INVENTORY_CANOPY_FLAT_THRESHOLD),
    ("van_wagner_cbd", EXAMPLE_INVENTORY_CANOPY_VAN_WAGNER_CBD),
    ("total_branchwood_fraction", EXAMPLE_INVENTORY_CANOPY_TOTAL_BRANCHWOOD),
    ("precomputed_fuel_column", EXAMPLE_INVENTORY_CANOPY_COLUMN_FUEL),
    ("lidar_crown_radii", EXAMPLE_INVENTORY_CANOPY_LIDAR_CROWNS),
]
