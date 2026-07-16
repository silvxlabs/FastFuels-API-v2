"""
api/v2/resources/grids/duet/examples.py

OpenAPI request examples for DUET surface fuel grids.
"""

CREATE_DUET_OPENAPI_EXAMPLES = {
    "uncalibrated": {
        "summary": "Raw DUET output (no calibration)",
        "description": (
            "Stores DUET's values as computed. The spatial pattern is "
            "meaningful — litter under crowns, grass in gaps — but the "
            "magnitudes are not physical fuel loadings. Use this to inspect "
            "the pattern, not to drive a fire model."
        ),
        "value": {
            "name": "DUET surface fuels (raw)",
            "source_grid_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "years_since_burn": 25,
            "bands": ["fuel_load.grass", "fuel_load.litter"],
        },
    },
    "calibrated_from_values": {
        "summary": "Calibrated against known loadings",
        "description": (
            "Rescales DUET's pattern to loadings measured or assumed for this "
            "stand. `maxmin` suits sparse or limited fuel data; `meansd` "
            "assumes the targets come from a dataset large enough to be "
            "approximately normal."
        ),
        "value": {
            "name": "DUET surface fuels (calibrated)",
            "source_grid_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "years_since_burn": 25,
            "wind_direction": 270,
            "wind_variability": 30,
            "bands": [
                "fuel_load.grass",
                "fuel_load.litter",
                "fuel_depth.grass",
                "fuel_depth.litter",
            ],
            "calibration": {
                "fuel_load": {
                    "grass": {
                        "source": "values",
                        "method": "meansd",
                        "mean": 0.5,
                        "sd": 0.25,
                    },
                    "litter": {
                        "source": "values",
                        "method": "maxmin",
                        "max": 5.0,
                        "min": 0.0,
                    },
                },
                "fuel_depth": {
                    "grass": {"source": "values", "method": "constant", "value": 0.3},
                    "litter": {"source": "values", "method": "constant", "value": 0.06},
                },
            },
        },
    },
    "separated_litter": {
        "summary": "Coniferous and deciduous litter as separate bands",
        "description": (
            "DUET tracks litter per species and duet-tools separates it into "
            "coniferous and deciduous layers. Useful in mixed stands where the "
            "two litter types burn differently."
        ),
        "value": {
            "name": "DUET mixed-stand litter",
            "source_grid_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "years_since_burn": 15,
            "bands": [
                "fuel_load.litter.coniferous",
                "fuel_load.litter.deciduous",
                "fuel_load.grass",
                "fuel_load.total",
            ],
        },
    },
}
