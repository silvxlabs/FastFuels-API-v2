"""
Example request bodies for resample endpoints.

These examples are used in:
1. OpenAPI/Swagger documentation - Users see these as example payloads
2. Integration tests - Each example is tested to ensure documentation stays accurate

All examples assume a valid source_grid_id exists. The domain_id is
propagated from the source grid automatically.
Replace placeholder IDs with actual values when testing.
"""

EXAMPLE_RESAMPLE_MINIMAL = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "domain", "resolution": 2.0},
}

EXAMPLE_RESAMPLE_WITH_OVERRIDES = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "domain", "resolution": 2.0, "method": "bilinear"},
    "method_overrides": {"fbfm": "nearest"},
    "name": "Resampled fuels at 2m",
    "description": "30m LANDFIRE resampled to 2m with nearest for categorical FBFM",
    "tags": ["resampled", "2m"],
}

EXAMPLE_RESAMPLE_ALL_NEAREST = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "domain", "resolution": 5.0, "method": "nearest"},
    "name": "Nearest-neighbor resample at 5m",
}

EXAMPLE_RESAMPLE_MODE_DOWNSAMPLING = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "domain", "resolution": 60.0, "method": "mode"},
    "name": "Downsampled categorical grid at 60m",
    "description": "Downsample using mode (most frequent value) for categorical data",
}

EXAMPLE_RESAMPLE_TARGET_GRID = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "grid", "grid_id": "grid_xyz789"},
    "name": "Aligned to existing grid",
    "description": "Exact lattice match to an existing grid (CRS, transform, shape).",
}

EXAMPLE_RESAMPLE_TARGET_GRID_NEW_RESOLUTION = {
    "source_grid_id": "grid_abc123",
    "alignment": {
        "target": "grid",
        "grid_id": "grid_xyz789",
        "resolution": 1.0,
        "method": "bilinear",
    },
    "name": "Same anchor as target grid, 1m cell size",
    "description": (
        "Reuse the target grid's CRS and origin but resample to 1m. "
        "Output cells nest cleanly inside the target grid's pixels."
    ),
}

EXAMPLE_RESAMPLE_NATIVE = {
    "source_grid_id": "grid_abc123",
    "alignment": {"target": "native", "resolution": 5.0},
    "name": "Resolution change preserving source anchor",
}

CREATE_RESAMPLE_OPENAPI_EXAMPLES = {
    "minimal": {
        "value": EXAMPLE_RESAMPLE_MINIMAL,
        "summary": "Minimal request (domain anchor)",
        "description": (
            "Resample a grid to 2m at the domain origin using role-aware "
            "default resampling. The source grid must have status "
            "'completed' and a georeference."
        ),
    },
    "with_overrides": {
        "value": EXAMPLE_RESAMPLE_WITH_OVERRIDES,
        "summary": "Per-band method overrides",
        "description": (
            "Resample to 2m using bilinear by default, but "
            "nearest-neighbor for the categorical 'fbfm' band. This is "
            "useful when a grid has both continuous and categorical bands."
        ),
    },
    "all_nearest": {
        "value": EXAMPLE_RESAMPLE_ALL_NEAREST,
        "summary": "All nearest-neighbor",
        "description": (
            "Resample to 5m using nearest-neighbor for all "
            "bands. Appropriate for purely categorical grids."
        ),
    },
    "mode_downsampling": {
        "value": EXAMPLE_RESAMPLE_MODE_DOWNSAMPLING,
        "summary": "Mode aggregation for downsampling",
        "description": (
            "Downsample to 60m using mode (most frequent value). "
            "Appropriate for categorical grids where you want the "
            "dominant category in each output pixel."
        ),
    },
    "target_grid": {
        "value": EXAMPLE_RESAMPLE_TARGET_GRID,
        "summary": "Align to an existing grid",
        "description": (
            "Produce a grid byte-equal in CRS, transform, and shape to the "
            "target grid. Useful for exact composition with an existing "
            "lattice."
        ),
    },
    "target_grid_new_resolution": {
        "value": EXAMPLE_RESAMPLE_TARGET_GRID_NEW_RESOLUTION,
        "summary": "Same anchor as target grid, new resolution",
        "description": (
            "Reuse the target grid's CRS and origin, but resample to a "
            "different cell size. Output cells nest cleanly inside the "
            "target grid's pixels (when the new resolution divides cleanly)."
        ),
    },
    "native": {
        "value": EXAMPLE_RESAMPLE_NATIVE,
        "summary": "Resolution change preserving source anchor",
        "description": (
            "Resample to a new resolution but preserve the source "
            "raster's pixel anchor. Output won't compose with "
            "domain-anchored grids without further alignment."
        ),
    },
}

ALL_RESAMPLE_EXAMPLE_VALUES = [
    ("minimal", EXAMPLE_RESAMPLE_MINIMAL),
    ("with_overrides", EXAMPLE_RESAMPLE_WITH_OVERRIDES),
    ("all_nearest", EXAMPLE_RESAMPLE_ALL_NEAREST),
    ("mode_downsampling", EXAMPLE_RESAMPLE_MODE_DOWNSAMPLING),
    ("target_grid", EXAMPLE_RESAMPLE_TARGET_GRID),
    ("target_grid_new_resolution", EXAMPLE_RESAMPLE_TARGET_GRID_NEW_RESOLUTION),
    ("native", EXAMPLE_RESAMPLE_NATIVE),
]
