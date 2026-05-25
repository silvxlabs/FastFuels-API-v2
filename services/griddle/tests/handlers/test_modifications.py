"""
Unit tests for griddle/modifications.py.

Build tiny in-memory grids and assert per-cell behavior for attribute
conditions, inline-geometry spatial conditions, feature-reference spatial
conditions, modifier semantics, the non-negative clamp, band resolution, and
the per-invocation feature cache.
"""

import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from griddle.modifications import (
    _resolve_band,
    apply_modifications,
)
from shapely.geometry import LineString, MultiPolygon, Polygon

from lib.errors import ProcessingError
from lib.firestore import DocumentNotFoundError

# 10×10 grid at 1 m resolution. Origin at (0, 10) so cells run y in [0,10] and
# x in [0,10], with cell centers at half-integers (0.5, 1.5, ..., 9.5).
GRID_TRANSFORM = Affine(1, 0, 0, 0, -1, 10)
GRID_SHAPE = (10, 10)
GRID_CRS = "EPSG:32612"


def _flat_dataset(value: float = 1.0) -> xr.Dataset:
    """Single 2D var `fuel_load.1hr`."""
    arr = np.full(GRID_SHAPE, value, dtype=np.float32)
    ds = xr.Dataset({"fuel_load.1hr": xr.DataArray(arr, dims=["y", "x"])})
    return ds.rio.write_crs(GRID_CRS).rio.write_transform(GRID_TRANSFORM)


def _layerset_dataset(value: float = 1.0) -> xr.Dataset:
    """3D var `shrub` with band coord {loading, height}."""
    arr = np.full((2,) + GRID_SHAPE, value, dtype=np.float32)
    ds = xr.Dataset(
        {
            "shrub": xr.DataArray(
                arr,
                dims=["band", "y", "x"],
                coords={"band": ["loading", "height"]},
            )
        }
    )
    return ds.rio.write_crs(GRID_CRS).rio.write_transform(GRID_TRANSFORM)


# ---------------------------------------------------------------- modifiers


def test_replace_modifier_writes_value_under_mask():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 1.0}],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 0.0).all()


def test_multiply_add_subtract_divide_modifiers():
    for modifier, value, expected in [
        ("multiply", 0.5, 2.0),
        ("add", 1.0, 5.0),
        ("subtract", 1.0, 3.0),
        ("divide", 2.0, 2.0),
    ]:
        ds = _flat_dataset(4.0)
        mods = [
            {
                "conditions": [
                    {"band": "fuel_load.1hr", "operator": "gt", "value": 0.0}
                ],
                "actions": [
                    {"band": "fuel_load.1hr", "modifier": modifier, "value": value}
                ],
            }
        ]
        apply_modifications(ds, mods, "d")
        assert (ds["fuel_load.1hr"].values == expected).all(), modifier


def test_subtract_clamps_to_zero():
    """Non-replace modifiers must not produce negative values."""
    ds = _flat_dataset(0.1)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "gt", "value": 0.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "subtract", "value": 5.0}
            ],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 0.0).all()


def test_replace_does_not_clamp():
    """Replace honors the user-set value, including negative."""
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "gt", "value": 0.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "replace", "value": -1.0}
            ],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == -1.0).all()


# --------------------------------------------------- attribute conditions


def test_attribute_condition_gt():
    ds = _flat_dataset(5.0)
    ds["fuel_load.1hr"].values[0, 0] = 0.1
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "gt", "value": 1.0}],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert ds["fuel_load.1hr"].values[0, 0] == pytest.approx(0.1)
    assert (ds["fuel_load.1hr"].values[1:].flatten() == 0.0).all()


def test_attribute_condition_list_eq():
    ds = _flat_dataset(0.0)
    ds["fuel_load.1hr"].values[0, 0] = 1.0
    ds["fuel_load.1hr"].values[0, 1] = 2.0
    ds["fuel_load.1hr"].values[0, 2] = 3.0
    mods = [
        {
            "conditions": [
                {"band": "fuel_load.1hr", "operator": "eq", "value": [1.0, 2.0]}
            ],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "replace", "value": 99.0}
            ],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert ds["fuel_load.1hr"].values[0, 0] == 99.0
    assert ds["fuel_load.1hr"].values[0, 1] == 99.0
    assert ds["fuel_load.1hr"].values[0, 2] == 3.0


def test_attribute_condition_list_with_invalid_op():
    ds = _flat_dataset(0.0)
    mods = [
        {
            "conditions": [
                {"band": "fuel_load.1hr", "operator": "gt", "value": [1.0, 2.0]}
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    with pytest.raises(ProcessingError) as exc:
        apply_modifications(ds, mods, "d")
    assert exc.value.code == "INVALID_OPERATOR"


# ------------------------------------------------------ spatial: inline


def test_centroid_within_polygon():
    """Cells whose centroid is inside the polygon are masked."""
    ds = _flat_dataset(1.0)
    # Polygon enclosing centers (2.5, 7.5) × (2.5, 7.5) → cells x∈[2,7], y∈[2,7]
    poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "target": "centroid",
                    "geometry": poly.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    inside = (ds["fuel_load.1hr"].values == 0.0).sum()
    # The polygon spans y∈[2,8], x∈[2,8]; cell centers at half-integers from 0.5
    # to 9.5 → centroids inside are (2.5..7.5) × 6 = 36
    assert inside == 36


def test_centroid_within_polygon_stringified_coordinates():
    """Coordinates arrive JSON-stringified (Firestore storage form).

    The API stringifies inline-geometry coordinates before the Firestore write
    because Firestore rejects nested arrays. Griddle reads the modification dict
    back with ``coordinates`` as a JSON string and must deserialize it before
    handing the geometry to shapely. Same expectation as
    ``test_centroid_within_polygon`` (36 cells masked).
    """
    ds = _flat_dataset(1.0)
    poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)])
    geometry = dict(poly.__geo_interface__)
    geometry["coordinates"] = json.dumps(geometry["coordinates"])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "target": "centroid",
                    "geometry": geometry,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    inside = (ds["fuel_load.1hr"].values == 0.0).sum()
    assert inside == 36


def test_cell_intersects_polygon():
    """Cells touched by the polygon are masked (any-overlap)."""
    ds = _flat_dataset(1.0)
    # Polygon at (2.3, 7.7) — boundary crosses cell edges
    poly = Polygon([(2.3, 2.3), (7.7, 2.3), (7.7, 7.7), (2.3, 7.7), (2.3, 2.3)])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "intersects",
                    "target": "cell",
                    "geometry": poly.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 0.0).sum() == 36  # 6×6 touched


def test_cell_within_polygon_strict():
    """Strict within: only cells whose footprint is fully inside."""
    ds = _flat_dataset(1.0)
    # Same polygon as the intersects test — only the inner 4×4 is fully inside.
    poly = Polygon([(2.3, 2.3), (7.7, 2.3), (7.7, 7.7), (2.3, 7.7), (2.3, 2.3)])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "target": "cell",
                    "geometry": poly.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 0.0).sum() == 16  # 4×4 fully inside


def test_cell_within_linestring_yields_empty_mask():
    """A 1D geometry has no area; no cell can be 'fully inside' it."""
    ds = _flat_dataset(1.0)
    line = LineString([(0, 5), (10, 5)])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "within",
                    "target": "cell",
                    "geometry": line.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 1.0).all()  # nothing masked


def test_outside_inverts_mask():
    ds = _flat_dataset(1.0)
    poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)])
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "outside",
                    "target": "centroid",
                    "geometry": poly.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    # 100 cells total - 36 inside = 64 outside
    assert (ds["fuel_load.1hr"].values == 0.0).sum() == 64


def test_buffer_widens_mask():
    ds_no_buffer = _flat_dataset(1.0)
    ds_with_buffer = _flat_dataset(1.0)
    line = LineString([(0.5, 5.5), (9.5, 5.5)])

    def make_mod(buffer_m):
        return [
            {
                "conditions": [
                    {
                        "source": "geometry",
                        "operator": "intersects",
                        "target": "cell",
                        "geometry": line.__geo_interface__,
                        "buffer_m": buffer_m,
                    }
                ],
                "actions": [
                    {"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}
                ],
            }
        ]

    apply_modifications(ds_no_buffer, make_mod(0), "d")
    apply_modifications(ds_with_buffer, make_mod(2.0), "d")
    masked_no_buffer = (ds_no_buffer["fuel_load.1hr"].values == 0.0).sum()
    masked_with_buffer = (ds_with_buffer["fuel_load.1hr"].values == 0.0).sum()
    assert masked_with_buffer > masked_no_buffer


def test_inline_geometry_crs_reproject():
    """Geometry supplied in EPSG:4326 must vector-reproject into the grid CRS."""
    ds = _flat_dataset(1.0)
    # Grid is centered at the origin in UTM 12N. A WGS84 polygon at lon/lat
    # of that origin should land at the grid origin.
    minx, miny = 0, 0
    maxx, maxy = 10, 10
    gs = gpd.GeoSeries(
        [Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])],
        crs=GRID_CRS,
    ).to_crs("EPSG:4326")
    wgs_poly = gs.iloc[0]
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "intersects",
                    "target": "cell",
                    "geometry": wgs_poly.__geo_interface__,
                    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    # The entire grid is inside the polygon
    assert (ds["fuel_load.1hr"].values == 0.0).all()


# ------------------------------------------------------ band resolution


def test_resolve_band_flat():
    ds = _flat_dataset()
    var, coord = _resolve_band(ds, "fuel_load.1hr")
    assert var == "fuel_load.1hr"
    assert coord is None


def test_resolve_band_layerset():
    ds = _layerset_dataset()
    var, coord = _resolve_band(ds, "shrub.loading")
    assert var == "shrub"
    assert coord == "loading"


def test_resolve_band_unknown_raises():
    ds = _flat_dataset()
    with pytest.raises(ProcessingError) as exc:
        _resolve_band(ds, "no.such.band")
    assert exc.value.code == "UNKNOWN_BAND"


def test_layerset_band_modifier_writes_through_band_coord():
    """Mutating `shrub.loading` only touches that band, not `shrub.height`."""
    ds = _layerset_dataset(value=5.0)
    mods = [
        {
            "conditions": [{"band": "shrub.loading", "operator": "gt", "value": 0.0}],
            "actions": [{"band": "shrub.loading", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["shrub"].sel(band="loading").values == 0.0).all()
    assert (ds["shrub"].sel(band="height").values == 5.0).all()


# -------------------------------------------------- feature negative cases


def _patch_feature(domain_id="d", status="completed"):
    """Build a fake snapshot + to_dict() result."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "domain_id": domain_id,
        "status": status,
    }
    return (MagicMock(), snapshot)


def test_feature_not_found():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "target": "cell",
                    "feature_id": "missing",
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    with patch(
        "griddle.modifications.get_document",
        side_effect=DocumentNotFoundError("missing"),
    ):
        with pytest.raises(ProcessingError) as exc:
            apply_modifications(ds, mods, "d")
    assert exc.value.code == "FEATURE_NOT_FOUND"


def test_feature_domain_mismatch():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "target": "cell",
                    "feature_id": "f1",
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    with patch(
        "griddle.modifications.get_document",
        return_value=_patch_feature(domain_id="other-domain"),
    ):
        with pytest.raises(ProcessingError) as exc:
            apply_modifications(ds, mods, "d")
    assert exc.value.code == "FEATURE_DOMAIN_MISMATCH"


def test_feature_not_ready():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "target": "cell",
                    "feature_id": "f1",
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    with patch(
        "griddle.modifications.get_document",
        return_value=_patch_feature(status="running"),
    ):
        with pytest.raises(ProcessingError) as exc:
            apply_modifications(ds, mods, "d")
    assert exc.value.code == "FEATURE_NOT_READY"


# ----------------------------------------------------- feature cache


def test_feature_cache_reads_once_per_buffer():
    """Two conditions referencing the same (feature_id, buffer_m) load once."""
    ds = _flat_dataset(1.0)
    line = LineString([(0, 5), (10, 5)])
    feature_gdf = gpd.GeoDataFrame(geometry=[line], crs=GRID_CRS)

    mods = [
        {
            "conditions": [
                {
                    "source": "feature",
                    "operator": "intersects",
                    "target": "cell",
                    "feature_id": "f1",
                },
                {
                    "source": "feature",
                    "operator": "intersects",
                    "target": "cell",
                    "feature_id": "f1",
                },
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]

    with (
        patch(
            "griddle.modifications.get_document",
            return_value=_patch_feature(),
        ) as mock_get,
        patch(
            "griddle.modifications.gpd.read_parquet",
            return_value=feature_gdf,
        ) as mock_read,
    ):
        apply_modifications(ds, mods, "d")

    assert mock_get.call_count == 1
    assert mock_read.call_count == 1


# ----------------------------------------------- multi-modification


def test_multi_modification_composes_in_order():
    """Mod 2 sees mod 1's mutation."""
    ds = _flat_dataset(4.0)
    mods = [
        # 4 * 2 = 8
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "ge", "value": 4.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "multiply", "value": 2.0}
            ],
        },
        # Only cells that are now ≥ 8 — i.e. all of them — get subtract 1.0
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "ge", "value": 8.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "subtract", "value": 1.0}
            ],
        },
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 7.0).all()


def test_multi_modification_second_no_match():
    """Mod 2 condition is false after mod 1 mutates — only mod 1 takes effect."""
    ds = _flat_dataset(4.0)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 4.0}],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        },
        # After mod 1 nothing has value 4.0 anymore
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 4.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "replace", "value": 99.0}
            ],
        },
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 0.0).all()


def test_multiple_actions_per_modification():
    """One rule writes to two bands under the same mask."""
    ds = _flat_dataset(1.0)
    ds["fuel_depth"] = xr.DataArray(
        np.full(GRID_SHAPE, 2.0, dtype=np.float32), dims=["y", "x"]
    )
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 1.0}],
            "actions": [
                {"band": "fuel_load.1hr", "modifier": "multiply", "value": 5.0},
                {"band": "fuel_depth", "modifier": "add", "value": 1.0},
            ],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 5.0).all()
    assert (ds["fuel_depth"].values == 3.0).all()


# ----------------------------------------------- empty-edge short-circuits


def test_empty_modifications_list_is_noop():
    ds = _flat_dataset(1.0)
    apply_modifications(ds, [], "d")
    assert (ds["fuel_load.1hr"].values == 1.0).all()


def test_empty_conditions_skips_rule():
    """A rule with no conditions does nothing — same as no-match."""
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 1.0).all()


def test_empty_actions_skips_rule():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 1.0}],
            "actions": [],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 1.0).all()


def test_no_match_conditions_is_noop():
    ds = _flat_dataset(1.0)
    mods = [
        {
            "conditions": [{"band": "fuel_load.1hr", "operator": "eq", "value": 999.0}],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    assert (ds["fuel_load.1hr"].values == 1.0).all()


# --------------------------------------------------- missing spatial combos


def test_centroid_intersects_equals_centroid_within_for_polygon():
    """For a point-target, intersects is the same query as within."""
    ds_within = _flat_dataset(1.0)
    ds_intersects = _flat_dataset(1.0)
    poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)])

    def mod(op):
        return [
            {
                "conditions": [
                    {
                        "source": "geometry",
                        "operator": op,
                        "target": "centroid",
                        "geometry": poly.__geo_interface__,
                    }
                ],
                "actions": [
                    {"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}
                ],
            }
        ]

    apply_modifications(ds_within, mod("within"), "d")
    apply_modifications(ds_intersects, mod("intersects"), "d")
    np.testing.assert_array_equal(
        ds_within["fuel_load.1hr"].values, ds_intersects["fuel_load.1hr"].values
    )


def test_cell_outside_inverts_intersects():
    """cell+outside selects every cell intersects does NOT select."""
    ds_intersects = _flat_dataset(1.0)
    ds_outside = _flat_dataset(1.0)
    poly = Polygon([(2.3, 2.3), (7.7, 2.3), (7.7, 7.7), (2.3, 7.7), (2.3, 2.3)])

    def mod(op):
        return [
            {
                "conditions": [
                    {
                        "source": "geometry",
                        "operator": op,
                        "target": "cell",
                        "geometry": poly.__geo_interface__,
                    }
                ],
                "actions": [
                    {"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}
                ],
            }
        ]

    apply_modifications(ds_intersects, mod("intersects"), "d")
    apply_modifications(ds_outside, mod("outside"), "d")
    intersects_mask = ds_intersects["fuel_load.1hr"].values == 0.0
    outside_mask = ds_outside["fuel_load.1hr"].values == 0.0
    np.testing.assert_array_equal(intersects_mask, ~outside_mask)


# --------------------------------------------------- multipolygon


def test_multipolygon_masks_disjoint_regions():
    ds = _flat_dataset(1.0)
    # Two disjoint 2×2 cell footprints (one top-left, one bottom-right).
    multi = MultiPolygon(
        [
            Polygon([(0, 8), (2, 8), (2, 10), (0, 10), (0, 8)]),
            Polygon([(8, 0), (10, 0), (10, 2), (8, 2), (8, 0)]),
        ]
    )
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "intersects",
                    "target": "cell",
                    "geometry": multi.__geo_interface__,
                }
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    masked = ds["fuel_load.1hr"].values == 0.0
    assert masked[0:2, 0:2].all()  # top-left 2×2
    assert masked[8:10, 8:10].all()  # bottom-right 2×2
    assert masked.sum() == 8  # exactly the two 2×2 patches


# --------------------------------------------------- combined spatial + attr


def test_combined_spatial_and_attribute_conditions():
    """A rule with both a spatial and an attribute condition AND-s them."""
    ds = _flat_dataset(1.0)
    # Distinguish two halves of the grid via the attribute condition.
    ds["fuel_load.1hr"].values[:, :5] = 2.0  # left half = 2.0
    poly = Polygon([(0, 5), (10, 5), (10, 10), (0, 10), (0, 5)])  # top half
    mods = [
        {
            "conditions": [
                {
                    "source": "geometry",
                    "operator": "intersects",
                    "target": "cell",
                    "geometry": poly.__geo_interface__,
                },
                {"band": "fuel_load.1hr", "operator": "eq", "value": 2.0},
            ],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0.0}],
        }
    ]
    apply_modifications(ds, mods, "d")
    # Top-left quadrant (5×5) should be zeroed; everywhere else untouched.
    arr = ds["fuel_load.1hr"].values
    assert (arr[:5, :5] == 0.0).all()  # top-left = was 2.0, now 0.0
    assert (arr[:5, 5:] == 1.0).all()  # top-right = was 1.0, attribute mismatch
    assert (arr[5:, :5] == 2.0).all()  # bottom-left = was 2.0, outside polygon
    assert (arr[5:, 5:] == 1.0).all()  # bottom-right = neither
