"""Tests for the compose handler."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.compose import compose_grid

from lib.errors import ProcessingError


def _make_ds(values: dict[str, np.ndarray], nodata: dict[str, float] | None = None):
    first = next(iter(values.values()))
    height, width = first.shape
    y = 30.0 - np.arange(height, dtype=float) * 10.0
    x = 10.0 + np.arange(width, dtype=float) * 10.0
    data_vars = {}
    for key, arr in values.items():
        da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
        if nodata and key in nodata:
            da = da.rio.write_nodata(nodata[key])
        data_vars[key] = da
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs("EPSG:32611")
    ds = ds.rio.write_transform()
    return ds


def _make_3d_ds(values: dict[str, np.ndarray]):
    first = next(iter(values.values()))
    depth, height, width = first.shape
    z = np.arange(depth, dtype=float)
    y = 30.0 - np.arange(height, dtype=float) * 10.0
    x = 10.0 + np.arange(width, dtype=float) * 10.0
    data_vars = {
        key: xr.DataArray(arr, dims=("z", "y", "x"), coords={"z": z, "y": y, "x": x})
        for key, arr in values.items()
    }
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs("EPSG:32611")
    ds = ds.rio.write_transform()
    return ds


def _source(inputs, select=None, compute=None):
    return {
        "name": "compose",
        "inputs": inputs,
        "select": select or [],
        "compute": compute or [],
    }


def _grid(*bands):
    band_defs = []
    for i, band in enumerate(bands):
        if isinstance(band, dict):
            band_defs.append({"index": i, **band})
        else:
            band_defs.append(
                {"key": band, "type": "continuous", "unit": "kg/m**2", "index": i}
            )
    return {
        "id": "grid_out",
        "domain_id": "domain-a",
        "owner_id": "owner-a",
        "bands": band_defs,
    }


def _snapshot(data=None):
    snap = MagicMock()
    snap.to_dict.return_value = data or {
        "id": "unused",
        "status": "completed",
        "domain_id": "domain-a",
        "owner_id": "owner-a",
        "checksum": "checksum-a",
        "bands": [
            {"key": "fbfm", "type": "categorical", "unit": None, "index": 0},
            {
                "key": "fuel_load.1hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 1,
            },
            {
                "key": "fuel_load.10hr",
                "type": "continuous",
                "unit": "kg/m**2",
                "index": 2,
            },
            {"key": "fuel_depth", "type": "continuous", "unit": "m", "index": 3},
            {"key": "savr.1hr", "type": "continuous", "unit": "1/m", "index": 4},
            {
                "key": "moisture_of_extinction",
                "type": "continuous",
                "unit": "%",
                "index": 5,
            },
            {"key": "dead_grass", "type": "continuous", "unit": "kg/m**2", "index": 6},
            {"key": "litter", "type": "continuous", "unit": "kg/m**2", "index": 7},
            {"key": "elevation", "type": "continuous", "unit": "m", "index": 8},
            {"key": "height", "type": "continuous", "unit": "m", "index": 9},
        ],
    }
    return snap


def _grid_doc(*bands, checksum="checksum-a", status="completed", domain_id="domain-a"):
    return {
        "id": "grid-a",
        "status": status,
        "domain_id": domain_id,
        "owner_id": "owner-a",
        "checksum": checksum,
        "bands": [{"index": i, **band} for i, band in enumerate(bands)],
    }


class TestComposeGrid:
    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_select_band(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds({"fuel_load.1hr": np.full((3, 3), 2.0)})

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                select=[{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.full((3, 3), 2.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_multi_grid_add(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.side_effect = [
            _make_ds({"fuel_load.1hr": np.full((3, 3), 2.0)}),
            _make_ds({"fuel_load.1hr": np.full((3, 3), 3.0)}),
        ]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                compute=[
                    {
                        "output": "fuel_load.1hr",
                        "operator": "add",
                        "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.full((3, 3), 5.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_selective_compose_with_max_output_order(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.side_effect = [
            _make_ds(
                {
                    "fbfm": np.array([[1, 2], [3, 4]]),
                    "fuel_depth": np.full((2, 2), 0.3),
                    "savr.1hr": np.full((2, 2), 1800.0),
                    "fuel_load.1hr": np.full((2, 2), 2.0),
                    "fuel_load.10hr": np.array([[1.0, 4.0], [2.0, 6.0]]),
                }
            ),
            _make_ds(
                {
                    "fuel_load.1hr": np.full((2, 2), 3.0),
                    "fuel_load.10hr": np.array([[5.0, 3.0], [7.0, 1.0]]),
                }
            ),
        ]

        result = compose_grid(
            _grid("fbfm", "fuel_depth", "savr.1hr", "fuel_load.1hr", "fuel_load.10hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                select=[
                    {"output": "fbfm", "from": "a.fbfm"},
                    {"output": "fuel_depth", "from": "a.fuel_depth"},
                    {"output": "savr.1hr", "from": "a.savr.1hr"},
                ],
                compute=[
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
            ),
            MagicMock(),
        )

        assert list(result.data_vars) == [
            "fbfm",
            "fuel_depth",
            "savr.1hr",
            "fuel_load.1hr",
            "fuel_load.10hr",
        ]
        np.testing.assert_array_equal(result["fbfm"].values, np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.full((2, 2), 5.0)
        )
        np.testing.assert_array_equal(
            result["fuel_load.10hr"].values,
            np.array([[5.0, 4.0], [7.0, 6.0]]),
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_single_grid_intra_grid_band_math(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fuel_depth": np.full((2, 2), 0.4),
                "moisture_of_extinction": np.full((2, 2), 25.0),
                "dead_grass": np.array([[0.1, 0.2], [0.3, 0.4]]),
                "litter": np.array([[0.5, 0.6], [0.7, 0.8]]),
            }
        )

        result = compose_grid(
            _grid("fuel_depth", "moisture_of_extinction", "fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                select=[
                    {"output": "fuel_depth", "from": "a.fuel_depth"},
                    {
                        "output": "moisture_of_extinction",
                        "from": "a.moisture_of_extinction",
                    },
                ],
                compute=[
                    {
                        "output": "fuel_load.1hr",
                        "operator": "add",
                        "operands": ["a.dead_grass", "a.litter"],
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_allclose(
            result["fuel_load.1hr"].values,
            np.array([[0.6, 0.8], [1.0, 1.2]]),
        )
        np.testing.assert_array_equal(
            result["moisture_of_extinction"].values, np.full((2, 2), 25.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_compute_with_literal(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds({"fuel_load.1hr": np.full((3, 3), 2.0)})

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "fuel_load.1hr",
                        "operator": "multiply",
                        "operands": [
                            "a.fuel_load.1hr",
                            {"type": "literal", "value": 0.5},
                        ],
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.full((3, 3), 1.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_add_converts_typed_literal_unit(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (
            None,
            _snapshot(
                _grid_doc({"key": "elevation", "type": "continuous", "unit": "m"})
            ),
        )
        mock_load_zarr.return_value = _make_ds({"elevation": np.full((2, 2), 1.0)})

        result = compose_grid(
            _grid({"key": "elevation", "type": "continuous", "unit": "m"}),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "elevation",
                        "operator": "add",
                        "operands": [
                            "a.elevation",
                            {"type": "literal", "value": 100, "unit": "cm"},
                        ],
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(result["elevation"].values, np.full((2, 2), 2.0))

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_divide_converts_dimensionless_unit_factor(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (
            None,
            _snapshot(_grid_doc({"key": "height", "type": "continuous", "unit": "m"})),
        )
        mock_load_zarr.return_value = _make_ds({"height": np.full((2, 2), 1.0)})

        result = compose_grid(
            _grid({"key": "height_ratio", "type": "continuous", "unit": None}),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "height_ratio",
                        "operator": "divide",
                        "operands": [
                            "a.height",
                            {"type": "literal", "value": 100, "unit": "cm"},
                        ],
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["height_ratio"].values, np.full((2, 2), 1.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_conditional_literal_fallback(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fbfm": np.array([[91, 91, 1], [1, 1, 1], [91, 1, 1]]),
                "fuel_load.1hr": np.full((3, 3), 2.0),
            }
        )

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                select=[
                    {
                        "output": "fuel_load.1hr",
                        "from": "a.fuel_load.1hr",
                        "conditions": [
                            {"band": "a.fbfm", "operator": "eq", "value": 91}
                        ],
                        "else": 0,
                    }
                ],
            ),
            MagicMock(),
        )

        expected = np.array([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_array_equal(result["fuel_load.1hr"].values, expected)

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_conditional_compute_fallback_to_band(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.side_effect = [
            _make_ds({"fuel_load.1hr": np.array([[2.0, 0.0], [4.0, 6.0]])}),
            _make_ds({"fuel_load.1hr": np.array([[6.0, 8.0], [0.0, -1.0]])}),
        ]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                compute=[
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
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values,
            np.array([[4.0, 0.0], [4.0, 6.0]]),
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_set_membership_selects_between_sources(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.side_effect = [
            _make_ds(
                {
                    "fbfm": np.array([[101, 102, 201], [103, 202, 104]]),
                    "fuel_load.1hr": np.full((2, 3), 1.0),
                }
            ),
            _make_ds({"fuel_load.1hr": np.full((2, 3), 9.0)}),
        ]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                select=[
                    {
                        "output": "fuel_load.1hr",
                        "from": "b.fuel_load.1hr",
                        "conditions": [
                            {
                                "band": "a.fbfm",
                                "operator": "in",
                                "value": [101, 102, 103, 104],
                            }
                        ],
                        "else": "a.fuel_load.1hr",
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values,
            np.array([[9.0, 9.0, 1.0], [9.0, 1.0, 9.0]]),
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_categorical_string_condition_raises(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fbfm": np.array([[101, 102], [103, 104]]),
                "fuel_load.1hr": np.full((2, 2), 1.0),
            }
        )

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid("fuel_load.1hr"),
                _source(
                    [{"grid_id": "grid_a", "alias": "a"}],
                    select=[
                        {
                            "output": "fuel_load.1hr",
                            "from": "a.fuel_load.1hr",
                            "conditions": [
                                {"band": "a.fbfm", "operator": "eq", "value": "GR1"}
                            ],
                            "else": 0,
                        }
                    ],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_INVALID_CONDITION"

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_inline_compute_fallback(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.side_effect = [
            _make_ds({"fuel_load.1hr": np.array([[2.0, 4.0], [6.0, 8.0]])}),
            _make_ds({"fuel_load.1hr": np.array([[0.0, 10.0], [0.0, 12.0]])}),
        ]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                select=[
                    {
                        "output": "fuel_load.1hr",
                        "from": "a.fuel_load.1hr",
                        "conditions": [
                            {"band": "b.fuel_load.1hr", "operator": "eq", "value": 0}
                        ],
                        "else": {
                            "operator": "average",
                            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
                        },
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values,
            np.array([[2.0, 7.0], [6.0, 10.0]]),
        )

    @patch("griddle.handlers.compose._evaluate_spatial_condition")
    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_mixed_attribute_and_spatial_conditions_are_anded(
        self, mock_load_zarr, mock_get_document, mock_spatial_condition
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_spatial_condition.return_value = np.array(
            [[True, False], [True, True]], dtype=bool
        )
        mock_load_zarr.side_effect = [
            _make_ds(
                {
                    "fuel_load.1hr": np.array([[2.0, 2.0], [0.0, 4.0]]),
                    "fbfm": np.array([[1, 2], [3, 4]]),
                    "fuel_depth": np.full((2, 2), 0.3),
                }
            ),
            _make_ds({"fuel_load.1hr": np.array([[6.0, 6.0], [6.0, 8.0]])}),
        ]

        result = compose_grid(
            _grid("fbfm", "fuel_depth", "fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                select=[
                    {"output": "fbfm", "from": "a.fbfm"},
                    {"output": "fuel_depth", "from": "a.fuel_depth"},
                ],
                compute=[
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
                                            [0.0, 0.0],
                                            [1.0, 0.0],
                                            [1.0, 1.0],
                                            [0.0, 1.0],
                                            [0.0, 0.0],
                                        ]
                                    ],
                                },
                            },
                        ],
                        "else": "a.fuel_load.1hr",
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values,
            np.array([[4.0, 2.0], [0.0, 6.0]]),
        )
        mock_spatial_condition.assert_called_once()

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_inline_spatial_condition_decodes_stringified_coordinates(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {"fuel_load.1hr": np.array([[1.0, 2.0], [3.0, 4.0]])}
        )
        coords = [
            [
                [-100.0, -100.0],
                [100.0, -100.0],
                [100.0, 100.0],
                [-100.0, 100.0],
                [-100.0, -100.0],
            ]
        ]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                select=[
                    {
                        "output": "fuel_load.1hr",
                        "from": "a.fuel_load.1hr",
                        "conditions": [
                            {
                                "source": "geometry",
                                "operator": "within",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": json.dumps(coords),
                                },
                            }
                        ],
                        "else": 0,
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_compute_nodata_propagates_to_nan(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fuel_load.1hr": np.array(
                    [[1.0, -9999.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                )
            },
            nodata={"fuel_load.1hr": -9999.0},
        )

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "fuel_load.1hr",
                        "operator": "multiply",
                        "operands": ["a.fuel_load.1hr", 2],
                    }
                ],
            ),
            MagicMock(),
        )

        assert np.isnan(result["fuel_load.1hr"].values[0, 1])
        assert result["fuel_load.1hr"].values[0, 0] == 2.0

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_nodata_detected_before_unit_scaling(
        self, mock_load_zarr, mock_get_document
    ):
        # The nodata sentinel must be read from the raw band — before astype
        # and before the cm->m factor would scale -9999 to -99.99 and hide it.
        mock_get_document.return_value = (
            None,
            _snapshot(
                _grid_doc({"key": "depth_cm", "type": "continuous", "unit": "cm"})
            ),
        )
        mock_load_zarr.return_value = _make_ds(
            {"depth_cm": np.array([[100.0, -9999.0], [200.0, 300.0]])},
            nodata={"depth_cm": -9999.0},
        )

        result = compose_grid(
            _grid({"key": "depth_out", "type": "continuous", "unit": "m"}),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "depth_out",
                        "operator": "add",
                        "operands": [
                            "a.depth_cm",
                            {"type": "literal", "value": 0, "unit": "cm"},
                        ],
                    }
                ],
            ),
            MagicMock(),
        )

        values = result["depth_out"].values
        assert np.isnan(values[0, 1])
        np.testing.assert_array_equal(
            values[[0, 1, 1], [0, 0, 1]], np.array([1.0, 2.0, 3.0])
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_divide_by_zero_raises(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fuel_load.1hr": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "fuel_depth": np.array([[1.0, 0.0], [2.0, 2.0]]),
            }
        )

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid({"key": "fuel_ratio", "type": "continuous", "unit": "kg/m**3"}),
                _source(
                    [{"grid_id": "grid_a", "alias": "a"}],
                    compute=[
                        {
                            "output": "fuel_ratio",
                            "operator": "divide",
                            "operands": ["a.fuel_load.1hr", "a.fuel_depth"],
                        }
                    ],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_NON_FINITE_RESULT"

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_divide_by_zero_excluded_by_condition_does_not_raise(
        self, mock_load_zarr, mock_get_document
    ):
        # The zero denominator at [0, 1] makes the divide produce inf, but the
        # condition routes that cell to the fallback, so the job must succeed.
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_ds(
            {
                "fuel_load.1hr": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "fuel_depth": np.array([[1.0, 0.0], [2.0, 2.0]]),
            }
        )

        result = compose_grid(
            _grid({"key": "fuel_ratio", "type": "continuous", "unit": "kg/m**3"}),
            _source(
                [{"grid_id": "grid_a", "alias": "a"}],
                compute=[
                    {
                        "output": "fuel_ratio",
                        "operator": "divide",
                        "operands": ["a.fuel_load.1hr", "a.fuel_depth"],
                        "conditions": [
                            {"band": "a.fuel_depth", "operator": "gt", "value": 0}
                        ],
                        "else": 0,
                    }
                ],
            ),
            MagicMock(),
        )

        np.testing.assert_array_equal(
            result["fuel_ratio"].values,
            np.array([[1.0, 0.0], [1.5, 2.0]]),
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_alignment_mismatch_raises(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        first = _make_ds({"fuel_load.1hr": np.ones((3, 3))})
        second = _make_ds({"fuel_load.1hr": np.ones((2, 2))})
        mock_load_zarr.side_effect = [first, second]

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid("fuel_load.1hr"),
                _source(
                    [
                        {"grid_id": "grid_a", "alias": "a"},
                        {"grid_id": "grid_b", "alias": "b"},
                    ],
                    compute=[
                        {
                            "output": "fuel_load.1hr",
                            "operator": "add",
                            "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
                        }
                    ],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_ALIGNMENT_MISMATCH"

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_checksum_mismatch_raises_before_loading_zarr(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (
            None,
            _snapshot(_grid_doc(checksum="new-checksum")),
        )

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid("fuel_load.1hr"),
                _source(
                    [
                        {
                            "grid_id": "grid_a",
                            "alias": "a",
                            "source_grid_checksum": "old-checksum",
                        }
                    ],
                    select=[{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_INPUT_CHANGED"
        mock_load_zarr.assert_not_called()

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_incomplete_input_raises_before_loading_zarr(
        self, mock_load_zarr, mock_get_document
    ):
        mock_get_document.return_value = (
            None,
            _snapshot(_grid_doc(status="running")),
        )

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid("fuel_load.1hr"),
                _source(
                    [{"grid_id": "grid_a", "alias": "a"}],
                    select=[{"output": "fuel_load.1hr", "from": "a.fuel_load.1hr"}],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_INPUT_NOT_COMPLETED"
        mock_load_zarr.assert_not_called()

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_subnanometer_coord_drift_still_aligns(
        self, mock_load_zarr, mock_get_document
    ):
        # Sub-nanometer coordinate drift passes the 1e-9 transform tolerance but
        # would make xarray's label-based arithmetic inner-join to an empty
        # result without coordinate normalization.
        mock_get_document.return_value = (None, _snapshot())
        first = _make_ds({"fuel_load.1hr": np.full((3, 3), 2.0)})
        drifted = _make_ds({"fuel_load.1hr": np.full((3, 3), 3.0)})
        drifted = drifted.assign_coords(x=drifted["x"] + 1e-10, y=drifted["y"] + 1e-10)
        mock_load_zarr.side_effect = [first, drifted]

        result = compose_grid(
            _grid("fuel_load.1hr"),
            _source(
                [
                    {"grid_id": "grid_a", "alias": "a"},
                    {"grid_id": "grid_b", "alias": "b"},
                ],
                compute=[
                    {
                        "output": "fuel_load.1hr",
                        "operator": "add",
                        "operands": ["a.fuel_load.1hr", "b.fuel_load.1hr"],
                    }
                ],
            ),
            MagicMock(),
        )

        assert result["fuel_load.1hr"].shape == (3, 3)
        np.testing.assert_array_equal(
            result["fuel_load.1hr"].values, np.full((3, 3), 5.0)
        )

    @patch("griddle.handlers.compose.get_document")
    @patch("griddle.handlers.compose.load_zarr")
    def test_3d_input_raises(self, mock_load_zarr, mock_get_document):
        mock_get_document.return_value = (None, _snapshot())
        mock_load_zarr.return_value = _make_3d_ds({"bulk_density": np.ones((2, 3, 3))})

        with pytest.raises(ProcessingError) as exc_info:
            compose_grid(
                _grid("bulk_density"),
                _source(
                    [{"grid_id": "grid_a", "alias": "a"}],
                    select=[{"output": "bulk_density", "from": "a.bulk_density"}],
                ),
                MagicMock(),
            )

        assert exc_info.value.code == "COMPOSE_UNSUPPORTED_DIMENSIONALITY"
