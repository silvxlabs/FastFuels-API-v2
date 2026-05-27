"""
Tests for the FBFM40 lookup handler.

Tests cover:
- SB40 table loading and completeness
- Imperial-to-metric unit conversions with spot-checks
- Single and multi-band lookups with mock source grids
- Non-burnable and out-of-range code handling
- Spatial metadata propagation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pint
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from griddle.handlers.lookup import (
    BAND_KEY_TO_COLUMN,
    UNIT_CONVERSIONS,
    _convert_to_metric,
    _get_conversion_key,
    _load_sb40_table,
    fbfm40_lookup,
)

from lib.errors import ProcessingError
from lib.zarr_utils import load_zarr, save_zarr

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


class TestSb40TableLoading:
    """Tests for SB40 CSV loading."""

    def test_table_loads_all_46_models(self):
        """SB40 table has all 46 fuel models (5 NB + 41 burnable)."""
        table = _load_sb40_table()
        expected_keys = [
            91,
            92,
            93,
            98,
            99,  # NB
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,  # GR
            121,
            122,
            123,
            124,  # GS
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,  # SH
            161,
            162,
            163,
            164,
            165,  # TU
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,  # TL
            201,
            202,
            203,
            204,  # SB
        ]
        # Check that non-zero values exist for burnable models
        for key in expected_keys[5:]:  # Skip NB models
            assert (
                table["fuel_load_1hr"][key] > 0 or table["fuel_load_10hr"][key] > 0
            ), f"Burnable model {key} should have some fuel load"

    def test_table_has_all_columns(self):
        """Table contains all expected quantity columns."""
        table = _load_sb40_table()
        expected_columns = list(BAND_KEY_TO_COLUMN.values())
        for col in expected_columns:
            assert col in table, f"Missing column: {col}"

    def test_nb_models_are_zeros(self):
        """Non-burnable models (NB1-NB9) have all zero values."""
        table = _load_sb40_table()
        nb_keys = [91, 92, 93, 98, 99]
        for key in nb_keys:
            for col in table:
                assert table[col][key] == 0.0, (
                    f"NB model {key} should have zero for {col}, got {table[col][key]}"
                )

    def test_gr1_values_match_sb40(self):
        """GR1 (key=101) values match Scott-Burgan 40 reference for all 14 columns."""
        table = _load_sb40_table()
        assert table["fuel_load_1hr"][101] == pytest.approx(0.10)
        assert table["fuel_load_10hr"][101] == pytest.approx(0.0)
        assert table["fuel_load_100hr"][101] == pytest.approx(0.0)
        assert table["fuel_load_live_herb"][101] == pytest.approx(0.30)
        assert table["fuel_load_live_woody"][101] == pytest.approx(0.0)
        assert table["savr_1hr"][101] == pytest.approx(2200)
        assert table["savr_10hr"][101] == pytest.approx(109)
        assert table["savr_100hr"][101] == pytest.approx(30)
        assert table["savr_live_herb"][101] == pytest.approx(2000)
        assert table["savr_live_woody"][101] == pytest.approx(9999)
        assert table["fuel_depth"][101] == pytest.approx(0.4)
        assert table["moisture_of_extinction"][101] == pytest.approx(15)
        assert table["heat_content"][101] == pytest.approx(8000)
        assert table["is_dynamic"][101] == pytest.approx(1.0)

    def test_gr6_heat_content_is_9000(self):
        """GR6 (key=106) has heat content of 9000 BTU/lb (only exception)."""
        table = _load_sb40_table()
        assert table["heat_content"][106] == pytest.approx(9000)

    def test_savr_10hr_constant_109(self):
        """All burnable models have savr_10hr = 109 1/ft."""
        table = _load_sb40_table()
        burnable_keys = (
            list(range(101, 110))
            + list(range(121, 125))
            + list(range(141, 150))
            + list(range(161, 166))
            + list(range(181, 190))
            + list(range(201, 205))
        )
        for key in burnable_keys:
            assert table["savr_10hr"][key] == pytest.approx(109), (
                f"Model {key} savr_10hr should be 109, got {table['savr_10hr'][key]}"
            )

    def test_savr_100hr_constant_30(self):
        """All burnable models have savr_100hr = 30 1/ft."""
        table = _load_sb40_table()
        burnable_keys = (
            list(range(101, 110))
            + list(range(121, 125))
            + list(range(141, 150))
            + list(range(161, 166))
            + list(range(181, 190))
            + list(range(201, 205))
        )
        for key in burnable_keys:
            assert table["savr_100hr"][key] == pytest.approx(30), (
                f"Model {key} savr_100hr should be 30, got {table['savr_100hr'][key]}"
            )

    def test_dynamic_models(self):
        """Dynamic models are correctly flagged."""
        table = _load_sb40_table()
        # All GR models are dynamic
        for key in range(101, 110):
            assert table["is_dynamic"][key] == 1.0, f"GR model {key} should be dynamic"
        # All GS models are dynamic
        for key in range(121, 125):
            assert table["is_dynamic"][key] == 1.0, f"GS model {key} should be dynamic"
        # SH1 and SH9 are dynamic
        assert table["is_dynamic"][141] == 1.0  # SH1
        assert table["is_dynamic"][149] == 1.0  # SH9
        # SH2-SH8 are NOT dynamic
        for key in range(142, 149):
            assert table["is_dynamic"][key] == 0.0, (
                f"SH model {key} should not be dynamic"
            )
        # TU1, TU3 are dynamic; TU2, TU4, TU5 are NOT
        assert table["is_dynamic"][161] == 1.0  # TU1
        assert table["is_dynamic"][162] == 0.0  # TU2
        assert table["is_dynamic"][163] == 1.0  # TU3
        assert table["is_dynamic"][164] == 0.0  # TU4
        assert table["is_dynamic"][165] == 0.0  # TU5
        # TL models are NOT dynamic
        for key in range(181, 190):
            assert table["is_dynamic"][key] == 0.0, (
                f"TL model {key} should not be dynamic"
            )
        # SB models are NOT dynamic
        for key in range(201, 205):
            assert table["is_dynamic"][key] == 0.0, (
                f"SB model {key} should not be dynamic"
            )


class TestUnitConversion:
    """Tests for imperial-to-metric unit conversion."""

    def test_fuel_load_conversion(self):
        """Fuel load: tons/acre → kg/m**2."""
        # GR1: 0.10 t/ac
        imperial = np.array([0.10])
        metric = _convert_to_metric(imperial, "fuel_load_1hr")
        expected = Q_(0.10, "short_ton / acre").to("kg / m**2").magnitude
        assert metric[0] == pytest.approx(expected, rel=1e-6)

    def test_savr_conversion(self):
        """SAVR: 1/ft → 1/m."""
        imperial = np.array([2200.0])
        metric = _convert_to_metric(imperial, "savr_1hr")
        expected = Q_(2200.0, "1 / ft").to("1 / m").magnitude
        assert metric[0] == pytest.approx(expected, rel=1e-6)

    def test_fuel_depth_conversion(self):
        """Fuel depth: ft → m."""
        imperial = np.array([0.4])
        metric = _convert_to_metric(imperial, "fuel_depth")
        expected = Q_(0.4, "ft").to("m").magnitude
        assert metric[0] == pytest.approx(expected, rel=1e-6)

    def test_heat_content_conversion(self):
        """Heat content: BTU/lb → kJ/kg."""
        imperial = np.array([8000.0])
        metric = _convert_to_metric(imperial, "heat_content")
        expected = Q_(8000.0, "BTU / lb").to("kJ / kg").magnitude
        assert metric[0] == pytest.approx(expected, rel=1e-4)

    def test_moisture_of_extinction_no_conversion(self):
        """Moisture of extinction: no conversion (unitless percentage)."""
        values = np.array([15.0, 30.0, 40.0])
        result = _convert_to_metric(values, "moisture_of_extinction")
        np.testing.assert_array_equal(result, values)

    def test_is_dynamic_no_conversion(self):
        """is_dynamic: no conversion (boolean 0/1)."""
        values = np.array([0.0, 1.0, 1.0])
        result = _convert_to_metric(values, "is_dynamic")
        np.testing.assert_array_equal(result, values)


def _make_mock_source_ds(fbfm_codes, y_coords=None, x_coords=None, crs="EPSG:32610"):
    """Create a mock xarray Dataset that mimics load_zarr output."""
    if y_coords is None:
        y_coords = np.arange(fbfm_codes.shape[0], dtype=np.float64) * 30.0
    if x_coords is None:
        x_coords = np.arange(fbfm_codes.shape[1], dtype=np.float64) * 30.0

    da = xr.DataArray(
        data=fbfm_codes.astype(np.int16),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform()
    da = da.rio.write_nodata(np.int16(32767))
    ds = da.to_dataset(name="FBFM")
    return ds


ALL_FBFM40_KEYS = [
    91,
    92,
    93,
    98,
    99,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    121,
    122,
    123,
    124,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    161,
    162,
    163,
    164,
    165,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    201,
    202,
    203,
    204,
]

ALL_BAND_KEYS = list(BAND_KEY_TO_COLUMN.keys())


class TestFbfm40Lookup:
    """Tests for the fbfm40_lookup function."""

    @patch("griddle.handlers.lookup.load_zarr")
    def test_returns_dataset(self, mock_load_zarr):
        """Returns a Dataset, not a DataArray."""
        codes = np.array([[101, 102]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}, {"key": "fuel_depth"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert isinstance(result, xr.Dataset)

    @patch("griddle.handlers.lookup.load_zarr")
    def test_dataset_variables_match_band_keys(self, mock_load_zarr):
        """Each band key becomes a named variable in the Dataset."""
        codes = np.array([[101, 102]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}, {"key": "fuel_depth"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert set(result.data_vars) == {"fuel_load.1hr", "fuel_depth"}
        for var in result.data_vars:
            assert result[var].dims == ("y", "x")

    @patch("griddle.handlers.lookup.load_zarr")
    def test_all_codes_all_bands(self, mock_load_zarr):
        """Every FBFM40 code produces the correct metric value for every band.

        Feeds all 46 codes through the full pipeline requesting all 14
        bands, then independently computes the expected metric value
        from the raw SB40 table + pint conversion and compares every cell.
        """
        table = _load_sb40_table()
        codes = np.array([ALL_FBFM40_KEYS])  # shape (1, 46)
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": k} for k in ALL_BAND_KEYS]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == set(ALL_BAND_KEYS)

        for band_key in ALL_BAND_KEYS:
            column = BAND_KEY_TO_COLUMN[band_key]
            conv_key = _get_conversion_key(column)
            src_unit, dst_unit = UNIT_CONVERSIONS[conv_key]

            for col_idx, fbfm_key in enumerate(ALL_FBFM40_KEYS):
                imperial_val = table[column][fbfm_key]

                if src_unit is None:
                    expected = imperial_val
                else:
                    expected = Q_(imperial_val, src_unit).to(dst_unit).magnitude

                actual = result[band_key].values[0, col_idx]
                assert actual == pytest.approx(expected, rel=1e-6, abs=1e-12), (
                    f"Mismatch for code {fbfm_key}, band {band_key}: "
                    f"expected {expected}, got {actual}"
                )

    @patch("griddle.handlers.lookup.load_zarr")
    def test_single_band(self, mock_load_zarr):
        """Single band lookup produces correct shape and values."""
        codes = np.array([[101, 102], [103, 91]])  # GR1, GR2, GR3, NB1
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert set(result.data_vars) == {"fuel_load.1hr"}
        assert result["fuel_load.1hr"].dims == ("y", "x")
        assert result["fuel_load.1hr"].shape == (2, 2)

        # GR1 fuel_load_1hr = 0.10 t/ac converted to metric
        expected_gr1 = Q_(0.10, "short_ton / acre").to("kg / m**2").magnitude
        assert result["fuel_load.1hr"].values[0, 0] == pytest.approx(
            expected_gr1, rel=1e-6
        )

        # NB1 should be zero
        assert result["fuel_load.1hr"].values[1, 1] == 0.0

    @patch("griddle.handlers.lookup.load_zarr")
    def test_multi_band_output(self, mock_load_zarr):
        """Multiple bands produce one variable per band."""
        codes = np.array([[101, 102], [103, 104]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [
            {"key": "fuel_load.1hr"},
            {"key": "savr.1hr"},
            {"key": "fuel_depth"},
            {"key": "heat_content"},
            {"key": "is_dynamic"},
        ]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert set(result.data_vars) == {
            "fuel_load.1hr",
            "savr.1hr",
            "fuel_depth",
            "heat_content",
            "is_dynamic",
        }
        for var in result.data_vars:
            assert result[var].shape == (2, 2)

    @patch("griddle.handlers.lookup.load_zarr")
    def test_nonburnable_codes_produce_zeros(self, mock_load_zarr):
        """NB codes (91-99) produce zero values for all bands."""
        codes = np.array([[91, 92], [93, 99]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [
            {"key": "fuel_load.1hr"},
            {"key": "savr.1hr"},
            {"key": "fuel_depth"},
        ]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        for var in result.data_vars:
            np.testing.assert_array_equal(result[var].values, 0.0)

    @patch("griddle.handlers.lookup.load_zarr")
    def test_invalid_codes_raise_error(self, mock_load_zarr):
        """Out-of-range or unknown FBFM codes fail with a descriptive error."""
        codes = np.array([[101, 255], [999, -1]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        with pytest.raises(ProcessingError) as exc_info:
            fbfm40_lookup("test-grid-id", bands, progress)

        assert exc_info.value.code == "INVALID_FBFM_CODES"
        assert "-1" in exc_info.value.message
        assert "255" in exc_info.value.message
        assert "999" in exc_info.value.message

    @patch("griddle.handlers.lookup.load_zarr")
    def test_zero_nodata_codes_raise_error(self, mock_load_zarr):
        """Code 0 (nodata) is not a valid FBFM40 code and raises an error."""
        codes = np.array([[0, 101]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        with pytest.raises(ProcessingError) as exc_info:
            fbfm40_lookup("test-grid-id", bands, progress)

        assert exc_info.value.code == "INVALID_FBFM_CODES"
        assert "0" in exc_info.value.message

    @patch("griddle.handlers.lookup.load_zarr")
    def test_spatial_metadata_inherited(self, mock_load_zarr):
        """Output inherits CRS from source grid."""
        codes = np.array([[101, 102], [103, 104]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes, crs="EPSG:32610")
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        assert result.rio.crs is not None
        assert result.rio.crs.to_epsg() == 32610

    @patch("griddle.handlers.lookup.load_zarr")
    def test_y_x_coordinates_preserved(self, mock_load_zarr):
        """Output preserves y and x coordinates from source grid."""
        codes = np.array([[101, 102], [103, 104]])
        y = np.array([5000000.0, 4999970.0])
        x = np.array([500000.0, 500030.0])
        mock_load_zarr.return_value = _make_mock_source_ds(
            codes, y_coords=y, x_coords=x
        )
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        np.testing.assert_array_equal(result.coords["y"].values, y)
        np.testing.assert_array_equal(result.coords["x"].values, x)

    @patch("griddle.handlers.lookup.load_zarr")
    def test_nodata_declared(self, mock_load_zarr):
        """All output variables have nodata declared."""
        codes = np.array([[101, 102], [103, 104]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": k} for k in BAND_KEY_TO_COLUMN]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        for var in result.data_vars:
            assert result[var].rio.nodata is not None

    @patch("griddle.handlers.lookup.load_zarr")
    def test_progress_callbacks(self, mock_load_zarr):
        """Handler calls progress at expected stages."""
        codes = np.array([[101]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        fbfm40_lookup("test-grid-id", bands, progress)

        assert progress.call_count >= 3
        messages = [call[0][0] for call in progress.call_args_list]
        assert any("Loading" in m for m in messages)
        assert any("Looking up" in m or "Lookup" in m for m in messages)

    @patch("griddle.handlers.lookup.load_zarr")
    def test_source_grid_not_found_raises(self, mock_load_zarr):
        """Missing source grid raises ProcessingError."""
        mock_load_zarr.side_effect = FileNotFoundError("not found")
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        with pytest.raises(ProcessingError) as exc_info:
            fbfm40_lookup("missing-grid", bands, progress)

        assert exc_info.value.code == "SOURCE_GRID_NOT_FOUND"


class TestLookupZarrRoundTrip:
    """Verify lookup output survives a zarr save/load cycle.

    This is the exact test that would have caught both production bugs:
    1. spatial_ref demotion (decode_coords regression)
    2. DataArray saved as __xarray_dataarray_variable__
    """

    @patch("griddle.handlers.lookup.load_zarr")
    def test_round_trip_preserves_named_variables(self, mock_load_zarr, tmp_path):
        """Each band is a separate data_var after round-trip, not __xarray_dataarray_variable__."""
        codes = np.array([[101, 102], [103, 104]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [
            {"key": "fuel_load.1hr"},
            {"key": "fuel_depth"},
            {"key": "savr.1hr"},
        ]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        save_zarr(str(tmp_path / "lookup.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "lookup.zarr"))

        assert set(loaded.data_vars) == {"fuel_load.1hr", "fuel_depth", "savr.1hr"}
        assert "__xarray_dataarray_variable__" not in loaded.data_vars

    @patch("griddle.handlers.lookup.load_zarr")
    def test_round_trip_spatial_ref_is_coordinate(self, mock_load_zarr, tmp_path):
        """spatial_ref is a coordinate after round-trip, not a data variable."""
        codes = np.array([[101, 102]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        save_zarr(str(tmp_path / "lookup.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "lookup.zarr"))

        assert "spatial_ref" in loaded.coords
        assert "spatial_ref" not in loaded.data_vars

    @patch("griddle.handlers.lookup.load_zarr")
    def test_round_trip_crs_preserved(self, mock_load_zarr, tmp_path):
        """CRS survives save/load cycle."""
        codes = np.array([[101, 102]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes, crs="EPSG:32610")
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        save_zarr(str(tmp_path / "lookup.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "lookup.zarr"))

        assert loaded.rio.crs is not None
        assert loaded.rio.crs.to_epsg() == 32610

    @patch("griddle.handlers.lookup.load_zarr")
    def test_round_trip_nodata_declared(self, mock_load_zarr, tmp_path):
        """Nodata survives save/load cycle."""
        codes = np.array([[101, 102]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": k} for k in BAND_KEY_TO_COLUMN]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        save_zarr(str(tmp_path / "lookup.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "lookup.zarr"))

        for var in loaded.data_vars:
            assert loaded[var].rio.nodata is not None

    @patch("griddle.handlers.lookup.load_zarr")
    def test_round_trip_to_raster_succeeds(self, mock_load_zarr, tmp_path):
        """Dataset.rio.to_raster() works after round-trip — the exact operation the exporter performs."""
        codes = np.array([[101, 102], [103, 104]])
        mock_load_zarr.return_value = _make_mock_source_ds(codes)
        progress = MagicMock()

        bands = [{"key": "fuel_load.1hr"}, {"key": "fuel_depth"}]
        result = fbfm40_lookup("test-grid-id", bands, progress)

        save_zarr(str(tmp_path / "lookup.zarr"), result, chunk_shape=(512, 512))
        loaded = load_zarr(str(tmp_path / "lookup.zarr"))

        # Full Dataset to_raster — the exact operation the exporter performs
        out_path = str(tmp_path / "multiband.tif")
        loaded.rio.to_raster(out_path)
        assert (tmp_path / "multiband.tif").exists()
