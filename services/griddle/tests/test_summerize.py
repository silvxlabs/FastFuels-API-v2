"""
Tests for griddle.summarize module.
"""

import math

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from griddle.summarize import (
    _nodata_mask,
    _summarize_categorical,
    _summarize_continuous,
    summarize_dataset,
)

from lib.errors import ProcessingError


def _make_da(data: np.ndarray, nodata=None) -> xr.DataArray:
    """Build a 2D DataArray with CRS and transform."""
    h, w = data.shape
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
    )
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.write_transform(Affine(10, 0, 0, 0, -10, 0))
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def _make_ds(variables: dict) -> xr.Dataset:
    """Build a Dataset from a dict of {name: DataArray}."""
    ds = xr.Dataset(variables)
    first = next(iter(variables.values()))
    ds = ds.rio.write_crs(first.rio.crs)
    ds = ds.rio.write_transform(first.rio.transform())
    return ds


# ---------------------------------------------------------------------------
# _nodata_mask
# ---------------------------------------------------------------------------


class TestNodataMask:
    """_nodata_mask builds correct boolean masks."""

    def test_none_nodata_returns_all_false(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = _nodata_mask(arr, None)
        assert not mask.any()

    def test_nan_nodata_masks_nan_cells(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = _nodata_mask(arr, np.nan)
        assert mask[0, 1]
        assert not mask[0, 0]

    def test_integer_sentinel_masks_matching_cells(self):
        arr = np.array([[101, 32767], [103, 32767]], dtype=np.int16)
        mask = _nodata_mask(arr, 32767)
        assert mask[0, 1]
        assert mask[1, 1]
        assert not mask[0, 0]

    def test_float_sentinel_masks_matching_cells(self):
        arr = np.array([[1.0, -999999.0], [3.0, 4.0]])
        mask = _nodata_mask(arr, -999999.0)
        assert mask[0, 1]
        assert not mask[0, 0]


# ---------------------------------------------------------------------------
# _summarize_continuous
# ---------------------------------------------------------------------------


class TestSummarizeContinuous:
    """_summarize_continuous computes correct stats for continuous bands."""

    def test_basic_stats(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        da = _make_da(data)
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert result["type"] == "continuous"
        assert result["count"] == 4
        assert result["nodata_count"] == 0
        assert result["min"] == 1.0
        assert result["max"] == 4.0
        assert math.isclose(result["mean"], 2.5)
        assert result["std"] is not None

    def test_nan_nodata_excluded(self):
        data = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
        da = _make_da(data, nodata=np.float32("nan"))
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert result["count"] == 2
        assert result["nodata_count"] == 2
        assert result["min"] == 1.0
        assert result["max"] == 3.0

    def test_integer_sentinel_excluded(self):
        data = np.array([[100, 200], [300, 32767]], dtype=np.int16)
        da = _make_da(data, nodata=32767)
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert result["count"] == 3
        assert result["nodata_count"] == 1
        assert result["max"] == 300.0

    def test_float_sentinel_excluded(self):
        data = np.array([[100.0, -999999.0], [300.0, 400.0]], dtype=np.float32)
        da = _make_da(data, nodata=-999999.0)
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert result["count"] == 3
        assert result["nodata_count"] == 1
        assert result["min"] == 100.0

    def test_all_nodata_returns_none_scalars(self):
        data = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
        da = _make_da(data, nodata=np.float32("nan"))
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert result["count"] == 0
        assert result["nodata_count"] == 4
        assert result["min"] is None
        assert result["max"] is None
        assert result["mean"] is None
        assert result["std"] is None

    def test_std_is_zero_for_constant_array(self):
        data = np.full((4, 4), 5.0, dtype=np.float32)
        da = _make_da(data)
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert math.isclose(result["std"], 0.0, abs_tol=1e-6)

    def test_mean_matches_numpy(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, (10, 10)).astype(np.float32)
        da = _make_da(data)
        result = _summarize_continuous(da, chunk_shape=(512, 512))

        assert math.isclose(result["mean"], float(data.mean()), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# _summarize_categorical
# ---------------------------------------------------------------------------


class TestSummarizeCategorical:
    """_summarize_categorical computes correct stats for categorical bands."""

    def test_basic_stats(self):
        data = np.array([[101, 102], [103, 101]], dtype=np.int16)
        da = _make_da(data)
        result = _summarize_categorical(da, chunk_shape=(512, 512))

        assert result["type"] == "categorical"
        assert result["count"] == 4
        assert result["nodata_count"] == 0
        assert result["unique_count"] == 3

    def test_integer_sentinel_excluded(self):
        data = np.array([[101, 32767], [103, 32767]], dtype=np.int16)
        da = _make_da(data, nodata=32767)
        result = _summarize_categorical(da, chunk_shape=(512, 512))

        assert result["count"] == 2
        assert result["nodata_count"] == 2
        assert result["unique_count"] == 2

    def test_zero_sentinel_excluded(self):
        data = np.array([[1, 0], [2, 0]], dtype=np.int64)
        da = _make_da(data, nodata=0)
        result = _summarize_categorical(da, chunk_shape=(512, 512))

        assert result["count"] == 2
        assert result["nodata_count"] == 2
        assert result["unique_count"] == 2

    def test_no_nodata_counts_all_cells(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.int16)
        da = _make_da(data)
        result = _summarize_categorical(da, chunk_shape=(512, 512))

        assert result["count"] == 4
        assert result["nodata_count"] == 0

    def test_all_nodata_returns_zero_counts(self):
        data = np.full((3, 3), 32767, dtype=np.int16)
        da = _make_da(data, nodata=32767)
        result = _summarize_categorical(da, chunk_shape=(512, 512))

        assert result["count"] == 0
        assert result["nodata_count"] == 9
        assert result["unique_count"] == 0


# ---------------------------------------------------------------------------
# summarize_dataset
# ---------------------------------------------------------------------------


class TestSummarizeDataset:
    """summarize_dataset dispatches correctly and returns keyed results."""

    def test_continuous_band(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        da = _make_da(data)
        ds = _make_ds({"elevation": da})
        result = summarize_dataset(
            ds, [{"key": "elevation", "type": "continuous"}], chunk_shape=(512, 512)
        )

        assert "elevation" in result
        assert result["elevation"]["type"] == "continuous"
        assert result["elevation"]["count"] == 4

    def test_categorical_band(self):
        data = np.array([[101, 102], [103, 101]], dtype=np.int16)
        da = _make_da(data, nodata=32767)
        ds = _make_ds({"fbfm": da})
        result = summarize_dataset(
            ds, [{"key": "fbfm", "type": "categorical"}], chunk_shape=(512, 512)
        )

        assert "fbfm" in result
        assert result["fbfm"]["type"] == "categorical"
        assert result["fbfm"]["unique_count"] == 3

    def test_multiple_bands(self):
        elev = _make_da(np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32))
        fbfm = _make_da(np.array([[101, 102], [103, 101]], dtype=np.int16))
        ds = _make_ds({"elevation": elev, "fbfm": fbfm})
        bands = [
            {"key": "elevation", "type": "continuous"},
            {"key": "fbfm", "type": "categorical"},
        ]
        result = summarize_dataset(ds, bands, chunk_shape=(512, 512))

        assert set(result.keys()) == {"elevation", "fbfm"}

    def test_unknown_band_type_raises(self):
        da = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        ds = _make_ds({"elevation": da})
        with pytest.raises(ValueError, match="Unknown band type"):
            summarize_dataset(
                ds, [{"key": "elevation", "type": "unknown"}], chunk_shape=(512, 512)
            )

    def test_layerset_dot_notation_key(self):
        """Pre-#672 layerset keys resolve to variable.sel(band=...)."""
        data = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
        )
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={
                "band": ["loading", "height"],
                "y": [1.0, 0.0],
                "x": [0.0, 1.0],
            },
        )
        da = da.rio.write_crs("EPSG:5070")
        da = da.rio.write_transform(Affine(10, 0, 0, 0, -10, 0))
        ds = _make_ds({"herb": da})
        bands = [
            {"key": "herb.loading", "type": "continuous"},
            {"key": "herb.height", "type": "continuous"},
        ]
        result = summarize_dataset(ds, bands, chunk_shape=(512, 512))

        assert "herb.loading" in result
        assert "herb.height" in result
        assert result["herb.loading"]["count"] == 4
        assert result["herb.loading"]["min"] == 1.0
        assert result["herb.height"]["min"] == 5.0

    def test_legacy_layerset_key_with_dotted_fuel_type(self):
        """Pre-#672 keys split on the last dot: properties never contain one."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": ["loading"], "y": [1.0, 0.0], "x": [0.0, 1.0]},
        )
        da = da.rio.write_crs("EPSG:5070")
        ds = _make_ds({"tl.3": da})
        bands = [{"key": "tl.3.loading", "type": "continuous"}]
        result = summarize_dataset(ds, bands, chunk_shape=(512, 512))
        assert result["tl.3.loading"]["min"] == 1.0

    def test_flat_layerset_key(self):
        """Layerset grids from #672 on store each band as its own variable."""
        da = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        ds = _make_ds({"fuel_load.herb": da})
        result = summarize_dataset(
            ds,
            [{"key": "fuel_load.herb", "type": "continuous"}],
            chunk_shape=(512, 512),
        )
        assert result["fuel_load.herb"]["max"] == 4.0

    @pytest.mark.parametrize(
        "listed_key",
        [
            # Pre-#682 Fosberg key vs the post-#682 output (#727).
            "fuel_moisture.dead.1hr",
            "elevation",
        ],
    )
    def test_band_missing_from_data_raises_processing_error(self, listed_key):
        da = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        ds = _make_ds({"fuel_moisture.1hr": da})
        with pytest.raises(ProcessingError) as exc:
            summarize_dataset(
                ds,
                [{"key": listed_key, "type": "continuous"}],
                chunk_shape=(512, 512),
            )
        assert exc.value.code == "BAND_NOT_PRODUCED"
        assert listed_key in exc.value.message

    def test_legacy_layerset_key_missing_band_coord_raises_processing_error(self):
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": ["loading"], "y": [1.0, 0.0], "x": [0.0, 1.0]},
        )
        ds = _make_ds({"herb": da.rio.write_crs("EPSG:5070")})
        with pytest.raises(ProcessingError) as exc:
            summarize_dataset(
                ds,
                [{"key": "herb.height", "type": "continuous"}],
                chunk_shape=(512, 512),
            )
        assert exc.value.code == "BAND_NOT_PRODUCED"
