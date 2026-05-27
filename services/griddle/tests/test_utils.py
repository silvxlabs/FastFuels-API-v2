"""
Unit tests for griddle.utils — infer_nodata and to_dataset.
"""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from griddle.utils import infer_nodata, to_dataset


def _make_da(dtype, nodata=None, with_crs=True, with_transform=True):
    """Build a minimal 2D DataArray for testing."""
    data = np.zeros((4, 4), dtype=dtype)
    da = xr.DataArray(data, dims=["y", "x"])
    if with_crs:
        da = da.rio.write_crs("EPSG:32612")
    if with_transform:
        da = da.rio.write_transform(Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


class TestInferNodataDtypeOnly:
    def test_float32_returns_float32_nan(self):
        result = infer_nodata(np.dtype("float32"))
        assert np.isnan(result)
        assert np.dtype(type(result)) == np.dtype("float32")

    def test_float64_returns_float64_nan(self):
        result = infer_nodata(np.dtype("float64"))
        assert np.isnan(result)
        assert np.dtype(type(result)) == np.dtype("float64")

    def test_int16_returns_dtype_max(self):
        result = infer_nodata(np.dtype("int16"))
        assert result == np.iinfo(np.int16).max

    def test_int32_returns_dtype_max(self):
        result = infer_nodata(np.dtype("int32"))
        assert result == np.iinfo(np.int32).max

    def test_uint32_returns_dtype_max(self):
        result = infer_nodata(np.dtype("uint32"))
        assert result == np.iinfo(np.uint32).max

    def test_int64_returns_dtype_max(self):
        result = infer_nodata(np.dtype("int64"))
        assert result == np.iinfo(np.int64).max

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Cannot infer nodata"):
            infer_nodata(np.dtype("complex64"))

    def test_accepts_dtype_string(self):
        """infer_nodata should accept dtype strings as well as np.dtype objects."""
        result = infer_nodata("float32")
        assert np.isnan(result)


class TestInferNodataWithDataArray:
    def test_returns_existing_nodata_cast_to_dtype(self):
        """If da.rio.nodata is already set, return it cast to the array dtype."""
        da = _make_da("int16", nodata=np.int16(32767))
        result = infer_nodata(da.dtype, da)
        assert result == np.int16(32767)
        assert np.dtype(type(result)) == np.dtype("int16")

    def test_falls_back_to_dtype_inference_when_nodata_none(self):
        da = _make_da("int16")  # no nodata
        result = infer_nodata(da.dtype, da)
        assert result == np.iinfo(np.int16).max

    def test_float32_existing_nodata_preserved(self):
        da = _make_da("float32", nodata=-999999.0)
        result = infer_nodata(da.dtype, da)
        assert result == np.float32(-999999.0)

    def test_none_dataarray_falls_through_to_dtype(self):
        result = infer_nodata(np.dtype("int32"), None)
        assert result == np.iinfo(np.int32).max


class TestToDatasetValidation:
    def test_empty_variables_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            to_dataset({})

    def test_missing_nodata_raises(self):
        da = _make_da("float32")  # no nodata
        with pytest.raises(ValueError, match="have no nodata set"):
            to_dataset({"elevation": da})

    def test_missing_nodata_lists_all_offending_variables(self):
        da1 = _make_da("float32")
        da2 = _make_da("int16")
        with pytest.raises(ValueError) as exc_info:
            to_dataset({"elevation": da1, "slope": da2})
        assert "elevation" in str(exc_info.value)
        assert "slope" in str(exc_info.value)

    def test_missing_crs_raises(self):
        da = _make_da("float32", nodata=np.float32("nan"), with_crs=False)
        with pytest.raises(ValueError, match="No CRS"):
            to_dataset({"elevation": da})

    def test_missing_transform_raises(self):
        da = _make_da("float32", nodata=np.float32("nan"), with_transform=False)
        with pytest.raises(ValueError, match="No transform"):
            to_dataset({"elevation": da})


class TestToDatasetOutput:
    def test_returns_dataset(self):
        da = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da})
        assert isinstance(result, xr.Dataset)

    def test_variables_present(self):
        da = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da})
        assert "elevation" in result.data_vars

    def test_crs_written(self):
        da = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da})
        assert result.rio.crs is not None
        assert result.rio.crs.to_epsg() == 32612

    def test_transform_written(self):
        da = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da})
        assert result.rio.transform() is not None

    def test_nodata_in_attrs(self):
        """Nodata is in da.attrs after to_dataset."""
        da = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da})
        assert "_FillValue" in result["elevation"].attrs

    def test_nodata_value_correct(self):
        da = _make_da("int16", nodata=np.int16(32767))
        result = to_dataset({"fbfm": da})
        assert result["fbfm"].attrs["_FillValue"] == 32767

    def test_multiple_variables(self):
        da1 = _make_da("float32", nodata=np.float32("nan"))
        da2 = _make_da("float32", nodata=np.float32("nan"))
        result = to_dataset({"elevation": da1, "slope": da2})
        assert set(result.data_vars) == {"elevation", "slope"}
        for var in result.data_vars:
            assert "_FillValue" in result[var].attrs


class TestToDatasetExplicitSpatialMetadata:
    def test_explicit_crs_used(self):
        """CRS passed explicitly should override missing CRS on DataArrays."""
        da = _make_da(
            "float32", nodata=np.float32("nan"), with_crs=False, with_transform=True
        )
        result = to_dataset({"band": da}, crs="EPSG:32612")
        assert result.rio.crs.to_epsg() == 32612

    def test_explicit_transform_used(self):
        """Transform passed explicitly should override missing transform on DataArrays."""
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        da = _make_da(
            "float32", nodata=np.float32("nan"), with_crs=True, with_transform=False
        )
        result = to_dataset({"band": da}, transform=transform)
        assert result.rio.transform() == transform

    def test_missing_crs_with_no_explicit_raises(self):
        da = _make_da(
            "float32", nodata=np.float32("nan"), with_crs=False, with_transform=True
        )
        with pytest.raises(ValueError, match="No CRS"):
            to_dataset({"band": da})
