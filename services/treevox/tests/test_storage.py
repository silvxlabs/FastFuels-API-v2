"""Unit tests for treevox.storage — xarray-backed 3D zarr I/O.

Uses local zarr DirectoryStore (tmp_path) rather than GCS.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from treevox import storage

# Helpers


def _init(
    tmp_path,
    nx=20,
    ny=20,
    nz=4,
    hr=1.0,
    vr=1.0,
    crs="EPSG:32610",
    keys=("volume_fraction",),
    chunk_shape=None,
):
    """Create a zarr store in tmp_path and return its path."""
    path = str(tmp_path / "grid.zarr")
    x_coords = np.arange(nx, dtype="float64") * hr + hr / 2
    y_coords = ny * hr - (np.arange(ny, dtype="float64") * hr + hr / 2)
    z_coords = np.arange(nz, dtype="float64") * vr + vr / 2
    if chunk_shape is None:
        chunk_shape = (nz, min(10, ny), min(10, nx))
    storage.init_store(
        path, x_coords, y_coords, z_coords, hr, vr, crs, 0.0, list(keys), chunk_shape
    )
    return path


# BAND_SPECS


class TestBandSpecs:
    def test_keys_match_tree_bands(self):
        expected = {
            "volume_fraction",
            "bulk_density.foliage.live",
            "bulk_density.foliage.dead",
            "bulk_density.branchwood.live",
            "bulk_density.branchwood.dead",
            "bulk_density.fine.live",
            "bulk_density.fine.dead",
            "leaf_area_density",
            "savr.foliage",
            "fuel_moisture.live",
            "fuel_moisture.dead",
            "spcd",
            "tree_id",
        }
        assert set(storage.BAND_SPECS) == expected

    @pytest.mark.parametrize(
        "key,dtype,fill",
        [
            ("volume_fraction", "float32", 0.0),
            ("bulk_density.foliage.live", "float32", 0.0),
            ("bulk_density.foliage.dead", "float32", 0.0),
            ("bulk_density.branchwood.live", "float32", 0.0),
            ("bulk_density.branchwood.dead", "float32", 0.0),
            ("bulk_density.fine.live", "float32", 0.0),
            ("bulk_density.fine.dead", "float32", 0.0),
            ("savr.foliage", "float32", 0.0),
            ("fuel_moisture.live", "float32", 0.0),
            ("fuel_moisture.dead", "float32", 0.0),
            ("spcd", "uint16", 0),
            ("tree_id", "int32", -1),
        ],
    )
    def test_per_band_dtype_and_fill(self, key, dtype, fill):
        assert storage.BAND_SPECS[key] == (dtype, fill)

    def test_additive_bands_are_known_band_specs(self):
        assert storage.ADDITIVE_BANDS <= set(storage.BAND_SPECS)


# init_store


class TestInitStore:
    def test_creates_requested_bands_with_correct_dtype_and_fill(self, tmp_path):
        path = _init(tmp_path, keys=("volume_fraction", "tree_id", "spcd"))

        ds = xr.open_zarr(path, consolidated=False, decode_coords="all")
        keys = ["volume_fraction", "tree_id", "spcd"]
        assert set(keys).issubset(set(ds.data_vars) | set(ds.coords))
        assert ds["volume_fraction"].dtype == np.float32
        assert ds["tree_id"].dtype == np.int32
        assert ds["spcd"].dtype == np.uint16

        assert np.all(ds["volume_fraction"].values == 0.0)
        assert np.all(ds["tree_id"].values == -1)
        assert np.all(ds["spcd"].values == 0)

    def test_shape_matches_dimensions(self, tmp_path):
        path = _init(tmp_path, nx=20, ny=30, nz=5, chunk_shape=(5, 10, 10))
        ds = xr.open_zarr(path, consolidated=False)
        assert ds["volume_fraction"].shape == (5, 30, 20)

    def test_crs_and_attrs_persist(self, tmp_path):
        path = _init(tmp_path, crs="EPSG:32611")

        ds = xr.open_zarr(path, consolidated=False)
        assert "spatial_ref" in set(ds.data_vars) | set(ds.coords)
        spatial_ref = (
            ds["spatial_ref"]
            if "spatial_ref" in ds.data_vars
            else ds.coords["spatial_ref"]
        )
        attrs_str = str(spatial_ref.attrs)
        assert "32611" in attrs_str

        assert ds.attrs["z_origin"] == 0.0
        assert ds.attrs["z_resolution"] == 1.0

    def test_subset_of_bands_only(self, tmp_path):
        """Only requested bands should be created — not all BAND_SPECS entries."""
        path = _init(tmp_path, keys=("volume_fraction",))

        ds = xr.open_zarr(path, consolidated=False, decode_coords="all")
        requested_bands_present = set(ds.data_vars) & set(storage.BAND_SPECS)
        assert requested_bands_present == {"volume_fraction"}

    def test_writes_consolidated_metadata(self, tmp_path):
        """init_store must write `.zmetadata` so subsequent reads can use
        `consolidated=True` without a directory listing per batch."""
        path = _init(tmp_path, keys=("volume_fraction",))
        ds = xr.open_zarr(path, consolidated=True)
        assert "volume_fraction" in ds.data_vars


# read_union


class TestReadUnion:
    def test_materializes_into_memory(self, tmp_path):
        """Workers must never receive lazy dask arrays."""
        path = _init(tmp_path, nx=40, ny=40, chunk_shape=(4, 20, 20))

        ds = storage.read_union(path, slice(0, 20), slice(0, 20))
        assert not ds.chunks
        assert isinstance(ds["volume_fraction"].variable.data, np.ndarray)

    def test_returns_correct_slice(self, tmp_path):
        path = _init(tmp_path, nx=40, ny=40, chunk_shape=(4, 20, 20))

        ds = storage.read_union(path, slice(5, 15), slice(10, 30))
        assert ds["volume_fraction"].shape == (4, 10, 20)


# write_union


class TestWriteUnion:
    def test_misaligned_region_succeeds_via_align_chunks(self, tmp_path):
        """Halo unions never align with on-disk chunks.

        Writing a region that crosses chunk boundaries (e.g. [5:25] when
        chunks are 20 wide) must succeed because align_chunks=True rechunks
        the write.
        """
        path = _init(tmp_path, nx=40, ny=40, nz=4, chunk_shape=(4, 20, 20))
        ds = storage.read_union(path, slice(5, 25), slice(5, 25))
        ds["volume_fraction"].values[:] = 1.0
        storage.write_union(path, ds, slice(5, 25), slice(5, 25))

        readback = xr.open_zarr(path, consolidated=False)
        assert np.all(
            readback["volume_fraction"].isel(y=slice(5, 25), x=slice(5, 25)).values
            == 1.0
        )

    def test_drops_coord_variables(self, tmp_path):
        """Region writes must not overwrite index coordinates."""
        path = _init(tmp_path, nx=40, ny=40, nz=4, chunk_shape=(4, 20, 20))
        ds = storage.read_union(path, slice(0, 20), slice(0, 20))

        orig_x = xr.open_zarr(path, consolidated=False).x.values.copy()
        orig_y = xr.open_zarr(path, consolidated=False).y.values.copy()

        ds = ds.assign_coords(x=ds.x.values + 1000, y=ds.y.values + 1000)
        ds["volume_fraction"].values[:] = 2.0
        storage.write_union(path, ds, slice(0, 20), slice(0, 20))

        readback = xr.open_zarr(path, consolidated=False)
        np.testing.assert_array_equal(readback.x.values, orig_x)
        np.testing.assert_array_equal(readback.y.values, orig_y)
        assert np.all(
            readback["volume_fraction"].isel(y=slice(0, 20), x=slice(0, 20)).values
            == 2.0
        )

    def test_round_trip_preserves_per_band_dtypes(self, tmp_path):
        path = _init(
            tmp_path,
            nx=40,
            ny=40,
            nz=4,
            keys=("volume_fraction", "tree_id", "spcd"),
            chunk_shape=(4, 20, 20),
        )

        ds = storage.read_union(path, slice(0, 20), slice(0, 20))
        ds["volume_fraction"].values[:] = 3.14
        ds["tree_id"].values[:] = 42
        ds["spcd"].values[:] = 202
        storage.write_union(path, ds, slice(0, 20), slice(0, 20))

        readback = xr.open_zarr(path, consolidated=False)
        assert readback["volume_fraction"].dtype == np.float32
        assert readback["tree_id"].dtype == np.int32
        assert readback["spcd"].dtype == np.uint16
        assert readback["volume_fraction"].values.max() == np.float32(3.14)
        assert readback["tree_id"].values.max() == 42
        assert readback["spcd"].values.max() == 202


# masked_merge


class TestMaskedMerge:
    def _union_ds(
        self, nz=2, ny=10, nx=10, keys=("spcd", "tree_id", "volume_fraction")
    ):
        data_vars = {}
        for k in keys:
            dtype, fill = storage.BAND_SPECS[k]
            data_vars[k] = (
                ("z", "y", "x"),
                np.full((nz, ny, nx), fill, dtype=dtype),
            )
        return xr.Dataset(data_vars)

    def test_tree_id_fill_minus_one_does_not_overwrite_real_zero(self):
        """v1 used `data > 0` which misclassifies tree_id=0 as fill."""
        union = self._union_ds(keys=("tree_id",))
        buffer = np.full((2, 5, 5), -1, dtype="int32")
        buffer[:, 0, 0] = 0

        result = {
            "chunk_location": (0, 0),
            "y_slice": slice(0, 5),
            "x_slice": slice(0, 5),
            "buffers": {"tree_id": buffer},
        }
        merged = storage.masked_merge(union, [result], slice(0, 10), slice(0, 10))

        assert merged["tree_id"].values[0, 0, 0] == 0
        assert merged["tree_id"].values[0, 0, 1] == -1

    def test_spcd_fill_zero_does_not_get_overwritten_by_real_spcd_zero(self):
        """With `data != fill_value` mask, spcd=0 (real) is distinguishable.

        Since fill=0 and a legitimate species code of 0 are value-identical,
        `data != fill_value` gives `False` for both. The invariant we assert
        is narrower: real spcd=N (N>0) writes overwrite; real spcd=0 writes do
        not (can't differentiate from fill). Documents the limitation.
        """
        union = self._union_ds(keys=("spcd",))
        buffer = np.zeros((2, 5, 5), dtype="uint16")
        buffer[:, 0, 0] = 131

        result = {
            "chunk_location": (0, 0),
            "y_slice": slice(0, 5),
            "x_slice": slice(0, 5),
            "buffers": {"spcd": buffer},
        }
        merged = storage.masked_merge(union, [result], slice(0, 10), slice(0, 10))

        assert merged["spcd"].values[0, 0, 0] == 131
        assert merged["spcd"].values[0, 0, 1] == 0

    def test_relative_slice_placement(self):
        """Results' absolute slices are translated to union-relative positions."""
        union = self._union_ds(nz=1, ny=10, nx=10, keys=("volume_fraction",))

        buf_a = np.zeros((1, 5, 5), dtype="float32")
        buf_a[:, 2, 2] = 7.0
        result_a = {
            "chunk_location": (0, 0),
            "y_slice": slice(0, 5),
            "x_slice": slice(0, 5),
            "buffers": {"volume_fraction": buf_a},
        }

        buf_b = np.zeros((1, 5, 5), dtype="float32")
        buf_b[:, 1, 1] = 9.0
        result_b = {
            "chunk_location": (0, 1),
            "y_slice": slice(0, 5),
            "x_slice": slice(5, 10),
            "buffers": {"volume_fraction": buf_b},
        }

        merged = storage.masked_merge(
            union, [result_a, result_b], slice(0, 10), slice(0, 10)
        )
        assert merged["volume_fraction"].values[0, 2, 2] == 7.0
        assert merged["volume_fraction"].values[0, 1, 6] == 9.0

    def test_overlapping_additive_halos_sum_worker_deltas(self):
        """Adjacent chunks start from the same union, so only deltas should add."""
        union = self._union_ds(nz=1, ny=6, nx=6, keys=("volume_fraction",))
        union["volume_fraction"].values[0, 2, 2] = 5.0

        buf_a = union["volume_fraction"].values[:, 0:5, 0:5].copy()
        buf_a[0, 2, 2] += 2.0
        result_a = {
            "chunk_location": (0, 0),
            "y_slice": slice(0, 5),
            "x_slice": slice(0, 5),
            "buffers": {"volume_fraction": buf_a},
        }

        buf_b = union["volume_fraction"].values[:, 0:5, 2:6].copy()
        buf_b[0, 2, 0] += 3.0
        result_b = {
            "chunk_location": (0, 1),
            "y_slice": slice(0, 5),
            "x_slice": slice(2, 6),
            "buffers": {"volume_fraction": buf_b},
        }

        merged = storage.masked_merge(
            union, [result_a, result_b], slice(0, 6), slice(0, 6)
        )

        assert merged["volume_fraction"].values[0, 2, 2] == 10.0

    def test_overwrite_halo_carryover_does_not_revert_previous_worker(self):
        """An untouched halo carrying pre-batch data should not overwrite a write."""
        union = self._union_ds(nz=1, ny=6, nx=6, keys=("spcd",))
        union["spcd"].values[0, 2, 2] = 131

        buf_a = union["spcd"].values[:, 0:5, 0:5].copy()
        buf_a[0, 2, 2] = 202
        result_a = {
            "chunk_location": (0, 0),
            "y_slice": slice(0, 5),
            "x_slice": slice(0, 5),
            "buffers": {"spcd": buf_a},
        }

        buf_b = union["spcd"].values[:, 0:5, 2:6].copy()
        result_b = {
            "chunk_location": (0, 1),
            "y_slice": slice(0, 5),
            "x_slice": slice(2, 6),
            "buffers": {"spcd": buf_b},
        }

        merged = storage.masked_merge(
            union, [result_a, result_b], slice(0, 6), slice(0, 6)
        )

        assert merged["spcd"].values[0, 2, 2] == 202

    def test_worker_error_raises(self):
        union = self._union_ds(keys=("volume_fraction",))
        result = {"chunk_location": (0, 0), "error": "boom"}
        with pytest.raises(RuntimeError, match="boom"):
            storage.masked_merge(union, [result], slice(0, 10), slice(0, 10))


# consolidate_metadata


class TestConsolidateMetadata:
    def test_store_is_readable_with_consolidated_metadata(self, tmp_path):
        path = _init(tmp_path, nx=20, ny=20, nz=2, chunk_shape=(2, 10, 10))
        storage.consolidate_metadata(path)

        ds = xr.open_zarr(path, consolidated=True)
        assert "volume_fraction" in ds.data_vars


# delete_zarr


class TestDeleteZarr:
    def test_best_effort_on_missing_path(self, monkeypatch):
        """delete_zarr wraps delete_directory in try/except; never raises."""

        def raising_delete(path):
            raise FileNotFoundError(path)

        monkeypatch.setattr(storage, "delete_directory", raising_delete)

        storage.delete_zarr("gs://nonexistent/path")

    def test_delegates_to_delete_directory(self, monkeypatch):
        calls = []
        monkeypatch.setattr(storage, "delete_directory", lambda p: calls.append(p))

        storage.delete_zarr("gs://my-bucket/my-grid")
        assert calls == ["gs://my-bucket/my-grid"]


# gcs_path


class TestGcsPath:
    def test_builds_uri_from_grids_bucket(self):
        path = storage.gcs_path("abc123")
        assert path.startswith("gs://")
        assert path.endswith("/abc123")
