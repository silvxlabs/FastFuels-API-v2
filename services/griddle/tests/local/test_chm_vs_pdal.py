"""Cross-check the point-cloud CHM against PDAL. Local only.

Griddle deliberately has no PDAL — see LAKITU.md on why the native stack is not
worth hosting — so the ground filter and the height-above-ground rasterisation
are our own implementations of published algorithms. This test is what keeps
them honest: it runs the same cloud through both and reports the difference.

Three comparisons, each on a real 3DEP cloud:

- **ground, classified** — our minimum surface over ASPRS class 2 against
  ``filters.range`` plus ``writers.gdal`` with ``output_type=min``. Both are a
  per-cell minimum over the same returns, so agreement should be near exact and
  any difference is our gap filling.
- **ground, derived** — our progressive morphological filter against
  ``filters.pmf`` at the same parameters. This is the comparison the handler's
  docstring claims and never checked.
- **canopy height** — our CHM against ``filters.hag_delaunay`` plus a per-cell
  maximum. These are *not* the same algorithm: PDAL triangulates the ground
  returns, we rasterise them and sample bilinearly. A difference is expected;
  the question is whether it is small.

GeoTIFFs of both surfaces and their difference are written to
``--pdal-output`` (default: a temp directory, printed at the end) so a
disagreement can be looked at rather than argued about.

Running it::

    conda create -y -n pdal-compare -c conda-forge pdal
    conda run -n pdal-compare pdal --version           # confirm it works
    PDAL_BIN="$(conda run -n pdal-compare which pdal)" \\
        uv run --active pytest tests/local -v -s

The cloud is downloaded from GCS once and cached, so the first run needs
credentials and a few minutes.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from griddle.handlers import chm_point_cloud

PDAL_BIN = os.environ.get("PDAL_BIN") or shutil.which("pdal")

pytestmark = [
    pytest.mark.local,
    pytest.mark.skipif(
        not PDAL_BIN, reason="needs a local PDAL; set PDAL_BIN or put it on PATH"
    ),
]

# The static integration cloud: 10.5M points over the Blackfoot domain, carrying
# real ASPRS ground classification.
SOURCE_CLOUD = "static-test-blackfoot-3dep/cloud.laz"

# A window of it, small enough that PDAL's Delaunay pass stays quick and large
# enough to hold real terrain and canopy.
SAMPLE_EDGE_M = 400.0

# 1 m, so our metre-derived filter windows land on PDAL's cell-count defaults
# (max_window_size 33, cell_size 1) with nothing to reconcile.
RESOLUTION = 1.0

CACHE = Path(os.environ.get("PDAL_COMPARE_CACHE", "/tmp/griddle-pdal-compare"))


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_laz():
    """A cropped copy of the static cloud, cached between runs."""
    import laspy

    from lib.config import POINT_CLOUDS_BUCKET
    from lib.gcs import get_gcsfs_client

    CACHE.mkdir(parents=True, exist_ok=True)
    cropped = CACHE / "sample.laz"
    if cropped.exists():
        return cropped

    whole = CACHE / "whole.laz"
    if not whole.exists():
        source = f"{POINT_CLOUDS_BUCKET}/{SOURCE_CLOUD}"
        print(f"\ndownloading {source} (once)")
        with get_gcsfs_client().open(source, "rb") as stream:
            whole.write_bytes(stream.read())

    las = laspy.read(str(whole))
    # Anchor the window on whole metres so the lattice has no fractional offset
    # to reconcile between the two implementations.
    min_x = np.floor(np.min(las.x)) + 50.0
    min_y = np.floor(np.min(las.y)) + 50.0
    keep = (
        (las.x >= min_x)
        & (las.x < min_x + SAMPLE_EDGE_M)
        & (las.y >= min_y)
        & (las.y < min_y + SAMPLE_EDGE_M)
    )
    subset = laspy.LasData(las.header)
    subset.points = las.points[keep]
    subset.write(str(cropped))
    print(f"cropped to {keep.sum():,} points at ({min_x}, {min_y})")
    return cropped


@pytest.fixture(scope="module")
def cloud_dataset(sample_laz, tmp_path_factory):
    """The cropped cloud in the partitioned Parquet layout griddle reads."""
    import laspy
    import pyarrow as pa
    import pyarrow.parquet as pq

    las = laspy.read(str(sample_laz))
    root = tmp_path_factory.mktemp("cloud") / "cloud.parquet"
    tile_m = 500.0
    mins = [float(np.min(las.x)), float(np.min(las.y)), float(np.min(las.z))]
    maxs = [float(np.max(las.x)), float(np.max(las.y)), float(np.max(las.z))]
    scales, offsets = [0.001] * 3, mins

    raw = {
        "X": np.round((np.asarray(las.x) - offsets[0]) / scales[0]).astype(np.int32),
        "Y": np.round((np.asarray(las.y) - offsets[1]) / scales[1]).astype(np.int32),
        "Z": np.round((np.asarray(las.z) - offsets[2]) / scales[2]).astype(np.int32),
        "classification": np.asarray(las.classification).astype(np.uint8),
    }
    tile_x = ((np.asarray(las.x) - mins[0]) / tile_m).astype(np.int32)
    tile_y = ((np.asarray(las.y) - mins[1]) / tile_m).astype(np.int32)

    for tx in np.unique(tile_x):
        for ty in np.unique(tile_y):
            member = (tile_x == tx) & (tile_y == ty)
            if not member.any():
                continue
            directory = root / f"tile_x={tx}" / f"tile_y={ty}"
            directory.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "lod": pa.array(np.zeros(member.sum(), np.uint8)),
                    "X": pa.array(raw["X"][member]),
                    "Y": pa.array(raw["Y"][member]),
                    "Z": pa.array(raw["Z"][member]),
                    "intensity": pa.array(np.zeros(member.sum(), np.uint16)),
                    "classification": pa.array(raw["classification"][member]),
                }
            )
            pq.write_table(table, directory / "part-00000.parquet")

    manifest = {
        "tile_m": tile_m,
        "mins": mins,
        "maxs": maxs,
        "scales": scales,
        "offsets": offsets,
    }
    (root / "_manifest.json").write_text(json.dumps(manifest))
    return root, manifest


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    directory = Path(
        os.environ.get("PDAL_COMPARE_OUTPUT") or tmp_path_factory.mktemp("compare")
    )
    directory.mkdir(parents=True, exist_ok=True)
    print(f"\nGeoTIFFs -> {directory}")
    return directory


# --- running the two implementations ----------------------------------------


def _run_ours(cloud_dataset, point_classes):
    """Our handler, reading the local dataset instead of GCS."""
    import geopandas as gpd
    from shapely.geometry import box

    root, manifest = cloud_dataset
    mins, maxs = manifest["mins"], manifest["maxs"]
    roi = gpd.GeoDataFrame(
        geometry=[box(mins[0], mins[1], maxs[0], maxs[1])], crs="EPSG:32612"
    )

    captured = []
    real_fill = chm_point_cloud._fill_gaps

    def capture_ground(surface, max_cells):
        # The ground handed to the height pass, which is what PDAL's is
        # comparable to. Captured rather than recomputed so the comparison sees
        # exactly what produced the CHM.
        #
        # Every call is kept and the whole-grid one picked out afterwards: on the
        # derived path this also runs inside `map_overlap`, once per block and
        # once on a zero-size array dask uses to infer the dtype.
        filled = real_fill(surface, max_cells)
        captured.append(filled)
        return filled

    with (
        patch.object(chm_point_cloud, "_cloud_prefix", return_value=str(root)),
        patch.object(chm_point_cloud, "get_gcsfs_client", return_value=None),
        patch.object(chm_point_cloud, "_read_manifest", return_value=manifest),
        patch.object(chm_point_cloud, "_fill_gaps", side_effect=capture_ground),
    ):
        dataset, provenance = chm_point_cloud.fetch_point_cloud_chm(
            roi=roi,
            point_cloud_id="local-compare",
            point_classes=point_classes,
            alignment={"target": "domain", "resolution": RESOLUTION},
            progress=lambda *_args, **_kwargs: None,
        )
    shape = dataset["chm"].shape
    whole = [surface for surface in captured if surface.shape == shape]
    assert whole, f"no whole-grid ground captured; saw {[s.shape for s in captured]}"
    return dataset, provenance, whole[-1]


def _pdal(pipeline, workdir):
    """Run a PDAL pipeline, failing loudly with its own diagnostics."""
    path = workdir / "pipeline.json"
    path.write_text(json.dumps(pipeline))
    result = subprocess.run(
        [PDAL_BIN, "pipeline", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"pdal pipeline failed ({result.returncode})\n{result.stderr}"
        )


def _strip_classification():
    """Blank the classification so PDAL derives ground from geometry alone.

    Return numbers go with it: `filters.pmf` refuses a cloud where some points
    carry them and others do not, and blanking the classification on a real
    tile leaves exactly that mix.
    """
    return {
        "type": "filters.assign",
        "value": [
            "Classification = 1",
            "ReturnNumber = 1",
            "NumberOfReturns = 1",
        ],
    }


def _pdal_points(stages, workdir, name, extra_dims=""):
    """Run a pipeline that ends in a LAS file and read the points back.

    PDAL does the filtering; the rasterising is done here, by our code, in both
    arms of every comparison. That is deliberate. `writers.gdal` bins a point
    into every cell whose centre lies within ``radius`` — ``resolution *
    sqrt(2)`` by default — so its per-cell minimum is really a neighbourhood
    minimum and its maximum a neighbourhood maximum. Measured against it, our
    ground read 0.28 m high and our canopy 2.0 m low, none of which is an
    algorithmic difference. Comparing point sets instead isolates the thing
    under test: which returns each implementation calls ground, and what height
    it gives them.
    """
    import laspy

    target = workdir / f"{name}.las"
    writer = {"type": "writers.las", "filename": str(target)}
    if extra_dims:
        writer["extra_dims"] = extra_dims
    _pdal([*stages, writer], workdir)
    return laspy.read(str(target))


def _rasterise(dataset, x, y, values, how):
    """Put values on our lattice with our own cell assignment and reduction."""
    transform = dataset.rio.transform()
    height, width = dataset["chm"].shape
    lattice = (transform.c, transform.f, height, width)
    fill = np.inf if how is np.minimum else -np.inf
    raster = np.full(height * width, fill, dtype=np.float32)
    index, inside = chm_point_cloud._cell_indices(
        np.asarray(x), np.asarray(y), lattice, RESOLUTION
    )
    how.at(raster, index[inside], np.asarray(values)[inside].astype(np.float32))
    raster[~np.isfinite(raster)] = np.nan
    return raster.reshape(height, width)


def _pmf_stage():
    """`filters.pmf` at the parameters our `_pmf` implements."""
    return {
        "type": "filters.pmf",
        "cell_size": RESOLUTION,
        "max_window_size": 33,
        "slope": 1.0,
        "initial_distance": 0.15,
        "max_distance": 2.5,
        "exponential": True,
    }


def _compare(name, ours, theirs, output_dir, reference_dataset):
    """Report the difference, write all three rasters, return the statistics."""
    import rioxarray  # noqa: F401
    import xarray as xr

    assert ours.shape == theirs.shape, f"{name}: {ours.shape} vs {theirs.shape}"

    both = np.isfinite(ours) & np.isfinite(theirs)
    assert both.any(), f"{name}: the two rasters share no cells"
    difference = np.where(both, ours - theirs, np.nan)
    stats = {
        "cells": int(both.sum()),
        "coverage_ours": float(np.isfinite(ours).mean()),
        "coverage_pdal": float(np.isfinite(theirs).mean()),
        "mean_abs": float(np.nanmean(np.abs(difference))),
        "median_abs": float(np.nanmedian(np.abs(difference))),
        "rmse": float(np.sqrt(np.nanmean(difference**2))),
        "p95_abs": float(np.nanpercentile(np.abs(difference), 95)),
        "max_abs": float(np.nanmax(np.abs(difference))),
        "bias": float(np.nanmean(difference)),
        # Over the cells the two share, not over the grid: counting nodata as a
        # disagreement would report coverage, not agreement.
        "within_10cm": float(np.mean(np.abs(difference[both]) < 0.10)),
        "within_25cm": float(np.mean(np.abs(difference[both]) < 0.25)),
        "within_50cm": float(np.mean(np.abs(difference[both]) < 0.50)),
        "over_1m": float(np.mean(np.abs(difference[both]) > 1.0)),
    }

    template = reference_dataset["chm"]
    for label, values in (("ours", ours), ("pdal", theirs), ("diff", difference)):
        raster = xr.DataArray(
            values.astype(np.float32), dims=template.dims, coords=template.coords
        )
        raster = raster.rio.write_crs(reference_dataset.rio.crs)
        raster = raster.rio.write_transform(reference_dataset.rio.transform())
        raster.rio.to_raster(output_dir / f"{name}_{label}.tif")

    print(f"\n[{name}]")
    for key, value in stats.items():
        print(f"  {key:<16}{value:>12.4f}")
    return stats


# --- the comparisons --------------------------------------------------------


class TestGroundSurface:
    """Our ground against PDAL's, which is the input every height depends on."""

    def test_classified_ground_matches_pdal(self, cloud_dataset, output_dir, tmp_path):
        """Same returns, same reduction — so this one has to be exact.

        Nothing algorithmic differs here: both take the lowest class-2 return in
        each cell. What it checks is the path around that — the Parquet reader,
        the millimetre coordinate decoding and the cell assignment — against a
        cloud PDAL read straight from the LAZ.
        """
        dataset, provenance, ground = _run_ours(cloud_dataset, point_classes=[2])
        assert provenance["ground_source"] == "classification"

        points = _pdal_points(
            [
                str(CACHE / "sample.laz"),
                {"type": "filters.range", "limits": "Classification[2:2]"},
            ],
            tmp_path,
            "ground_classified",
        )
        theirs = _rasterise(dataset, points.x, points.y, points.z, np.minimum)

        stats = _compare("ground_classified", ground, theirs, output_dir, dataset)
        # Ours is gap-filled and PDAL's is not, so ours covers more.
        assert stats["coverage_ours"] >= stats["coverage_pdal"]
        # Where both have a real return the cells are identical, save for a
        # handful: we re-quantise coordinates to millimetres on the way into
        # Parquet, and a return sitting exactly on a cell boundary can land on
        # the far side of it afterwards, taking its cell's minimum with it.
        assert stats["median_abs"] == 0.0
        assert stats["p95_abs"] == 0.0
        assert stats["within_10cm"] > 0.994
        assert stats["over_1m"] < 0.0002
        assert abs(stats["bias"]) < 0.002

    def test_derived_ground_matches_pdal_pmf(self, cloud_dataset, output_dir, tmp_path):
        """Our progressive morphological filter against ``filters.pmf`` itself.

        The handler's docstring claims to implement Zhang et al. 2003 at PDAL's
        parameters. This is the only thing that checks the claim.
        """
        dataset, provenance, ground = _run_ours(cloud_dataset, point_classes=[1])
        assert provenance["ground_source"] == "derived"

        points = _pdal_points(
            [
                str(CACHE / "sample.laz"),
                _strip_classification(),
                _pmf_stage(),
                {"type": "filters.range", "limits": "Classification[2:2]"},
            ],
            tmp_path,
            "ground_pmf",
        )
        print(f"\n  pdal pmf kept {len(points.x):,} ground returns")
        theirs = _rasterise(dataset, points.x, points.y, points.z, np.minimum)

        stats = _compare("ground_derived", ground, theirs, output_dir, dataset)
        # Measured 0.00 / 0.04 / 0.285 / 98.3%; these sit just above that, so a
        # real drift in the filter shows up rather than hiding under slack.
        assert stats["median_abs"] == 0.0
        assert stats["p95_abs"] < 0.06
        assert stats["rmse"] < 0.35
        assert stats["within_10cm"] > 0.98
        assert abs(stats["bias"]) < 0.02


# Just above what was measured on this fixture, so a real change in either
# implementation trips them rather than disappearing into slack. Derived ground
# is looser than classified because two ground estimates disagree underneath the
# canopy rather than one.
THRESHOLDS = {
    "chm_classified": {
        "median_abs": 0.10,
        "p95_abs": 0.40,
        "rmse": 0.45,
        "mean_abs": 0.16,
        "bias": 0.15,
    },
    "chm_derived": {
        "median_abs": 0.11,
        "p95_abs": 0.50,
        "rmse": 0.65,
        "mean_abs": 0.22,
        "bias": 0.19,
    },
}


class TestCanopyHeight:
    """The CHM itself, over both ground sources."""

    @pytest.mark.parametrize(
        ("point_classes", "derive_ground", "name"),
        [([2], False, "chm_classified"), ([1], True, "chm_derived")],
    )
    def test_chm_matches_pdal(
        self, cloud_dataset, output_dir, tmp_path, point_classes, derive_ground, name
    ):
        """Not the same algorithm, so the bar is 'negligible', not 'identical'.

        PDAL interpolates ground by triangulating the returns; we rasterise them
        and sample the raster bilinearly. Both then take the tallest return in a
        cell, so the canopy side is common and the difference is whatever the two
        ground surfaces disagree by underneath it.
        """
        dataset, _, _ = _run_ours(cloud_dataset, point_classes=point_classes)
        ours = dataset["chm"].values.astype(np.float64)

        stages = [str(CACHE / "sample.laz")]
        if derive_ground:
            stages += [_strip_classification(), _pmf_stage()]
        stages.append({"type": "filters.hag_delaunay"})
        points = _pdal_points(
            stages, tmp_path, name, extra_dims="HeightAboveGround=float"
        )

        # The same height window our handler applies, so the comparison is of
        # the surfaces rather than of each one's outlier policy.
        height = np.asarray(points.HeightAboveGround)
        keep = (
            (height >= chm_point_cloud.MIN_CANOPY_HEIGHT_M)
            & (height < chm_point_cloud.MAX_CANOPY_HEIGHT_M)
            & np.isfinite(height)
        )
        theirs = _rasterise(
            dataset, points.x[keep], points.y[keep], height[keep], np.maximum
        )

        stats = _compare(name, ours, theirs, output_dir, dataset)
        for metric, limit in THRESHOLDS[name].items():
            value = abs(stats[metric]) if metric == "bias" else stats[metric]
            assert value < limit, f"{metric} {value:.4f} >= {limit}"
        assert stats["within_50cm"] > (0.95 if derive_ground else 0.98)
