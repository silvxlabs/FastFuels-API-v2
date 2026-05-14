"""
Visual verification of alignment lattice geometry.

Renders one PDF per scenario into ``tests/handlers/figures/`` so the
output of the alignment system can be inspected by eye. These are not
assertion tests — the lattice math is asserted in
``test_alignment_handlers.py``, ``services/lib/tests/test_alignment.py``,
and ``services/lib/tests/test_raster.py``. This file exists so a
reviewer can confirm the diagrams match the documented behavior at a
glance.

Every diagram uses production code to derive the output lattice:

- ``target='domain'`` and ``target='grid'`` scenarios call
  ``lib.alignment.resolve_alignment_destination`` directly and read the
  destination transform/shape it returns.
- ``target='native'`` scenarios write a tmp GeoTIFF whose pixel anchor
  is offset from the domain (matching real-world fetches), call
  ``lib.raster.RasterConnection.extract_window``, and read the actual
  output's CRS / transform / shape via rioxarray. This exercises the
  CRS-only-override and default-clip branches end-to-end.

There is no separately-implemented "what I think the output should
look like" code in this file — the diagrams are the production system's
output, plotted.

Skipped unless ``DEPLOYMENT_ENV=local`` (the default for developer
workstations); CI / cloud runners set ``DEPLOYMENT_ENV`` to something
else and skip the file. Also requires the ``viz`` extra
(``uv sync --extra test --extra viz``); skipped automatically when
matplotlib is not installed.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import box

from lib.alignment import resolve_alignment_destination
from lib.config import DEPLOYMENT_ENV
from lib.raster import RasterConnection

pytestmark = [
    pytest.mark.skipif(
        DEPLOYMENT_ENV != "local",
        reason="alignment visualizations only run locally (DEPLOYMENT_ENV=local)",
    ),
]

FIGURES_DIR = Path(__file__).parent / "figures"

_DOMAIN_BOUNDS = (720000.0, 5190000.0, 720600.0, 5190400.0)


def _domain_gdf(crs="EPSG:32611"):
    return gpd.GeoDataFrame(geometry=[box(*_DOMAIN_BOUNDS)], crs=crs)


# The default source anchor is intentionally offset from the domain
# lower-left by a sub-pixel-but-visible amount (77.3 m east, 43.7 m
# north of the source LL relative to the domain LL — about 2.5 source
# cells in each direction). This makes the difference between
# target='native' (preserves source-pixel anchor) and target='domain'
# (snaps to domain LL) visible at the rendered scale rather than
# microscopic. Real-world fetches almost never have a pixel grid that
# coincidentally aligns with a user-defined domain; the visualizations
# should reflect that reality rather than the trivial special case.
_DEFAULT_SOURCE_ANCHOR = (720000.0 - 77.3, 5190000.0 - 43.7)


def _source_lattice(
    source_resolution: float = 30.0,
    anchor: tuple[float, float] = _DEFAULT_SOURCE_ANCHOR,
    extent_meters: tuple[float, float] = (750.0, 600.0),
) -> tuple[Affine, tuple[int, int]]:
    """Build (transform, shape) for a synthetic source raster anchored at
    ``anchor`` (lower-left), spanning ``extent_meters`` (w, h) at the given
    cell size. Used by visualization tests to vary source-pixel anchoring."""
    width = int(round(extent_meters[0] / source_resolution))
    height = int(round(extent_meters[1] / source_resolution))
    transform = from_bounds(
        anchor[0],
        anchor[1],
        anchor[0] + width * source_resolution,
        anchor[1] + height * source_resolution,
        width,
        height,
    )
    return transform, (height, width)


def _write_source_geotiff(
    path: Path,
    *,
    crs: str = "EPSG:32611",
    source_resolution: float = 30.0,
    anchor: tuple[float, float] = None,
    extent_meters: tuple[float, float] = (750.0, 600.0),
) -> tuple[Affine, tuple[int, int]]:
    """Write a small GeoTIFF whose pixel grid is anchored at ``anchor``
    (lower-left). Returns (transform, shape) for plotting. The content is
    just a deterministic ramp — the visualizations only care about the
    geometry, not the pixel values."""
    if anchor is None:
        anchor = _DEFAULT_SOURCE_ANCHOR
    width = int(round(extent_meters[0] / source_resolution))
    height = int(round(extent_meters[1] / source_resolution))
    transform = from_origin(
        anchor[0],
        anchor[1] + height * source_resolution,
        source_resolution,
        source_resolution,
    )
    data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    return transform, (height, width)


def _run_native_extract(
    raster_path: Path,
    domain_gdf: gpd.GeoDataFrame,
    *,
    destination_resolution: float | None = None,
    extent_buffer_cells: int = 0,
) -> tuple[Affine, tuple[int, int]]:
    """Call the production ``RasterConnection.extract_window`` for a
    ``target='native'`` scenario and read back the output's actual
    transform and shape via rioxarray. ``destination_resolution=None``
    exercises the default-clip branch; a value exercises the CRS-only
    override branch. This is the same code path the handlers run."""
    conn = RasterConnection(str(raster_path))
    kwargs: dict = {
        "roi": domain_gdf,
        "interpolation_padding_cells": extent_buffer_cells,
    }
    if destination_resolution is not None:
        kwargs["destination_crs"] = domain_gdf.crs
        kwargs["destination_resolution"] = destination_resolution
    out = conn.extract_window(**kwargs)
    out = out.squeeze("band", drop=True) if "band" in out.dims else out
    h, w = out.rio.shape
    return out.rio.transform(), (h, w)


def _resolve(
    alignment,
    target_grid_doc=None,
    source_resolution=30.0,
    extent_buffer_cells=0,
):
    """Run the alignment helper with the shared visualization domain so
    every figure uses identical inputs."""
    return resolve_alignment_destination(
        alignment,
        _domain_gdf(),
        target_grid_doc,
        source_resolution,
        extent_buffer_cells=extent_buffer_cells,
    )


def _bounds_from_lattice(transform, shape):
    """Return (minx, miny, maxx, maxy) for a north-up lattice."""
    h, w = shape
    minx = transform.c
    maxy = transform.f
    maxx = minx + w * transform.a
    miny = maxy + h * transform.e
    return minx, miny, maxx, maxy


def _shift_bounds(bounds, dx, dy):
    return (bounds[0] - dx, bounds[1] - dy, bounds[2] - dx, bounds[3] - dy)


def _draw_lattice(
    ax, transform, shape, *, origin, color, label, linewidth, alpha, ls="-"
):
    """Draw a lattice's bounding box plus every cell line in domain-relative
    coordinates (subtract ``origin = (minx, miny)`` before plotting)."""
    from matplotlib.patches import Rectangle

    minx, miny, maxx, maxy = _shift_bounds(
        _bounds_from_lattice(transform, shape), *origin
    )
    h, w = shape
    dx = transform.a
    dy = -transform.e

    ax.add_patch(
        Rectangle(
            (minx, miny),
            maxx - minx,
            maxy - miny,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=ls,
            alpha=alpha,
            label=label,
        )
    )

    # Density-aware line styling: a coarse lattice gets bold lines; a
    # very fine lattice gets thinner translucent ones. Same color so the
    # count stays visible when overlaid on a coarser lattice.
    max_dim = max(h, w)
    line_lw = max(0.15, min(0.6, 12.0 / max_dim))
    line_alpha = max(0.18, min(0.55, 18.0 / max_dim)) * alpha

    for i in range(0, w + 1):
        x = minx + i * dx
        ax.plot([x, x], [miny, maxy], color=color, alpha=line_alpha, linewidth=line_lw)
    for j in range(0, h + 1):
        y = miny + j * dy
        ax.plot([minx, maxx], [y, y], color=color, alpha=line_alpha, linewidth=line_lw)


def _render_alignment_diagram(
    out_path,
    *,
    title,
    description,
    domain_bounds,
    layers,
):
    """Render a lattice diagram with all coordinates expressed relative to
    the domain's lower-left corner.

    ``layers`` is a list of dicts each with keys: ``transform`` (Affine),
    ``shape`` (h, w), ``color``, ``label``, ``linewidth``, ``alpha``,
    optional ``ls``. Drawing order matches list order — coarsest first
    so finer lattices overlay cleanly on top.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    domain_minx, domain_miny, domain_maxx, domain_maxy = domain_bounds
    origin = (domain_minx, domain_miny)
    domain_w = domain_maxx - domain_minx
    domain_h = domain_maxy - domain_miny

    fig, ax = plt.subplots(figsize=(11, 7.5))

    for layer in layers:
        _draw_lattice(
            ax,
            layer["transform"],
            layer["shape"],
            origin=origin,
            color=layer["color"],
            label=layer["label"],
            linewidth=layer["linewidth"],
            alpha=layer["alpha"],
            ls=layer.get("ls", "-"),
        )

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            domain_w,
            domain_h,
            fill=False,
            edgecolor="#d62728",
            linewidth=2.5,
            label=f"domain ({domain_w:g}×{domain_h:g} m)",
        )
    )

    ax.set_xlabel("x — meters east of domain lower-left")
    ax.set_ylabel("y — meters north of domain lower-left")
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, framealpha=1.0)
    fig.text(
        0.02,
        0.02,
        description,
        fontsize=9,
        wrap=True,
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.07, right=0.74, top=0.92, bottom=0.16)
    fig.savefig(out_path)
    plt.close(fig)


@pytest.fixture(scope="module")
def figures_dir():
    pytest.importorskip("matplotlib")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


@pytest.fixture(scope="module")
def source_raster(tmp_path_factory):
    """Write a small GeoTIFF with the default offset source anchor and
    yield (path, transform, shape). Used by ``target='native'``
    visualizations to drive the production ``RasterConnection.extract_window``
    end-to-end."""
    path = tmp_path_factory.mktemp("source") / "source.tif"
    transform, shape = _write_source_geotiff(path)
    return path, transform, shape


class TestAlignmentVisualizations:
    """Render one PDF per alignment scenario.

    Coordinates are expressed relative to the domain's lower-left corner
    so axis labels are short integers, not 7-digit UTM eastings. Layers:
    source pixel grid (gray), domain bbox (red), output lattice (green),
    and — for ``target='grid'`` — the target grid (blue dashed). Buffer
    comparisons add a purple "unbuffered" reference layer.
    """

    # ──────────────────────────────────────────────────────────────────
    # The six base scenarios — every combination of target × resolution.
    # ──────────────────────────────────────────────────────────────────

    def test_native_no_resolution(self, figures_dir, source_raster):
        raster_path, src_transform, src_shape = source_raster
        # extract_window's default branch (no destination override): reproject
        # to ROI CRS at the source's native cell size and clip to ROI ± buffer.
        out_transform, out_shape = _run_native_extract(raster_path, _domain_gdf())
        _render_alignment_diagram(
            figures_dir / "01_native_no_resolution.pdf",
            title="target='native', resolution = (none)",
            description=(
                "Pre-#205 behavior — the helper returns {} so no destination "
                "override is passed. extract_window reprojects to ROI CRS at "
                "the source's native 30 m cell size and clips to the ROI. "
                "The output cells inherit the source pixel anchor (offset "
                "from the domain LL by the same fraction as the source), "
                "so the output bbox doesn't sit on the domain origin — it "
                "snaps outward to the source-pixel grid. Output transform "
                "and shape come from a real RasterConnection.extract_window "
                "call against a tmp GeoTIFF."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": out_transform,
                    "shape": out_shape,
                    "color": "#2ca02c",
                    "label": (
                        f"output (30 m cells, {out_shape[0]}×{out_shape[1]}, "
                        "anchor preserved from source)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_native_with_resolution(self, figures_dir, source_raster):
        raster_path, src_transform, src_shape = source_raster
        out_transform, out_shape = _run_native_extract(
            raster_path, _domain_gdf(), destination_resolution=5.0
        )
        _render_alignment_diagram(
            figures_dir / "02_native_with_resolution_5m.pdf",
            title="target='native', resolution = 5 m",
            description=(
                "Resolution change preserving the source pixel anchor. "
                "The helper returns {destination_crs} only; extract_window "
                "passes resolution=5 to rio.reproject, then clips to the "
                "ROI ± buffer (buffer = 0 here). Output transform and "
                "shape come from a real RasterConnection.extract_window "
                "call against a tmp GeoTIFF — the slight overshoot past "
                "the domain comes from clip_box snapping outward to the "
                "5 m source-pixel-anchored grid."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": out_transform,
                    "shape": out_shape,
                    "color": "#2ca02c",
                    "label": (
                        f"output (5 m cells, {out_shape[0]}×{out_shape[1]}, "
                        "anchor preserved from source)"
                    ),
                    "linewidth": 1.5,
                    "alpha": 0.95,
                },
            ],
        )

    def test_domain_no_resolution(self, figures_dir):
        src_transform, src_shape = _source_lattice()
        dest = _resolve({"target": "domain"}, source_resolution=30.0)
        _render_alignment_diagram(
            figures_dir / "03_domain_no_resolution.pdf",
            title="target='domain', resolution = (default = source native)",
            description=(
                "When resolution is omitted, the output cell size falls "
                "back to the source raster's native resolution (30 m here). "
                "The anchor is the domain's lower-left corner and the "
                "lattice covers the bbox via ceil()."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"output (30 m cells, "
                        f"{dest['destination_shape'][0]}×"
                        f"{dest['destination_shape'][1]}, anchored at domain LL)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_domain_with_resolution(self, figures_dir):
        src_transform, src_shape = _source_lattice()
        dest = _resolve({"target": "domain", "resolution": 2.0})
        _render_alignment_diagram(
            figures_dir / "04_domain_with_resolution_2m.pdf",
            title="target='domain', resolution = 2 m",
            description=(
                "Output cells anchored at the domain's lower-left corner, "
                "covering the bbox via ceil(). 15 output cells nest inside "
                "each 30 m source cell. Composes by integer slicing with "
                "any other domain-anchored grid at this resolution."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"output (2 m cells, "
                        f"{dest['destination_shape'][0]}×"
                        f"{dest['destination_shape'][1]}, anchored at domain LL)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_grid_no_resolution(self, figures_dir):
        src_transform, src_shape = _source_lattice()
        target_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                # 5 m cells, 40×60, anchored at (720100, 5190100) lower-left.
                "transform": (5.0, 0.0, 720100.0, 0.0, -5.0, 5190300.0),
                "shape": (40, 60),
            }
        }
        dest = _resolve({"target": "grid", "grid_id": "x"}, target_grid_doc=target_doc)
        target_transform = Affine(*target_doc["georeference"]["transform"])
        _render_alignment_diagram(
            figures_dir / "05_grid_no_resolution.pdf",
            title="target='grid', resolution = (none)",
            description=(
                "Output is byte-equal to the target grid's CRS, transform, "
                "and shape. The output lattice sits exactly on top of the "
                "target lattice — no offset, no shape drift. Useful when "
                "you want a new fetch to compose with an existing grid."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": target_transform,
                    "shape": tuple(target_doc["georeference"]["shape"]),
                    "color": "#1f77b4",
                    "label": "target grid (5 m cells, 40×60)",
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"output (5 m cells, "
                        f"{dest['destination_shape'][0]}×"
                        f"{dest['destination_shape'][1]}, exact match)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_grid_with_resolution(self, figures_dir):
        src_transform, src_shape = _source_lattice()
        target_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                # 30 m cells, 10×10, anchored at (720100, 5190200) lower-left.
                "transform": (30.0, 0.0, 720100.0, 0.0, -30.0, 5190500.0),
                "shape": (10, 10),
            }
        }
        dest = _resolve(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            target_grid_doc=target_doc,
        )
        target_transform = Affine(*target_doc["georeference"]["transform"])
        _render_alignment_diagram(
            figures_dir / "06_grid_with_resolution_1m.pdf",
            title="target='grid', resolution = 1 m",
            description=(
                "Output keeps the target grid's CRS and origin (lower-left) "
                "but resamples to a 1 m cell size. Each target 30 m cell "
                "is exactly 30×30 output cells, so the output nests cleanly "
                "inside the target lattice. Useful for nesting a fine "
                "fetch (e.g. 0.6 m CHM) inside an existing 30 m fuels grid."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": target_transform,
                    "shape": tuple(target_doc["georeference"]["shape"]),
                    "color": "#1f77b4",
                    "label": "target grid (30 m cells, 10×10)",
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"output (1 m cells, "
                        f"{dest['destination_shape'][0]}×"
                        f"{dest['destination_shape'][1]})"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    # ──────────────────────────────────────────────────────────────────
    # Beyond the base 6: buffer expansion and the composition motivation.
    # ──────────────────────────────────────────────────────────────────

    def test_native_with_buffer(self, figures_dir, source_raster):
        """Buffer for ``target='native'`` with an explicit resolution.
        ``extract_window`` clips the post-reproject output to ROI ± N
        cells at the requested ``destination_resolution``, then snaps to
        the source-pixel-anchored grid. Origin and extent shift by an
        integer multiple of the destination resolution, so buffered and
        unbuffered output cells still nest cleanly."""
        raster_path, src_transform, src_shape = source_raster
        unbuf_transform, unbuf_shape = _run_native_extract(
            raster_path, _domain_gdf(), destination_resolution=5.0
        )
        buf_transform, buf_shape = _run_native_extract(
            raster_path,
            _domain_gdf(),
            destination_resolution=5.0,
            extent_buffer_cells=8,
        )
        _render_alignment_diagram(
            figures_dir / "07_native_with_buffer.pdf",
            title="target='native', resolution = 5 m, extent_buffer_cells = 8",
            description=(
                "extract_window's CRS-only override branch reprojects to "
                "ROI CRS at the requested 5 m resolution preserving the "
                "source pixel anchor, then clips to ROI ± 8 cells × 5 m "
                "= ±40 m. Output shape grows from "
                f"{unbuf_shape[0]}×{unbuf_shape[1]} to "
                f"{buf_shape[0]}×{buf_shape[1]}. Both outputs share the "
                "same source-pixel anchor — buffered cells nest with "
                "unbuffered cells. Output transforms come from real "
                "RasterConnection.extract_window calls."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": unbuf_transform,
                    "shape": unbuf_shape,
                    "color": "#9467bd",
                    "label": (
                        f"unbuffered output (5 m cells, "
                        f"{unbuf_shape[0]}×{unbuf_shape[1]})"
                    ),
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": buf_transform,
                    "shape": buf_shape,
                    "color": "#2ca02c",
                    "label": (
                        f"buffered output (5 m cells, {buf_shape[0]}×{buf_shape[1]})"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_domain_with_buffer(self, figures_dir):
        """Buffer expands the destination lattice by N output cells on
        every side. Origin shifts by exactly N*resolution so buffered
        cells still nest with the unbuffered lattice."""
        src_transform, src_shape = _source_lattice()
        unbuffered = _resolve({"target": "domain", "resolution": 2.0})
        buffered = _resolve(
            {"target": "domain", "resolution": 2.0}, extent_buffer_cells=10
        )
        _render_alignment_diagram(
            figures_dir / "08_domain_with_buffer.pdf",
            title="target='domain', resolution = 2 m, extent_buffer_cells = 10",
            description=(
                "extent_buffer_cells expands the destination lattice by "
                "N output cells on every side. Here the buffer adds "
                "10 cells × 2 m = 20 m on each edge. Shape grows from "
                f"{unbuffered['destination_shape'][0]}×"
                f"{unbuffered['destination_shape'][1]} to "
                f"{buffered['destination_shape'][0]}×"
                f"{buffered['destination_shape'][1]}. Origin shifts by an "
                "integer multiple of the cell size, so buffered and "
                "unbuffered lattices still nest cleanly."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": unbuffered["destination_transform"],
                    "shape": unbuffered["destination_shape"],
                    "color": "#9467bd",
                    "label": (
                        f"unbuffered output (2 m cells, "
                        f"{unbuffered['destination_shape'][0]}×"
                        f"{unbuffered['destination_shape'][1]})"
                    ),
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": buffered["destination_transform"],
                    "shape": buffered["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"buffered output (2 m cells, "
                        f"{buffered['destination_shape'][0]}×"
                        f"{buffered['destination_shape'][1]})"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_grid_with_resolution_and_buffer(self, figures_dir):
        """Grid alignment with explicit resolution + buffer. The buffer
        expands inside the *new* (post-resolution) cells, keeping the
        target's CRS and (shifted) origin."""
        src_transform, src_shape = _source_lattice()
        target_doc = {
            "georeference": {
                "crs": "EPSG:32611",
                "transform": (30.0, 0.0, 720100.0, 0.0, -30.0, 5190500.0),
                "shape": (10, 10),
            }
        }
        unbuffered = _resolve(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            target_grid_doc=target_doc,
        )
        buffered = _resolve(
            {"target": "grid", "grid_id": "x", "resolution": 1.0},
            target_grid_doc=target_doc,
            extent_buffer_cells=8,
        )
        target_transform = Affine(*target_doc["georeference"]["transform"])
        _render_alignment_diagram(
            figures_dir / "09_grid_with_buffer.pdf",
            title="target='grid', resolution = 1 m, extent_buffer_cells = 8",
            description=(
                "Buffer for a grid-aligned fetch with an explicit "
                "resolution: 8 cells × 1 m = 8 m added on every side. "
                "Output shape grows from "
                f"{unbuffered['destination_shape'][0]}×"
                f"{unbuffered['destination_shape'][1]} to "
                f"{buffered['destination_shape'][0]}×"
                f"{buffered['destination_shape'][1]} and the origin shifts "
                "by 8 m outside the target's lower-left corner. Buffer "
                "cells still nest with the target lattice."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": target_transform,
                    "shape": tuple(target_doc["georeference"]["shape"]),
                    "color": "#1f77b4",
                    "label": "target grid (30 m cells, 10×10)",
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": unbuffered["destination_transform"],
                    "shape": unbuffered["destination_shape"],
                    "color": "#9467bd",
                    "label": (
                        f"unbuffered output (1 m, "
                        f"{unbuffered['destination_shape'][0]}×"
                        f"{unbuffered['destination_shape'][1]})"
                    ),
                    "linewidth": 1.5,
                    "alpha": 0.85,
                    "ls": "--",
                },
                {
                    "transform": buffered["destination_transform"],
                    "shape": buffered["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"buffered output (1 m, "
                        f"{buffered['destination_shape'][0]}×"
                        f"{buffered['destination_shape'][1]})"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_native_vs_domain_side_by_side(self, figures_dir, source_raster):
        """Direct side-by-side of the two anchor strategies. Both fetches
        request a 30 m output on the same domain so the comparison is
        apples-to-apples; the output resolution is intentionally coarse
        so the source-pixel anchor's offset (≈77 m east, ≈44 m north of
        the domain LL) is visible at the rendered scale rather than
        sub-pixel."""
        raster_path, src_transform, src_shape = source_raster
        domain_dest = _resolve(
            {"target": "domain", "resolution": 30.0}, source_resolution=30.0
        )
        native_out_transform, native_out_shape = _run_native_extract(
            raster_path, _domain_gdf(), destination_resolution=30.0
        )
        _render_alignment_diagram(
            figures_dir / "10_native_vs_domain_side_by_side.pdf",
            title="target='native' vs target='domain' at the same resolution",
            description=(
                "Both fetches request a 30 m output on the same domain. "
                "target='native' (purple) inherits the source pixel "
                "anchor and ends up offset from the domain LL by the same "
                "fractional amount as the source. target='domain' (green) "
                "snaps to the domain LL and covers the bbox via ceil(). "
                "Two domain-aligned grids on the same domain will share a "
                "lattice exactly; two native-aligned grids generally won't "
                "— which is why target='domain' is the default for "
                "cross-source composition."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": "source (30 m cells, offset anchor)",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                {
                    "transform": native_out_transform,
                    "shape": native_out_shape,
                    "color": "#9467bd",
                    "label": (
                        f"target='native', 30 m output "
                        f"({native_out_shape[0]}×{native_out_shape[1]}, "
                        "anchor offset)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.9,
                    "ls": "--",
                },
                {
                    "transform": domain_dest["destination_transform"],
                    "shape": domain_dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": (
                        f"target='domain', 30 m output "
                        f"({domain_dest['destination_shape'][0]}×"
                        f"{domain_dest['destination_shape'][1]}, "
                        "anchored at domain LL)"
                    ),
                    "linewidth": 1.8,
                    "alpha": 0.95,
                },
            ],
        )

    def test_two_domain_grids_compose(self, figures_dir):
        """Two grids on the same domain with the same resolution and
        target='domain' share an exact lattice — the property that lets
        QUIC-Fire / LCP exports stitch by integer arithmetic with no
        second reprojection."""
        src_transform, src_shape = _source_lattice()
        # Imagine grid A is FBFM40 and grid B is topography. Both at 2 m.
        dest = _resolve({"target": "domain", "resolution": 2.0})
        _render_alignment_diagram(
            figures_dir / "11_two_domain_grids_compose.pdf",
            title="Two domain-aligned grids at the same resolution → identical lattice",
            description=(
                "FBFM40 and topography fetched separately, both with "
                "target='domain', resolution=2 m, same domain. They land "
                "on the same lattice — same CRS, transform, shape — so "
                "downstream code (combined exports, grid-to-grid math) "
                "can stitch them by integer slicing with no second "
                "reprojection. This is the design's core promise."
            ),
            domain_bounds=_DOMAIN_BOUNDS,
            layers=[
                {
                    "transform": src_transform,
                    "shape": src_shape,
                    "color": "#888888",
                    "label": f"source (30 m cells, {src_shape[0]}×{src_shape[1]})",
                    "linewidth": 1.2,
                    "alpha": 0.55,
                },
                # Two outputs drawn slightly differently so they're
                # visibly co-located in the legend, even though they
                # occupy the exact same lattice.
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#1f77b4",
                    "label": "FBFM40 grid A (2 m, target='domain')",
                    "linewidth": 2.4,
                    "alpha": 0.7,
                    "ls": "-",
                },
                {
                    "transform": dest["destination_transform"],
                    "shape": dest["destination_shape"],
                    "color": "#2ca02c",
                    "label": "topography grid B (2 m, target='domain')",
                    "linewidth": 1.2,
                    "alpha": 0.95,
                    "ls": "--",
                },
            ],
        )
