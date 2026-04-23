"""Visualize foliage bulk density from tmp.zarr to verify integration tests.

Reads `bulk_density.foliage` (nz, ny, nx) from the zarr store and produces:
  - 2D canopy height map (highest z with non-zero bulk density)
  - 2D column-summed bulk density (kg/m^3 summed over z)
  - 3D PyVista rendering of canopy voxels (PNG + optional interactive window)
  - Vertical (x-z) cross-section through mid-y, averaged over a few rows

Pass --interactive (or -i) to open the PyVista 3D scene in a window.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import xarray as xr
from matplotlib.colors import Normalize

ZARR_PATH = Path(__file__).resolve().parents[1] / "tmp.zarr"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 600


def load_bulk_density(path: Path):
    ds = xr.open_zarr(str(path), consolidated=True)
    da = ds["bulk_density.foliage"].load()
    bd = np.nan_to_num(da.values, nan=0.0).astype(np.float32)
    x = ds["x"].values
    y = ds["y"].values
    z = ds["z"].values
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    dz = float(abs(z[1] - z[0])) if z.size > 1 else 1.0
    return bd, x, y, z, dx, dy, dz


def compute_canopy_height(bd: np.ndarray, dz: float) -> np.ndarray:
    nz, ny, nx = bd.shape
    canopy_height = np.zeros((ny, nx), dtype=np.float32)
    for zi in range(nz - 1, -1, -1):
        mask = (bd[zi] > 0) & (canopy_height == 0)
        canopy_height[mask] = (zi + 1) * dz
    return canopy_height


def _extent(x: np.ndarray, y: np.ndarray) -> list:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return [x[0] - dx / 2, x[-1] + dx / 2, y[-1] - dy / 2, y[0] + dy / 2]


def plot_2d_map(
    data, x, y, title, cbar_label, cmap, out, vmin=None, vmax=None, mask=None
):
    fig, ax = plt.subplots(figsize=(12, 8))
    d = data.astype(np.float64).copy()
    if mask is not None:
        d[mask] = np.nan
    im = ax.imshow(
        d,
        extent=_extent(x, y),
        origin="upper",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _build_canopy_mesh(bd, x, y, dx, dy, dz):
    nz, ny, nx = bd.shape
    x_origin = float(x.min() - dx / 2)
    y_origin = float(y.min() - dy / 2)
    bd_flipped = bd[:, ::-1, :]
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
        origin=(x_origin, y_origin, 0.0),
    )
    grid.cell_data["bulk_density"] = bd_flipped.flatten()
    return grid.threshold(value=1e-4, scalars="bulk_density")


def plot_canopy_3d(bd, x, y, dx, dy, dz, out):
    threshed = _build_canopy_mesh(bd, x, y, dx, dy, dz)
    pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    pl.add_mesh(
        threshed,
        scalars="bulk_density",
        cmap="Greens",
        opacity=0.5,
        clim=[0, float(bd.max())],
        scalar_bar_args={"title": "Canopy Bulk Density (kg/m^3)"},
    )
    pl.add_axes()
    pl.set_background("white")
    pl.camera_position = "iso"
    pl.screenshot(str(out))
    pl.close()
    print(f"Saved: {out}")


def show_canopy_interactive(bd, x, y, dx, dy, dz):
    threshed = _build_canopy_mesh(bd, x, y, dx, dy, dz)
    pl = pv.Plotter(window_size=[1280, 900])
    pl.add_mesh(
        threshed,
        scalars="bulk_density",
        cmap="Greens",
        opacity=0.5,
        clim=[0, float(bd.max())],
        scalar_bar_args={"title": "Canopy Bulk Density (kg/m^3)"},
    )
    pl.add_axes()
    pl.set_background("white")
    pl.camera_position = "iso"
    print("Opening interactive viewer — close the window to exit.")
    pl.show()


def plot_cross_section(bd, x, dx, dz, out, y_avg_width=10):
    nz, ny, nx = bd.shape
    y_mid = ny // 2
    half = y_avg_width // 2
    y_lo = max(y_mid - half, 0)
    y_hi = min(y_lo + y_avg_width, ny)
    slab = bd[:, y_lo:y_hi, :].mean(axis=1)  # (nz, nx)

    x_left = float(x.min() - dx / 2)
    x_right = float(x.max() + dx / 2)
    z_top = nz * dz

    fig, ax = plt.subplots(figsize=(14, max(3.0, 14 * z_top / (x_right - x_left))))
    bd_max = float(bd.max())
    display = np.where(slab > 0, slab, np.nan)
    im = ax.imshow(
        display,
        extent=[x_left, x_right, 0, z_top],
        origin="lower",
        cmap="Greens",
        vmin=0,
        vmax=bd_max,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cbar.set_label("Canopy Bulk Density (kg/m^3)", fontsize=11)
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Height Above Ground (m)", fontsize=12)
    ax.set_title(
        f"Vertical Cross-Section (y-average over {y_avg_width} rows)", fontsize=13
    )
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="open interactive PyVista viewer after saving figures",
    )
    args = parser.parse_args()

    print(f"Reading {ZARR_PATH}...")
    bd, x, y, z, dx, dy, dz = load_bulk_density(ZARR_PATH)
    print(f"  shape (nz, ny, nx) = {bd.shape}")
    print(f"  spacing dx={dx} dy={dy} dz={dz}")
    print(
        f"  min={bd.min():.4f}  max={bd.max():.4f}  nonzero cells={int((bd > 0).sum())}"
    )

    canopy_height = compute_canopy_height(bd, dz)
    plot_2d_map(
        canopy_height,
        x,
        y,
        title="Canopy Height (from foliage bulk density)",
        cbar_label="Height (m)",
        cmap="Greens",
        out=OUTPUT_DIR / "canopy_height.png",
        mask=(canopy_height == 0),
    )

    column_bd = bd.sum(axis=0) * dz  # integrated foliage mass per m^2
    plot_2d_map(
        column_bd,
        x,
        y,
        title="Integrated Foliage Load (column sum of bulk density * dz)",
        cbar_label="Foliage Load (kg/m^2)",
        cmap="Greens",
        out=OUTPUT_DIR / "foliage_load_column.png",
        mask=(column_bd == 0),
    )

    plot_canopy_3d(bd, x, y, dx, dy, dz, OUTPUT_DIR / "canopy_3d.png")
    plot_cross_section(bd, x, dx, dz, OUTPUT_DIR / "cross_section.png")

    print(f"\nDone. Figures in {OUTPUT_DIR}")

    if args.interactive:
        show_canopy_interactive(bd, x, y, dx, dy, dz)


if __name__ == "__main__":
    main()
