"""
Shared helpers for point cloud uploader tests.

Synthesizes tiny LAS/LAZ files with laspy so tests assert against exact,
known-by-construction metadata (CRS, point count, classification codes) rather
than a committed opaque binary fixture.
"""

import laspy
import numpy as np
import pyproj


def make_test_las(
    path: str,
    n: int = 100,
    epsg: int = 32612,
    classes: tuple[int, ...] = (1, 2, 5),
    x0: float = 500000.0,
    y0: float = 4300000.0,
    span: float = 1000.0,
    z0: float = 1800.0,
    z_span: float = 100.0,
    with_srs: bool = True,
) -> dict:
    """Write a tiny LAS/LAZ at ``path`` with a known CRS and classification set.

    The extension of ``path`` (``.las`` / ``.laz``) selects compression. LAS 1.4
    / point format 6 is used so ``Classification`` is a full byte (codes > 31 and
    high-vegetation class 5 round-trip without the legacy 4-bit clamp).

    Returns the ground-truth dict the handler should reproduce.
    """
    rng = np.random.default_rng(0)
    x = x0 + rng.uniform(0, span, n)
    y = y0 + rng.uniform(0, span, n)
    z = z0 + rng.uniform(0, z_span, n)
    classification = np.array(
        [classes[i % len(classes)] for i in range(n)], dtype=np.uint8
    )

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = [x0, y0, z0]
    header.scales = [0.01, 0.01, 0.01]
    if with_srs:
        header.add_crs(pyproj.CRS.from_epsg(epsg))

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification
    las.write(path)

    return {
        "crs": f"EPSG:{epsg}",
        "point_count": n,
        "point_classes": sorted({int(c) for c in classes}),
        "xy_area": float((x.max() - x.min()) * (y.max() - y.min())),
        "min_z": float(z.min()),
        "max_z": float(z.max()),
        "x": x,
        "y": y,
        "z": z,
    }
