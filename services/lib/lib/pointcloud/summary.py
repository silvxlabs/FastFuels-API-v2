"""The statistics a point cloud resource reports about itself.

Shared because both writers report the same fields for the same resource type:
`point_count`, `point_classes` and `density` on `summary`, and the 3D extent on
`georeference.bounds`. A cloud fetched from 3DEP and one a user uploaded should
describe themselves identically.
"""

import numpy as np


class PointSummary:
    """Folds written points into the statistics the resource reports.

    Accumulated on the way past rather than read back afterwards, so what is
    reported always describes what was stored.

    The per-point reduction happens wherever the points already are — the write
    workers, which hold each flush anyway — and this only folds the handful of
    scalars each one returns. It used to run on the parent thread, six strided
    reductions and a scatter over every point in the cloud, on the one thread
    that also routes every point and drains every queue.
    """

    def __init__(self, scales, offsets):
        self._scales = np.asarray(scales)
        self._offsets = np.asarray(offsets)
        self.count = 0
        # Classification is a uint8, so a flag per value beats accumulating a
        # set: no sort, no Python-level set union, per fold.
        self._seen_class = np.zeros(256, dtype=bool)
        self._mins = np.full(3, np.iinfo(np.int32).max, dtype=np.int64)
        self._maxs = np.full(3, np.iinfo(np.int32).min, dtype=np.int64)

    def fold(self, mins, maxs, classes, count):
        """Merge one flush's extremes in.

        Reduces over the stored integers and scales only the six surviving
        scalars at the end, so nothing here is per-point.

        Args:
            mins: Per-axis minima of X, Y, Z, as stored integers.
            maxs: Per-axis maxima of X, Y, Z, as stored integers.
            classes: The classification values the flush contained.
            count: How many points it held.
        """
        np.minimum(self._mins, mins, out=self._mins)
        np.maximum(self._maxs, maxs, out=self._maxs)
        self._seen_class[classes] = True
        self.count += count

    def bounds(self) -> list[float]:
        """``[min_x, min_y, min_z, max_x, max_y, max_z]`` in world units."""
        if self.count == 0:
            return [*self._offsets.tolist(), *self._offsets.tolist()]
        mins = self._mins * self._scales + self._offsets
        maxs = self._maxs * self._scales + self._offsets
        return [*mins.tolist(), *maxs.tolist()]

    def summary(self) -> dict:
        """Point count, the ASPRS classes present, and points per square metre."""
        bounds = self.bounds()
        area = (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) if self.count else 0.0
        return {
            "point_count": self.count,
            "point_classes": [int(c) for c in np.flatnonzero(self._seen_class)],
            "density": float(self.count / area) if area > 0 else 0.0,
        }
