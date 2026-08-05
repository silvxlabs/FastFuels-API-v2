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
    """

    def __init__(self, scales, offsets):
        self._scales = np.asarray(scales)
        self._offsets = np.asarray(offsets)
        self.count = 0
        # Classification is a uint8, so a flag per value beats accumulating a
        # set: no sort, no Python-level set union, per record.
        self._seen_class = np.zeros(256, dtype=bool)
        self._mins = np.full(3, np.iinfo(np.int32).max, dtype=np.int64)
        self._maxs = np.full(3, np.iinfo(np.int32).min, dtype=np.int64)

    def observe(self, records):
        """Fold each record's extremes in, then pass it straight through.

        Reduces over the stored integers and scales only the six surviving
        scalars at the end. Scaling every point here would repeat, on the
        busiest thread in the process, work the writer already does to route the
        point.
        """
        for record in records:
            for axis, name in enumerate(("X", "Y", "Z")):
                column = record[name]
                self._mins[axis] = min(self._mins[axis], int(column.min()))
                self._maxs[axis] = max(self._maxs[axis], int(column.max()))
            self._seen_class[record["classification"]] = True
            self.count += record.size
            yield record

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
