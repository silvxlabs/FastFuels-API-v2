"""Unit tests for the node plan and the tile schedule it produces.

The plan decides two things the writer cannot recover on its own: the order
nodes are read in, and the point at which each tile can be written whole. Both
are cheap to get subtly wrong and expensive to notice — a tile claimed one node
too early is not a corrupt dataset, just a second file, which is exactly the
cost this is meant to remove.
"""

from lakitu.handlers.threedep import plan_nodes

TILE_M = 500.0
BOUNDS = (0.0, 0.0, 4000.0, 4000.0)


class FakeNode:
    """Enough of an EptNode for the planner: horizontal bounds and a depth."""

    def __init__(self, x0, y0, x1, y1, depth):
        self.bounds = (x0, y0, 0.0, x1, y1, 1.0)
        self.depth = depth


class Identity:
    """Stands in for a pyproj transformer when the CRSs already agree."""

    def transform(self, xs, ys):
        return list(xs), list(ys)


def plan(all_nodes, transformers=None):
    sources = transformers or {index: Identity() for index, _ in all_nodes}
    return plan_nodes(all_nodes, sources, BOUNDS, TILE_M)


def test_nodes_wider_than_a_tile_are_read_before_the_sweep():
    """A node spanning many tiles has to land before the sweep, not during it.

    Left in the sweep it arrives after tiles it covers have been written, and
    every one it reopens costs a file. On the real 64 km2 domain these are 0.2%
    of nodes and 3.25% of points, so hoisting them is nearly free.
    """
    coarse = FakeNode(0, 0, 4000, 4000, 2)
    fine = [FakeNode(x, 0, x + 100, 100, 12) for x in (0, 1000, 2000, 3000)]

    ordered, _ = plan([(0, fine + [coarse])])

    assert ordered[0][1] is coarse
    assert [node for _, node in ordered[1:]] == fine


def test_the_sweep_runs_once_across_every_acquisition():
    """Ordering per acquisition sweeps the grid twice and reopens seam tiles."""
    left = [FakeNode(0, y, 100, y + 100, 12) for y in (0, 1000)]
    right = [FakeNode(600, y, 700, y + 100, 12) for y in (0, 1000)]

    ordered, _ = plan([(0, left), (1, right)])

    # Both acquisitions' row 0 before either one's row 2.
    assert [source for source, _ in ordered] == [0, 1, 0, 1]


def test_a_tile_is_never_due_before_a_node_that_reaches_it():
    """The schedule is what the writer trusts; due-too-early is a split tile."""
    nodes = [FakeNode(x, 0, x + 100, 100, 12) for x in (0, 600, 1200, 1800)]

    ordered, schedule = plan([(0, nodes)])

    for index, (_, node) in enumerate(ordered):
        tile = (int(node.bounds[0] // TILE_M), 0)
        assert schedule[tile] >= index


def test_every_tile_a_node_touches_is_scheduled():
    """Including the halo. A tile absent from the schedule is one the writer
    only ever writes under buffer pressure, which is the case being removed."""
    node = FakeNode(400, 400, 1100, 1100, 12)

    _, schedule = plan([(0, [node])])

    for tx in range(3):
        for ty in range(3):
            assert (tx, ty) in schedule


def test_schedule_is_clamped_to_the_output_extent():
    """A coarse node may cover an acquisition far larger than the domain."""
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    coarse = FakeNode(-20_000, -20_000, 20_000, 20_000, 0)

    _, schedule = plan_nodes([(0, [coarse])], {0: Identity()}, bounds, TILE_M)

    assert set(schedule) == {(tx, ty) for tx in range(3) for ty in range(3)}


def test_the_last_node_overall_owns_every_tile_it_covers():
    """A coarse node read first must not leave tiles due at index 0."""
    coarse = FakeNode(0, 0, 2000, 2000, 2)
    fine = [FakeNode(x, 0, x + 100, 100, 12) for x in (0, 600, 1200)]

    ordered, schedule = plan([(0, [coarse] + fine)])

    assert ordered[0][1] is coarse
    assert schedule[(0, 0)] > 0


def test_an_empty_walk_plans_nothing():
    ordered, schedule = plan([(0, [])])

    assert ordered == []
    assert schedule == {}
