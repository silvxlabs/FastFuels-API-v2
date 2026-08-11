"""The per-node point chain: decode, reproject, clip, normalize.

Kept in its own module, and deliberately import-light at module scope, because
forkserver re-imports it in every child. Everything heavy is imported inside the
worker functions.

This runs in processes rather than threads because decode is GIL-bound in
practice. The docstring it replaces claimed LAZ decompression holds the
interpreter lock; that is false as applied -- `laspy.read` with no backend
argument selects LazrsParallel, which decompresses on a rayon pool. It still
does not help, for a different reason: LAZ parallelism is per chunk, chunks hold
50,000 points, and real 3DEP nodes hold 20-37k, so a node is a single chunk with
nothing to split. Eight concurrent decoders measured 1.00 effective cores.
Threading the chain anyway cost +41% CPU for ~17% wall; processes gave 30% wall
for +4% CPU.
"""

import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import shapely

from lakitu.ept import decode_node, fetch_nodes
from lib.config import LAKITU_CHAIN_WORKERS, LAKITU_DOWNLOAD_WORKERS
from lib.pointcloud.schema import point_dtype

# Per-worker state, set once by the initializer so each task carries only its
# compressed bytes.
_W = {}


def _chain_init(sources, dst_crs_wkt, header_bounds, point_format_id):
    """Set up one worker for every acquisition, not just one.

    The per-source transformer and clip used to be initializer arguments, which
    meant one pool per acquisition and so one node order per acquisition. The
    writer can only finish a tile once, so a seam tile was written, closed and
    reopened on the second sweep. Keying them by source index instead lets a
    single pool serve one globally ordered plan.
    """
    # An exception here kills the worker and the parent only ever sees
    # BrokenProcessPool, so say what actually failed before going down.
    try:
        from pyproj import CRS, Transformer

        from lib.laz import build_output_header

        _W["sources"] = [
            {
                "transformer": Transformer.from_crs(
                    src_crs_wkt, dst_crs_wkt, always_xy=True
                ),
                "clip": shapely.from_wkb(clip_wkb) if clip_wkb else None,
                "bounds": clip_bounds,
            }
            for src_crs_wkt, clip_wkb, clip_bounds in sources
        ]
        # Rebuilt rather than pickled: laspy headers hold VLRs and a CRS object,
        # and reconstructing from the same inputs is cheaper and exact. add_crs
        # wants a pyproj CRS, so the WKT has to be revived first.
        _W["header"] = build_output_header(
            CRS.from_wkt(dst_crs_wkt), header_bounds, point_format_id=point_format_id
        )
        # The canonical format carries RGB only when a source declared it, so
        # this decides the record layout once for the whole cloud.
        _W["dtype"] = point_dtype("red" in _W["header"].point_format.dimension_names)
    except BaseException:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _chain_work(source, node, payload):
    """Decode, reproject, clip and normalize one node. Runs in a child process.

    Returns a compact `point_dtype` array, or None when the node contributes
    nothing.
    """
    from lib.crs import reproject
    from lib.laz import normalize_record

    src = _W["sources"][source]
    points = decode_node(node, payload)
    min_x, min_y, max_x, max_y = src["bounds"]
    x, y = reproject(src["transformer"], np.asarray(points.x), np.asarray(points.y))
    z = np.asarray(points.z)

    # Cheap rectangle test first; the polygon test only runs for the points that
    # survive it, and only where two acquisitions must be arbitrated.
    keep = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    clip = src["clip"]
    if clip is not None and keep.any():
        # Boundary-exclusive: on a shared edge between two acquisitions,
        # dropping a point is safer than duplicating it.
        keep[keep] = shapely.contains_xy(clip, x[keep], y[keep])
    if not keep.any():
        return None

    # normalize_record already returns the canonical point format, so this is a
    # field-by-field narrowing rather than a conversion.
    record = normalize_record(
        points.points[keep], _W["header"], x[keep], y[keep], z[keep]
    )
    dtype = _W["dtype"]
    out = np.empty(len(record), dtype=dtype)
    for name in dtype.names:
        out[name] = record.array[name]
    return out


def stream_records(
    session,
    plan,
    sources,
    dst_crs_wkt,
    header_bounds,
    point_format_id,
    *,
    workers=LAKITU_CHAIN_WORKERS,
    download_workers=LAKITU_DOWNLOAD_WORKERS,
    batch=250,
    on_node=None,
):
    """Fetch every planned node and yield ``(plan index, records)``.

    The index is what lets the writer know which tiles are finished. Nodes are
    downloaded on a thread pool and come back in completion order, so a node's
    position in the stream says nothing about the plan; the index does, and a
    node that contributes no points still yields its index with None so the
    tiles waiting on it are not stranded.

    Args:
        session: Requests session for the EPT archive.
        plan: ``(source index, node)`` in read order, from `plan_nodes` -- one
            sequence across every acquisition, so the tile sweep happens once.
        sources: Per source index, ``(src_crs_wkt, clip_wkb, clip_bounds)``.
            ``clip_wkb`` is the polygon a point must fall inside, or None when
            the bounds alone decide -- the common case of one acquisition under
            an axis-aligned domain, where the polygon test is far more expensive
            than the rectangle it sits in.
        dst_crs_wkt: Target CRS as WKT -- the domain's.
        header_bounds: Horizontal bounds for the output header.
        point_format_id: LAS point format every acquisition is normalized to.
        workers: Chain processes. Must not exceed the vCPU allocation.
        download_workers: Concurrent downloads. A separate knob from `workers`:
            downloads are network-bound, and once the chain moved to processes
            fetch wait hit 54 s of a 167 s job until this was raised.
        batch: Nodes per fetch_nodes call.
        on_node: Optional callback invoked once per completed node.

    Yields:
        ``(plan index, records)``, records being a `schema.point_dtype` array or
        None, in node completion order.

    Raises:
        ProcessingError: If a node cannot be downloaded or decoded.
    """
    if not plan:
        return
    # forkserver, not fork: the caller has a collector thread running, and
    # forking a threaded process can inherit a lock held by a thread the child
    # does not have.
    ctx = multiprocessing.get_context("forkserver")
    # Nodes are frozen dataclasses carrying their own base URL, so one map
    # serves every acquisition without the batches having to stay separate.
    where = {node: (index, source) for index, (source, node) in enumerate(plan)}
    nodes = [node for _, node in plan]

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_chain_init,
        initargs=(sources, dst_crs_wkt, header_bounds, point_format_id),
    ) as pool:
        # Bounded so submissions cannot outrun the pool and grow without limit;
        # results are drained in order.
        inflight = deque()

        def drain():
            index, future = inflight.popleft()
            record = future.result()
            if on_node is not None:
                on_node()
            return index, record

        for off in range(0, len(nodes), batch):
            for node, payload in fetch_nodes(
                session,
                nodes[off : off + batch],
                raw=True,
                max_workers=download_workers,
            ):
                index, source = where[node]
                inflight.append(
                    (index, pool.submit(_chain_work, source, node, payload))
                )
                if len(inflight) >= workers * 2:
                    yield drain()
        while inflight:
            yield drain()
