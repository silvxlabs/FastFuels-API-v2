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

import io
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import shapely

from lakitu.ept import fetch_nodes
from lib.config import LAKITU_CHAIN_WORKERS, LAKITU_DOWNLOAD_WORKERS
from lib.pointcloud.schema import point_dtype

# Per-worker state, set once by the initializer so each task carries only its
# compressed bytes.
_W = {}


def _chain_init(
    src_crs_wkt, dst_crs_wkt, header_bounds, point_format_id, clip_wkb, clip_bounds
):
    # An exception here kills the worker and the parent only ever sees
    # BrokenProcessPool, so say what actually failed before going down.
    try:
        from pyproj import CRS, Transformer

        from lib.laz import build_output_header

        _W["transformer"] = Transformer.from_crs(
            src_crs_wkt, dst_crs_wkt, always_xy=True
        )
        # Rebuilt rather than pickled: laspy headers hold VLRs and a CRS object,
        # and reconstructing from the same inputs is cheaper and exact. add_crs
        # wants a pyproj CRS, so the WKT has to be revived first.
        _W["header"] = build_output_header(
            CRS.from_wkt(dst_crs_wkt), header_bounds, point_format_id=point_format_id
        )
        _W["clip"] = shapely.from_wkb(clip_wkb) if clip_wkb else None
        _W["bounds"] = clip_bounds
        # The canonical format carries RGB only when a source declared it, so
        # this decides the record layout once for the whole cloud.
        _W["dtype"] = point_dtype("red" in _W["header"].point_format.dimension_names)
    except BaseException:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _chain_work(payload):
    """Decode, reproject, clip and normalize one node. Runs in a child process.

    Returns a compact `point_dtype` array, or None when the node contributes
    nothing.
    """
    import laspy

    from lib.crs import reproject
    from lib.laz import normalize_record

    points = laspy.read(io.BytesIO(payload))
    min_x, min_y, max_x, max_y = _W["bounds"]
    x, y = reproject(_W["transformer"], np.asarray(points.x), np.asarray(points.y))
    z = np.asarray(points.z)

    # Cheap rectangle test first; the polygon test only runs for the points that
    # survive it, and only where two acquisitions must be arbitrated.
    keep = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    clip = _W["clip"]
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
    dst_crs_wkt,
    header_bounds,
    point_format_id,
    *,
    workers=LAKITU_CHAIN_WORKERS,
    download_workers=LAKITU_DOWNLOAD_WORKERS,
    batch=250,
    on_node=None,
):
    """Fetch every planned node and yield normalized PDRF-6 records.

    Args:
        session: Requests session for the EPT archive.
        plan: Sequence of ``(meta, nodes, clip, bounds)`` per acquisition, in the
            domain CRS. ``clip`` is the polygon a point must fall inside, or None
            when ``bounds`` alone decides -- which is the common case of one
            acquisition under an axis-aligned domain, where the polygon test is
            far more expensive than the rectangle it sits in.
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
        PDRF-6 structured arrays, in node completion order.

    Raises:
        ProcessingError: If a node cannot be downloaded or decoded.
    """
    # forkserver, not fork: the caller has a collector thread running, and
    # forking a threaded process can inherit a lock held by a thread the child
    # does not have.
    ctx = multiprocessing.get_context("forkserver")

    for meta, nodes, clip, bounds in plan:
        if not nodes:
            continue
        args = (
            meta.crs.to_wkt(),
            dst_crs_wkt,
            header_bounds,
            point_format_id,
            clip.wkb if clip is not None else None,
            bounds,
        )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_chain_init, initargs=args
        ) as pool:
            # Bounded so submissions cannot outrun the pool and grow without
            # limit; results are drained in order.
            inflight = deque()
            for off in range(0, len(nodes), batch):
                for _node, payload in fetch_nodes(
                    session,
                    nodes[off : off + batch],
                    raw=True,
                    max_workers=download_workers,
                ):
                    inflight.append(pool.submit(_chain_work, payload))
                    if len(inflight) >= workers * 2:
                        record = inflight.popleft().result()
                        if on_node is not None:
                            on_node()
                        if record is not None:
                            yield record
            while inflight:
                record = inflight.popleft().result()
                if on_node is not None:
                    on_node()
                if record is not None:
                    yield record
