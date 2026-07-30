"""
Entwine Point Tile (EPT) reader.

EPT is how USGS publishes 3DEP lidar: an octree of ordinary LAZ files, one per
node, addressed by a ``depth-x-y-z`` key, with a JSON index describing which
nodes exist and how many points each holds. Reading a small area means walking
the index for the nodes that overlap it and fetching only those files.

Points in an EPT tree are **additive across depths**. A node at depth 3 holds a
coarse sample of its own volume and its children hold the rest; full density is
the union of every overlapping node at every depth, not the deepest level alone.

Nothing here needs PDAL. EPT node files are plain LAZ that laspy reads directly,
which is what keeps this service on a pure-pip image.
"""

import io
import logging
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import laspy
import requests
from pyproj import CRS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lib.errors import ProcessingError

logger = logging.getLogger(__name__)

# Structural sanity bound. Depth 24 subdivides a 100 km cube to under a
# centimetre, so a tree claiming to go deeper is malformed, not merely dense.
MAX_DEPTH = 24

# A large domain over a dense acquisition legitimately needs many index pages;
# far more than this means the walk is not converging.
MAX_HIERARCHY_PAGES = 512

MAX_FETCH_WORKERS = 16

# Decoded nodes waiting to be consumed. Bounds how much decompressed point data
# is resident while the consumer is busy.
_NODE_QUEUE_DEPTH = 8


@dataclass(frozen=True)
class EptMetadata:
    """Contents of an acquisition's ``ept.json``."""

    url: str
    base_url: str
    bounds: tuple[float, float, float, float, float, float]
    bounds_conforming: tuple[float, float, float, float, float, float]
    crs: CRS
    vertical_crs: str | None
    point_count: int
    dimension_names: tuple[str, ...]


@dataclass(frozen=True)
class EptNode:
    """One octree node: a key, a point count, and the volume it covers."""

    key: str
    depth: int
    count: int
    bounds: tuple[float, float, float, float, float, float]
    base_url: str

    @property
    def data_url(self) -> str:
        return f"{self.base_url}/ept-data/{self.key}.laz"


def make_session() -> requests.Session:
    """Build a session that survives the throttling anonymous S3 applies."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=MAX_FETCH_WORKERS, pool_maxsize=32
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_metadata(session: requests.Session, url: str) -> EptMetadata:
    """Read and validate an acquisition's ``ept.json``.

    Every field this reader depends on is checked here rather than assumed, so
    an acquisition we cannot read fails with a clear reason instead of an
    obscure error several layers down.

    Args:
        session: HTTP session to use.
        url: Full URL of the acquisition's ``ept.json``.

    Returns:
        The parsed metadata.

    Raises:
        ProcessingError: EPT_METADATA_ERROR if the file cannot be read or
            describes a tree this reader does not support.
    """
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        document = response.json()
    except Exception as e:
        raise ProcessingError(
            code="EPT_METADATA_ERROR",
            message="Could not read the USGS 3DEP lidar index for this area.",
            suggestion="Try again shortly, or select a different acquisition.",
            traceback=f"{url}: {e}",
        ) from e

    data_type = document.get("dataType")
    if data_type != "laszip":
        raise ProcessingError(
            code="EPT_METADATA_ERROR",
            message=(
                "This USGS 3DEP acquisition is published in a format that "
                "cannot be read."
            ),
            suggestion="Select a different acquisition.",
            traceback=f"{url}: unsupported dataType {data_type!r}",
        )

    hierarchy_type = document.get("hierarchyType")
    if hierarchy_type != "json":
        raise ProcessingError(
            code="EPT_METADATA_ERROR",
            message=(
                "This USGS 3DEP acquisition uses an index format that cannot be read."
            ),
            suggestion="Select a different acquisition.",
            traceback=f"{url}: unsupported hierarchyType {hierarchy_type!r}",
        )

    srs = document.get("srs") or {}
    crs = _parse_srs(srs, url)

    return EptMetadata(
        url=url,
        base_url=url.rsplit("/", 1)[0],
        bounds=tuple(document["bounds"]),
        bounds_conforming=tuple(document.get("boundsConforming") or document["bounds"]),
        crs=crs,
        # Reported, never applied: elevations pass through untouched, so this
        # only records what they are measured from. Most acquisitions declare
        # nothing, and inventing a datum for those would be a guess presented
        # as a fact.
        vertical_crs=(
            f"{srs['authority']}:{srs['vertical']}"
            if srs.get("authority") and srs.get("vertical")
            else None
        ),
        point_count=int(document.get("points", 0)),
        # The schema names every attribute the tree carries, so the output
        # format can be chosen without downloading a node to look.
        dimension_names=tuple(
            entry["name"] for entry in document.get("schema", []) if "name" in entry
        ),
    )


def _parse_srs(srs: dict, url: str) -> CRS:
    """Resolve the tree's CRS from its ``srs`` block.

    3DEP publishes in EPSG:3857, but that is a property of the data rather than
    of the format, so it is read rather than assumed.
    """
    authority, horizontal = srs.get("authority"), srs.get("horizontal")
    if authority and horizontal:
        try:
            return CRS.from_user_input(f"{authority}:{horizontal}")
        except Exception:
            pass
    if srs.get("wkt"):
        try:
            return CRS.from_wkt(srs["wkt"])
        except Exception:
            pass
    raise ProcessingError(
        code="EPT_METADATA_ERROR",
        message=(
            "This USGS 3DEP acquisition does not declare a coordinate "
            "reference system, so its points cannot be placed."
        ),
        suggestion="Select a different acquisition.",
        traceback=f"{url}: unusable srs {srs!r}",
    )


def node_bounds(
    root: tuple[float, ...], depth: int, x: int, y: int, z: int
) -> tuple[float, float, float, float, float, float]:
    """Return the bounds of an octree node within a root cube.

    Computed directly from the root rather than by accumulating per level, so
    deep keys do not drift.
    """
    divisions = 2**depth
    min_x, min_y, min_z, max_x, max_y, max_z = root
    span_x = (max_x - min_x) / divisions
    span_y = (max_y - min_y) / divisions
    span_z = (max_z - min_z) / divisions
    return (
        min_x + x * span_x,
        min_y + y * span_y,
        min_z + z * span_z,
        min_x + (x + 1) * span_x,
        min_y + (y + 1) * span_y,
        min_z + (z + 1) * span_z,
    )


def _overlaps_2d(
    node: tuple[float, ...], query: tuple[float, float, float, float]
) -> bool:
    """Test horizontal overlap between a node and a query box.

    Deliberately ignores z. An octree subdivides in all three axes, so at depth
    8 a node covers one thin slice of the elevation range; a domain has no
    elevation extent to test against and wants every slice. Testing z here
    would silently select nothing.

    Strict inequalities, because adjacent nodes share a face and touching is
    not overlapping — including them would fetch a ring of nodes whose points
    all get masked away.
    """
    return (
        node[3] > query[0]
        and node[0] < query[2]
        and node[4] > query[1]
        and node[1] < query[3]
    )


def walk_hierarchy(
    session: requests.Session,
    metadata: EptMetadata,
    query: tuple[float, float, float, float],
) -> list[EptNode]:
    """Find every node overlapping a query box, at every depth.

    Args:
        session: HTTP session to use.
        metadata: Acquisition metadata from ``fetch_metadata``.
        query: Horizontal query box ``(min_x, min_y, max_x, max_y)`` in the
            acquisition's CRS.

    Returns:
        Overlapping nodes with points, shallowest first.

    Raises:
        ProcessingError: EPT_METADATA_ERROR if the index is malformed or does
            not converge.
    """
    hierarchy: dict[str, int] = {}
    visited_pages: set[str] = set()

    def load_page(key: str) -> None:
        if key in visited_pages:
            raise ProcessingError(
                code="EPT_METADATA_ERROR",
                message="The USGS 3DEP index for this area is malformed.",
                suggestion="Select a different acquisition.",
                traceback=f"{metadata.url}: hierarchy page {key} referenced twice",
            )
        if len(visited_pages) >= MAX_HIERARCHY_PAGES:
            raise ProcessingError(
                code="EPT_METADATA_ERROR",
                message="The USGS 3DEP index for this area is too large to read.",
                suggestion="Use a smaller domain.",
                traceback=f"{metadata.url}: exceeded {MAX_HIERARCHY_PAGES} pages",
            )
        visited_pages.add(key)
        url = f"{metadata.base_url}/ept-hierarchy/{key}.json"
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            page = response.json()
        except Exception as e:
            raise ProcessingError(
                code="EPT_METADATA_ERROR",
                message="Could not read the USGS 3DEP index for this area.",
                suggestion="Try again shortly.",
                traceback=f"{url}: {e}",
            ) from e
        # The page must overwrite the -1 placeholder that pointed at it, so
        # merge the page over what is already known, never the other way round.
        hierarchy.update(page)

    load_page("0-0-0-0")

    selected: list[EptNode] = []
    pending = ["0-0-0-0"]

    while pending:
        key = pending.pop()
        depth, x, y, z = (int(part) for part in key.split("-"))

        bounds = node_bounds(metadata.bounds, depth, x, y, z)
        if not _overlaps_2d(bounds, query):
            continue

        count = hierarchy.get(key)
        if count is None:
            continue

        # Depth is judged only for nodes the index actually claims. Children are
        # pushed speculatively one level past whatever exists, so checking
        # before the lookup above would fail a tree whose deepest real level is
        # exactly MAX_DEPTH.
        if depth > MAX_DEPTH:
            raise ProcessingError(
                code="EPT_METADATA_ERROR",
                message="The USGS 3DEP index for this area is malformed.",
                suggestion="Select a different acquisition.",
                traceback=f"{metadata.url}: hierarchy deeper than {MAX_DEPTH}",
            )

        if count == -1:
            # A placeholder: this node's real count and its children live in
            # their own index page. Load it and reconsider the same key.
            load_page(key)
            resolved = hierarchy.get(key)
            if resolved is None or resolved == -1:
                raise ProcessingError(
                    code="EPT_METADATA_ERROR",
                    message="The USGS 3DEP index for this area is malformed.",
                    suggestion="Select a different acquisition.",
                    traceback=(
                        f"{metadata.url}: page {key} did not resolve its own count"
                    ),
                )
            pending.append(key)
            continue

        if count > 0:
            selected.append(
                EptNode(
                    key=key,
                    depth=depth,
                    count=count,
                    bounds=bounds,
                    base_url=metadata.base_url,
                )
            )

        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    pending.append(
                        f"{depth + 1}-{x * 2 + dx}-{y * 2 + dy}-{z * 2 + dz}"
                    )

    selected.sort(key=lambda node: (node.depth, node.key))
    return selected


def fetch_nodes(
    session: requests.Session,
    nodes: list[EptNode],
    max_workers: int = MAX_FETCH_WORKERS,
) -> Iterator[tuple[EptNode, laspy.LasData]]:
    """Download and decode nodes, overlapping network with decoding.

    Downloads run on a thread pool because they are network-bound and release
    the interpreter lock; decoding happens on the consuming thread because LAZ
    decompression holds the lock and gains nothing from more threads. The queue
    between them bounds how many downloaded files are resident at once.

    Args:
        session: HTTP session to use.
        nodes: Nodes to read.
        max_workers: Concurrent downloads.

    Yields:
        ``(node, points)`` in completion order.

    Raises:
        ProcessingError: EPT_NODE_MISSING if a node the index promised is
            absent, EPT_FETCH_FAILED if it cannot be downloaded.
    """
    if not nodes:
        return

    results: queue.Queue = queue.Queue(maxsize=_NODE_QUEUE_DEPTH)
    stop = threading.Event()

    def download(node: EptNode) -> None:
        if stop.is_set():
            return
        try:
            response = session.get(node.data_url, timeout=300)
            if response.status_code == 404:
                # The index said this node holds points, so a missing file is a
                # hole in the data we would otherwise pass off as complete.
                raise ProcessingError(
                    code="EPT_NODE_MISSING",
                    message=(
                        "Part of the USGS 3DEP lidar for this area is missing "
                        "from the USGS archive."
                    ),
                    suggestion=("Try again later, or select a different acquisition."),
                    traceback=f"404 for {node.data_url} (index count {node.count})",
                )
            response.raise_for_status()
            results.put((node, response.content))
        except ProcessingError as e:
            results.put(e)
        except Exception as e:
            results.put(
                ProcessingError(
                    code="EPT_FETCH_FAILED",
                    message="Could not download USGS 3DEP lidar for this area.",
                    suggestion="Try again shortly.",
                    traceback=f"{node.data_url}: {e}",
                )
            )

    def produce() -> None:
        # The sentinel must be enqueued whatever happens. The consumer waits on
        # results.get() with no timeout, so a producer that dies without one —
        # a thread that cannot start under memory pressure, say — would park the
        # job until Cloud Run times it out, with the traceback going nowhere but
        # this daemon thread's stderr.
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(download, nodes))
        except BaseException as e:
            results.put(
                ProcessingError(
                    code="EPT_FETCH_FAILED",
                    message="Could not download USGS 3DEP lidar for this area.",
                    suggestion="Try again shortly.",
                    traceback=f"node download pool failed: {e!r}",
                )
            )
        finally:
            results.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    try:
        while True:
            item = results.get()
            if item is None:
                break
            if isinstance(item, ProcessingError):
                raise item
            node, payload = item
            # Decoding belongs to the same ladder as the download: a truncated
            # or corrupt node is a bad file from the archive, not a bug here,
            # and it deserves the same message as a failed fetch.
            try:
                points = laspy.read(io.BytesIO(payload))
            except Exception as e:
                raise ProcessingError(
                    code="EPT_FETCH_FAILED",
                    message="USGS 3DEP lidar for this area could not be read.",
                    suggestion="Try again shortly.",
                    traceback=f"{node.data_url}: {e}",
                ) from e
            yield node, points
    finally:
        stop.set()
        # Drain so the producer never blocks on a full queue while unwinding.
        while producer.is_alive():
            try:
                results.get(timeout=0.1)
            except queue.Empty:
                pass
        producer.join(timeout=5)
