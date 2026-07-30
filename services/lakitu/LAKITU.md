# Lakitu design notes

Contributor-facing notes on why this service looks the way it does. User-facing
documentation lives in OpenAPI and on the docs site.

## Why a separate service

Griddle is grid-shaped end to end: it loads a document from `grids-v2`, its
dispatch returns an `xr.Dataset`, and the result is written as zarr with chunk
and band bookkeeping. A point cloud job shares none of that — a different
collection, a different artifact, different completion fields — so hosting it
in griddle would mean branching the entry point on payload type and running two
unrelated job contracts through one deployment.

Separating it also lets the two be sized independently. A 3DEP fetch is
network-bound and holds its whole output in memory; a grid job is neither.

## No PDAL

EPT nodes are ordinary LAZ files, which `laspy` reads directly, so the only
thing PDAL would add is octree traversal — about a hundred lines in `ept.py`.

That matters because PDAL is expensive to host. The PyPI `pdal` package is an
sdist that builds against libpdal, and Debian dropped `pdal` after bullseye, so
adding it to a `python:3.13-slim` image means a conda/micromamba layer. Every
other worker here is an eight-line uv Dockerfile, and this one stays that way.

If COPC output is ever wanted, that calculus changes — a COPC writer does need
the native stack — and this service is where that would be isolated.

## Memory is the binding constraint

A LAZ writer has to seek back on close to backfill the header and chunk table.
GCS streams are not seekable and a worker's local filesystem is RAM, so the
output is assembled in an in-memory buffer and peak memory tracks the point
count.

`LAKITU_MAX_POINTS` (default 200M) is what bounds it. The budget is checked
twice: the API rejects a request whose catalog-derived estimate is over, and the
worker re-checks from the octree index *before downloading anything*. The second
check is much the sharper of the two, since it counts the nodes that actually
exist rather than assuming a uniform density across a published extent.

Neither check is exact, and the worker's has one subtlety worth knowing. Nodes
are selected by bounding box but kept by contribution polygon, so on an
irregular seam two acquisitions' boxes both approximate the whole domain.
Summing raw node counts would charge the domain to each of them and roughly
double the total — enough to reject a fetch that was well inside the budget — so
each acquisition's count is scaled by its contribution's share of its own box.

Cloud Run sizing follows from measurement rather than guesswork:

| Stage | Throughput | Scales with more vCPU? |
|---|---|---|
| Node download | network-bound | no, but threads help |
| LAZ decode (lazrs) | ~8.5M pts/s | **no — holds the GIL** |
| pyproj transform | ~15M pts/s | yes, but it is ~4% of the work |
| LAZ write | ~29M pts/s | no, inherently serial |

So the fetch runs downloads on a thread pool and does decode, transform, and
write on the consuming thread. A process pool would multiply the memory that is
already the limit. 8 GiB / 2 vCPU is the cheapest Cloud Run tier with real
headroom over the ~3 GB peak; 16 GiB would force 4 vCPU, buying two cores the
GIL cannot use.

## Correctness notes

**Depth.** EPT is additive: a node holds a coarse sample of its own volume and
its children hold the rest. Full density is every overlapping node at every
depth, not the deepest level.

**Never test z.** The octree subdivides in three axes, so a deep node covers one
thin elevation slice. Domains have no elevation extent and want every slice —
an AABB test including z selects nothing at all. `_overlaps_2d` is horizontal
on purpose.

**Point formats must be normalized.** laspy refuses to write a record whose
point format differs from the file header's, and it compares extra dimensions
too, so two acquisitions can disagree even at the same format id. Everything is
converted to one canonical format (`lib.laz`) before writing. `OriginId` is
dropped: it indexes one acquisition's own source-file list, so after a merge the
same value means two different flightlines.

**Seams must not double up.** Acquisitions overlap freely — a real domain
measured 144% when per-acquisition coverage was summed. `lib.entwine` assigns
each acquisition a disjoint *contribution* polygon (what it covers minus what
earlier ones already do) and each is read only within its own. Masking is
boundary-exclusive: on a shared edge, dropping a point is safer than
duplicating one.

**Densify before reprojecting.** A straight domain edge is a curve in the
acquisition's CRS, so transforming only the corners yields a query box that
clips the bulge. Edges are segmented first.

## Vertical datum

3DEP mostly publishes orthometric heights on NAVD88, but that is a property of
each survey, not of the program. The domain CRS is horizontal, so there is
nothing to transform and elevations pass through exactly as published.

`georeference.vertical_crs` therefore *labels* rather than converts, and it is
filled in only from what a survey's own `ept.json` declares in `srs.vertical`.
Most declare nothing, so null is the common answer — which is the honest one,
since an uploaded cloud may carry ellipsoidal heights and the two would sit tens
of metres apart with nothing to say why. A merge across surveys that disagree
reports null too, rather than picking a winner.

**Known limitation:** a few USGS acquisitions store z in US survey feet, and
`ept.json` carries no reliable flag for it. `georeference.bounds` exposes the z
range, so a wildly wrong elevation is visible, but nothing detects it
automatically.
