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
CPU-bound across a fleet of worker processes; a grid job is neither.

## No PDAL

EPT nodes are ordinary LAZ files, which `laspy` reads directly, so the only
thing PDAL would add is octree traversal — about a hundred lines in `ept.py`.

That matters because PDAL is expensive to host. The PyPI `pdal` package is an
sdist that builds against libpdal, and Debian dropped `pdal` after bullseye, so
adding it to a `python:3.13-slim` image means a conda/micromamba layer. Every
other worker here is an eight-line uv Dockerfile, and this one stays that way.

If COPC output is ever wanted, that calculus changes — a COPC writer does need
the native stack — and this service is where that would be isolated.

## Output is a partitioned Parquet dataset

`<id>/cloud.parquet/tile_x=<i>/tile_y=<j>/part-*.parquet`, plus a `_metadata`
footer carrying every row group's statistics and a `_manifest.json`. Not a
single LAZ, and not COPC.

Partitions *are* the output, which is what makes one pass enough. COPC needs a
second pass not for its LOD but for physical grouping: every node must be one
contiguous byte range inside one file. Independent partition files let a point
be placed and given its level together, so nothing spills and nothing is
assembled whole. Peak memory stopped tracking the point count as a result — a
16 km² fetch peaks at 1.4 GB and a 64 km² one at 1.5 GB.

Each part is written one row group per LOD level so a `lod <= k` filter prunes
on statistics; pushdown prunes row groups, not rows. The pyramid is voxel-based,
so a coarse level samples vertical structure rather than collapsing each column
to whichever point happened to arrive first.

Measured against the LAZ writer it replaces, at 16 km² / 343.6M points: 630s →
126s and 2.42 GB → 2.76 GB. `gps_time` is not stored — it was 23% of the file
and nothing reads it. Colour is not stored either, which is a real limitation:
a colour-carrying source is promoted to LAS format 7 or 8 so the LAZ path never
drops RGB, and this schema has nowhere to put it.

## The point budget

`LAKITU_MAX_POINTS` (default 200M) bounds a fetch. It was sized when the output
had to fit in memory, which is no longer true — 64 km² is 1.04B points and
writes in 488s — so the number is now a policy choice about job duration rather
than a memory guard. The budget is checked
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

## Why processes, not threads

An earlier version of this file claimed LAZ decode "holds the GIL" and used that
to justify keeping decode on the consuming thread. The claim is false as stated:
`laspy.read` with no backend argument selects `LazBackend.LazrsParallel`, which
decompresses on a rayon pool.

The conclusion was accidentally right, for a different reason. **LAZ parallelism
is per chunk**, chunks hold 50,000 points, and real 3DEP nodes hold 20-37k — one
chunk each, with nothing to split. Eight concurrent decoders measured **1.00
effective cores**. So threads genuinely cannot parallelise decode, and the only
axis left is across nodes in separate processes.

Threading the chain anyway cost **+41% CPU for ~17% wall**, GIL contention
burning futex traffic under gVisor. Moving it to processes gave 30% wall for +4%
CPU. Two process pools do the work now: `chain.py` decodes, reprojects, clips
and normalizes one node per task, and `parquet_writer.py` runs tile-pinned
workers that own LOD assignment, encode and upload for the tiles hashed to them.
Tiles are pinned because a tile's LOD grids must survive its repeated flushes.

Download concurrency is a **separate knob** from chain concurrency. Downloads
are network-bound and were never the limiter until the chain got fast; then
fetch wait hit 54s of a 167s job until `LAKITU_DOWNLOAD_WORKERS` was raised.

Worker counts are sharply peaked at the core count and must track the Cloud Run
vCPU allocation — 3 chain workers on 2 vCPU cost 2.4x the CPU of 2 for identical
output. `os.cpu_count()` cannot be used to infer it: it reports host cores, not
the quota, and read 4 on a 2 vCPU service. Hence the explicit
`LAKITU_WRITE_WORKERS` / `LAKITU_CHAIN_WORKERS` config.

At 16 km², 8 vCPU is 126s against 234s at 4 vCPU and 486s at 2 vCPU — 20% more
vCPU-seconds than 4 vCPU for 1.7x the speed, and cheaper than the LAZ writer at
every shape.

Cloud Tasks cancels an attempt at its dispatch deadline (600s by default for an
HTTP target) and retries it, and this worker treats any retry as terminal, so
that deadline — not Cloud Run's timeout — is the real ceiling on a fetch. The
service's timeout is set to match it rather than exceed it. Measured fetches run
about three minutes at the point-budget ceiling, so this has not been close to
binding; if it ever is, the answer is a smaller `LAKITU_MAX_POINTS`.


## Correctness notes

**Depth.** EPT is additive: a node holds a coarse sample of its own volume and
its children hold the rest. Full density is every overlapping node at every
depth, not the deepest level.

**Never test z.** The octree subdivides in three axes, so a deep node covers one
thin elevation slice. Domains have no elevation extent and want every slice —
an AABB test including z selects nothing at all. `_overlaps_2d` is horizontal
on purpose.

**Point formats must be normalized.** Two acquisitions can carry different LAS
point formats, and so different dimensions, which a single output schema cannot
represent. Everything is converted to one canonical format (`lib.laz`) before
writing. `OriginId` is dropped: it indexes one acquisition's own source-file
list, so after a merge the same value means two different flightlines.

**Coordinates are not identity.** Distinct returns do share a cubic millimetre —
5,697 pairs in the two-acquisition seam fixture, differing in intensity,
classification and source id. A de-duplication check has to compare every stored
attribute, not position.

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

The datum is deliberately **not recorded on the resource**. Nothing consumes it:
every use of z in FastFuels is a difference — #330's CHM is height above ground
from ground points in the same cloud, so the datum cancels — and no path
compares a point cloud's absolute z against anything else's. It is also not
lost: `source.datasets` names the acquisitions, so each `ept.json` can be
re-read if an absolute-elevation use ever appears. Sampling 40 of the catalog's
2,277 acquisitions found *none* declaring `srs.vertical`, so a field for it
would have read null on essentially every 3DEP cloud regardless.

**Known limitation:** a few USGS acquisitions store z in US survey feet, and
`ept.json` carries no reliable flag for it. `georeference.bounds` exposes the z
range, so a wildly wrong elevation is visible, but nothing detects it
automatically.
