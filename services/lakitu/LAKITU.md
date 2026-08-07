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

`<id>/cloud.parquet/tile_x=<i>/tile_y=<j>/part-*.parquet`, plus a
`_manifest.json` carrying the tiling and coordinate scaling. Not a single LAZ,
and not COPC.

Partitions *are* the output, which is what makes one pass enough. COPC needs a
second pass not for its LOD but for physical grouping: every node must be one
contiguous byte range inside one file. Independent partition files let a point
be placed and given its level together, so nothing spills and nothing is
assembled whole. Peak memory stopped tracking the point count as a result — a
16 km² fetch peaks at 1.4 GB and a 64 km² one at 1.5 GB.

Each part is written one row group per LOD level so a `lod <= k` filter prunes
on statistics; pushdown prunes row groups, not rows. The pyramid is a stride —
level `k` holds 1 in `4**(5-k)` of a tile — so the levels are nested, exactly
4× apart, and unbiased: a preview's class mix matches the full cloud. It was a
per-tile voxel grid until the grid was found to cost `side**2 * nz` cells per
live tile, 6.4 GB across a 64 km² domain, and to skew the class mix it sampled.

Which level a given point lands in is not reproducible across runs: the stride
runs over arrival order within a flush, so two runs of the same domain put
1,022,619 and 1,021,204 points in level 0. The totals are exact and the sample
is unbiased either way; nothing should depend on a preview being byte-identical.

## A tile is one file, and that is scheduled

Tiles are the partition; part files are how many times a tile was written. Those
came apart badly. The parent used to flush the largest live tile whenever its
buffer filled, which makes file count scale with **data volume** rather than with
area — all records pass through one fixed window, so more data means more
flushes, and more area means more tiles live per flush. Measured at 64 km²:
2,953 files over 260 tiles, up to 33 for a single tile, against 157 over 72 at
16 km².

That is paid on every read. A tile split 33 ways has its coarsest level in 33
places, so a whole-cloud `lod <= 0` preview — about a million points — spent
**243 s** opening 3,210 objects.

`plan_nodes` now returns, with the node order, the index of the last node that
can put a point in each tile; the writer holds a tile until that node has
arrived and then writes it whole. Two things make that possible:

- **Coarse nodes are read before the sweep.** A node's span halves with depth,
  so shallow ones cover far more than a tile: at 64 km², depth ≤ 9 is 0.2% of
  nodes and 3.25% of points but spans up to 400 km. Left in the sweep, each one
  reopens tiles the sweep has passed and no tile is ever final.
- **One order across all acquisitions.** Ordering within each acquisition sweeps
  the grid once per acquisition and reopens every seam tile.

Nodes arrive in download-completion order, not plan order, so the writer tracks
a watermark — the highest node index with no gap below it — and a tile is final
only when its last toucher is under that. Morton order over the tiles was tried
and is worse than row-major: its quads must be descended and returned to, so
tiles on a quad boundary stay open across the recursion.

Measured at 64 km², identical point counts across every arm:

| flush | budget | files | per tile | peak RSS | wall |
|---|---|---|---|---|---|
| eviction | 192 MiB | 2,953 | 11.4 | 2.72 GB | 317 s |
| schedule | 192 MiB | 1,069 | 4.1 | 2.96 GB | 309 s |
| **schedule** | **512 MiB** | **449** | **1.7** | **3.92 GB** | **365 s** |
| schedule | 1024 MiB | 359 | 1.4 | 5.12 GB | 385 s |

The schedule is free; the budget is not. 512 MiB ships because it costs 15% wall
for 6.6× fewer files, and takes the `lod <= 0` preview from 243 s to **26.6 s**.
Size a budget from measured RSS rather than from the budget: buffers hold lists
of small arrays and `np.concatenate` doubles at flush, so RSS grew about 3× the
budget increase. The floor is above 260 because `MAX_TILE_BYTES` still splits the
densest tiles, which is the part-file size cap working as intended.

Measured against the LAZ writer it replaces, at 16 km² / 343.6M points: 630s →
113s and 2.42 GB → 2.23 GB. `gps_time` is not stored — it was 23% of the file
and nothing reads it. Colour is not stored either, which is a real limitation:
a colour-carrying source is promoted to LAS format 7 or 8 so the LAZ path never
drops RGB, and this schema has nowhere to put it.

## How many points a fetch reads

There is no cap. `LAKITU_MAX_POINTS` used to bound a fetch at 200M, sized when
the whole output had to fit in memory. Parquet is written out-of-core, so that
reason is gone and the limit was removed rather than re-derived: 64 km² is 1.04B
points and writes in 488s. What still bounds a fetch is the Cloud Tasks dispatch
deadline (see below), which is a real ceiling rather than a proxy for one.

Two estimates survive the removal, both advisory. The API reports
`estimated_point_count` from catalog density over the domain area, for callers
sizing a request. The worker computes its own from the octree index and logs it,
which is sharper because it counts the nodes that actually exist.

The worker's has one subtlety worth knowing. Nodes are selected by bounding box
but kept by contribution polygon, so on an irregular seam two acquisitions'
boxes both approximate the whole domain. Summing raw node counts would charge
the domain to each of them and roughly double the total, so each acquisition's
count is scaled by its contribution's share of its own box.

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
and normalizes one node per task, and `lib/pointcloud/writer.py` runs workers
that take any flush off one shared queue and carry it to GCS.

Neither pool is partitioned. The write workers used to own a fixed subset of
tiles, because a tile's LOD grids had to survive its repeated flushes; a stride
carries no state between flushes, so nothing is owned. Pinning is actively wrong
once points arrive in spatial order, since consecutive flushes are neighbouring
tiles and a hash would land them on the same worker. The chain pool likewise
serves every acquisition rather than one per acquisition — the per-source
transformer and clip are task arguments, not initializer arguments — because one
pool per acquisition forces one node order per acquisition, and the tile
schedule needs a single order across all of them.

Download concurrency is a **separate knob** from chain concurrency. Downloads
are network-bound and were never the limiter until the chain got fast; then
fetch wait hit 54s of a 167s job until `LAKITU_DOWNLOAD_WORKERS` was raised.

Worker counts must track the Cloud Run vCPU allocation, but not by matching it.
At 2 vCPU they are sharply peaked at the core count — 3 chain workers there cost
2.4x the CPU of 2 for identical output. At 8 vCPU the opposite holds: the
shipping 6 chain + 8 write, which is 14 processes on 8 cores, beat every smaller
combination tried — 4+4 by 13%, 3+3 by 30%. Oversubscription stops being waste
once the parent thread is the limiter, because the parent's serial work sets the
floor and the workers only have to stay ahead of it.

`os.cpu_count()` cannot be used to infer the allocation anyway: it reports host
cores, not the quota, and read 4 on a 2 vCPU service. Hence the explicit
`LAKITU_WRITE_WORKERS` / `LAKITU_CHAIN_WORKERS` config.

At 16 km², measured together on one build: 140s at 8 vCPU against 234s at 4 and
486s at 2 — 20% more vCPU-seconds than 4 vCPU for 1.7x the speed, and cheaper
than the LAZ writer at every shape, because Cloud Run bills allocated vCPU x
request time rather than CPU consumed.

That curve is also where the ceiling shows. 2 → 4 vCPU is near-linear and 4 → 8
is not, because past roughly 4 vCPU the serial parent thread — routing every
point, concatenating every flush, draining both queues — sets the floor rather
than the workers. Work removed from a *worker* is worth almost nothing at 8 vCPU
and close to its full CPU cost at 2.

Cloud Tasks cancels an attempt at its dispatch deadline (600s by default for an
HTTP target) and retries it, and this worker treats any retry as terminal, so
that deadline — not Cloud Run's timeout — is the real ceiling on a fetch. The
service's timeout is set to match it rather than exceed it. With the point cap
removed this is the only ceiling left, and it is closer than it used to be:
64 km² / 1.04B points writes in 488s, about 80% of the deadline. A domain past
that fails at the deadline as a generic `UNEXPECTED_FAILURE`, since a retry is
indistinguishable from any other cause here. Raise the deadline and the
service's timeout together, or the extra time is never reached.


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
